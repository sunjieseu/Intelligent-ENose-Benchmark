#!/usr/bin/env python3
"""
Second-dataset validation on GSALC (UCI ID 1081, CQU low-concentration array).

Distinct platform from UCSD drift: 10 MOS sensors (vs 16), 6 gases at ppb
level, 90 samples. Used to test whether benchmark conclusions transfer across
arrays rather than to re-rank the UCSD protocol.

Protocol:
  - Features: the SAME 8 statistical descriptors per sensor as the UCSD
    128-dim protocol, applied to each sensor's 900-point response curve,
    giving 10 x 8 = 80-dim vectors (sensor-count is the only difference).
  - Static split: stratified 60/30 train/test (small dataset).
  - Concentration-drift split: train on 50+100 ppb, test on 200 ppb only,
    which simulates a concentration-induced distribution shift.
  - Few-shot: 5-way 3-shot episodes drawn from held-out gases; support and
    query pools strictly separated.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transfer_learning import TCA, DANN
from models.few_shot import PrototypicalNetwork, RelationNetwork, create_fewshot_episode

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('gsalc')
logger.setLevel(logging.INFO)

SEED = 42
N_SENSORS = 10
PTS_PER_SENSOR = 900


def extract_8d(curve):
    """The 8 UCSD-protocol descriptors on a single response curve."""
    c = np.asarray(curve, dtype=float)
    n = len(c)
    baseline = np.median(c[: max(n // 10, 1)])
    max_resp = np.max(c) - baseline
    rel_max = max_resp / (abs(baseline) + 1e-10)
    slope = np.mean(np.diff(c)) if n > 1 else 0.0
    integral = np.trapezoid(c) if n > 1 else 0.0
    max_idx = int(np.argmax(c))
    time_to_max = max_idx / max(n - 1, 1)
    n_tail = max(int(n * 0.1), 1)
    steady = np.mean(c[-n_tail:]) - baseline
    n_rec = max(int(n * 0.2), 2)
    rec_slope = np.mean(np.diff(c[-n_rec:])) if n >= n_rec else 0.0
    area_ratio = integral / (abs(max_resp) + 1e-10)
    return [max_resp, rel_max, slope, integral, time_to_max, steady, rec_slope, area_ratio]


def load_gsalc(csv_path):
    df = pd.read_csv(csv_path, header=None)
    y = df.iloc[:, 0].astype(str).values
    conc = df.iloc[:, 1].astype(str).str.replace('ppb', '').astype(int).values
    curves = df.iloc[:, 2:].values
    X = []
    for row in curves:
        row = np.asarray(row, dtype=float)
        feats = []
        for s in range(N_SENSORS):
            seg = row[s * PTS_PER_SENSOR:(s + 1) * PTS_PER_SENSOR]
            feats.extend(extract_8d(seg))
        X.append(feats)
    classes = sorted(set(y))
    cls_map = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_map[v] for v in y])
    return np.asarray(X), y_idx, conc, classes


class MLPEnc(torch.nn.Module):
    def __init__(self, in_dim, hid=64, n_cls=6):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, hid),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hid, n_cls))

    def fit(self, X, y, epochs=300, lr=1e-3):
        torch.manual_seed(SEED)
        Xs = torch.tensor(X, dtype=torch.float32)
        ys = torch.tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            F.cross_entropy(self.net(Xs), ys).backward()
            opt.step()
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()


def scale_fit(X):
    return (X.mean(0), X.std(0) + 1e-12)


def scale_app(X, s):
    return (X - s[0]) / s[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()

    X, y, conc, classes = load_gsalc(args.csv)
    logger.info(f"GSALC: {X.shape[0]} samples, {X.shape[1]} dims, classes={classes}")

    results = {}

    # ---------------- static split (stratified) ----------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=1 / 3, random_state=SEED,
                                          stratify=y)
    s = scale_fit(Xtr)
    Xtr_s, Xte_s = scale_app(Xtr, s), scale_app(Xte, s)

    results['svm'] = {'acc': accuracy_score(
        yte, SVC(kernel='rbf').fit(Xtr_s, ytr).predict(Xte_s))}
    results['mlp'] = {'acc': accuracy_score(
        yte, MLPEnc(X.shape[1]).fit(Xtr_s, ytr).predict(Xte_s))}

    # few-shot (support/query strictly separated via held-out gases)
    # gases 0,1 -> query pool; gases 2..5 -> episode pool
    ep_mask = np.isin(y, [2, 3, 4, 5])
    q_mask = np.isin(y, [0, 1])
    X_ep, y_ep = X[ep_mask], y[ep_mask]
    X_q, y_q = X[q_mask], y[q_mask]
    s_ep = scale_fit(X_ep)
    X_ep_s = scale_app(X_ep, s_ep)

    accs_p, accs_r = [], []
    for m_name in ('protonet', 'relationnet'):
        accs = []
        for ep in range(20):
            np.random.seed(SEED + ep)
            if m_name == 'protonet':
                m = PrototypicalNetwork(input_dim=X.shape[1], hidden_dim=64,
                                        embedding_dim=32)
            else:
                m = RelationNetwork(input_dim=X.shape[1], hidden_dim=64)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(60):
                support, query = create_fewshot_episode(
                    X_ep_s, y_ep, n_way=4, k_shot=3, n_query=3)
                if len(support[1]) == 0:
                    continue
                m.train_episode(support, query, opt)
            # evaluate on the held-out gas pool (support/query disjoint pools)
            if m_name == 'protonet':
                Xq_s = torch.FloatTensor(scale_app(X_q, s_ep))
                protos = m.compute_prototypes(Xq_s, torch.LongTensor(y_q),
                                              len(np.unique(y_q)))
                # build prototypes from a 3-shot subset of query gases, test on rest
                idx_by = [np.where(y_q == c)[0][:3] for c in np.unique(y_q)]
                sup_idx = np.concatenate(idx_by)
                protos = m.compute_prototypes(Xq_s[sup_idx], torch.LongTensor(y_q[sup_idx]),
                                              len(np.unique(y_q)))
                rest = np.setdiff1d(np.arange(len(y_q)), sup_idx)
                pred = m.predict(scale_app(X_q[rest], s_ep), protos)
                accs.append(accuracy_score(y_q[rest], pred))
            else:
                idx_by = [np.where(y_q == c)[0][:3] for c in np.unique(y_q)]
                sup_idx = np.concatenate(idx_by)
                rest = np.setdiff1d(np.arange(len(y_q)), sup_idx)
                support = (scale_app(X_q[sup_idx], s_ep), y_q[sup_idx])
                pred = m.predict(scale_app(X_q[rest], s_ep), support)
                accs.append(accuracy_score(y_q[rest], pred))
        results[m_name] = {'acc': float(np.mean(accs))}

    # ---------------- concentration-drift split ----------------
    tr = conc < 200
    te = conc == 200
    Xtr2, ytr2, Xte2, yte2 = X[tr], y[tr], X[te], y[te]
    s2 = scale_fit(Xtr2)
    Xtr2_s, Xte2_s = scale_app(Xtr2, s2), scale_app(Xte2, s2)

    results['drift_svm'] = {'acc': accuracy_score(
        yte2, SVC(kernel='rbf').fit(Xtr2_s, ytr2).predict(Xte2_s))}
    results['drift_mlp'] = {'acc': accuracy_score(
        yte2, MLPEnc(X.shape[1]).fit(Xtr2_s, ytr2).predict(Xte2_s))}

    # TCA with unlabeled target (drift split); gamma=1/d as standard scaling
    tca = TCA(n_components=10, gamma=1.0 / X.shape[1])
    Xt = tca.fit_transform(Xtr2_s, Xte2_s)
    svm = SVC(kernel='rbf').fit(Xt[:len(Xtr2_s)], ytr2)
    results['drift_tca'] = {'acc': accuracy_score(
        yte2, svm.predict(tca.transform(Xte2_s)))}

    # DANN with unlabeled target (drift split)
    dann = DANN(input_dim=X.shape[1], hidden_dim=64,
                num_classes=len(classes), alpha=1.0)
    dann.fit(Xtr2_s, ytr2, Xte2_s, epochs=50, batch_size=16)
    results['drift_dann'] = {'acc': accuracy_score(yte2, dann.predict(Xte2_s))}

    # few-shot under the concentration-drift split: episode pool = train
    # concentrations (50/100 ppb) of all gases; evaluation = 200 ppb pool,
    # so support and query come from disjoint concentration regimes.
    X_ep2 = Xtr2_s
    y_ep2 = ytr2
    for m_name in ('drift_protonet', 'drift_relationnet'):
        accs = []
        for ep in range(20):
            np.random.seed(SEED + ep)
            if m_name == 'drift_protonet':
                m = PrototypicalNetwork(input_dim=X.shape[1], hidden_dim=64,
                                        embedding_dim=32)
            else:
                m = RelationNetwork(input_dim=X.shape[1], hidden_dim=64)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(60):
                support, query = create_fewshot_episode(
                    X_ep2, y_ep2, n_way=6, k_shot=3, n_query=3)
                if len(support[1]) == 0:
                    continue
                m.train_episode(support, query, opt)
            # 3-shot support drawn from the 200-ppb evaluation pool per class
            idx_by = [np.where(yte2 == c)[0][:3] for c in np.unique(yte2)]
            sup_idx = np.concatenate(idx_by)
            rest = np.setdiff1d(np.arange(len(yte2)), sup_idx)
            if m_name == 'drift_protonet':
                Xq = torch.FloatTensor(Xte2_s)
                protos = m.compute_prototypes(Xq[sup_idx], torch.LongTensor(yte2[sup_idx]),
                                              len(np.unique(yte2)))
                pred = m.predict(Xte2_s[rest], protos)
            else:
                support = (Xte2_s[sup_idx], yte2[sup_idx])
                pred = m.predict(Xte2_s[rest], support)
            accs.append(accuracy_score(yte2[rest], pred))
        results[m_name] = {'acc': float(np.mean(accs))}

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'gsalc_validation.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'dataset': 'gsalc', 'seed': SEED,
                   'classes': classes, 'results': results}, f, indent=2)
    for k, v in results.items():
        logger.info(f"{k}: {v['acc']:.4f}")
    logger.info(f"Saved -> {out}")


if __name__ == '__main__':
    main()
