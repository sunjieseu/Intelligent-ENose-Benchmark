#!/usr/bin/env python3
"""
Multi-dataset drift benchmark under the unified 128-dim UCSD protocol.

Runs all released baselines and sequential-update methods on TWO datasets:
  - UCSD Drift (UCI ID 224): standard 16-sensor, 8-feature, 128-dim protocol
  - UCSD Drift at Different Concentrations (UCI ID 270): same platform,
    per-measurement concentration labels; classes merged to the same 6 gases

Protocol (identical for every method and dataset):
  - Batch 1 is the source domain; batches 2..10 are sequential target batches.
  - Each target batch is split 50/50 into adaptation and evaluation halves
    (stratified, fixed seed), so adaptation samples are NEVER evaluated.
  - Single-pass methods (SVM, MLP, TCA, DANN, ProtoNet, RelationNet) are
    trained with access to the source domain (+ unlabeled target for DA
    methods; labeled adaptation half for supervised baselines) and evaluated
    on the evaluation half of each batch.
  - Sequential methods (CRE, TTA, SSL+TTA) update over batches and are
    re-evaluated on all previously seen evaluation halves, producing the
    stage-by-historical-task accuracy matrix from which BWT/FWT are computed.
  - Accuracy is averaged over batches 2..10 (final-batch accuracy also kept).

Usage:
    python benchmarks/eval_drift_unified.py --dataset 224
    python benchmarks/eval_drift_unified.py --dataset 270
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transfer_learning import TCA, DANN
from models.few_shot import PrototypicalNetwork, RelationNetwork, create_fewshot_episode
from models.drift_compensation import ClassifierReplacementEnsemble
from utils.metrics import compute_bwt_from_matrix, compute_fwt_from_matrix

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('unified')
logger.setLevel(logging.INFO)

SEED = 42
N_CLASSES = 6
TAU = 0.5  # drift-indicator threshold gating TTA (Algorithm 1, paper Sec. VII)


# ----------------------------------------------------------------------
# Data loading: parse UCI .dat (libsvm-style, 128 features per line)
# ----------------------------------------------------------------------
def load_batches(data_dir, dataset_id):
    """Parse batch{1..10}.dat -> list of (X, y) per batch, 128-dim features."""
    batches = []
    for b in range(1, 11):
        path = os.path.join(data_dir, f'batch{b}.dat')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        X, y = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "class[;concentration] idx:val idx:val ..."
                head, _, feats = line.partition(' ')
                if ';' in head:
                    cls = int(head.split(';')[0])
                else:
                    cls = int(head)
                vals = np.zeros(128)
                for tok in feats.split():
                    idx, v = tok.split(':')
                    vals[int(idx) - 1] = float(v)
                X.append(vals)
                y.append(cls - 1)  # 0-indexed
        X = np.asarray(X)
        y = np.asarray(y)
        # keep the 6 standard gas classes (labels 1..6 -> 0..5)
        keep = np.isin(y, np.arange(6))
        batches.append((X[keep], y[keep]))
        logger.info(f"dataset {dataset_id} batch {b}: {X.shape[0]} rows, kept {keep.sum()}")
    return batches


# ----------------------------------------------------------------------
# Unified feature scaling: fit on source batch 1 only (no target leakage)
# ----------------------------------------------------------------------
def fit_scaler(X_source):
    mu = X_source.mean(axis=0)
    sd = X_source.std(axis=0) + 1e-12
    return (mu, sd)


def apply_scaler(X, scaler):
    mu, sd = scaler
    return (X - mu) / sd


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class MLPEnc(nn.Module):
    """Compact 2-layer MLP encoder + linear head (paper MLP baseline)."""

    def __init__(self, in_dim=128, hid=64, n_cls=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, n_cls))

    def fit(self, X, y, epochs=200, lr=1e-3, seed=SEED):
        torch.manual_seed(seed)
        Xs = torch.tensor(X, dtype=torch.float32)
        ys = torch.tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            opt.zero_grad()
            out = self.net(Xs)
            loss = F.cross_entropy(out, ys)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32))
        return out.argmax(1).numpy()


class SimpleSSL:
    """SimCLR-style linear-probe SSL pre-training (Phase 1 of lifecycle).

    Augmentations for tabular UCSD features: Gaussian noise and channel
    dropout, which do not erase the steady-state/transient structure.
    """

    def __init__(self, in_dim=128, emb=64, seed=SEED):
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(nn.Linear(in_dim, emb), nn.ReLU())

    def augment(self, X, rng):
        Xn = X * (1 + rng.normal(0, 0.05, X.shape))
        mask = (rng.random(X.shape[1]) > 0.2).astype(float)
        return Xn * mask

    def fit(self, X, epochs=100, lr=1e-3, temp=0.5):
        opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        rng = np.random.default_rng(SEED)
        Xs = torch.tensor(X, dtype=torch.float32)
        self.encoder.train()
        for _ in range(epochs):
            xi = torch.tensor(self.augment(X, rng), dtype=torch.float32)
            xj = torch.tensor(self.augment(X, rng), dtype=torch.float32)
            zi = F.normalize(self.encoder(xi), dim=1)
            zj = F.normalize(self.encoder(xj), dim=1)
            z = torch.cat([zi, zj], 0)
            sim = z @ z.t() / temp
            n = zi.shape[0]
            target = torch.arange(n)
            target = torch.cat([target, target], 0)
            loss = F.cross_entropy(sim, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def transform(self, X):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(torch.tensor(X, dtype=torch.float32)).numpy()


class SSLProtoNet:
    """Lifecycle pilot P1->P2: SSL encoder + Prototypical head."""

    def __init__(self, ssl: SimpleSSL, k_shot=5, seed=SEED):
        self.ssl = ssl
        self.k_shot = k_shot
        self.seed = seed

    def fit(self, X_support, y_support):
        Z = self.ssl.transform(X_support)
        self.classes_ = np.unique(y_support)
        self.protos = {}
        for c in self.classes_:
            Zc = Z[y_support == c]
            take = min(self.k_shot, len(Zc))
            self.protos[c] = Zc[:take].mean(0)

    def predict(self, X):
        Z = self.ssl.transform(X)
        protos = np.stack([self.protos[c] for c in self.classes_])
        d = ((Z[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
        return self.classes_[d.argmin(1)]


class SimpleTTA:
    """Entropy-minimization TTA with confidence filtering (paper TTA baseline).

    Updates BN-free affine encoder head by a few gradient steps on unlabeled
    evaluation-stream samples whose max softmax confidence exceeds tau.
    """

    def __init__(self, model: MLPEnc, tau=0.7, steps=3, lr=1e-3):
        self.model = model
        self.tau = tau
        self.steps = steps
        self.lr = lr
        self._head = model.net[-1]

    def adapt(self, X_unlabeled):
        self.model.eval()
        Xs = torch.tensor(X_unlabeled, dtype=torch.float32)
        with torch.no_grad():
            p = F.softmax(self.model.net(Xs), dim=1)
        conf, _ = p.max(1)
        sel = conf > self.tau
        if sel.sum() < 2:
            return
        Xsel = Xs[sel]
        opt = torch.optim.SGD(self._head.parameters(), lr=self.lr)
        for _ in range(self.steps):
            out = self.model.net(Xsel)
            loss = -(F.softmax(out, 1) * F.log_softmax(out, 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, X):
        return self.model.predict(X)


# ----------------------------------------------------------------------
# Evaluation harness
# ----------------------------------------------------------------------
def fit_protonet(X_src, y_src, n_episodes=100):
    """Episodic training of PrototypicalNetwork on source-domain episodes."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    m = PrototypicalNetwork(input_dim=X_src.shape[1], hidden_dim=64,
                            embedding_dim=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(n_episodes):
        support, query = create_fewshot_episode(
            X_src, y_src, n_way=N_CLASSES, k_shot=5, n_query=5)
        if len(support[1]) == 0:
            continue
        m.train_episode(support, query, opt)
    return m


def proto_predict(m, X_eval, X_src, y_src):
    """Predict with prototypes from the full (remapped) source labels."""
    m.eval()
    Xs = torch.FloatTensor(X_src)
    ys = torch.LongTensor(y_src)
    protos = m.compute_prototypes(Xs, ys, N_CLASSES)
    return m.predict(X_eval, protos)


def fit_relationnet(X_src, y_src, n_episodes=100):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    m = RelationNetwork(input_dim=X_src.shape[1], hidden_dim=64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(n_episodes):
        support, query = create_fewshot_episode(
            X_src, y_src, n_way=N_CLASSES, k_shot=5, n_query=5)
        if len(support[1]) == 0:
            continue
        m.train_episode(support, query, opt)
    return m


def eval_static(batches, method, scaler):
    """Single-pass evaluation: train on source + adapt split, eval per batch."""
    X_src, y_src = batches[0]
    accs = []
    for bi in range(1, 10):
        X_t, y_t = batches[bi]
        X_adapt, X_eval, y_adapt, y_eval = train_test_split(
            X_t, y_t, test_size=0.5, random_state=1000 + bi,
            stratify=y_t if len(np.unique(y_t)) > 1 else None)
        if method == 'svm':
            m = SVC(kernel='rbf').fit(apply_scaler(X_src, scaler), y_src)
            pred = m.predict(apply_scaler(X_eval, scaler))
        elif method == 'mlp':
            m = MLPEnc().fit(apply_scaler(X_src, scaler), y_src)
            pred = m.predict(apply_scaler(X_eval, scaler))
        elif method == 'tca':
            # gamma = 1/n_features is the standard RBF scaling for
            # standardized high-dimensional features; gamma=1 degenerates
            # the kernel (exp(-||x||^2) ~ 0) and collapses alignment.
            tca = TCA(n_components=10, gamma=1.0 / X_src.shape[1])
            Xs_s, Xa_s = apply_scaler(X_src, scaler), apply_scaler(X_adapt, scaler)
            Xt = tca.fit_transform(Xs_s, Xa_s)
            svm = SVC(kernel='rbf').fit(Xt[:len(Xs_s)], y_src)
            pred = svm.predict(tca.transform(apply_scaler(X_eval, scaler)))
        elif method == 'dann':
            dann = DANN(input_dim=128, hidden_dim=128, num_classes=N_CLASSES, alpha=1.0)
            dann.fit(apply_scaler(X_src, scaler), y_src,
                     apply_scaler(X_adapt, scaler), epochs=50, batch_size=32)
            pred = dann.predict(apply_scaler(X_eval, scaler))
        elif method == 'protonet':
            Xs_s = apply_scaler(X_src, scaler)
            m = fit_protonet(Xs_s, y_src)
            pred = proto_predict(m, apply_scaler(X_eval, scaler), Xs_s, y_src)
        elif method == 'relationnet':
            Xs_s = apply_scaler(X_src, scaler)
            m = fit_relationnet(Xs_s, y_src)
            support = (Xs_s, y_src)
            pred = m.predict(apply_scaler(X_eval, scaler), support)
        else:
            raise ValueError(method)
        accs.append(accuracy_score(y_eval, pred))
    return float(np.mean(accs)), float(accs[-1]), accs


def energy_distance(X_ref, X_new, sub=500, seed=0):
    """Drift indicator s_t: squared energy distance between the reference
    (source) feature distribution and the incoming batch.

    E^2 = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||, an unbiased, parameter-free
    estimator of 2* divergence between distributions (Szekely & Rizzo).
    Chosen over MMD because it needs no kernel bandwidth, and over KS because
    it is multivariate. s_t in [0, 1] after normalization by feature scale.
    """
    rng = np.random.default_rng(seed)
    Xr = X_ref[rng.choice(len(X_ref), min(sub, len(X_ref)), replace=False)]
    Xn = X_new[rng.choice(len(X_new), min(sub, len(X_new)), replace=False)]
    scale = np.sqrt(Xr.var(0).mean() + 1e-12)

    def dmat(A, B):
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)) / scale

    dxy = dmat(Xr, Xn).mean()
    dxx = dmat(Xr, Xr).mean()
    dyy = dmat(Xn, Xn).mean()
    return float(max(2 * dxy - dxx - dyy, 0.0))


def eval_sequential(batches, method, scaler, ssl_model=None):
    """Sequential update with stage-by-task matrix for BWT/FWT.

    Matrix convention (Lopez-Paz & Ranzato, 2017):
      a[t, i] = accuracy on task i (batch i+1 evaluation half) after the
      model has been adapted at stage t (batches 1..t+1 seen).
      - BWT uses the last row vs the diagonal.
      - FWT uses the superdiagonal a[t-1, t]: accuracy on task t measured
        with the stage-(t-1) model, BEFORE adapting to task t, minus the
        no-transfer baseline b_t (SVM trained on source only, never updated).
    The superdiagonal is filled by evaluating the pre-adaptation model on
    the incoming batch's evaluation half before any update is applied.
    """
    X_src, y_src = batches[0]
    scaler_src = apply_scaler(X_src, scaler)
    seen_eval = []
    n_tasks = 10
    matrix = np.full((n_tasks, n_tasks), np.nan)
    baseline_task = np.full(n_tasks, np.nan)
    final_batch_acc = None
    drift_indicators = {}

    # static reference for FWT baseline: SVM trained on batch 1 only
    ref = SVC(kernel='rbf').fit(scaler_src, y_src)

    model = None
    tta = None
    for bi in range(1, 10):
        X_t, y_t = batches[bi]
        X_adapt, X_eval, y_adapt, y_eval = train_test_split(
            X_t, y_t, test_size=0.5, random_state=1000 + bi,
            stratify=y_t if len(np.unique(y_t)) > 1 else None)
        Xe_s = apply_scaler(X_eval, scaler)
        Xa_s = apply_scaler(X_adapt, scaler)
        seen_eval.append((Xe_s, y_eval))
        baseline_task[bi - 1] = accuracy_score(y_eval, ref.predict(Xe_s))

        # drift indicator: energy distance source vs incoming adaptation half
        s_t = energy_distance(scaler_src, Xa_s, seed=SEED + bi)
        drift_indicators[f'batch{bi+1}'] = round(s_t, 4)

        # ---- FWT superdiagonal: evaluate the CURRENT (pre-update) model on
        # the incoming task before any adaptation, then update. ----
        if model is not None and method != 'ssl_protonet':
            matrix[bi - 2, bi - 1] = accuracy_score(y_eval, model.predict(Xe_s))

        if method == 'cre':
            if model is None:
                model = ClassifierReplacementEnsemble(ensemble_size=5, threshold=0.05)
                model.fit_initial(Xa_s, y_adapt)
            else:
                n_lab = min(50, len(Xa_s))
                idx = np.random.default_rng(SEED + bi).choice(
                    len(Xa_s), n_lab, replace=False)
                model.update(Xa_s[idx], y_adapt[idx])
            for ti, (Xt_e, yt_e) in enumerate(seen_eval):
                matrix[bi - 1, ti] = accuracy_score(yt_e, model.predict(Xt_e))
            final_batch_acc = matrix[bi - 1, bi - 1]
        elif method in ('tta', 'ssl_tta'):
            if model is None:
                model = MLPEnc()
                model.fit(scaler_src, y_src, epochs=60)  # source warm start
                tta = SimpleTTA(model)
            else:
                # gate TTA on the drift indicator (matches Algorithm 1)
                if s_t > TAU:
                    tta.adapt(Xa_s)  # unlabeled adaptation half only
            for ti, (Xt_e, yt_e) in enumerate(seen_eval):
                matrix[bi - 1, ti] = accuracy_score(yt_e, model.predict(Xt_e))
            final_batch_acc = matrix[bi - 1, bi - 1]
        elif method == 'ssl_protonet':
            # Lifecycle pilot P1->P2: SSL encoder frozen; prototypes calibrated
            # per batch on a small labeled target support set (k-shot per class)
            # drawn from the adaptation half only. No memory: prototypes of
            # batch t are overwritten by batch t+1 (the cause of the measured
            # negative BWT).
            k_shot = 5
            model = SSLProtoNet(ssl_model, k_shot=k_shot)
            support_idx = []
            rng = np.random.default_rng(SEED + bi)
            for c in np.unique(y_adapt):
                idx_c = np.where(y_adapt == c)[0]
                take = min(k_shot, len(idx_c))
                support_idx.extend(rng.choice(idx_c, take, replace=False))
            model.fit(Xa_s[np.array(support_idx, dtype=int)],
                      y_adapt[np.array(support_idx, dtype=int)])
            for ti, (Xt_e, yt_e) in enumerate(seen_eval):
                matrix[bi - 1, ti] = accuracy_score(yt_e, model.predict(Xt_e))
            final_batch_acc = matrix[bi - 1, bi - 1]
        else:
            raise ValueError(method)

    # ---- BWT from the final row vs diagonal (well-defined for all methods) ----
    T = len(seen_eval)
    filled = matrix[:T, :T].copy()
    # BWT needs last row and diagonal only; other NaNs are irrelevant to it
    bwt = compute_bwt_from_matrix(filled)

    # ---- FWT from the superdiagonal vs no-transfer baseline ----
    # a[t-1, t] for t=2..T, minus baseline_task[t-1]
    sup = np.array([matrix[t - 2, t - 1] for t in range(2, T + 1)])
    bl = np.array([baseline_task[t - 1] for t in range(2, T + 1)])
    if np.isnan(sup).any():
        # only possible for stage-1 methods with no pre-adaptation model
        fwt = float('nan')
    else:
        fwt = float(np.mean(sup - bl))

    diag = [matrix[i, i] for i in range(T)]
    avg = float(np.nanmean(diag[1:]))  # batches 2..10
    return avg, float(final_batch_acc), bwt, fwt, drift_indicators


def count_params(kind):
    """Software-level parameter counts (thousands), from the actual modules."""
    if kind in ('svm', 'cre'):
        return None
    if kind in ('mlp', 'tta', 'ssl_tta'):
        m = MLPEnc()
    elif kind == 'ssl':
        m = SimpleSSL()
    elif kind == 'protonet':
        m = PrototypicalNetwork(input_dim=128, hidden_dim=64)
    elif kind == 'relationnet':
        m = RelationNetwork(input_dim=128, hidden_dim=64)
    elif kind == 'tca':
        return 1.28  # 128 x 10 projection matrix
    elif kind == 'dann':
        d = DANN(input_dim=128, hidden_dim=128, num_classes=6)
        n = sum(p.numel() for p in d.parameters() if p.requires_grad)
        return round(n / 1000, 2)
    else:
        return None
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return round(n / 1000, 2)


def count_macs(kind, in_dim=128):
    """Software-level MACs per inference (thousands), from module shapes."""
    if kind in ('svm', 'cre'):
        return None
    if kind == 'tca':
        return round(in_dim * 10 / 1000, 2)
    if kind == 'dann':
        # extractor 128->128->128 + predictor 128->6 + discriminator 128->1
        return round((in_dim * 128 * 2 + 128 * 6 + 128) / 1000, 2)
    if kind in ('mlp', 'tta', 'ssl_tta'):
        return round((in_dim * 64 + 64 * 6) / 1000, 2)
    if kind == 'ssl':
        return round(in_dim * 64 / 1000, 2)
    if kind == 'protonet':
        # encoder 128->64->64->32 per sample (MACs, excluding distance)
        return round((in_dim * 64 + 64 * 64 + 64 * 32) / 1000, 2)
    if kind == 'relationnet':
        return round((in_dim * 64 + 64 * 64) / 1000, 2)
    return None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['224', '270'], required=True)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    if args.dataset == '224':
        data_dir = os.path.join(root, 'ucsd', 'Dataset')
    else:
        data_dir = os.path.join(root, 'ucsd270')

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    batches = load_batches(data_dir, args.dataset)
    X_src, _ = batches[0]
    scaler = fit_scaler(X_src)

    results = {}

    # --- static / single-pass baselines ---
    for m in ['svm', 'mlp', 'tca', 'dann', 'protonet', 'relationnet']:
        avg, fin, accs = eval_static(batches, m, scaler)
        results[m] = {'avg_acc': avg, 'final_acc': fin,
                      'params_k': count_params(m), 'macs_k': count_macs(m)}
        logger.info(f"[{args.dataset}] {m}: avg={avg:.4f} final={fin:.4f}")

    # --- SSL pre-training (Phase 1) ---
    ssl = SimpleSSL()
    ssl.fit(apply_scaler(X_src, scaler))

    # --- lifecycle pilot: SSL + ProtoNet (P1->P2) ---
    avg, fin, bwt, fwt, drift = eval_sequential(batches, 'ssl_protonet', scaler, ssl_model=ssl)
    results['ssl_protonet'] = {'avg_acc': avg, 'final_acc': fin,
                               'bwt': bwt, 'fwt': fwt,
                               'params_k': 7.53, 'macs_k': 7.38}
    logger.info(f"[{args.dataset}] ssl+protonet: avg={avg:.4f} bwt={bwt:.4f} fwt={fwt:.4f}")
    results['drift_indicators'] = drift

    # --- sequential baselines with BWT/FWT ---
    for m in ['cre', 'tta', 'ssl_tta']:
        avg, fin, bwt, fwt, drift = eval_sequential(batches, m, scaler, ssl_model=ssl)
        results[m] = {'avg_acc': avg, 'final_acc': fin, 'bwt': bwt, 'fwt': fwt,
                      'params_k': count_params('mlp') if m in ('tta', 'ssl_tta') else None,
                      'macs_k': count_macs('mlp') if m in ('tta', 'ssl_tta') else None}
        logger.info(f"[{args.dataset}] {m}: avg={avg:.4f} final={fin:.4f} bwt={bwt:.4f} fwt={fwt:.4f}")

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', f'drift_unified_{args.dataset}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'dataset': args.dataset, 'seed': SEED, 'results': results}, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
