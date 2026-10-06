#!/usr/bin/env python3
"""Third drift dataset: UCI Twin Gas Sensor Arrays (UCI ID 361).

Cross-device drift evaluation under the same deployment-aware protocol as
the UCSD benchmark (eval_revision.py):

  - Source domain: board 1 (160 measurements, 4 gases x 10 conc. x 4 reps)
  - Target stages: boards 2..5 in order (cross-device + cross-day drift;
    board 4/5 carry 2 repetitions only = 80 samples)
  - Scaler fitted on the source board only (no target leakage)
  - Per-stage stratified 50/50 adaptation/evaluation halves
  - Same access groups and model configurations as Table 3 of the paper:
    source-only (SVM, MLP), unlabeled target (TCA, DANN, TTA, SSL+TTA),
    few labeled target (ProtoNet, RelationNet, SSL+ProtoNet,
    lifecycle-frozen), labeled updates (CRE), lifecycle-TTA, -replay
  - BWT/FWT over the stage-by-stage matrix; proto-validity diagnostic;
    energy-distance trigger values recorded per stage
  - Five seeds (7, 21, 42, 87, 123), mean +/- s.d.

Usage:
    python benchmarks/eval_twin.py --seeds 7 21 42 87 123
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmarks.eval_drift_unified as base
from benchmarks.eval_revision import predict_proto, predict_head, adapt_head, blend_prototypes
from models.transfer_learning import TCA, DANN
from models.few_shot import PrototypicalNetwork, RelationNetwork, create_fewshot_episode
from models.drift_compensation import ClassifierReplacementEnsemble
from datasets.twin_loader import load_twin

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('twin')
logger.setLevel(logging.INFO)

IN_DIM = 64
N_CLASSES = 4
SEED = 42


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    base.SEED = seed


def fit_protonet4(X_src, y_src, n_episodes=100, k_shot=5, seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = PrototypicalNetwork(input_dim=IN_DIM, hidden_dim=64, embedding_dim=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(n_episodes):
        support, query = create_fewshot_episode(
            X_src, y_src, n_way=N_CLASSES, k_shot=k_shot, n_query=5)
        if len(support[1]) == 0:
            continue
        m.train_episode(support, query, opt)
    return m


def fit_relationnet4(X_src, y_src, n_episodes=100, k_shot=5, seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = RelationNetwork(input_dim=IN_DIM, hidden_dim=64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(n_episodes):
        support, query = create_fewshot_episode(
            X_src, y_src, n_way=N_CLASSES, k_shot=k_shot, n_query=5)
        if len(support[1]) == 0:
            continue
        m.train_episode(support, query, opt)
    return m


def proto_predict4(m, X_eval, X_src, y_src):
    m.eval()
    protos = m.compute_prototypes(torch.FloatTensor(X_src),
                                  torch.LongTensor(y_src), N_CLASSES)
    return m.predict(X_eval, protos)


def eval_static_p(stages, method, scaler, seed, test_size=0.5):
    """Single-pass methods evaluated on each board's evaluation half."""
    X_src, y_src = stages[0]
    accs = []
    for bi in range(1, len(stages)):
        X_t, y_t = stages[bi]
        X_adapt, X_eval, y_adapt, y_eval = train_test_split(
            X_t, y_t, test_size=test_size, random_state=seed * 1000 + bi,
            stratify=y_t)
        if method == 'svm':
            m = SVC(kernel='rbf').fit(base.apply_scaler(X_src, scaler), y_src)
            pred = m.predict(base.apply_scaler(X_eval, scaler))
        elif method == 'mlp':
            m = base.MLPEnc(in_dim=IN_DIM, n_cls=N_CLASSES).fit(
                base.apply_scaler(X_src, scaler), y_src, seed=seed)
            pred = m.predict(base.apply_scaler(X_eval, scaler))
        elif method == 'tca':
            tca = TCA(n_components=10, gamma=1.0 / IN_DIM)
            Xs_s, Xa_s = base.apply_scaler(X_src, scaler), base.apply_scaler(X_adapt, scaler)
            Xt = tca.fit_transform(Xs_s, Xa_s)
            svm = SVC(kernel='rbf').fit(Xt[:len(Xs_s)], y_src)
            pred = svm.predict(tca.transform(base.apply_scaler(X_eval, scaler)))
        elif method == 'dann':
            torch.manual_seed(seed)
            dann = DANN(input_dim=IN_DIM, hidden_dim=128,
                        num_classes=N_CLASSES, alpha=1.0)
            dann.fit(base.apply_scaler(X_src, scaler), y_src,
                     base.apply_scaler(X_adapt, scaler), epochs=50, batch_size=32)
            pred = dann.predict(base.apply_scaler(X_eval, scaler))
        elif method == 'protonet':
            Xs_s = base.apply_scaler(X_src, scaler)
            m = fit_protonet4(Xs_s, y_src, seed=seed)
            pred = proto_predict4(m, base.apply_scaler(X_eval, scaler), Xs_s, y_src)
        elif method == 'relationnet':
            Xs_s = base.apply_scaler(X_src, scaler)
            m = fit_relationnet4(Xs_s, y_src, seed=seed)
            pred = m.predict(base.apply_scaler(X_eval, scaler), (Xs_s, y_src))
        else:
            raise ValueError(method)
        accs.append(accuracy_score(y_eval, pred))
    return float(np.mean(accs))


def eval_sequential_p(stages, method, scaler, seed, test_size=0.5, k_shot=5,
                      tau=0.5, blend_alpha=0.2, ssl_model=None):
    """Sequential evaluation over boards 2..5 with lifecycle variants.
    Generalization of benchmarks/eval_revision.py:eval_sequential_p to
    n_stages = len(stages) - 1 with stage matrix for BWT/FWT.
    (Kept as a thin wrapper; implementation is in eval_sequential_impl below.)"""
    return eval_sequential_impl(stages, method, scaler, seed, test_size,
                                k_shot, tau, blend_alpha, ssl_model)



def eval_sequential_impl(stages, method, scaler, seed, test_size, k_shot,
                          tau, blend_alpha, ssl_model):
    """Full self-contained reimplementation (single loop, correct matrix)."""
    X_src, y_src = stages[0]
    scaler_src = base.apply_scaler(X_src, scaler)
    ns = len(stages)
    matrix = np.full((ns - 1, ns - 1), np.nan)
    baseline_task = np.full(ns - 1, np.nan)
    proto_validity, energy_vals = {}, {}
    n_tta_triggers = 0
    ref = SVC(kernel='rbf').fit(scaler_src, y_src)
    model, tta = None, None
    seen_eval = []

    def predict_m(X):
        if isinstance(model, dict):
            if model['mode'] in ('lifecycle_frozen', 'lifecycle_replay'):
                return predict_proto(ssl_model, model, X)
            return predict_head(ssl_model, model, X)
        return model.predict(X)

    for bi in range(1, ns):
        X_t, y_t = stages[bi]
        X_adapt, X_eval, y_adapt, y_eval = train_test_split(
            X_t, y_t, test_size=test_size, random_state=seed * 1000 + bi,
            stratify=y_t)
        Xe_s = base.apply_scaler(X_eval, scaler)
        Xa_s = base.apply_scaler(X_adapt, scaler)
        seen_eval.append((Xe_s, y_eval))
        baseline_task[bi - 1] = accuracy_score(y_eval, ref.predict(Xe_s))
        s_t = base.energy_distance(scaler_src, Xa_s, seed=seed + bi)
        energy_vals[f'board{bi+1}'] = round(s_t, 4)

        if model is not None and method != 'ssl_protonet':
            matrix[bi - 2, bi - 1] = accuracy_score(y_eval, predict_m(Xe_s))

        if method == 'cre':
            if model is None:
                model = ClassifierReplacementEnsemble(ensemble_size=5, threshold=0.05)
                model.fit_initial(Xa_s, y_adapt)
            else:
                n_lab = min(50, len(Xa_s))
                idx = np.random.default_rng(seed + bi).choice(len(Xa_s), n_lab, replace=False)
                model.update(Xa_s[idx], y_adapt[idx])
        elif method in ('tta', 'ssl_tta'):
            if model is None:
                model = base.MLPEnc(in_dim=IN_DIM, n_cls=N_CLASSES)
                model.fit(scaler_src, y_src, epochs=60, seed=seed)
                tta = base.SimpleTTA(model)
            else:
                if s_t > tau:
                    tta.adapt(Xa_s)
                    n_tta_triggers += 1
        elif method == 'ssl_protonet':
            model = base.SSLProtoNet(ssl_model, k_shot=k_shot, seed=seed)
            rng = np.random.default_rng(seed + bi)
            support_idx = []
            for c in np.unique(y_adapt):
                idx_c = np.where(y_adapt == c)[0]
                take = min(k_shot, len(idx_c))
                support_idx.extend(rng.choice(idx_c, take, replace=False))
            model.fit(Xa_s[np.array(support_idx, dtype=int)],
                      y_adapt[np.array(support_idx, dtype=int)])
        elif method in ('lifecycle_frozen', 'lifecycle_tta', 'lifecycle_replay'):
            if model is None:
                rng = np.random.default_rng(seed + bi)
                support_idx = []
                for c in np.unique(y_adapt):
                    idx_c = np.where(y_adapt == c)[0]
                    take = min(k_shot, len(idx_c))
                    support_idx.extend(rng.choice(idx_c, take, replace=False))
                idx = np.array(support_idx, dtype=int)
                Xs, ys = Xa_s[idx], y_adapt[idx]
                Zs = ssl_model.transform(Xs)
                protos = {int(c): Zs[ys == c].mean(0) for c in np.unique(ys)}
                torch.manual_seed(seed)
                emb_dim = ssl_model.encoder[0].out_features
                head = nn.Linear(emb_dim, N_CLASSES)
                opt = torch.optim.Adam(head.parameters(), lr=1e-2)
                Zt = torch.tensor(Zs, dtype=torch.float32)
                yt = torch.tensor(ys, dtype=torch.long)
                head.train()
                for _ in range(150):
                    opt.zero_grad()
                    F.cross_entropy(head(Zt), yt).backward()
                    opt.step()
                head.eval()
                model = {'head': head, 'protos': protos, 'mode': method}
            else:
                if s_t > tau:
                    proto_validity[f'board{bi+1}'] = round(
                        accuracy_score(y_eval, predict_proto(ssl_model, model, Xe_s)), 4)
                    if method == 'lifecycle_tta':
                        adapt_head(ssl_model, model, Xa_s)
                        n_tta_triggers += 1
                    elif method == 'lifecycle_replay':
                        adapt_head(ssl_model, model, Xa_s)
                        blend_prototypes(ssl_model, model, Xa_s, blend_alpha)
                        n_tta_triggers += 1

        if method == 'lifecycle_frozen':
            proto_validity[f'board{bi+1}'] = round(
                accuracy_score(y_eval, predict_proto(ssl_model, model, Xe_s)), 4)
        for ti, (Xt_e, yt_e) in enumerate(seen_eval):
            matrix[bi - 1, ti] = accuracy_score(yt_e, predict_m(Xt_e))

    T = ns - 1
    filled = matrix[:T, :T]
    bwt = float(np.nanmean([filled[T - 1, i] - filled[i, i] for i in range(T - 1)]))
    sup = np.array([matrix[t - 2, t - 1] for t in range(2, T + 1)])
    bl = np.array([baseline_task[t - 1] for t in range(2, T + 1)])
    fwt = float('nan') if np.isnan(sup).any() else float(np.mean(sup - bl))
    diag = [matrix[i, i] for i in range(T)]
    return {'avg_acc': float(np.nanmean(diag[1:])), 'bwt': bwt, 'fwt': fwt,
            'proto_validity': proto_validity, 'energy_distance': energy_vals,
            'n_tta_triggers': n_tta_triggers,
            'final_acc': float(matrix[T - 1, T - 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    ap.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 21, 42, 87, 123])
    ap.add_argument('--test-size', type=float, default=0.5)
    ap.add_argument('--tau', type=float, default=0.5)
    args = ap.parse_args()

    X, y, unit, rep, gases = load_twin(os.path.join(args.data_root, 'twin'))
    logger.info(f'Twin dataset: {X.shape[0]} samples, {X.shape[1]} dims, gases={gases}')

    # source = board 1; target stages = boards 2..5 in order
    stages = [(X[unit == b], y[unit == b]) for b in sorted(set(unit.tolist()))]
    for i, (Xb, yb) in enumerate(stages):
        logger.info(f'board {i+1}: {len(yb)} samples')
    scaler = base.fit_scaler(stages[0][0])

    all_res = {}
    for seed in args.seeds:
        logger.info(f'=== seed {seed} ===')
        set_seeds(seed)
        torch.manual_seed(seed)
        res = {}
        for m in ['svm', 'mlp', 'tca', 'dann', 'protonet', 'relationnet']:
            res[m] = {'avg_acc': eval_static_p(stages, m, scaler, seed,
                                               args.test_size)}
            logger.info(f'{m}: {res[m]["avg_acc"]:.4f}')
        ssl = base.SimpleSSL(in_dim=IN_DIM, emb=64, seed=seed)
        ssl.fit(base.apply_scaler(stages[0][0], scaler))
        for m in ['cre', 'tta', 'ssl_tta', 'ssl_protonet',
                  'lifecycle_frozen', 'lifecycle_tta', 'lifecycle_replay']:
            r = eval_sequential_impl(stages, m, scaler, seed, args.test_size,
                                     k_shot=5, tau=args.tau, blend_alpha=0.2,
                                     ssl_model=ssl)
            res[m] = r
            logger.info(f'{m}: avg={r["avg_acc"]:.4f} bwt={r["bwt"]:.4f} '
                        f'fwt={r["fwt"]:.4f}')
        all_res[str(seed)] = res

    agg = {}
    for m in list(all_res[str(args.seeds[0])].keys()):
        entry = {}
        for key in ('avg_acc', 'bwt', 'fwt'):
            vals = [all_res[s].get(m, {}).get(key) for s in all_res]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                entry[key + '_mean'] = float(np.mean(vals))
                entry[key + '_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg[m] = entry
    out = {'dataset': 'twin_gas_sensor_arrays (UCI 361)',
           'protocol': ('source=board1; targets=boards 2..5; source-only '
                        'normalization; stratified 50/50 adapt/eval halves; '
                        f'tau={args.tau}'),
           'gases': gases, 'seeds': args.seeds, 'per_seed': all_res,
           'aggregate': agg}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'twin_multiseed.json')
    json.dump(out, open(path, 'w'), indent=2)
    logger.info(f'Saved -> {path}')


if __name__ == '__main__':
    main()
