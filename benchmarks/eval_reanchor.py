#!/usr/bin/env python3
"""Published-protocol re-anchoring experiment (manuscript Results 2.3).

Re-runs identical model families under the literature-style pooled
evaluation and compares them with the deployment-aware strict protocol
of eval_revision.py, holding everything else fixed:

  strict (paper protocol) : source = batch 1; scaler fit on batch 1 only;
      per-batch stratified 50/50 adapt/eval; evaluation on held-out
      halves of batches 2..10.  (Numbers already in revision_multiseed.json)
  pooled (literature style): all ten batches merged; scaler fit on the
      pooled set; stratified per-seed 90/10 train/test splits, five seeds.
      This mirrors the common practice of random-shuffle cross-validation
      over the full drift sequence, in which training samples already carry
      every drift stage's statistics.

Same architectures, same hyperparameters, same seeds as the strict runs,
so the only difference is the evaluation protocol. Saves
results/reanchor_multiseed.json.

Usage:
    python benchmarks/eval_reanchor.py --seeds 7 21 42 87 123
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmarks.eval_drift_unified as base
from benchmarks.eval_twin import fit_protonet4, fit_relationnet4, proto_predict4

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('reanchor')
logger.setLevel(logging.INFO)

N_CLASSES = 6


def fit_protonet6(X, y, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    from models.few_shot import PrototypicalNetwork, create_fewshot_episode
    m = PrototypicalNetwork(input_dim=128, hidden_dim=64, embedding_dim=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(100):
        sup, que = create_fewshot_episode(X, y, n_way=6, k_shot=5, n_query=5)
        if len(sup[1]) == 0:
            continue
        m.train_episode(sup, que, opt)
    return m


def fit_relationnet6(X, y, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    from models.few_shot import RelationNetwork, create_fewshot_episode
    m = RelationNetwork(input_dim=128, hidden_dim=64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(100):
        sup, que = create_fewshot_episode(X, y, n_way=6, k_shot=5, n_query=5)
        if len(sup[1]) == 0:
            continue
        m.train_episode(sup, que, opt)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    ap.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 21, 42, 87, 123])
    args = ap.parse_args()

    batches = base.load_batches(os.path.join(args.data_root, 'ucsd', 'Dataset'), 224)
    X_all = np.concatenate([b[0] for b in batches])
    y_all = np.concatenate([b[1] for b in batches])
    logger.info(f'pooled set: {X_all.shape[0]} samples, {X_all.shape[1]} dims')

    methods = ['svm', 'mlp', 'protonet', 'relationnet']
    per_seed = {}
    for seed in args.seeds:
        logger.info(f'=== seed {seed} ===')
        np.random.seed(seed)
        torch.manual_seed(seed)
        # literature-style pooled split: literature statistics in preprocessing
        scaler = base.fit_scaler(X_all)
        Xp = base.apply_scaler(X_all, scaler)
        Xtr, Xte, ytr, yte = train_test_split(
            Xp, y_all, test_size=0.1, random_state=seed, stratify=y_all)
        res = {}
        for m_name in methods:
            if m_name == 'svm':
                m = SVC(kernel='rbf').fit(Xtr, ytr)
                pred = m.predict(Xte)
            elif m_name == 'mlp':
                m = base.MLPEnc(in_dim=128, n_cls=N_CLASSES).fit(Xtr, ytr, seed=seed)
                pred = m.predict(Xte)
            elif m_name == 'protonet':
                m = fit_protonet6(Xtr, ytr, seed)
                base_m = m
                m.eval()
                protos = m.compute_prototypes(torch.FloatTensor(Xtr),
                                              torch.LongTensor(ytr), N_CLASSES)
                pred = m.predict(Xte, protos)
            else:  # relationnet
                m = fit_relationnet6(Xtr, ytr, seed)
                pred = m.predict(Xte, (Xtr, ytr))
            res[m_name] = {'acc': float(accuracy_score(yte, pred))}
            logger.info(f'{m_name}: {res[m_name]["acc"]:.4f}')
        per_seed[str(seed)] = res

    agg = {}
    for m_name in methods:
        vals = [per_seed[s][m_name]['acc'] for s in per_seed]
        entry = {'acc_mean': float(np.mean(vals)),
                 'acc_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        strict = json.load(open(os.path.join(args.out_dir, 'revision_multiseed.json')))
        sagg = strict['aggregate'][m_name]
        entry['strict_mean'] = sagg.get('acc_mean', sagg.get('avg_acc_mean'))
        entry['strict_std'] = sagg.get('acc_std', sagg.get('avg_acc_std'))
        entry['gap_pp'] = float((entry['acc_mean'] - entry['strict_mean']) * 100)
        agg[m_name] = entry

    out = {'experiment': 'pooled literature-style random 90/10 splits, five seeds, '
                         'pooled-set normalization; vs strict deployment-aware '
                         'protocol (revision_multiseed.json)',
           'seeds': args.seeds, 'n_pooled': int(X_all.shape[0]),
           'per_seed': per_seed, 'aggregate': agg}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'reanchor_multiseed.json')
    json.dump(out, open(path, 'w'), indent=2)
    for m_name, a in agg.items():
        logger.info(f'{m_name}: pooled {a["acc_mean"]*100:.2f}±{a["acc_std"]*100:.2f} '
                    f'vs strict {a["strict_mean"]*100:.2f}±{a["strict_std"]*100:.2f} '
                    f'(gap {a["gap_pp"]:+.1f} pp)')
    logger.info(f'Saved -> {path}')


if __name__ == '__main__':
    main()
