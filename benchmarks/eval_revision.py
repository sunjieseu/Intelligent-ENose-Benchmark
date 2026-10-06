#!/usr/bin/env python3
"""
Revision experiments (rebuttal-driven, on top of eval_drift_unified.py):

  R2-5  multi-seed statistics: 5 seeds x all unified-protocol methods
  R1-3  protocol sensitivity:  k-shot sweep, split-ratio sweep, TTA tau sweep
  R1-5  hyperparameter selection protocol: batch-2 inner validation grids
  R1-6 / R2-1 / R2-2  end-to-end lifecycle runs:
        lifecycle_frozen   = SSL -> few-shot ProtoNet on batch 2, frozen forever
        lifecycle_tta      = SSL -> few-shot head on batch 2 -> per-batch TTA
        lifecycle_replay   = lifecycle_tta + prototype replay blending (alpha)
        + prototype-validity diagnostics (cross-stage interaction)
  R2-4  strictly chronological prequential stream evaluation (tta_chrono)

All variants keep the unified protocol of eval_drift_unified.py: batch 1 is
the source domain, batches 2..10 arrive sequentially, adaptation and
evaluation halves are disjoint, normalization is fitted on batch 1 only.

Usage:
    python benchmarks/eval_revision.py --dataset 224 --seeds 7 21 42 87 123
    python benchmarks/eval_revision.py --dataset 224 --sensitivity
    python benchmarks/eval_revision.py --dataset 224 --chronological
    python benchmarks/eval_revision.py --all
"""

import os
import sys
import json
import argparse
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmarks.eval_drift_unified as base
from models.transfer_learning import TCA, DANN
from models.few_shot import PrototypicalNetwork, RelationNetwork, create_fewshot_episode
from models.drift_compensation import (ClassifierReplacementEnsemble,
                                       LightweightCoTTA, LightweightEATA,
                                       LightweightRoTTA)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('revision')
logger.setLevel(logging.INFO)

N_CLASSES = 6
SEQUENTIAL_METHODS = [
    'cre', 'tta', 'ssl_tta', 'sar_tta', 'cotta_tta', 'eata_tta', 'rotta_tta',
    'ssl_protonet', 'lifecycle_frozen', 'lifecycle_tta', 'lifecycle_replay',
    'lifecycle_memory'
]


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    base.SEED = seed  # modules use base.SEED defaults


def seed_std(vals):
    """Unbiased sample standard deviation for repeated random-seed runs."""
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


# ----------------------------------------------------------------------
# Parameterized re-implementations of the unified harness
# ----------------------------------------------------------------------
def eval_static_p(batches, method, scaler, seed, test_size, return_per_batch=False):
    X_src, y_src = batches[0]
    accs = []
    for bi in range(1, 10):
        X_t, y_t = batches[bi]
        X_adapt, X_eval, y_adapt, y_eval = train_test_split(
            X_t, y_t, test_size=test_size, random_state=seed * 1000 + bi,
            stratify=y_t if len(np.unique(y_t)) > 1 else None)
        if method == 'svm':
            m = SVC(kernel='rbf').fit(base.apply_scaler(X_src, scaler), y_src)
            pred = m.predict(base.apply_scaler(X_eval, scaler))
        elif method == 'mlp':
            m = base.MLPEnc().fit(base.apply_scaler(X_src, scaler), y_src, seed=seed)
            pred = m.predict(base.apply_scaler(X_eval, scaler))
        elif method == 'tca':
            tca = TCA(n_components=10, gamma=1.0 / X_src.shape[1])
            Xs_s, Xa_s = base.apply_scaler(X_src, scaler), base.apply_scaler(X_adapt, scaler)
            Xt = tca.fit_transform(Xs_s, Xa_s)
            svm = SVC(kernel='rbf').fit(Xt[:len(Xs_s)], y_src)
            pred = svm.predict(tca.transform(base.apply_scaler(X_eval, scaler)))
        elif method == 'dann':
            torch.manual_seed(seed)
            dann = DANN(input_dim=128, hidden_dim=128, num_classes=N_CLASSES, alpha=1.0)
            dann.fit(base.apply_scaler(X_src, scaler), y_src,
                     base.apply_scaler(X_adapt, scaler), epochs=50, batch_size=32)
            pred = dann.predict(base.apply_scaler(X_eval, scaler))
        elif method == 'protonet':
            np.random.seed(seed)
            Xs_s = base.apply_scaler(X_src, scaler)
            m = base.fit_protonet(Xs_s, y_src)
            pred = base.proto_predict(m, base.apply_scaler(X_eval, scaler), Xs_s, y_src)
        elif method == 'relationnet':
            np.random.seed(seed)
            Xs_s = base.apply_scaler(X_src, scaler)
            m = base.fit_relationnet(Xs_s, y_src)
            pred = m.predict(base.apply_scaler(X_eval, scaler), (Xs_s, y_src))
        else:
            raise ValueError(method)
        accs.append(accuracy_score(y_eval, pred))
    if return_per_batch:
        per_batch = {f'batch{i + 2}': float(acc) for i, acc in enumerate(accs)}
        return {
            'avg_acc': float(np.mean(accs)),
            'avg_acc_b3_b10': float(np.mean(accs[1:])),
            'per_batch_acc': per_batch,
        }
    return float(np.mean(accs))


def eval_sequential_p(batches, method, scaler, seed, test_size, k_shot=5,
                      tau=0.5, blend_alpha=0.2, ssl_model=None):
    """Sequential evaluation with lifecycle end-to-end variants.

    Lifecycle variants (Algorithm-1 end-to-end instantiations):
      lifecycle_frozen : Phase1 SSL -> Phase2 few-shot protos on batch 2,
                         frozen forever (no Phase-3 update).
      lifecycle_tta    : + Phase3 entropy-min TTA on the few-shot head
                         (encoder frozen, prototypes fixed -> measures the
                         class-structure preservation question of R2-2).
      lifecycle_replay : + Phase3 TTA + prototype replay blending
                         (alpha-blend of confident pseudo-labeled cluster
                         means; predictions use the blended prototypes,
                         giving the forgetting-mitigation stage).
    Prediction rule: frozen/replay -> nearest prototype; tta -> head.
    """
    X_src, y_src = batches[0]
    scaler_src = base.apply_scaler(X_src, scaler)
    seen_eval = []
    n_tasks = 10
    matrix = np.full((n_tasks, n_tasks), np.nan)
    baseline_task = np.full(n_tasks, np.nan)
    proto_validity = {}   # frozen-prototype acc per batch (R2-2 diagnostics)
    n_tta_triggers = 0
    ref = SVC(kernel='rbf').fit(scaler_src, y_src)

    model = None
    tta = None
    tta_stats = {'n_selected': 0, 'n_skipped': 0,
                 'mean_entropy_before': [], 'mean_entropy_after': []}

    def mlp_features(mlp, X):
        mlp.eval()
        with torch.no_grad():
            z = mlp.net[:-1](torch.tensor(X, dtype=torch.float32))
        return z

    def record_tta_stats(stats):
        if not stats:
            return
        tta_stats['n_selected'] += int(stats.get('n_selected', 0))
        if stats.get('n_selected', 0) == 0:
            tta_stats['n_skipped'] += 1
        tta_stats['mean_entropy_before'].append(float(stats.get('mean_entropy_before', 0.0)))
        tta_stats['mean_entropy_after'].append(float(stats.get('mean_entropy_after', 0.0)))

    def predict_m(X):
        if isinstance(model, dict):
            if model['mode'] in ('lifecycle_frozen', 'lifecycle_replay', 'lifecycle_memory'):
                return predict_proto(ssl_model, model, X)
            return predict_head(ssl_model, model, X)
        return model.predict(X)

    for bi in range(1, 10):
        X_t, y_t = batches[bi]
        X_adapt, X_eval, y_adapt, y_eval = train_test_split(
            X_t, y_t, test_size=test_size, random_state=seed * 1000 + bi,
            stratify=y_t if len(np.unique(y_t)) > 1 else None)
        Xe_s = base.apply_scaler(X_eval, scaler)
        Xa_s = base.apply_scaler(X_adapt, scaler)
        seen_eval.append((Xe_s, y_eval))
        baseline_task[bi - 1] = accuracy_score(y_eval, ref.predict(Xe_s))

        s_t = base.energy_distance(scaler_src, Xa_s, seed=seed + bi)

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
                model = base.MLPEnc()
                model.fit(scaler_src, y_src, epochs=60, seed=seed)
                tta = base.SimpleTTA(model)
            else:
                if s_t > tau:
                    tta.adapt(Xa_s)
                    n_tta_triggers += 1
        elif method == 'sar_tta':
            if model is None:
                Zs = ssl_model.transform(scaler_src)
                torch.manual_seed(seed)
                emb_dim = ssl_model.encoder[0].out_features
                head = nn.Linear(emb_dim, N_CLASSES)
                opt = torch.optim.Adam(head.parameters(), lr=1e-2)
                Zt = torch.tensor(Zs, dtype=torch.float32)
                yt = torch.tensor(y_src, dtype=torch.long)
                head.train()
                for _ in range(150):
                    opt.zero_grad()
                    F.cross_entropy(head(Zt), yt).backward()
                    opt.step()
                head.eval()
                model = {'head': head, 'mode': method,
                         'anchor_state': [p.detach().clone() for p in head.parameters()]}
            else:
                if s_t > tau:
                    n_tta_triggers += int(adapt_head_sar(ssl_model, model, Xa_s) > 0)
        elif method in ('cotta_tta', 'eata_tta', 'rotta_tta'):
            if model is None:
                model = base.MLPEnc()
                model.fit(scaler_src, y_src, epochs=60, seed=seed)
                head = model.net[-1]
                if method == 'cotta_tta':
                    tta = LightweightCoTTA(head)
                elif method == 'eata_tta':
                    tta = LightweightEATA(head)
                else:
                    tta = LightweightRoTTA(head)
            else:
                if s_t > tau:
                    stats = tta.adapt(mlp_features(model, Xa_s))
                    record_tta_stats(stats)
                    n_tta_triggers += int(stats.get('n_selected', 0) > 0)
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
        elif method in ('lifecycle_frozen', 'lifecycle_tta', 'lifecycle_replay', 'lifecycle_memory'):
            if model is None:
                # Phase 2: few-shot calibration on batch-2 adaptation half only
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
                if method == 'lifecycle_memory':
                    model['anchor_protos'] = {c: p.copy() for c, p in protos.items()}
            else:
                if s_t > tau:
                    # frozen-prototype validity diagnostic (pre-update state)
                    proto_validity[f'batch{bi+1}'] = round(
                        accuracy_score(y_eval, predict_proto(ssl_model, model, Xe_s)), 4)
                    if method == 'lifecycle_tta':
                        adapt_head(ssl_model, model, Xa_s)
                        n_tta_triggers += 1
                    elif method == 'lifecycle_replay':
                        adapt_head(ssl_model, model, Xa_s)
                        blend_prototypes(ssl_model, model, Xa_s, blend_alpha)
                        n_tta_triggers += 1
                    elif method == 'lifecycle_memory':
                        adapt_head(ssl_model, model, Xa_s)
                        updated = update_memory_prototypes(ssl_model, model, Xa_s)
                        n_tta_triggers += int(updated > 0)
        else:
            raise ValueError(method)

        if method == 'lifecycle_frozen':
            # prototypes never updated; validity monitored only
            proto_validity[f'batch{bi+1}'] = round(
                accuracy_score(y_eval, predict_proto(ssl_model, model, Xe_s)), 4)
        for ti, (Xt_e, yt_e) in enumerate(seen_eval):
            matrix[bi - 1, ti] = accuracy_score(yt_e, predict_m(Xt_e))

    T = 9
    filled = matrix[:T, :T]
    bwt = float(np.nanmean([filled[T - 1, i] - filled[i, i] for i in range(T - 1)]))
    sup = np.array([matrix[t - 2, t - 1] for t in range(2, T + 1)])
    bl = np.array([baseline_task[t - 1] for t in range(2, T + 1)])
    fwt = float('nan') if np.isnan(sup).any() else float(np.mean(sup - bl))
    diag = [matrix[i, i] for i in range(T)]
    stats_out = {
        'n_selected': int(tta_stats['n_selected']),
        'n_skipped': int(tta_stats['n_skipped']),
        'mean_entropy_before': float(np.mean(tta_stats['mean_entropy_before']))
        if tta_stats['mean_entropy_before'] else 0.0,
        'mean_entropy_after': float(np.mean(tta_stats['mean_entropy_after']))
        if tta_stats['mean_entropy_after'] else 0.0,
    }
    return {'avg_acc': float(np.nanmean(diag[1:])), 'bwt': bwt, 'fwt': fwt,
            'proto_validity': proto_validity, 'n_tta_triggers': n_tta_triggers,
            'final_acc': float(matrix[T - 1, T - 1]), 'tta_stats': stats_out}


def predict_proto(ssl_model, model, X):
    Z = ssl_model.transform(X)
    classes = sorted(model['protos'].keys())
    P = np.stack([model['protos'][c] for c in classes])
    d = ((Z[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return np.array(classes)[d.argmin(1)]


def predict_head(ssl_model, model, X):
    Z = torch.tensor(ssl_model.transform(X), dtype=torch.float32)
    model['head'].eval()
    with torch.no_grad():
        return model['head'](Z).argmax(1).numpy()


def adapt_head(ssl_model, model, Xa_s, conf_tau=0.7, steps=3, lr=1e-3):
    """Entropy-minimization TTA on the few-shot head only (encoder frozen)."""
    head = model['head']
    Z = torch.tensor(ssl_model.transform(Xa_s), dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        p = F.softmax(head(Z), dim=1)
    conf, _ = p.max(1)
    sel = conf > conf_tau
    if sel.sum() < 2:
        return
    Zs = Z[sel]
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    head.train()
    for _ in range(steps):
        out = head(Zs)
        loss = -(F.softmax(out, 1) * F.log_softmax(out, 1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()


def adapt_head_sar(ssl_model, model, Xa_s, conf_tau=0.7, margin=0.4,
                   steps=1, lr=5e-4, reg=1e-3):
    """Entropy-filtered robust TTA with an anchor penalty on the head."""
    head = model['head']
    Z = torch.tensor(ssl_model.transform(Xa_s), dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        out = head(Z)
        p = F.softmax(out, dim=1)
        conf, _ = p.max(1)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
    sel = (conf > conf_tau) & (entropy < margin)
    if sel.sum() < 2:
        return int(sel.sum().item())

    anchors = model.get('anchor_state')
    if anchors is None:
        anchors = [p.detach().clone() for p in head.parameters()]
        model['anchor_state'] = anchors
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    head.train()
    for _ in range(steps):
        logits = head(Z[sel])
        prob = F.softmax(logits, 1)
        loss = -(prob * F.log_softmax(logits, 1)).sum(1).mean()
        for param, anchor in zip(head.parameters(), anchors):
            loss = loss + reg * (param - anchor).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return int(sel.sum().item())


def blend_prototypes(ssl_model, model, Xa_s, alpha=0.2, conf_tau=0.9):
    """Replay-style prototype retention: blend new pseudo-labeled cluster
    means with the stored prototypes (encoder frozen, head already updated)."""
    Z = ssl_model.transform(Xa_s)
    head = model['head']
    with torch.no_grad():
        p = F.softmax(head(torch.tensor(Z, dtype=torch.float32)), dim=1)
    conf, pseudo = p.max(1)
    conf, pseudo = conf.numpy(), pseudo.numpy()
    for c in model['protos']:
        sel = (pseudo == c) & (conf > conf_tau)
        if sel.sum() > 0:
            new_mean = Z[sel].mean(0)
            model['protos'][c] = (1 - alpha) * model['protos'][c] + alpha * new_mean


def update_memory_prototypes(ssl_model, model, Xa_s, anchor_alpha=0.05,
                             memory_alpha=0.25, conf_tau=0.75):
    """Update prototype memory with confident clusters while retaining anchors."""
    Z = ssl_model.transform(Xa_s)
    head = model['head']
    with torch.no_grad():
        p = F.softmax(head(torch.tensor(Z, dtype=torch.float32)), dim=1)
    conf, pseudo = p.max(1)
    conf, pseudo = conf.numpy(), pseudo.numpy()
    updated = 0
    anchors = model.get('anchor_protos', model['protos'])
    for c in model['protos']:
        sel = (pseudo == c) & (conf > conf_tau)
        if sel.sum() == 0:
            continue
        new_mean = Z[sel].mean(0)
        retained = (1 - memory_alpha) * model['protos'][c] + memory_alpha * new_mean
        model['protos'][c] = (1 - anchor_alpha) * retained + anchor_alpha * anchors[c]
        updated += 1
    return updated


# ----------------------------------------------------------------------
# Chronological prequential stream evaluation (R2-4)
# ----------------------------------------------------------------------
def eval_chronological(batches, seed, conf_tau=0.7, window=16):
    """Strictly chronological prequential evaluation.

    No stratified split: every sample of every arriving batch is predicted
    first, then becomes available for adaptation. Class imbalance within a
    batch is therefore preserved rather than stratified away.
    """
    set_seeds(seed)
    X_src, y_src = batches[0]
    scaler = base.fit_scaler(X_src)
    scaler_src = base.apply_scaler(X_src, scaler)
    model = base.MLPEnc().fit(scaler_src, y_src, epochs=60, seed=seed)
    per_batch_acc, per_batch_maxclass = [], []
    seen_eval = []
    for bi in range(1, 10):
        X_t, y_t = base.apply_scaler(batches[bi][0], scaler), batches[bi][1]
        buf, correct, total = [], 0, 0
        opt = torch.optim.SGD(model.net[-1].parameters(), lr=1e-3)
        for x, y in zip(X_t, y_t):
            pred = model.predict(x[None, :])[0]
            correct += int(pred == y)
            total += 1
            buf.append(x)
            if len(buf) >= window:
                Xw = torch.tensor(np.array(buf), dtype=torch.float32)
                with torch.no_grad():
                    p = F.softmax(model.net(Xw), dim=1)
                conf, _ = p.max(1)
                Xs = Xw[conf > conf_tau]
                if Xs.shape[0] >= 2:
                    model.train()
                    for _ in range(3):
                        out = model.net(Xs)
                        loss = -(F.softmax(out, 1) * F.log_softmax(out, 1)).sum(1).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    model.eval()
                buf = []
        per_batch_acc.append(correct / total)
        vals = np.bincount(y_t, minlength=N_CLASSES)
        per_batch_maxclass.append(float(vals.max() / vals.sum()))
        seen_eval.append((X_t, y_t))
    return {'prequential_acc': float(np.mean(per_batch_acc)),
            'per_batch_acc': [round(a, 4) for a in per_batch_acc],
            'per_batch_max_class_ratio': [round(r, 4) for r in per_batch_maxclass]}


def aggregate_chronological_results(seeds, per_seed, legacy_seed=42):
    """Aggregate chronological runs while preserving the legacy seed-42 fields."""
    seed_keys = [str(seed) for seed in seeds]
    accs = [per_seed[s]['prequential_acc'] for s in seed_keys]
    per_batch = np.array([per_seed[s]['per_batch_acc'] for s in seed_keys], dtype=float)
    legacy = per_seed[str(legacy_seed)]
    out = dict(legacy)
    out['seeds'] = list(seeds)
    out['per_seed'] = per_seed
    out['aggregate'] = {
        'prequential_acc_mean': float(np.mean(accs)),
        'prequential_acc_std': seed_std(accs),
        'per_batch_acc_mean': [round(float(v), 4) for v in np.mean(per_batch, axis=0)],
        'per_batch_acc_std': [round(seed_std(per_batch[:, i]), 4)
                              for i in range(per_batch.shape[1])],
    }
    return out


# ----------------------------------------------------------------------
# Hyperparameter selection on a batch-2 internal validation split (R1-5)
# ----------------------------------------------------------------------
def hyperparam_selection(batches, seed=42):
    X_src, y_src = batches[0]
    scaler = base.fit_scaler(X_src)
    scaler_src = base.apply_scaler(X_src, scaler)
    X_t, y_t = batches[1]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_t, y_t, test_size=0.5, random_state=seed,
        stratify=y_t)
    X_tr_s, X_val_s = base.apply_scaler(X_tr, scaler), base.apply_scaler(X_val, scaler)

    out = {}
    # MLP hidden size (source-only model judged on drifted target validation half)
    best = None
    for hid in (32, 64, 128):
        m = base.MLPEnc(in_dim=128, hid=hid).fit(scaler_src, y_src, seed=seed)
        acc = accuracy_score(y_val, m.predict(X_val_s))
        if best is None or acc > best[1]:
            best = (hid, acc)
    out['mlp_hidden'] = {'grid': {hid: None for hid in (32, 64, 128)},
                         'selected': best[0], 'val_acc': round(best[1], 4)}

    # TTA steps / entropy-confidence tau (unsupervised adaptation on the
    # batch-2 TRAIN half; judged on the held-out validation half)
    base_mlp = base.MLPEnc().fit(scaler_src, y_src, epochs=60, seed=seed)
    best = None
    for steps in (1, 3, 5):
        for cf in (0.6, 0.7, 0.8):
            m2 = copy.deepcopy(base_mlp)
            tta = base.SimpleTTA(m2, tau=cf, steps=steps)
            tta.adapt(X_tr_s)
            acc = accuracy_score(y_val, m2.predict(X_val_s))
            if best is None or acc > best[1]:
                best = ((steps, cf), acc)
    out['tta'] = {'grid_steps': [1, 3, 5], 'grid_conf': [0.6, 0.7, 0.8],
                  'selected': {'steps': best[0][0], 'conf': best[0][1]},
                  'val_acc': round(best[1], 4)}

    # DANN alpha on batch 2
    best = None
    for alpha in (0.5, 1.0, 2.0):
        torch.manual_seed(seed)
        dann = DANN(input_dim=128, hidden_dim=128, num_classes=N_CLASSES, alpha=alpha)
        dann.fit(scaler_src, y_src, X_tr_s, epochs=50, batch_size=32)
        acc = accuracy_score(y_val, dann.predict(X_val_s))
        if best is None or acc > best[1]:
            best = (alpha, acc)
    out['dann_alpha'] = {'grid': [0.5, 1.0, 2.0], 'selected': best[0],
                         'val_acc': round(best[1], 4)}
    return out


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------
def run_multiseed(args):
    data_dir = os.path.join(args.data_root, 'ucsd', 'Dataset')
    batches = base.load_batches(data_dir, 224)
    scaler = base.fit_scaler(batches[0][0])
    all_res = {}
    for seed in args.seeds:
        logger.info(f'=== seed {seed} ===')
        set_seeds(seed)
        res = {}
        for m in ['svm', 'mlp', 'tca', 'dann', 'protonet', 'relationnet']:
            res[m] = eval_static_p(batches, m, scaler, seed, 0.5, return_per_batch=True)
            logger.info(f'{m}: {res[m]["avg_acc"]:.4f}')
        ssl = base.SimpleSSL(seed=seed)
        ssl.fit(base.apply_scaler(batches[0][0], scaler))
        for m in SEQUENTIAL_METHODS:
            r = eval_sequential_p(batches, m, scaler, seed, 0.5,
                                  ssl_model=ssl)
            res[m] = r
            logger.info(f'{m}: avg={r["avg_acc"]:.4f} bwt={r["bwt"]:.4f} fwt={r["fwt"]:.4f}')
        all_res[str(seed)] = res

    # aggregate mean +- std over seeds
    all_res = {str(s): r for s, r in all_res.items()}
    agg = {}
    for m in list(all_res[str(args.seeds[0])].keys()):
        entry = {}
        for key in ('avg_acc', 'avg_acc_b3_b10', 'bwt', 'fwt'):
            vals = [all_res[s].get(m, {}).get(key) for s in all_res]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                entry[key + '_mean'] = float(np.mean(vals))
                entry[key + '_std'] = seed_std(vals)
        agg[m] = entry
    out = {'seeds': args.seeds, 'per_seed': all_res, 'aggregate': agg}
    path = os.path.join(args.out_dir, 'revision_multiseed.json')
    json.dump(out, open(path, 'w'), indent=2)
    logger.info(f'Saved -> {path}')
    return agg


def run_sensitivity(args):
    data_dir = os.path.join(args.data_root, 'ucsd', 'Dataset')
    batches = base.load_batches(data_dir, 224)
    scaler = base.fit_scaler(batches[0][0])
    seed = 42
    set_seeds(seed)
    out = {}

    ssl = base.SimpleSSL(seed=seed)
    ssl.fit(base.apply_scaler(batches[0][0], scaler))

    # k-shot sweep (lifecycle pilot, sequential)
    ks = {}
    for k in (1, 3, 5, 10):
        set_seeds(seed)
        r = eval_sequential_p(batches, 'ssl_protonet', scaler, seed, 0.5,
                              k_shot=k, ssl_model=ssl)
        ks[str(k)] = r
        logger.info(f'k_shot={k}: avg={r["avg_acc"]:.4f} bwt={r["bwt"]:.4f}')
    out['k_shot'] = ks

    # split-ratio sweep
    sp = {}
    for ts in (0.3, 0.5, 0.7):
        sp[str(ts)] = {}
        for m in ('svm', 'dann'):
            set_seeds(seed)
            acc = eval_static_p(batches, m, scaler, seed, ts)
            sp[str(ts)][m] = {'avg_acc': acc}
        for m in ('ssl_protonet', 'cre'):
            set_seeds(seed)
            r = eval_sequential_p(batches, m, scaler, seed, ts, ssl_model=ssl)
            sp[str(ts)][m] = {'avg_acc': r['avg_acc'], 'bwt': r['bwt']}
        logger.info(f'test_size={ts}: {sp[str(ts)]}')
    out['split'] = sp

    # tau sweep (TTA gating)
    taus = {}
    for tau in (0.25, 0.5, 1.0, 2.0):
        set_seeds(seed)
        r = eval_sequential_p(batches, 'tta', scaler, seed, 0.5, tau=tau)
        taus[str(tau)] = {'avg_acc': r['avg_acc'], 'bwt': r['bwt'],
                          'n_triggers': r['n_tta_triggers']}
        logger.info(f'tau={tau}: avg={r["avg_acc"]:.4f} triggers={r["n_tta_triggers"]}')
    out['tau'] = taus

    out['hyperparam_selection'] = hyperparam_selection(batches, seed)
    path = os.path.join(args.out_dir, 'revision_sensitivity.json')
    json.dump(out, open(path, 'w'), indent=2)
    logger.info(f'Saved -> {path}')


def run_chronological(args):
    data_dir = os.path.join(args.data_root, 'ucsd', 'Dataset')
    batches = base.load_batches(data_dir, 224)
    per_seed = {}
    for seed in args.seeds:
        per_seed[str(seed)] = eval_chronological(batches, seed)
        logger.info(f'chronological seed={seed} acc={per_seed[str(seed)]["prequential_acc"]:.4f}')
    res = aggregate_chronological_results(args.seeds, per_seed)
    path = os.path.join(args.out_dir, 'revision_chronological.json')
    json.dump(res, open(path, 'w'), indent=2)
    logger.info('chronological prequential acc='
                f'{res["aggregate"]["prequential_acc_mean"]:.4f} '
                f'+/- {res["aggregate"]["prequential_acc_std"]:.4f}')
    logger.info(f'Saved -> {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='224')
    ap.add_argument('--data-root', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    ap.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 21, 42, 87, 123])
    ap.add_argument('--sensitivity', action='store_true')
    ap.add_argument('--chronological', action='store_true')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.all or not (args.sensitivity or args.chronological):
        run_multiseed(args)
    if args.all or args.sensitivity:
        run_sensitivity(args)
    if args.all or args.chronological:
        run_chronological(args)


if __name__ == '__main__':
    main()
