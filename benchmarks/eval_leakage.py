#!/usr/bin/env python3
"""
Leakage-quantification experiment (Review-4 comment 7).

Measures how much two protocol relaxations inflate UCSD drift accuracy:
  (a) JOINT normalization: scaler fitted on ALL batches (target leakage)
      vs the unified protocol's source-only normalization.
  (b) EVALUATION ON ADAPTATION DATA: methods evaluated on the same samples
      they adapted on (no adaptation/evaluation split).

Same classifier (SVM-RBF and DANN), same batches 2..10; only the protocol
changes. This provides the missing evidence that the 88-97% figures in the
gas-sensing literature and the ~40-55% unified-protocol figures differ
because of the protocol, not the algorithms.
"""
import os, sys, json
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmarks.eval_drift_unified as base
from models.transfer_learning import DANN

SEED = 42


def svm_eval(batches, scaler, eval_on_adapt):
    X_src, y_src = batches[0]
    m = SVC(kernel='rbf').fit(base.apply_scaler(X_src, scaler), y_src)
    accs = []
    for bi in range(1, 10):
        X_t, y_t = batches[bi]
        if eval_on_adapt:
            Xe, ye = X_t, y_t
        else:
            _, Xe, _, ye = train_test_split(
                X_t, y_t, test_size=0.5, random_state=SEED * 1000 + bi,
                stratify=y_t)
        accs.append(accuracy_score(ye, m.predict(base.apply_scaler(Xe, scaler))))
    return float(np.mean(accs))


def dann_eval(batches, scaler, eval_on_adapt):
    X_src, y_src = batches[0]
    accs = []
    for bi in range(1, 10):
        X_t, y_t = batches[bi]
        if eval_on_adapt:
            Xa, Xe, ye = X_t, X_t, y_t
        else:
            Xa, Xe, _, ye = train_test_split(
                X_t, y_t, test_size=0.5, random_state=SEED * 1000 + bi,
                stratify=y_t)
        torch.manual_seed(SEED)
        dann = DANN(input_dim=128, hidden_dim=128, num_classes=6, alpha=1.0)
        dann.fit(base.apply_scaler(X_src, scaler), y_src,
                 base.apply_scaler(Xa, scaler), epochs=50, batch_size=32)
        accs.append(accuracy_score(ye, dann.predict(base.apply_scaler(Xe, scaler))))
    return float(np.mean(accs))


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'ucsd', 'Dataset')
    batches = base.load_batches(data_dir, 224)

    src_scaler = base.fit_scaler(batches[0][0])
    X_all = np.vstack([X for X, _ in batches])
    joint_scaler = base.fit_scaler(X_all)

    res = {}
    res['svm_source_only_split'] = svm_eval(batches, src_scaler, False)
    res['svm_joint_norm'] = svm_eval(batches, joint_scaler, False)
    res['svm_joint_norm_eval_on_adapt'] = svm_eval(batches, joint_scaler, True)
    res['svm_source_only_eval_on_adapt'] = svm_eval(batches, src_scaler, True)
    # DANN adapts on unlabeled target; evaluation on the adaptation half leaks
    res['dann_source_only_split'] = dann_eval(batches, src_scaler, False)
    res['dann_joint_norm_eval_on_adapt'] = dann_eval(batches, joint_scaler, True)

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results', 'revision_leakage.json')
    json.dump(res, open(path, 'w'), indent=2)
    for k, v in res.items():
        print(f'{k}: {v*100:.2f}%')


if __name__ == '__main__':
    main()
