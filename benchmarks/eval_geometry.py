#!/usr/bin/env python3
"""Feature-space drift geometry diagnostics for the manuscript."""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmarks.eval_drift_unified as base


def _energy_distance(X_ref, X_cur, max_n=500, seed=0):
    rng = np.random.default_rng(seed)
    a = X_ref
    b = X_cur
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    sigma = float(np.sqrt(np.mean(np.var(X_ref, axis=0))))
    sigma = max(sigma, 1e-12)
    d_ab = pairwise_distances(a, b).mean() / sigma
    d_aa = pairwise_distances(a, a).mean() / sigma
    d_bb = pairwise_distances(b, b).mean() / sigma
    return float(max(2 * d_ab - d_aa - d_bb, 0.0))


def _effective_rank(X):
    centered = X - X.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eig = singular_values ** 2
    if eig.sum() <= 0:
        return 0.0
    p = eig / eig.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def _principal_angle(X_ref, X_cur, n_components=10):
    n = min(n_components, X_ref.shape[0] - 1, X_cur.shape[0] - 1, X_ref.shape[1])
    if n < 1:
        return float("nan")
    ref_basis = PCA(n_components=n).fit(X_ref).components_.T
    cur_basis = PCA(n_components=n).fit(X_cur).components_.T
    singular_values = np.linalg.svd(ref_basis.T @ cur_basis, compute_uv=False)
    min_cos = np.clip(singular_values.min(), -1.0, 1.0)
    return float(np.degrees(np.arccos(min_cos)))


def _centroid_stats(X, y, ref_centroids=None):
    labels = sorted(np.unique(y).astype(int).tolist())
    centroids = {c: X[y == c].mean(axis=0) for c in labels if np.sum(y == c) > 0}
    spreads = []
    for c, center in centroids.items():
        vals = X[y == c]
        if len(vals) > 0:
            spreads.append(np.linalg.norm(vals - center, axis=1).mean())
    within = float(np.mean(spreads)) if spreads else float("nan")
    centers = list(centroids.values())
    if len(centers) > 1:
        dist = pairwise_distances(np.vstack(centers))
        between = float(dist[np.triu_indices_from(dist, k=1)].mean())
    else:
        between = float("nan")
    drift = float("nan")
    if ref_centroids is not None:
        common = sorted(set(ref_centroids).intersection(centroids))
        if common:
            drift = float(np.mean([np.linalg.norm(centroids[c] - ref_centroids[c]) for c in common]))
    ratio = float(between / within) if within and within > 0 and np.isfinite(between) else float("nan")
    return centroids, within, between, ratio, drift


def _prototype_margin(X, y, prototypes):
    classes = sorted(prototypes)
    prototype_matrix = np.vstack([prototypes[c] for c in classes])
    dist = pairwise_distances(X, prototype_matrix)
    order = np.argsort(dist, axis=1)
    pred = np.array(classes)[order[:, 0]]
    acc = float(np.mean(pred == y))
    margins = []
    for i, yi in enumerate(y):
        if int(yi) not in prototypes or len(classes) < 2:
            continue
        true_idx = classes.index(int(yi))
        true_dist = dist[i, true_idx]
        other_dist = np.min(np.delete(dist[i], true_idx))
        margins.append(other_dist - true_dist)
    return acc, float(np.mean(margins)) if margins else float("nan")


def _load_batches(dataset):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if dataset == 224:
        data_dir = os.path.join(root, "ucsd", "Dataset")
    elif dataset == 270:
        data_dir = os.path.join(root, "ucsd270")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return base.load_batches(data_dir, dataset)


def compute_geometry(dataset=224, seed=42, test_size=0.5):
    batches = _load_batches(dataset)
    X_src, y_src = batches[0]
    scaler = base.fit_scaler(X_src)
    scaled = [(base.apply_scaler(X, scaler), y) for X, y in batches]
    X0, y0 = scaled[0]
    ref_centroids, _, _, _, _ = _centroid_stats(X0, y0)

    X_b2, y_b2 = batches[1]
    X_adapt, _, y_adapt, _ = train_test_split(
        X_b2,
        y_b2,
        test_size=test_size,
        random_state=seed * 1000 + 1,
        stratify=y_b2 if len(np.unique(y_b2)) > 1 else None,
    )
    X_adapt_s = base.apply_scaler(X_adapt, scaler)
    prototypes = {int(c): X_adapt_s[y_adapt == c].mean(axis=0) for c in np.unique(y_adapt)}

    out = {
        "dataset": dataset,
        "seed": seed,
        "batches": [],
        "energy_to_source": [],
        "energy_adjacent": [],
        "effective_rank": [],
        "principal_angle_deg": [],
        "centroid_drift": [],
        "within_class_spread": [],
        "between_class_distance": [],
        "separation_ratio": [],
        "prototype_accuracy": [],
        "prototype_margin": [],
    }
    previous = X0
    for i, (X, y) in enumerate(scaled):
        batch_id = i + 1
        _, within, between, ratio, drift = _centroid_stats(X, y, ref_centroids)
        pacc, pmargin = _prototype_margin(X, y, prototypes)
        out["batches"].append(batch_id)
        out["energy_to_source"].append(round(_energy_distance(X0, X, seed=seed + i), 4))
        adjacent = _energy_distance(previous, X, seed=seed + 100 + i) if i > 0 else 0.0
        out["energy_adjacent"].append(round(adjacent, 4))
        out["effective_rank"].append(round(_effective_rank(X), 4))
        angle = _principal_angle(X0, X) if i > 0 else 0.0
        out["principal_angle_deg"].append(round(angle, 4) if np.isfinite(angle) else None)
        out["centroid_drift"].append(round(drift, 4) if np.isfinite(drift) else None)
        out["within_class_spread"].append(round(within, 4) if np.isfinite(within) else None)
        out["between_class_distance"].append(round(between, 4) if np.isfinite(between) else None)
        out["separation_ratio"].append(round(ratio, 4) if np.isfinite(ratio) else None)
        out["prototype_accuracy"].append(round(pacc, 4))
        out["prototype_margin"].append(round(pmargin, 4) if np.isfinite(pmargin) else None)
        previous = X
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("results", "geometry_diagnostics.json"))
    args = parser.parse_args()
    out = compute_geometry(dataset=args.dataset, seed=args.seed)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(args.out)


if __name__ == "__main__":
    main()
