# NCS Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the manuscript and benchmark package toward a Nature Computational Science-style Resource/Benchmark by adding feature-space drift geometry diagnostics, aligning figures with reported methods, and narrowing unsupported deployment claims.

**Architecture:** Add one focused diagnostic script that reads the existing UCSD dataset through the current loader/scaler path and writes a JSON result file. Extend the existing figure generator to include the missing methods and a new geometry figure, then revise the manuscript around the new evidence without inventing hardware data or method performance.

**Tech Stack:** Python, NumPy, scikit-learn, Matplotlib, existing benchmark loaders, LaTeX (`pdflatex`).

**Spec:** Approved in chat on 2026-10-05 after reviewer-style assessment; no separate design document exists because the user approved the in-chat architectural design.

## Global Constraints

- Do not claim NCS acceptance or guaranteed publication.
- Do not fabricate MCU/TFLM/CMSIS-NN data; if no named MCU measurements exist, deployment claims must be lowered to component-level or deployment-protocol evidence.
- Do not tune a new positive baseline on UCSD evaluation batches just to beat MLP; report mechanism diagnostics and supported design principles instead.
- Preserve existing benchmark protocol: batch 1 is source, batches 2--10 are target stages, source-only scaling, disjoint adaptation/evaluation halves where applicable.
- All new manuscript claims must be backed by generated JSON or existing result JSONs.

---

### Task 1: Add Feature-Space Geometry Diagnostics

**Files:**
- Create: `benchmarks/eval_geometry.py`
- Create: `results/geometry_diagnostics.json` by running the script

**Interfaces:**
- Consumes: `benchmarks.eval_drift_unified.load_dataset`, `fit_scaler`, `apply_scaler`, and `energy_distance` if available; otherwise computes energy distance locally with the same formula.
- Produces: JSON with keys `batches`, `energy_to_source`, `energy_adjacent`, `effective_rank`, `principal_angle_deg`, `centroid_drift`, `separation_ratio`, and `prototype_margin`.

- [ ] **Step 1: Create a minimal diagnostics script**

Create `benchmarks/eval_geometry.py` with a CLI:

```python
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
    A = X_ref
    B = X_cur
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    sigma = float(np.sqrt(np.mean(np.var(X_ref, axis=0))))
    sigma = max(sigma, 1e-12)
    d_ab = pairwise_distances(A, B).mean() / sigma
    d_aa = pairwise_distances(A, A).mean() / sigma
    d_bb = pairwise_distances(B, B).mean() / sigma
    return float(max(2 * d_ab - d_aa - d_bb, 0.0))


def _effective_rank(X):
    centered = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eig = s ** 2
    if eig.sum() <= 0:
        return 0.0
    p = eig / eig.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def _principal_angle(X_ref, X_cur, n_components=10):
    n = min(n_components, X_ref.shape[0] - 1, X_cur.shape[0] - 1, X_ref.shape[1])
    if n < 1:
        return float('nan')
    pref = PCA(n_components=n).fit(X_ref).components_.T
    pcur = PCA(n_components=n).fit(X_cur).components_.T
    singular = np.linalg.svd(pref.T @ pcur, compute_uv=False)
    min_cos = np.clip(singular.min(), -1.0, 1.0)
    return float(np.degrees(np.arccos(min_cos)))


def _centroid_stats(X, y, ref_centroids=None):
    labels = sorted(np.unique(y).astype(int).tolist())
    centroids = {c: X[y == c].mean(axis=0) for c in labels if np.sum(y == c) > 0}
    spreads = []
    for c, center in centroids.items():
        vals = X[y == c]
        if len(vals) > 0:
            spreads.append(np.linalg.norm(vals - center, axis=1).mean())
    within = float(np.mean(spreads)) if spreads else float('nan')
    centers = list(centroids.values())
    if len(centers) > 1:
        dist = pairwise_distances(np.vstack(centers))
        between = float(dist[np.triu_indices_from(dist, k=1)].mean())
    else:
        between = float('nan')
    drift = float('nan')
    if ref_centroids is not None:
        common = sorted(set(ref_centroids).intersection(centroids))
        if common:
            drift = float(np.mean([np.linalg.norm(centroids[c] - ref_centroids[c]) for c in common]))
    ratio = float(between / within) if within and within > 0 and np.isfinite(between) else float('nan')
    return centroids, within, between, ratio, drift


def _prototype_margin(X, y, prototypes):
    classes = sorted(prototypes)
    P = np.vstack([prototypes[c] for c in classes])
    d = pairwise_distances(X, P)
    order = np.argsort(d, axis=1)
    pred = np.array(classes)[order[:, 0]]
    acc = float(np.mean(pred == y))
    margins = []
    for i, yi in enumerate(y):
        if int(yi) not in prototypes or len(classes) < 2:
            continue
        true_idx = classes.index(int(yi))
        true_d = d[i, true_idx]
        other_d = np.min(np.delete(d[i], true_idx))
        margins.append(other_d - true_d)
    return acc, float(np.mean(margins)) if margins else float('nan')


def compute_geometry(dataset=224, seed=42, test_size=0.5):
    batches = base.load_dataset(dataset)
    X_src, y_src = batches[0]
    scaler = base.fit_scaler(X_src)
    scaled = [(base.apply_scaler(X, scaler), y) for X, y in batches]
    X0, y0 = scaled[0]
    ref_centroids, _, _, _, _ = _centroid_stats(X0, y0)

    X_b2, y_b2 = batches[1]
    X_adapt, _, y_adapt, _ = train_test_split(
        X_b2, y_b2, test_size=test_size, random_state=seed * 1000 + 1,
        stratify=y_b2 if len(np.unique(y_b2)) > 1 else None)
    X_adapt_s = base.apply_scaler(X_adapt, scaler)
    prototypes = {int(c): X_adapt_s[y_adapt == c].mean(axis=0) for c in np.unique(y_adapt)}

    out = {
        'dataset': dataset,
        'seed': seed,
        'batches': [],
        'energy_to_source': [],
        'energy_adjacent': [],
        'effective_rank': [],
        'principal_angle_deg': [],
        'centroid_drift': [],
        'within_class_spread': [],
        'between_class_distance': [],
        'separation_ratio': [],
        'prototype_accuracy': [],
        'prototype_margin': []
    }
    prev = X0
    for i, (X, y) in enumerate(scaled):
        batch_id = i + 1
        centroids, within, between, ratio, drift = _centroid_stats(X, y, ref_centroids)
        pacc, pmargin = _prototype_margin(X, y, prototypes)
        out['batches'].append(batch_id)
        out['energy_to_source'].append(round(_energy_distance(X0, X, seed=seed + i), 4))
        out['energy_adjacent'].append(round(_energy_distance(prev, X, seed=seed + 100 + i), 4) if i > 0 else 0.0)
        out['effective_rank'].append(round(_effective_rank(X), 4))
        out['principal_angle_deg'].append(round(_principal_angle(X0, X), 4) if i > 0 else 0.0)
        out['centroid_drift'].append(round(drift, 4) if np.isfinite(drift) else None)
        out['within_class_spread'].append(round(within, 4) if np.isfinite(within) else None)
        out['between_class_distance'].append(round(between, 4) if np.isfinite(between) else None)
        out['separation_ratio'].append(round(ratio, 4) if np.isfinite(ratio) else None)
        out['prototype_accuracy'].append(round(pacc, 4))
        out['prototype_margin'].append(round(pmargin, 4) if np.isfinite(pmargin) else None)
        prev = X
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=os.path.join('results', 'geometry_diagnostics.json'))
    args = parser.parse_args()
    out = compute_geometry(dataset=args.dataset, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(args.out)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the diagnostics**

Run: `python benchmarks/eval_geometry.py --dataset 224 --seed 42 --out results/geometry_diagnostics.json`

Expected: command exits 0 and writes `results/geometry_diagnostics.json`.

- [ ] **Step 3: Inspect generated ranges**

Run: `python -c "import json; d=json.load(open('results/geometry_diagnostics.json')); print(d['batches']); print(d['energy_to_source']); print(d['prototype_accuracy'])"`

Expected: prints ten batches and ten numeric values for each diagnostic.

---

### Task 2: Update Manuscript Figures

**Files:**
- Modify: `figures/make_manuscript_figures.py`
- Create/Update: `../images/fig_main_results.pdf`
- Create: `../images/fig_geometry.pdf`

**Interfaces:**
- Consumes: `results/revision_multiseed.json`, `results/revision_chronological.json`, `results/revision_proto_validity.json`, and `results/geometry_diagnostics.json`.
- Produces: updated main result figure including `sar_tta` and `lifecycle_memory`, plus a new geometry figure.

- [ ] **Step 1: Add missing methods to main figure lists**

Modify `METHODS` so it includes:

```python
    ("SAR-style TTA", "SAR-style TTA", C_UNL, "sar_tta"),
```

after `SSL+TTA`, and:

```python
    ("Lifecycle-memory", "Lifecycle-memory", C_LC, "lifecycle_memory"),
```

after `Lifecycle-replay`. Update `GROUP_BREAKS` so visual group separators still fall after source-only, unlabeled, few-shot, labeled, and lifecycle blocks.

- [ ] **Step 2: Add missing sequential methods to BWT/FWT and cross-dataset scatter if available**

Add `sar_tta` to the UCSD BWT/FWT panel. Add `lifecycle_memory` to UCSD BWT/FWT panel. Only add keys to the twin-array cross-dataset scatter if `twin_multiseed.json` contains those keys; otherwise keep the cross-dataset scatter to methods shared by both datasets.

- [ ] **Step 3: Add `fig_geometry()`**

Add a function that loads `geometry_diagnostics.json` and plots three panels:

```python
def fig_geometry():
    gd = load_json("geometry_diagnostics.json")
    batches = gd["batches"]
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.985, top=0.9, bottom=0.18, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(batches, gd["energy_to_source"], "-o", color=C_UNL, lw=1.0, ms=2.6, label="to batch 1")
    ax.plot(batches, gd["energy_adjacent"], "-s", color=C_FEW, lw=1.0, ms=2.6, label="to previous batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Energy distance")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(fig, ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(batches, gd["principal_angle_deg"], "-o", color=C_UNL, lw=1.0, ms=2.6)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Principal angle to batch 1 (deg)")
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(fig, ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(batches, [v * 100 for v in gd["prototype_accuracy"]], "-o", color=C_BAD, lw=1.0, ms=2.6, label="accuracy")
    ax2 = ax.twinx()
    ax2.plot(batches, gd["prototype_margin"], "--s", color="0.3", lw=0.9, ms=2.4, label="margin")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Batch-2 prototype accuracy (%)")
    ax2.set_ylabel("Nearest-prototype margin")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    panel_label(fig, ax, "c")
    out = os.path.join(OUTDIR, "fig_geometry.pdf")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out
```

- [ ] **Step 4: Generate figures**

Run: `python figures/make_manuscript_figures.py`

Expected: command prints paths for `fig_main_results.pdf`, `fig_cross_checks.pdf`, `fig_audit.pdf`, and `fig_geometry.pdf`.

---

### Task 3: Revise Manuscript Claims and Add Geometry Section

**Files:**
- Modify: `../main_revision.tex`

**Interfaces:**
- Consumes: generated numbers from `results/geometry_diagnostics.json` and updated figures in `../images`.
- Produces: manuscript text that cites the new geometry figure and narrows deployment claims.

- [ ] **Step 1: Add the geometry figure after the lifecycle/chronological evidence**

Insert a `figure` block after the current cross-check figure or after the protocol sensitivity paragraph:

```latex
\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{images/fig_geometry.pdf}
\caption{Feature-space geometry of UCSD drift. \textbf{a}, Energy distance from each batch to the source batch and to the preceding batch, showing why a fixed-source trigger saturates on a long aging trajectory. \textbf{b}, Principal angle between the leading source subspace and each batch subspace, quantifying nonstationary descriptor geometry. \textbf{c}, Accuracy and margin of batch-2 prototypes when applied to later batches, linking prototype invalidation to loss of geometric separation.}
\label{fig:geometry}
\end{figure}
\FloatBarrier
```

- [ ] **Step 2: Add a short results paragraph grounded in JSON values**

Read `results/geometry_diagnostics.json` and add a paragraph that states the numeric ranges for source energy distance, adjacent energy distance, principal angle, and prototype accuracy/margin. Use only generated numbers.

- [ ] **Step 3: Update abstract and discussion**

Add one abstract sentence that the geometry diagnostics explain why fixed-source triggers saturate and batch-local prototypes fail. In Discussion, replace broad “next step” language with design principles supported by the diagnostics.

- [ ] **Step 4: Lower deployment wording if no MCU data exists**

If no named MCU data is present, keep Table 5 as host-level component screening and ensure title/abstract/discussion do not imply device-level deployment. Prefer wording such as `deployment-relevant protocol`, `component-level footprint`, and `not device-level deployability evidence`.

---

### Task 4: Verify Build and Consistency

**Files:**
- Read: `../main_revision.log`
- Read: `results/geometry_diagnostics.json`

**Interfaces:**
- Consumes: outputs from Tasks 1--3.
- Produces: verified PDF and consistency checks.

- [ ] **Step 1: Run tests for the benchmark package**

Run: `pytest tests -q`

Expected: existing tests pass.

- [ ] **Step 2: Compile the manuscript twice**

From `D:/papers/odor_endnote/P11/20260929_nature`, run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main_revision.tex
pdflatex -interaction=nonstopmode -halt-on-error main_revision.tex
```

Expected: both commands exit 0 and write `main_revision.pdf`.

- [ ] **Step 3: Check for unresolved layout/reference failures**

Search `main_revision.log` for `Overfull \\hbox`, `Label(s) may have changed`, and undefined references.

Expected: no matches for severe overfull boxes or unresolved references. Underfull boxes may remain in narrow table cells.

- [ ] **Step 4: Check text/result consistency**

Search manuscript for geometry numbers and confirm they match `results/geometry_diagnostics.json`. Search for `microcontroller`, `MCU`, and `x86-64` and confirm the text does not claim device-level evidence.

---

## Self-Review

- Spec coverage: the plan covers figure consistency, geometry diagnostics, manuscript revision, deployment claim narrowing, and verification.
- Placeholder scan: no TBD/TODO placeholders are present.
- Type consistency: `geometry_diagnostics.json` keys used by figure and manuscript tasks match Task 1 outputs.
