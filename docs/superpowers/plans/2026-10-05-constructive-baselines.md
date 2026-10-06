# Constructive Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a constructive lifecycle-memory baseline, one robust modern TTA control, and portable deployment-footprint accounting, then update the manuscript only from generated results.

**Architecture:** Extend the existing `benchmarks/eval_revision.py` harness with two method names that reuse the current UCSD split, scaler, sequential matrix, and metrics. Extend `benchmarks/eval_deployment.py` with model-independent parameter/MAC/state accounting rather than claiming device-level latency. Keep manuscript edits evidence-bound.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn, pytest, LaTeX.

**Spec:** Approved in chat on 2026-10-05: add `lifecycle_memory`, add one robust TTA baseline (`sar_tta`), add deployment parameter/MAC/state metrics, run real local experiments, update `main_revision.tex` from JSON.

## Global Constraints

- No production benchmark logic without a failing pytest first.
- Do not invent numbers; manuscript values must come from JSON written by benchmark scripts.
- Do not claim MCU/device latency without real device profiling.
- Keep changes surgical in existing benchmark files.

---

### Task 1: Lifecycle Memory Baseline

**Files:**
- Modify: `benchmarks/eval_revision.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: `update_memory_prototypes(ssl_model, model, Xa_s, anchor_alpha=0.05, memory_alpha=0.25, conf_tau=0.75) -> int`
- Produces method name: `lifecycle_memory` accepted by `eval_sequential_p`
- Consumes existing `predict_proto`, `adapt_head`, `eval_sequential_p`

- [ ] **Step 1: Write failing tests**

```python
def test_update_memory_prototypes_blends_toward_confident_clusters():
    import numpy as np
    import torch
    import torch.nn as nn
    from benchmarks.eval_revision import update_memory_prototypes

    class IdentitySSL:
        def transform(self, X):
            return np.asarray(X, dtype=float)

    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        head.bias.zero_()
    model = {
        'head': head,
        'protos': {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])},
        'anchor_protos': {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])},
        'mode': 'lifecycle_memory',
    }
    updated = update_memory_prototypes(
        IdentitySSL(), model, np.array([[4.0, 0.0], [0.0, 4.0]]),
        anchor_alpha=0.0, memory_alpha=0.5, conf_tau=0.75)
    assert updated == 2
    assert np.allclose(model['protos'][0], np.array([2.5, 0.0]))
    assert np.allclose(model['protos'][1], np.array([0.0, 2.5]))

def test_update_memory_prototypes_anchor_limits_drift():
    import numpy as np
    import torch
    import torch.nn as nn
    from benchmarks.eval_revision import update_memory_prototypes

    class IdentitySSL:
        def transform(self, X):
            return np.asarray(X, dtype=float)

    head = nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.bias.zero_()
    model = {
        'head': head,
        'protos': {0: np.array([1.0, 0.0])},
        'anchor_protos': {0: np.array([1.0, 0.0])},
        'mode': 'lifecycle_memory',
    }
    updated = update_memory_prototypes(
        IdentitySSL(), model, np.array([[5.0, 0.0]]),
        anchor_alpha=0.2, memory_alpha=0.5, conf_tau=0.0)
    assert updated == 1
    assert np.allclose(model['protos'][0], np.array([2.6, 0.0]))
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_models.py::TestRevisionBaselines -v`

Expected: FAIL because `update_memory_prototypes` is not defined.

- [ ] **Step 3: Implement minimal code**

Add `update_memory_prototypes`, initialize `anchor_protos`, handle `lifecycle_memory` alongside lifecycle variants, and add it to `run_multiseed`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_models.py::TestRevisionBaselines -v`

Expected: PASS.

---

### Task 2: Robust TTA Control

**Files:**
- Modify: `benchmarks/eval_revision.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: `adapt_head_sar(ssl_model, model, Xa_s, conf_tau=0.7, margin=0.4, steps=1, lr=5e-4, reg=1e-3) -> int`
- Produces method name: `sar_tta` accepted by `eval_sequential_p`

- [ ] **Step 1: Write failing test**

```python
def test_adapt_head_sar_skips_high_entropy_samples():
    import numpy as np
    import torch
    import torch.nn as nn
    from benchmarks.eval_revision import adapt_head_sar

    class IdentitySSL:
        def transform(self, X):
            return np.asarray(X, dtype=float)

    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    model = {'head': head, 'mode': 'sar_tta'}
    selected = adapt_head_sar(IdentitySSL(), model, np.array([[1.0, 0.0], [0.0, 1.0]]), conf_tau=0.9)
    assert selected == 0
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_models.py::TestRevisionBaselines::test_adapt_head_sar_skips_high_entropy_samples -v`

Expected: FAIL because `adapt_head_sar` is not defined.

- [ ] **Step 3: Implement minimal code**

Add entropy/confidence filtered adaptation with an L2 anchor to the initial head parameters, and route `sar_tta` through sequential evaluation.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_models.py::TestRevisionBaselines -v`

Expected: PASS.

---

### Task 3: Deployment Footprint Accounting

**Files:**
- Modify: `benchmarks/eval_deployment.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces: `linear_macs(module: nn.Module) -> int`
- Produces: `count_parameters(module: nn.Module) -> int`
- Produces JSON fields: `params`, `macs_per_sample`, `state_kb`

- [ ] **Step 1: Write failing test**

```python
def test_linear_macs_counts_linear_layers_only():
    import torch.nn as nn
    from benchmarks.eval_deployment import linear_macs, count_parameters

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    assert linear_macs(model) == 18
    assert count_parameters(model) == 23
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_models.py::TestRevisionBaselines::test_linear_macs_counts_linear_layers_only -v`

Expected: FAIL because the helper functions are missing.

- [ ] **Step 3: Implement minimal code**

Add helpers and include metrics in each neural-model result plus SVM support-vector state size.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_models.py::TestRevisionBaselines -v`

Expected: PASS.

---

### Task 4: Run Experiments and Update Manuscript

**Files:**
- Modify: `results/revision_multiseed.json`
- Modify: `results/deployment_footprint.json`
- Modify: `main_revision.tex`

**Interfaces:**
- Consumes generated aggregate JSON values from Tasks 1-3.
- Produces evidence-bound manuscript text and tables.

- [ ] **Step 1: Run tests**

Run: `pytest tests/test_models.py::TestRevisionBaselines -v`

Expected: PASS.

- [ ] **Step 2: Run UCSD multiseed benchmark**

Run: `python benchmarks/eval_revision.py --dataset 224 --seeds 7 21 42 87 123`

Expected: writes `results/revision_multiseed.json` including `lifecycle_memory` and `sar_tta`.

- [ ] **Step 3: Run deployment benchmark**

Run: `python benchmarks/eval_deployment.py`

Expected: writes `results/deployment_footprint.json` including `params`, `macs_per_sample`, and state-size fields.

- [ ] **Step 4: Update manuscript**

Use exact aggregate values from JSON in `main_revision.tex`; avoid claims stronger than the observed results.

- [ ] **Step 5: Verify manuscript references**

Run a LaTeX build if available, otherwise grep modified values and report that compilation was not run.

---

## Self-Review

- Spec coverage: constructive baseline, robust TTA control, and deployment-footprint critique are each covered by Tasks 1-4.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: helper names in tests match produced interfaces.
