"""Tests for revision benchmark result logging."""

import os
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_eval_static_p_returns_per_batch_values_when_requested():
    """Catch missing per-batch logging for static-method common-window audits."""
    from benchmarks.eval_revision import eval_static_p

    source_x = np.array([
        [-2.0, -2.0],
        [-2.0, -1.5],
        [2.0, 2.0],
        [2.0, 1.5],
    ])
    source_y = np.array([0, 0, 1, 1])

    batches = [(source_x, source_y)]
    for offset in range(1, 10):
        target_x = np.array([
            [-2.0 - offset * 0.01, -2.0],
            [-2.0 - offset * 0.01, -1.5],
            [2.0 + offset * 0.01, 2.0],
            [2.0 + offset * 0.01, 1.5],
        ])
        target_y = np.array([0, 0, 1, 1])
        batches.append((target_x, target_y))

    scaler = (np.zeros(2), np.ones(2))

    result = eval_static_p(batches, "svm", scaler, seed=7, test_size=0.5, return_per_batch=True)

    assert result == {
        "avg_acc": 1.0,
        "avg_acc_b3_b10": 1.0,
        "per_batch_acc": {
            "batch2": 1.0,
            "batch3": 1.0,
            "batch4": 1.0,
            "batch5": 1.0,
            "batch6": 1.0,
            "batch7": 1.0,
            "batch8": 1.0,
            "batch9": 1.0,
            "batch10": 1.0,
        },
    }


def test_sample_std_uses_unbiased_denominator_for_seed_summaries():
    """Catch population-SD underestimation in five-seed manuscript summaries."""
    from benchmarks.eval_revision import seed_std

    vals = [45.53, 50.26, 50.22, 39.25, 42.77]

    observed = seed_std(vals)

    assert round(observed, 2) == 4.78


def test_aggregate_chronological_preserves_seed42_compatibility():
    """Five-seed chronological summaries should keep legacy top-level fields."""
    from benchmarks.eval_revision import aggregate_chronological_results

    per_seed = {
        "7": {
            "prequential_acc": 0.40,
            "per_batch_acc": [0.1, 0.2],
            "per_batch_max_class_ratio": [0.5, 0.6],
        },
        "42": {
            "prequential_acc": 0.50,
            "per_batch_acc": [0.3, 0.4],
            "per_batch_max_class_ratio": [0.5, 0.6],
        },
    }

    result = aggregate_chronological_results([7, 42], per_seed, legacy_seed=42)

    assert result["prequential_acc"] == 0.50
    assert result["per_batch_acc"] == [0.3, 0.4]
    assert result["seeds"] == [7, 42]
    assert result["aggregate"]["prequential_acc_mean"] == 0.45
    assert round(result["aggregate"]["prequential_acc_std"], 4) == 0.0707
    assert result["aggregate"]["per_batch_acc_mean"] == [0.2, 0.3]


def test_ctta_light_method_keys_are_registered():
    """Modern TTA-light baselines should be part of the multiseed protocol."""
    from benchmarks.eval_revision import SEQUENTIAL_METHODS

    assert "cotta_tta" in SEQUENTIAL_METHODS
    assert "eata_tta" in SEQUENTIAL_METHODS
    assert "rotta_tta" in SEQUENTIAL_METHODS
