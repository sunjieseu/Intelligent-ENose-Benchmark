import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.eval_geometry import (
    _centroid_stats,
    _effective_rank,
    _prototype_margin,
)


def test_effective_rank_is_low_for_one_dimensional_data():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    rank = _effective_rank(x)

    assert 0.9 <= rank <= 1.1


def test_centroid_stats_report_drift_and_separation_ratio():
    x_ref = np.array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.2, 0.0]])
    y_ref = np.array([0, 0, 1, 1])
    ref_centroids, *_ = _centroid_stats(x_ref, y_ref)
    x_cur = x_ref + np.array([1.0, 0.0])

    _, within, between, ratio, drift = _centroid_stats(x_cur, y_ref, ref_centroids)

    assert within > 0
    assert between > within
    assert ratio > 1
    assert 0.9 <= drift <= 1.1


def test_prototype_margin_collapses_when_prototypes_are_wrong():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
    y = np.array([0, 0, 1, 1])
    good = {0: np.array([0.0, 0.0]), 1: np.array([3.0, 0.0])}
    swapped = {0: np.array([3.0, 0.0]), 1: np.array([0.0, 0.0])}

    good_acc, good_margin = _prototype_margin(x, y, good)
    bad_acc, bad_margin = _prototype_margin(x, y, swapped)

    assert good_acc == 1.0
    assert bad_acc == 0.0
    assert good_margin > 0
    assert bad_margin < 0
