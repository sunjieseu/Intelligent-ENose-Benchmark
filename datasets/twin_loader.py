# -*- coding: utf-8 -*-
"""
Loader for the UCI Twin Gas Sensor Arrays dataset (UCI ID 361).

Five nominally identical 8-channel MOX arrays (B1..B5) were exposed to
4 gases (CO, ethanol, ethylene, methane) x 10 concentration levels each,
following Fonollosa et al., Sensors and Actuators B 236 (2016).
Source files: data1/B{u}_G{gas}_F{conc}_R{rep}.txt, each a raw resistance
trace (100 Hz, ~530 s, columns: time + 8 channels in KOhm).

Feature convention: baseline-referenced relative response
    r(t) = (R(t) - R0) / R0,  R0 = median of the first 10% of samples
followed by the SAME 8 descriptors per channel as the UCSD/GSALC
protocols (max, rel. max, mean slope, integral, t_to_max, steady-state,
recovery slope, area ratio), downsampled to 10 Hz -> 64-dim vectors.

Extracted features are cached in twin_features64.npz after the first run.
"""

import os
import re
import logging
import numpy as np

logger = logging.getLogger('twin_loader')

RE = re.compile(r'B(\d)_G([A-Za-z]+)_F(\d+)_R(\d+)\.txt')
DOWNSAMPLE = 10  # 100 Hz -> 10 Hz


def extract_8d(curve):
    """The 8 benchmark descriptors on a single baseline-referenced curve."""
    c = np.asarray(curve, dtype=float)
    n = len(c)
    baseline = np.median(c[: max(n // 10, 1)])
    r = (c - baseline) / (abs(baseline) + 1e-12)
    max_resp = np.max(r) - r[: max(n // 10, 1)].mean()
    rel_max = max_resp / (abs(r[: max(n // 10, 1)].mean()) + 1e-12)
    slope = np.mean(np.diff(r)) if n > 1 else 0.0
    integral = np.trapezoid(r) if n > 1 else 0.0
    max_idx = int(np.argmax(r))
    time_to_max = max_idx / max(n - 1, 1)
    n_tail = max(int(n * 0.1), 1)
    steady = np.mean(r[-n_tail:])
    n_rec = max(int(n * 0.2), 2)
    rec_slope = np.mean(np.diff(r[-n_rec:])) if n >= n_rec else 0.0
    area_ratio = integral / (abs(max_resp) + 1e-12)
    return [max_resp, rel_max, slope, integral, time_to_max, steady,
            rec_slope, area_ratio]


def load_twin(data_dir, cache=True):
    """Parse all 640 experiment files.

    Args:
        data_dir: directory containing the unpacked UCI 361 'data1' folder
        cache: write/read twin_features64.npz beside data1/

    Returns:
        X      [640, 64] float64 feature matrix
        y      [640] int64 gas labels (0=CO, 1=Ea, 2=Ey, 3=Me)
        unit   [640] int64 board number 1..5
        rep    [640] int64 repetition index
        gases  list of gas names in label order
    """
    data1 = os.path.join(data_dir, 'data1') if os.path.isdir(
        os.path.join(data_dir, 'data1')) else data_dir
    cache_path = os.path.join(os.path.dirname(data1), 'twin_features64.npz')
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path)
        return z['X'], z['y'], z['unit'], z['rep'], list(z['gases'])

    rows = []
    for fname in sorted(os.listdir(data1)):
        m = RE.match(fname)
        if not m:
            continue
        b, g, c, r = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        arr = np.loadtxt(os.path.join(data1, fname))
        arr = arr[::DOWNSAMPLE]
        feats = []
        for ch in range(1, 9):
            feats.extend(extract_8d(arr[:, ch]))
        rows.append((b, g, c, r, np.asarray(feats, dtype=float)))

    gases = sorted(set(r[1] for r in rows))
    gmap = {g: i for i, g in enumerate(gases)}
    X = np.stack([r[4] for r in rows])
    y = np.array([gmap[r[1]] for r in rows])
    unit = np.array([int(r[0]) for r in rows])
    rep = np.array([r[3] for r in rows])
    logger.info(f'Twin: {X.shape[0]} samples, {X.shape[1]} dims, '
                f'gases={gases}, units={sorted(set(unit.tolist()))}')
    if cache:
        np.savez(cache_path, X=X, y=y, unit=unit, rep=rep,
                 gases=np.array(gases))
    return X, y, unit, rep, gases
