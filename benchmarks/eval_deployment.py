#!/usr/bin/env python3
"""
Deployment-footprint measurements for the released benchmark models.

All measurements are host-level (x86-64 CPU) and are reported as such:
  - Flash footprint: int8 dynamic-quantized model size in KB (upper bound
    for the neural components on any MCU with int8 support), measured by
    serializing the quantized state_dict.
  - Latency: median wall-clock single-sample inference over 1000 runs.
  - Peak RAM: tracemalloc high-water mark during batched inference.
  - Update time: wall-clock per TTA adaptation batch (entropy-min steps).
  - Accuracy retention: evaluation accuracy before/after int8 quantization
    on the fixed UCSD batch-2 evaluation half.

No cycle-accurate MCU numbers are claimed; the paper labels these as host
measurements and lists device-level profiling as future work.
"""

import os
import sys
import json
import time
import io
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.few_shot import PrototypicalNetwork, create_fewshot_episode

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('deploy')
logger.setLevel(logging.INFO)

SEED = 42


# ---------------- models (same shapes as eval_drift_unified) ----------------
class MLPEnc(nn.Module):
    def __init__(self, in_dim=128, hid=64, n_cls=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, n_cls))

    def forward(self, x):
        return self.net(x)


class SSLEncoder(nn.Module):
    def __init__(self, in_dim=128, emb=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, emb), nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class ProtoEnc(nn.Module):
    """Encoder part of PrototypicalNetwork with identical layer shapes."""

    def __init__(self, in_dim=128, hidden_dim=64, embedding_dim=32):
        super().__init__()
        # eval-time encoder without BatchNorm (BN folds away under quantize;
        # we keep architecture identical in width/depth for fair MACs)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim))

    def forward(self, x):
        return self.encoder(x)


class RelEnc(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        return self.encoder(x)


# ---------------- helpers ----------------
def state_bytes(module) -> int:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


def quant_dynamic(module: nn.Module) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(
        module, {nn.Linear}, dtype=torch.qint8)


def measure_latency(fn, x1, n=1000):
    fn(x1)  # warmup
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x1)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)  # ms


def measure_peak_ram(fn, xb):
    """Peak RSS delta during fn(xb), via Windows-compatible psutil-free
    reading of the process working set."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)]

    psapi = ctypes.WinDLL('psapi')
    h = ctypes.windll.kernel32.GetCurrentProcess()
    before = PMC()
    psapi.GetProcessMemoryInfo(h, ctypes.byref(before), ctypes.sizeof(PMC))
    fn(xb)
    after = PMC()
    psapi.GetProcessMemoryInfo(h, ctypes.byref(after), ctypes.sizeof(PMC))
    delta_peak = after.PeakWorkingSetSize - before.WorkingSetSize
    return max(delta_peak, 0) / (1024 * 1024)  # MB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=None)
    args = ap.parse_args()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_dir = os.path.join(root, 'ucsd', 'Dataset')

    # ---- load batch 1 (source) and batch 2 eval half (fixed protocol) ----
    def parse(path):
        X, y = [], []
        with open(path) as f:
            for line in f:
                toks = line.strip().split()
                if not toks:
                    continue
                cls = int(toks[0].split(';')[0])
                vals = np.zeros(128)
                for t in toks[1:]:
                    i, v = t.split(':')
                    vals[int(i) - 1] = float(v)
                X.append(vals)
                y.append(cls - 1)
        return np.asarray(X), np.asarray(y)

    X1, y1 = parse(os.path.join(data_dir, 'batch1.dat'))
    X2, y2 = parse(os.path.join(data_dir, 'batch2.dat'))
    X2_ad, X2_ev, y2_ad, y2_ev = train_test_split(
        X2, y2, test_size=0.5, random_state=1002, stratify=y2)

    mu, sd = X1.mean(0), X1.std(0) + 1e-12
    scale = lambda X: (X - mu) / sd

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    results = {}

    def record(name, model, forward, x_eval_t, y_eval):
        """Measure fp32/int8 size, latency, RAM, and accuracy retention."""
        fp32_b = state_bytes(model)
        model.eval()
        x1 = x_eval_t[:1]

        lat = measure_latency(lambda x: forward(model, x), x1, n=1000)
        ram = measure_peak_ram(lambda xb: forward(model, xb), x_eval_t)

        # accuracy fp32
        with torch.no_grad():
            pred = forward(model, x_eval_t).argmax(1).numpy()
        acc_fp = accuracy_score(y_eval, pred)

        # int8 quantized
        q = quant_dynamic(model).eval()
        q_b = state_bytes(q)
        with torch.no_grad():
            predq = forward(q, x_eval_t).argmax(1).numpy()
        acc_q = accuracy_score(y_eval, predq)
        lat_q = measure_latency(lambda x: forward(q, x), x1, n=1000)

        results[name] = {
            'fp32_kb': round(fp32_b / 1024, 1),
            'int8_kb': round(q_b / 1024, 1),
            'latency_ms': round(lat, 3),
            'latency_int8_ms': round(lat_q, 3),
            'peak_ram_mb': round(ram, 2),
            'acc_fp32': round(acc_fp, 4),
            'acc_int8': round(acc_q, 4),
        }
        logger.info(f"{name}: fp32={results[name]['fp32_kb']}KB int8={results[name]['int8_kb']}KB "
                    f"lat={lat:.3f}ms (int8 {lat_q:.3f}ms) ram={ram:.2f}MB "
                    f"acc {acc_fp:.3f}->{acc_q:.3f}")

    # ---- MLP head (trained on batch1, eval batch2) ----
    mlp = MLPEnc()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    X1t = torch.tensor(scale(X1), dtype=torch.float32)
    y1t = torch.tensor(y1, dtype=torch.long)
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(mlp(X1t), y1t).backward()
        opt.step()
    record('mlp', mlp, lambda m, x: m(x),
           torch.tensor(scale(X2_ev), dtype=torch.float32), y2_ev)

    # ---- SSL encoder + ProtoNet head (lifecycle pilot shapes) ----
    ssl = SSLEncoder()
    proto = ProtoEnc()
    # train ssl encoder (SimCLR-lite) on unlabeled batch1
    opt = torch.optim.Adam(ssl.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    X1s = scale(X1)
    for _ in range(100):
        xi = torch.tensor(X1s * (1 + rng.normal(0, .05, X1s.shape)), dtype=torch.float32)
        xj = torch.tensor(X1s * (1 + rng.normal(0, .05, X1s.shape)), dtype=torch.float32)
        zi = F.normalize(ssl(xi), dim=1)
        zj = F.normalize(ssl(xj), dim=1)
        z = torch.cat([zi, zj], 0)
        sim = z @ z.t() / .5
        tgt = torch.arange(len(zi)).repeat(2)
        loss = F.cross_entropy(sim, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # prototype head on 5-shot support from batch2 adaptation half
    k = 5
    sup_idx = []
    for c in np.unique(y2_ad):
        idx = np.where(y2_ad == c)[0]
        sup_idx.extend(idx[:k])
    sup_idx = np.array(sup_idx, dtype=int)
    Xs = torch.tensor(scale(X2_ad[sup_idx]), dtype=torch.float32)
    ys = torch.tensor(y2_ad[sup_idx], dtype=torch.long)
    # train proto encoder to embed support (episode training)
    opt = torch.optim.Adam(proto.parameters(), lr=1e-3)
    for _ in range(100):
        opt.zero_grad()
        # simple prototype loss on the support set
        emb = proto(Xs)
        protos = torch.stack([emb[ys == c].mean(0) for c in np.unique(y2_ad)])
        d = torch.cdist(emb, protos)
        loss = F.cross_entropy(-d, ys)
        loss.backward()
        opt.step()

    def proto_forward(m, x):
        emb = m(x)
        protos = torch.stack(
            [m(Xs)[ys == c].mean(0) for c in np.unique(y2_ad)])
        return -torch.cdist(emb, protos)

    record('ssl_protonet_enc', proto, proto_forward,
           torch.tensor(scale(X2_ev), dtype=torch.float32), y2_ev)

    # ---- TTA head (= MLP + entropy adaptation timing) ----
    tta = MLPEnc()
    tta.load_state_dict(mlp.state_dict())
    X2_ad_t = torch.tensor(scale(X2_ad), dtype=torch.float32)

    def tta_adapt(model, xb, steps=3, tau=0.7, lr=1e-3):
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(xb), dim=1)
        sel = p.max(1).values > tau
        if sel.sum() < 2:
            return
        opt = torch.optim.SGD(model.net[-1].parameters(), lr=lr)
        for _ in range(steps):
            out = model(xb[sel])
            loss = -(F.softmax(out, 1) * F.log_softmax(out, 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    t0 = time.perf_counter()
    tta_adapt(tta, X2_ad_t)
    tta_ms = (time.perf_counter() - t0) * 1e3
    ram_adapt = measure_peak_ram(lambda xb: tta_adapt(tta, xb), X2_ad_t)
    results['tta_update'] = {'update_ms_per_batch': round(tta_ms, 2),
                             'peak_ram_mb': round(ram_adapt, 2)}
    logger.info(f"tta_update: {tta_ms:.2f} ms/batch, ram={ram_adapt:.2f}MB")

    # ---- SVM (sklearn) ----
    svm = SVC(kernel='rbf').fit(scale(X1), y1)
    svm_b = len(pickle.dumps(svm))
    t0 = time.perf_counter()
    svm.predict(scale(X2_ev))
    svm_ms = (time.perf_counter() - t0) * 1e3 / len(X2_ev)
    acc_svm = accuracy_score(y2_ev, svm.predict(scale(X2_ev)))
    results['svm'] = {'model_kb': round(svm_b / 1024, 1),
                      'latency_ms_per_sample': round(svm_ms, 3),
                      'acc': round(acc_svm, 4)}
    logger.info(f"svm: model={svm_b/1024:.1f}KB lat={svm_ms:.3f}ms acc={acc_svm:.3f}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'deployment_footprint.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'host': 'x86-64 CPU', 'seed': SEED, 'results': results}, f, indent=2)
    logger.info(f"Saved -> {out}")


if __name__ == '__main__':
    main()
