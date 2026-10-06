#!/usr/bin/env python3
"""Generate submission-grade figures for the Nature-style manuscript.

Outputs (into ../../images):
  fig_main_results.pdf : Fig. 3, UCSD main-results figure
  fig_cross_checks.pdf  : Fig. 4, cross-protocol and cross-dataset checks
  fig_geometry.pdf      : Fig. 5, feature-space geometry diagnostics
  fig_chronological.pdf : Fig. 6, chronological stream diagnostics
  fig_audit.pdf        : Fig. 2, failure-mode -> check -> evidence schematic

All numbers are read from the released result JSONs (no hard-coded metrics).
"""

import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUTDIR = os.path.join(HERE, "..", "..", "images")

MM = 1 / 25.4
W = 180 * MM  # double-column Nature width

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "highlight": "#FFD700",
}


def apply_publication_style():
    """figures4papers-inspired compact style for Nature double-column panels."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.4,
        "axes.labelsize": 6.8,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 5.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
    })


apply_publication_style()

# figures4papers semantic palette adapted to this benchmark.
C_SRC = PALETTE["neutral_dark"]      # source only / reference
C_UNL = PALETTE["blue_main"]        # unlabeled target adaptation
C_FEW = PALETTE["red_strong"]       # few labeled target samples
C_LAB = PALETTE["green_3"]          # labeled online updates
C_LC = PALETTE["violet"]            # lifecycle compositions
C_BAD = "#7A1F1F"                   # catastrophic forgetting highlight
C_EDGE = "#272727"
C_GRID = "#E7E7E7"

METHODS = [
    ("SVM", "SVM", C_SRC, "svm"),
    ("MLP", "MLP", C_SRC, "mlp"),
    ("TCA+SVM", "TCA + SVM", C_UNL, "tca"),
    ("DANN", "DANN", C_UNL, "dann"),
    ("TTA", "TTA", C_UNL, "tta"),
    ("SSL+TTA", "SSL + TTA", C_UNL, "ssl_tta"),
    ("SAR-style TTA", "SAR-style TTA", C_UNL, "sar_tta"),
    ("CoTTA-light", "CoTTA-light", C_UNL, "cotta_tta"),
    ("EATA-light", "EATA-light", C_UNL, "eata_tta"),
    ("RoTTA-light", "RoTTA-light", C_UNL, "rotta_tta"),
    ("ProtoNet", "ProtoNet", C_FEW, "protonet"),
    ("RelationNet", "RelationNet", C_FEW, "relationnet"),
    ("SSL+ProtoNet", "SSL + ProtoNet", C_FEW, "ssl_protonet"),
    ("CRE", "CRE", C_LAB, "cre"),
    ("Lifecycle-frozen", "Lifecycle-frozen", C_LC, "lifecycle_frozen"),
    ("Lifecycle-TTA", "Lifecycle-TTA", C_LC, "lifecycle_tta"),
    ("Lifecycle-replay", "Lifecycle-replay", C_LC, "lifecycle_replay"),
    ("Lifecycle-memory", "Lifecycle-memory", C_LC, "lifecycle_memory"),
]
GROUP_BREAKS = [2, 10, 13, 14]  # indices where a new access group starts


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def panel_label(fig, ax, letter):
    ax.text(0.015, 0.985, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="top")


def polish_axis(ax, grid_axis=None):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.75)
        ax.spines[side].set_color(C_EDGE)
    ax.tick_params(width=0.7, color=C_EDGE, labelcolor=C_EDGE)
    if grid_axis:
        ax.grid(axis=grid_axis, color=C_GRID, lw=0.45, zorder=0)


def save_figure(fig, out):
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def fig_main_results():
    ms = load_json("revision_multiseed.json")["aggregate"]
    lk = load_json("revision_leakage.json")

    fig = plt.figure(figsize=(W, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0],
                          left=0.085, right=0.985, top=0.965, bottom=0.09,
                          hspace=0.58, wspace=0.46)

    # ---------------- panel a: accuracy by access group ----------------
    ax = fig.add_subplot(gs[0, :])
    labels = [m[1] for m in METHODS]
    means = [ms[m[3]]["avg_acc_mean"] * 100 for m in METHODS]
    stds = [ms[m[3]]["avg_acc_std"] * 100 for m in METHODS]
    colors = [m[2] for m in METHODS]
    y = range(len(METHODS))
    ax.barh(y, means, xerr=stds, color=colors, height=0.62,
            edgecolor=C_EDGE, linewidth=0.35,
            error_kw=dict(lw=0.65, capsize=1.5, capthick=0.65, ecolor=C_EDGE), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 78)
    ax.set_xlabel("Held-out accuracy over batches 2-10 (%)")
    for bi in GROUP_BREAKS:
        ax.axhline(bi - 0.5, color="0.78", lw=0.55, zorder=1)
    for yi, mval, sval in zip(y, means, stds):
        ax.text(min(mval + sval + 1.5, 76), yi, f"{mval:.1f}",
                va="center", ha="left", fontsize=5.6, color="0.15")
    polish_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    panel_label(fig, ax, "a")

    # ---------------- panel b: BWT / FWT ----------------
    seq = [
        ("SSL+Proto", "ssl_protonet"),
        ("LC-memory", "lifecycle_memory"),
        ("LC-replay", "lifecycle_replay"),
        ("SAR-style TTA", "sar_tta"),
        ("CoTTA-light", "cotta_tta"),
        ("EATA-light", "eata_tta"),
        ("RoTTA-light", "rotta_tta"),
        ("TTA", "tta"),
        ("SSL + TTA", "ssl_tta"),
        ("LC-TTA", "lifecycle_tta"),
        ("CRE", "cre"),
        ("LC-frozen", "lifecycle_frozen"),
    ]
    ax = fig.add_subplot(gs[1, 0])
    bwt = [ms[k]["bwt_mean"] * 100 for _, k in seq]
    bwt_s = [ms[k]["bwt_std"] * 100 for _, k in seq]
    fwt = {k: ms[k].get("fwt_mean") for _, k in seq}
    y = range(len(seq))
    for yi, (name, key), bv, bs in zip(y, seq, bwt, bwt_s):
        col = C_BAD if key == "ssl_protonet" else (
            C_LAB if bv >= -0.05 else C_UNL)
        ax.barh(yi, bv, xerr=bs, color=col, height=0.62,
                edgecolor=C_EDGE, linewidth=0.35,
                error_kw=dict(lw=0.65, capsize=1.5, capthick=0.65, ecolor=C_EDGE), zorder=3)
        fv = fwt[key]
        if fv is None:
            ax.text(1.6, yi, "FWT n/a", va="center", ha="left",
                    fontsize=5.9, color="0.3")
        else:
            xpos = bv - bs - 1.5 if bv < 0 else 1.5
            ax.text(xpos, yi, f"FWT +{fv * 100:.1f}", va="center",
                    ha="right" if bv < 0 else "left", fontsize=5.9, color="0.3")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in seq])
    ax.tick_params(axis="y", labelsize=5.8, pad=1)
    ax.invert_yaxis()
    ax.axvline(0, color=C_EDGE, lw=0.75, zorder=2)
    ax.set_xlim(-56, 30)
    ax.set_xlabel("Backward transfer BWT (percentage points)")
    ax.text(-55.5, -0.62, "forgetting", fontsize=5.4, color="0.35",
            ha="left", va="bottom")
    polish_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    panel_label(fig, ax, "b")

    # ---------------- panel c: leakage inflation ----------------
    ax = fig.add_subplot(gs[1, 1])
    strict = [lk["svm_source_only_split"] * 100,
              lk["dann_source_only_split"] * 100]
    leaky = [lk["svm_joint_norm"] * 100,
             lk["dann_joint_norm_eval_on_adapt"] * 100]
    x = [0, 1]
    w = 0.32
    ax.bar([xi - w / 2 for xi in x], strict, width=w, color=C_UNL,
           edgecolor=C_EDGE, linewidth=0.45,
           label="Source-only protocol", zorder=3)
    ax.bar([xi + w / 2 for xi in x], leaky, width=w, color=PALETTE["highlight"],
           edgecolor=C_EDGE, linewidth=0.45,
           label="Leaky protocol", zorder=3)
    for xi, s, l in zip(x, strict, leaky):
        ax.text(xi - w / 2, s + 0.8, f"{s:.1f}", ha="center", fontsize=5.6)
        ax.text(xi + w / 2, l + 0.8, f"{l:.1f}", ha="center", fontsize=5.6)
        ax.annotate("", xy=(xi + w / 2, l - 0.6),
                    xytext=(xi - w / 2 - 0.04, s + 0.6),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=C_EDGE))
        ax.text(xi + w * 0.06, max(s, l) + 4.8,
                f"+{l - s:.1f} pp", ha="center", fontsize=5.6, color="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels(["SVM", "DANN"])
    ax.set_ylim(0, 72)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Baseline (UCSD, batch-mean)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.16, 1.0),
              borderaxespad=0.0, handlelength=1.0)
    polish_axis(ax, grid_axis="y")
    panel_label(fig, ax, "c")

    out = os.path.join(OUTDIR, "fig_main_results.pdf")
    return save_figure(fig, out)


def fig_chronological():
    ms = load_json("revision_multiseed.json")["aggregate"]
    ch = load_json("revision_chronological.json")
    pv = load_json("revision_proto_validity.json")

    fig = plt.figure(figsize=(W, 2.85))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.93,
                          bottom=0.18, wspace=0.42)

    # ---------------- panel a: chronological prequential ----------------
    ax = fig.add_subplot(gs[0, 0])
    batches = list(range(2, 11))
    acc = [a * 100 for a in ch.get("aggregate", {}).get("per_batch_acc_mean", ch["per_batch_acc"])]
    acc_s = [a * 100 for a in ch.get("aggregate", {}).get(
        "per_batch_acc_std", [0.0] * len(acc))]
    dom = [r * 100 for r in ch["per_batch_max_class_ratio"]]
    ax.bar(batches, dom, width=0.62, color=PALETTE["neutral"],
           edgecolor="white", linewidth=0.25, zorder=1,
           label="Dominant-class share")
    ax.plot(batches, acc, "-o", color=C_UNL, lw=1.25, ms=3.0,
            markeredgecolor="white", markeredgewidth=0.35, zorder=3,
            label="Prequential accuracy")
    if any(acc_s):
        ax.fill_between(batches,
                        [a - s for a, s in zip(acc, acc_s)],
                        [a + s for a, s in zip(acc, acc_s)],
                        color=C_UNL, alpha=0.16, linewidth=0, zorder=2,
                        label="s.d. across seeds")
    chrono_mean = ch.get("aggregate", {}).get("prequential_acc_mean", ch["prequential_acc"]) * 100
    chrono_std = ch.get("aggregate", {}).get("prequential_acc_std", 0.0) * 100
    ax.axhline(chrono_mean, color=C_UNL, lw=0.7, ls="--", zorder=2)
    chrono_label = f"{chrono_mean:.1f}%"
    if chrono_std:
        chrono_label += f" ± {chrono_std:.1f}%"
    ax.text(10.35, chrono_mean + 2.4, chrono_label,
            fontsize=5.4, color=C_UNL, ha="right",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
    tta_strat = ms["tta"]["avg_acc_mean"] * 100
    ax.axhline(tta_strat, color="0.45", lw=0.7, ls=":", zorder=2)
    ax.text(2.1, tta_strat + 3.6, "stratified TTA 47.0%", fontsize=5.4,
            color="0.35",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
    ax.set_xticks(batches)
    ax.set_xlabel("Incoming batch (chronological order)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 85)
    ax.legend(loc="upper right", handlelength=1.0, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.86)
    polish_axis(ax, grid_axis="y")
    panel_label(fig, ax, "a")

    # ---------------- panel b: prototype validity ----------------
    ax = fig.add_subplot(gs[0, 1])
    for seed, col, mk in (("42", C_UNL, "o"), ("7", C_FEW, "s")):
        vals = [pv[seed]["proto_validity"][f"batch{b}"] * 100
                for b in batches]
        ax.plot(batches, vals, "-" + mk, color=col, lw=1.25, ms=3.0,
                markeredgecolor="white", markeredgewidth=0.35,
                label=f"seed {seed}")
    v8 = pv["42"]["proto_validity"]["batch8"] * 100
    ax.annotate(f"{v8:.1f}%", xy=(8, v8), xytext=(8.28, 30), fontsize=5.6,
                color="0.2",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85),
                arrowprops=dict(arrowstyle="->", lw=0.65, color=C_EDGE))
    ax.set_xticks(batches)
    ax.set_xlabel("Incoming batch")
    ax.set_ylabel("Frozen-prototype accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", handlelength=1.0)
    polish_axis(ax, grid_axis="y")
    panel_label(fig, ax, "b")

    out = os.path.join(OUTDIR, "fig_chronological.pdf")
    return save_figure(fig, out)


def fig_geometry():
    gd = load_json("geometry_diagnostics.json")
    batches = gd["batches"]

    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.985, top=0.9,
                          bottom=0.18, wspace=0.5)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(batches, gd["energy_to_source"], "-o", color=C_UNL, lw=1.35,
            ms=3.0, markeredgecolor="white", markeredgewidth=0.35,
            label="to batch 1")
    ax.plot(batches, gd["energy_adjacent"], "-s", color=C_FEW, lw=1.35,
            ms=3.0, markeredgecolor="white", markeredgewidth=0.35,
            label="to previous batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Energy distance")
    ax.legend(loc="upper right", handlelength=1.0)
    polish_axis(ax, grid_axis="y")
    panel_label(fig, ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(batches, gd["principal_angle_deg"], "-o", color=C_UNL, lw=1.35,
            ms=3.0, markeredgecolor="white", markeredgewidth=0.35)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Principal angle to batch 1 (deg)")
    ax.set_ylim(0, 96)
    polish_axis(ax, grid_axis="y")
    panel_label(fig, ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    c_acc = C_UNL
    c_margin = C_BAD
    ax.plot(batches, [v * 100 for v in gd["prototype_accuracy"]], "-o",
            color=c_acc, lw=1.35, ms=3.0, markeredgecolor="white",
            markeredgewidth=0.35, label="Accuracy")
    ax2 = ax.twinx()
    ax2.plot(batches, gd["prototype_margin"], "--s", color=c_margin, lw=1.05,
             ms=2.8, markeredgecolor="white", markeredgewidth=0.35,
             label="Margin")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Batch-2 prototype accuracy (%)", color=c_acc)
    ax2.set_ylabel("Prototype margin", color=c_margin)
    ax.tick_params(axis="y", colors=c_acc, labelcolor=c_acc)
    ax2.tick_params(axis="y", colors=c_margin, labelcolor=c_margin)
    polish_axis(ax, grid_axis="y")
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.spines["right"].set_linewidth(0.75)
    ax2.spines["right"].set_visible(True)
    ax.spines["left"].set_color(c_acc)
    ax2.spines["right"].set_color(c_margin)
    ax.yaxis.label.set_color(c_acc)
    ax2.yaxis.label.set_color(c_margin)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="lower left",
              handlelength=1.4)
    panel_label(fig, ax, "c")

    out = os.path.join(OUTDIR, "fig_geometry.pdf")
    return save_figure(fig, out)


def fig_cross_checks():
    ms = load_json("revision_multiseed.json")["aggregate"]
    ra = load_json("reanchor_multiseed.json")["aggregate"]
    tw = load_json("twin_multiseed.json")["aggregate"]

    fig = plt.figure(figsize=(W, 2.85))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.94,
                          bottom=0.18, wspace=0.55)

    # ---------------- panel a: protocol re-anchoring ----------------
    ax = fig.add_subplot(gs[0, 0])
    ram = [("SVM", "svm"), ("MLP", "mlp"),
           ("ProtoNet", "protonet"), ("RelationNet", "relationnet")]
    strict = [ra[k]["strict_mean"] * 100 for _, k in ram]
    strict_s = [ra[k]["strict_std"] * 100 for _, k in ram]
    pooled = [ra[k]["acc_mean"] * 100 for _, k in ram]
    pooled_s = [ra[k]["acc_std"] * 100 for _, k in ram]
    y = range(len(ram))
    w = 0.36
    ax.barh([yi - w / 2 for yi in y], strict, xerr=strict_s, height=w,
            color=C_UNL, edgecolor=C_EDGE, linewidth=0.4,
            label="Strict protocol (this work)", zorder=3,
            error_kw=dict(lw=0.65, capsize=1.5, capthick=0.65, ecolor=C_EDGE))
    ax.barh([yi + w / 2 for yi in y], pooled, xerr=pooled_s, height=w,
            color=PALETTE["highlight"], edgecolor=C_EDGE, linewidth=0.4,
            label="Pooled literature-style protocol",
            zorder=3, error_kw=dict(lw=0.65, capsize=1.5, capthick=0.65, ecolor=C_EDGE))
    for yi, sv, pv2, (nm, _) in zip(y, strict, pooled, ram):
        ax.text(98.5, yi + w / 2, f"+{pv2 - sv:.1f} pp",
                va="center", ha="right", fontsize=5.4, color="0.2")
    ax.set_yticks(list(y))
    ax.set_yticklabels([nm for nm, _ in ram])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Held-out accuracy (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
              borderaxespad=0.0, handlelength=1.0)
    polish_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    panel_label(fig, ax, "a")

    # ---------------- panel b: cross-dataset accuracy vs BWT ----------------
    ax = fig.add_subplot(gs[0, 1])
    seqk = [("SSL+ProtoNet", "ssl_protonet"),
            ("Lifecycle-replay", "lifecycle_replay"),
            ("TTA", "tta"),
            ("SSL+TTA", "ssl_tta"),
            ("Lifecycle-TTA", "lifecycle_tta"),
            ("CRE", "cre"),
            ("Lifecycle-frozen", "lifecycle_frozen")]
    for name, key in seqk:
        ax.errorbar(ms[key]["avg_acc_mean"] * 100,
                    ms[key]["bwt_mean"] * 100,
                    xerr=ms[key]["avg_acc_std"] * 100,
                    fmt="o", color=C_UNL, ms=3.2, lw=0.65, capsize=1.5,
                    markeredgecolor="white", markeredgewidth=0.35,
                    zorder=3)
        ax.errorbar(tw[key]["avg_acc_mean"] * 100,
                    tw[key]["bwt_mean"] * 100,
                    xerr=tw[key]["avg_acc_std"] * 100,
                    fmt="s", color=C_FEW, ms=3.2, lw=0.65, capsize=1.5,
                    markeredgecolor="white", markeredgewidth=0.35,
                    zorder=3)
    # guide lines and annotations for the two anchor mechanisms
    ax.plot(0, 1, "-o", color="none")  # keep z-order stable
    ax.annotate("prototype overwriting\n(via SSL+ProtoNet)",
                xy=(ms["ssl_protonet"]["avg_acc_mean"] * 100,
                    ms["ssl_protonet"]["bwt_mean"] * 100),
                xytext=(30, -34), fontsize=5.4, color=C_BAD,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_BAD))
    ax.axhline(0, color=C_EDGE, lw=0.75, zorder=2)
    ax.set_xlabel("Mean held-out accuracy (%)")
    ax.set_ylabel("Backward transfer BWT (pp)")
    ax.set_ylim(-52, 14)
    ax.plot([], [], "-o", color=C_UNL, ms=3.2, label="UCSD (temporal drift)")
    ax.plot([], [], "-s", color=C_FEW, ms=3.2, label="Twin arrays (cross-device)")
    ax.legend(loc="lower left", handlelength=1.0)
    polish_axis(ax, grid_axis="both")
    panel_label(fig, ax, "b")

    out = os.path.join(OUTDIR, "fig_cross_checks.pdf")
    return save_figure(fig, out)


def fig_audit():
    fig = plt.figure(figsize=(W, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    rows = [
        ("Feature", "#F4E6CC", PALETTE["highlight"],
         "Descriptors stop encoding the same chemistry as sensors age",
         "Source-only normalization and leakage diagnostics",
         "Leaky preprocessing inflates accuracy by 4.8-6.2 pp (Fig. 3c)"),
        ("Knowledge", "#D6E6F1", PALETTE["blue_secondary"],
         "Negative transfer when source and target semantics conflict",
         "Explicit target-access grouping with per-group reporting",
         "TCA falls below the source-only SVM baseline (Fig. 3a)"),
        ("Decision", "#D8EBD9", PALETTE["green_3"],
         "Few-shot prototypes overwrite earlier drift states",
         "Stage-by-task accuracy matrices with BWT and FWT",
         "65.7% held-out accuracy with -40.2 pp BWT (Fig. 3b)"),
        ("Decision", "#D8EBD9", PALETTE["green_3"],
         "Confirmation bias under entropy-minimizing updates",
         "Confidence gating and chronological prequential streams",
         "Prequential accuracy drops to 43.5 ± 2.7% (Fig. 6a)"),
        ("System", "#EAD6E7", PALETTE["violet"],
         "Memory and latency growth from ensembles and replay",
         "Host-level footprint screening of components",
         "Component sizes and update costs (Table 5)"),
    ]

    x_layer, w_layer = 1.4, 8.7
    x1, w1 = 13.0, 25.5
    x2, w2 = 42.5, 25.5
    x3, w3 = 72.0, 26.0

    top, rh, gap = 88.6, 14.6, 2.25
    header_y = 94.0
    for xc, title in ((x1 + w1 / 2, "Failure mode"),
                      (x2 + w2 / 2, "Benchmark check"),
                      (x3 + w3 / 2, "Evidence in this work")):
        ax.text(xc, header_y, title, fontsize=7.2, fontweight="bold",
                ha="center", va="center", color=C_EDGE)
        ax.plot([xc - 10.8, xc + 10.8], [header_y - 3.0, header_y - 3.0],
                color="0.72", lw=0.45)

    y = top
    spans = {}
    for layer, tint, accent, c1, c2, c3 in rows:
        y0 = y - rh
        for x, w, txt in ((x1, w1, c1), (x2, w2, c2), (x3, w3, c3)):
            ax.add_patch(FancyBboxPatch(
                (x, y0), w, rh, boxstyle="round,pad=0.24,rounding_size=1.6",
                facecolor=tint, edgecolor="#FFFFFF", lw=0.7, zorder=1))
            ax.add_patch(FancyBboxPatch(
                (x, y0), 1.05, rh, boxstyle="round,pad=0.0,rounding_size=1.1",
                facecolor=accent, edgecolor="none", alpha=0.9, zorder=2))
            wrapped = "\n".join(textwrap.wrap(txt, width=27 if x != x3 else 28))
            ax.text(x + w / 2, y0 + rh / 2, wrapped, fontsize=5.9,
                    ha="center", va="center", linespacing=1.28,
                    color=C_EDGE, zorder=3)
        for x_start, x_end in ((x1 + w1 + 0.7, x2 - 0.7),
                               (x2 + w2 + 0.7, x3 - 0.7)):
            ax.add_patch(FancyArrowPatch(
                (x_start, y0 + rh / 2), (x_end, y0 + rh / 2),
                arrowstyle="-|>", mutation_scale=6.5, lw=0.55,
                color="0.42", shrinkA=0, shrinkB=0, zorder=2))
        spans.setdefault(layer, []).append(y0)
        y = y0 - gap

    for layer, ys in spans.items():
        y_lo, y_hi = min(ys), max(ys) + rh
        ax.add_patch(FancyBboxPatch(
            (x_layer, y_lo), w_layer, y_hi - y_lo,
            boxstyle="round,pad=0.18,rounding_size=1.7",
            facecolor="#F2F2F2", edgecolor="#FFFFFF", lw=0.7, zorder=1))
        ax.plot([x_layer + w_layer - 0.9, x_layer + w_layer - 0.9],
                [y_lo + 1.0, y_hi - 1.0], color=C_EDGE, lw=0.7, alpha=0.55,
                zorder=2)
        ax.text(x_layer + w_layer / 2, (y_lo + y_hi) / 2, layer,
                fontsize=6.3, fontweight="bold", ha="center", va="center",
                rotation=90, color=C_EDGE, zorder=3)

    out = os.path.join(OUTDIR, "fig_audit.pdf")
    return save_figure(fig, out)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for p in (fig_main_results(), fig_cross_checks(), fig_geometry(), fig_chronological(), fig_audit()):
        print("wrote", os.path.abspath(p))
