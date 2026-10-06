"""Regression tests for manuscript figure layouts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fig3_uses_balanced_three_panel_layout():
    """Fig. 3 should avoid an unbalanced three-panel grid."""
    src = (ROOT / "figures" / "make_manuscript_figures.py").read_text(encoding="utf-8")

    assert "fig.add_gridspec(2, 2" in src
    assert "fig.add_subplot(gs[0, :])" in src
    assert "fig.add_subplot(gs[1, 0])" in src
    assert "fig.add_subplot(gs[1, 1])" in src
