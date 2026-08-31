"""Shared print-figure styling for CoRE paper plots.

Colorblind-safe (Okabe-Ito derived) palette, validated with the dataviz skill's
palette checker (light mode): the 4 pillar-ablation hues and the 3 energy-source
hues both PASS lightness/chroma/CVD/normal-vision separation. The single
orange-on-white contrast WARN is relieved by bar edge strokes and direct line
labels, and every figure has a numeric backing table in the appendix.

Kept in scripts/ permanently; imported by plot_gap_vs_constraints.py and
plot_ablation_by_rescount.py so both figures share one visual system.
"""
from __future__ import annotations

import matplotlib as mpl

# Ink (reference series / text). Not a categorical hue by design.
INK = "#1a1a1a"
GRID = "#d9d9d6"

# Energy sources (Fig. 1). Learned is the hero -> strong blue.
ENERGY = {
    "Learned": "#0072B2",   # blue
    "Distance": "#E69F00",  # orange
    "Uniform": "#009E73",   # green
}

# Component ablations grouped by the design principle they realize (Fig. 2).
# Two warm hues for the two semantic-factorization ablations, green for
# contextualization, blue for state-dependent pricing -- the hue families echo
# the pillar grouping.
FULL = INK
PILLAR = {
    "$-$Monolithic": "#E69F00",   # semantic factorization
    "$-$Identity": "#D55E00",     # semantic factorization
    "$-$No coupling": "#009E73",  # contextualization
    "$-$Static": "#0072B2",       # state-dependent pricing
}


def apply_style() -> None:
    """Global rcParams for compact, embeddable, single-column paper figures."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,          # embed TrueType (not Type3) for camera-ready
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#4d4d4d",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.color": "#4d4d4d",
        "ytick.color": "#4d4d4d",
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# Full text width (single-column ICLR) for a centered float (~5.5in).
COLWIDTH = 3.3
# Wrap size: native dimensions tuned so that, scaled into a ~0.42\linewidth
# wrapfigure box (~2.3in wide), the figure renders near 1:1 and its 7pt fonts
# stay legible beside the text -- the L2Seg wrapfigure recipe.
WRAPW = 2.45
