#!/usr/bin/env python3
"""Figure 1: mean reference gap vs. number of active constraints.

Visualizes Table `tab:constraint-count` (the 100 reference-scored CVRP-family
variants, gap by number of active constraints). Data are the published table
aggregates so the figure and table agree by construction; the story is that the
learned energy stays low and flat while the heuristic energies rise as
constraints stack.

Run:  python scripts/plot_gap_vs_constraints.py
Out:  figures/gap_vs_constraints.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from prism_plotstyle import apply_style, ENERGY, WRAPW

# ---- Data: Table tab:constraint-count (mean gap %, lower is better) --------- #
# Capacity counts as 1; each active A/O/MD/B/P/L/TW/PD attribute adds 1.
N_CONSTRAINTS = [1, 2, 3, 4, 5, 6, 7, 8]
N_VARIANTS    = [1, 7, 18, 26, 25, 16, 6, 1]
GAP = {
    "Uniform":  [3.46, 2.49, 2.95, 4.58, 5.61, 5.78, 5.57, 2.64],
    "Distance": [4.17, 1.97, 1.86, 3.07, 3.58, 4.24, 3.64, 1.32],
    "Learned":  [2.10, -0.34, -0.57, 0.62, 1.33, 1.71, 1.67, -0.26],
}
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(WRAPW, 1.6))

    ax.axhline(0.0, color="#b0b0ac", lw=0.8, zorder=1)

    # Draw baselines first (recessive), Learned last and prominent (hero).
    spec = [
        ("Uniform",  dict(lw=1.2, marker="o", ms=2.8, ls="--", zorder=3)),
        ("Distance", dict(lw=1.2, marker="s", ms=2.8, ls="--", zorder=3)),
        ("Learned",  dict(lw=1.9, marker="D", ms=3.6, ls="-",  zorder=5)),
    ]
    for name, kw in spec:
        ax.plot(N_CONSTRAINTS, GAP[name], color=ENERGY[name],
                markeredgecolor="white", markeredgewidth=0.6, **kw)

    # Direct labels at the right end (no legend box).
    label_y = {"Uniform": GAP["Uniform"][-2] + 0.5,
               "Distance": GAP["Distance"][-2] - 0.7,
               "Learned": GAP["Learned"][-2] - 0.9}
    for name in ("Uniform", "Distance", "Learned"):
        weight = "bold" if name == "Learned" else "normal"
        pretty = r"CoRE" if name == "Learned" else name
        ax.text(7.15, label_y[name], pretty, color=ENERGY[name],
                fontsize=6.5, fontweight=weight, ha="right", va="center")

    ax.set_xlabel("Number of active constraints")
    ax.set_ylabel("Mean gap to reference (%)")
    ax.set_xticks(N_CONSTRAINTS)
    ax.set_xlim(0.6, 8.4)
    ax.margins(y=0.12)

    out = ROOT / "figures" / "gap_vs_constraints.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    # Consistency echo (must match tab:constraint-count).
    print("Learned gap by #constraints:", GAP["Learned"])


if __name__ == "__main__":
    main()
