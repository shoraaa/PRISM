#!/usr/bin/env python3
"""Figure 2: architecture component ablation by number of active resources.

Recomputes the per-bucket mean reference gap for the full model and the four
pillar ablations directly from the run CSVs, reports any drift from the
published Table `tab:factorization` (without blocking), then draws a grouped bar over
{<=1, 2, >=3}-resource buckets. The two supporting-mechanism ablations (binding
gate, feasibility risk) stay in the table only; the figure shows the three
design principles: semantic factorization (Monolithic, Identity), cross-resource
contextualization (No coupling), and state-dependent pricing (Static).

All six ablation checkpoints share the training protocol (1000 epochs, constant
LR 5e-5, seed 1234, same 22 training variants), so this is a controlled
comparison.

Run:  python scripts/plot_ablation_by_rescount.py
Out:  figures/ablation_by_rescount.pdf
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from prism_eval.results import read_rows  # noqa: E402
from prism_eval.summary import method_gaps  # noqa: E402

from prism_plotstyle import apply_style, FULL, PILLAR, WRAPW  # noqa: E402
from problem_data import resource_count  # noqa: E402

# Full model (baseline=none run) + the four pillar ablations.
CSVS = {
    "CoRE": "urs.csv",
    "$-$Monolithic": "pretrained/monolithic-resource-field/results.csv",
    "$-$Identity": "pretrained/index-embedded-resource/results.csv",
    "$-$No coupling": "pretrained/no-resource-token/results.csv",
    "$-$Static": "pretrained/no-dynamic-field/results.csv",
}
ORDER = ["CoRE", "$-$Monolithic", "$-$Identity", "$-$No coupling", "$-$Static"]
BUCKETS = ["$\\leq 1$", "$2$", "$\\geq 3$"]

# Published Table tab:factorization values (<=1, 2, >=3) for the consistency
# assertion. Keyed as in the paper.
PUBLISHED = {
    "CoRE":            [-1.59, -0.18, 2.97],
    "$-$Monolithic":   [-1.15, 0.03, 3.25],
    "$-$Identity":     [-0.76, 0.49, 3.74],
    "$-$No coupling":  [-1.56, 0.16, 3.35],
    "$-$Static":       [-0.75, 0.43, 3.89],
}


def _fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _load(path):
    """{variant: PRISM oracle gap} for one eval CSV, tidy or legacy."""
    return method_gaps(read_rows(ROOT / path))


def _bucket(n: int) -> int:
    return 0 if n <= 1 else (1 if n == 2 else 2)


def compute() -> dict[str, list[float]]:
    data = {k: _load(v) for k, v in CSVS.items()}
    common = [
        v for v in data["CoRE"]
        if all(_fnum(data[k].get(v)) is not None for k in data)
    ]
    out: dict[str, list[float]] = {}
    for name in ORDER:
        buckets: list[list[float]] = [[], [], []]
        for v in common:
            buckets[_bucket(resource_count(v))].append(
                _fnum(data[name][v])
            )
        out[name] = [round(st.mean(b), 2) for b in buckets]
    return out


def main() -> None:
    means = compute()
    # Soft consistency check: report any drift from the numbers currently in the
    # paper (Table tab:factorization) but NEVER block regeneration. Re-run this
    # after refreshing the ablation CSVs and the figure just reflects them; if a
    # value moved, the message below says exactly what to update in the paper
    # (both tab:factorization and the PUBLISHED dict above).
    print("bucket means from current CSVs  [<=1, 2, >=3]:")
    for name in ORDER:
        print(f"  {name:<16} {means[name]}")
    drift = [
        (name, means[name], PUBLISHED[name])
        for name in ORDER
        if any(abs(g - w) >= 0.005 for g, w in zip(means[name], PUBLISHED[name]))
    ]
    if drift:
        print("\n[!] CSV values differ from the paper table -- update "
              "tab:factorization and PUBLISHED:")
        for name, got, want in drift:
            print(f"    {name:<16} paper {want} -> csv {got}")
    else:
        print("(matches the paper table)")

    apply_style()
    fig, ax = plt.subplots(figsize=(WRAPW, 1.95))

    x = np.arange(len(BUCKETS))
    n = len(ORDER)
    bw = 0.82 / n
    colors = {"CoRE": FULL, **PILLAR}
    ax.axhline(0.0, color="#b0b0ac", lw=0.8, zorder=1)
    for i, name in enumerate(ORDER):
        off = (i - (n - 1) / 2) * bw
        ax.bar(x + off, means[name], width=bw * 0.92, color=colors[name],
               edgecolor="white", linewidth=0.5, zorder=3,
               label=("CoRE (full)" if name == "CoRE" else name))

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS)
    ax.set_xlabel("Number of active resources")
    ax.set_ylabel("Mean gap to reference (%)")
    ax.legend(ncol=1, frameon=False, loc="upper left", handlelength=0.9,
              borderaxespad=0.15, labelspacing=0.2, handletextpad=0.4,
              fontsize=5.6)
    ax.margins(y=0.10)

    out = ROOT / "figures" / "ablation_by_rescount.pdf"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
