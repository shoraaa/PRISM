#!/usr/bin/env python3
"""Report over a results CSV: per-method summaries and pairwise comparisons.

``test.py`` prints this same report at the end of a run. Running it separately
lets a finished (or interrupted) CSV be re-read without re-solving anything,
and lets several runs be concatenated -- PRISM from one file, a baseline from
another -- since a row carries everything needed to place it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prism_eval import summary  # noqa: E402
from prism_eval.results import Row, is_prism, read_rows  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="results CSV files written by test.py (--csv)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help=(
            "method every other method is compared against (default: the "
            "first PRISM checkpoint in the file, which a multi-checkpoint run "
            "names prism:<label> rather than prism)"
        ),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="comma-separated subset of methods to report on",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="comma-separated subset of variants to report on",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="also print a variant x method table of objectives and gaps",
    )
    return parser.parse_args(argv)


def default_base(rows: list[Row]) -> str:
    """The checkpoint every other method is measured against.

    A run that compared several checkpoints has no method literally named
    ``prism``, so the base is the first PRISM-family method the file mentions.
    """
    for row in rows:
        if is_prism(row.method):
            return row.method
    return rows[0].method if rows else "prism"


def _filter(rows: list[Row], args: argparse.Namespace, base: str) -> list[Row]:
    if args.methods:
        keep = {name.strip() for name in args.methods.split(",")}
        keep.add(base)
        rows = [row for row in rows if row.method in keep]
    if args.variants:
        keep = {name.strip() for name in args.variants.split(",")}
        rows = [row for row in rows if row.variant in keep]
    return rows


def print_table(rows: list[Row], base: str) -> None:
    summaries = summary.variant_summaries(rows)
    methods = summary.methods(rows, base)
    variants = sorted({variant for variant, _method in summaries})
    width = max((len(variant) for variant in variants), default=7)
    header = f"{'variant':<{width}}  " + "  ".join(
        f"{method:>18}" for method in methods
    )
    print(header)
    print("-" * len(header))
    for variant in variants:
        cells = []
        for method in methods:
            entry = summaries.get((variant, method))
            if entry is None or entry.objective is None:
                cells.append(f"{'-':>18}")
                continue
            gap = entry.gap_pct
            cells.append(
                f"{entry.objective:>10.4g}"
                + (f" {gap:+6.2f}%" if gap is not None else " " * 7)
            )
        print(f"{variant:<{width}}  " + "  ".join(cells))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[Row] = []
    for path in args.csv:
        rows.extend(read_rows(path))
    base = args.base or default_base(rows)
    rows = _filter(rows, args, base)
    if not rows:
        raise SystemExit("no rows selected")
    summary.print_report(rows, base=base)
    if args.table:
        print()
        print_table(rows, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
