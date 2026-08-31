#!/usr/bin/env python3
"""Paired ablation evaluation, run + aggregate in one file.

Evaluates the original v6 policy and the resource-token-attention ablation
(--no-couple-resource-tokens) across several eval seeds, then reports the
learned policy's oracle gap (lower is better) as mean +/- std
so the effect can be read against seed noise instead of a single point.

Checkpoints (defaults, per current layout):
    pretrained/epoch1000_best.pt  = original v6                 -> "baseline"
    pretrained/best.pt            = --no-couple-resource-tokens -> "ablation"

Both are evaluated with identical variants/iterations/seed, so the only
difference is the trained field. Per-seed CSVs land in the output directory and
are skipped if already present (safe to resume). Pass --aggregate-only to
re-summarise existing CSVs without re-running any eval.

Examples:
    python scripts/ablation_eval.py
    python scripts/ablation_eval.py --variants cvrp,cvrptw --seeds 1234 2
    python scripts/ablation_eval.py --aggregate-only
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prism_eval.results import read_rows  # noqa: E402
from prism_eval.summary import method_gaps  # noqa: E402


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_eval(name: str, ckpt: Path, seed: int, args: argparse.Namespace) -> None:
    """Invoke test.py for one (model, seed); skip if its CSV already exists."""
    csv_path = args.outdir / f"{name}_seed{seed}.csv"
    if csv_path.exists():
        print(f"skip existing: {csv_path}")
        return
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    print(f"=== {name}  seed={seed}  variants={args.variants} ===")
    env = {**os.environ, "PYTHONPATH": "src"}
    cmd = [
        sys.executable, "test.py",
        "--checkpoint", str(ckpt),
        "--variants", args.variants,
        "--iterations", str(args.iterations),
        "--seed", str(seed),
        "--csv", str(csv_path),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def seed_gaps(csv_path: Path) -> tuple[dict[str, float], float | None]:
    """Return ({variant: PRISM's oracle gap}, overall_mean) for one eval CSV.

    Reads through the shared results reader, so a tidy CSV and an older wide
    one both work; variants without a reference are skipped.
    """
    per_variant = method_gaps(read_rows(csv_path))
    if not per_variant:
        return {}, None
    return per_variant, statistics.fmean(per_variant.values())


def fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f} (1 seed)"
    return f"{statistics.fmean(values):.3f} +/- {statistics.stdev(values):.3f}"


def aggregate(outdir: Path) -> int:
    # model -> seed -> {variant: oracle gap}
    data: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for csv_path in sorted(outdir.glob("*_seed*.csv")):
        model, _, seed = csv_path.stem.rpartition("_seed")
        per_variant, overall_mean = seed_gaps(csv_path)
        if overall_mean is None:
            print(f"warn: no referenced variants in {csv_path.name}", file=sys.stderr)
            continue
        data[model][seed] = per_variant

    if "baseline" not in data or "ablation" not in data:
        print(
            "need both baseline_seed*.csv and ablation_seed*.csv; found: "
            f"{sorted(data)}",
            file=sys.stderr,
        )
        return 1

    # Align on the variant set present with a reference in EVERY csv, so the
    # per-model overall means and the paired delta average the same variants. A
    # variant missing from any run (a failed eval, or no reference that run)
    # would otherwise make the model means incomparable.
    all_sets = [set(pv) for seeds in data.values() for pv in seeds.values()]
    common = set.intersection(*all_sets)
    if not common:
        print("no variant has a reference in every csv; nothing comparable",
              file=sys.stderr)
        return 1
    dropped = set.union(*all_sets) - common
    if dropped:
        print(
            f"note: {len(dropped)} variant(s) absent from some run, excluded "
            f"from overall/delta: {', '.join(sorted(dropped))}",
            file=sys.stderr,
        )

    def overall(per_variant: dict[str, float]) -> float:
        return statistics.fmean(per_variant[v] for v in common)

    print(f"\nOracle gap over {len(common)} shared variants,"
          " lower is better")
    print("=" * 52)
    for model in ("baseline", "ablation"):
        overalls = [overall(pv) for pv in data[model].values()]
        seeds = ",".join(sorted(data[model]))
        print(f"{model:>9}: {fmt(overalls):<28} seeds={seeds}")

    shared = sorted(set(data["baseline"]) & set(data["ablation"]))
    deltas = [
        overall(data["ablation"][s]) - overall(data["baseline"][s])
        for s in shared
    ]
    print("-" * 52)
    if not deltas:
        print("no shared seeds between models for a paired comparison")
        return 0

    print(f"paired delta (ablation - baseline): {fmt(deltas)}")
    mean_delta = statistics.fmean(deltas)
    verdict = (
        "baseline better" if mean_delta > 0
        else "ablation better" if mean_delta < 0
        else "tied"
    )
    if len(deltas) > 1:
        note = "  <-- within seed noise" if abs(mean_delta) < statistics.stdev(deltas) else ""
        print(f"  positive = baseline wins. {verdict}{note}")
    else:
        print(f"  positive = baseline wins. {verdict} (single seed: no error bar)")

    # Per-variant paired delta, to surface where any effect concentrates.
    print("-" * 52)
    print("per-variant paired delta (ablation - baseline), top 10 by |delta|:")
    variant_deltas: dict[str, list[float]] = defaultdict(list)
    for seed in shared:
        base_v = data["baseline"][seed]
        abl_v = data["ablation"][seed]
        for variant in common:
            variant_deltas[variant].append(abl_v[variant] - base_v[variant])
    ranked = sorted(
        variant_deltas.items(),
        key=lambda kv: abs(statistics.fmean(kv[1])),
        reverse=True,
    )
    for variant, vals in ranked[:10]:
        print(f"  {variant:<24} {fmt(vals)}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ckpt", type=Path,
        default=REPO_ROOT / "pretrained/epoch1000_best.pt",
        help="original v6 checkpoint",
    )
    parser.add_argument(
        "--ablation-ckpt", type=Path,
        default=REPO_ROOT / "pretrained/best.pt",
        help="--no-couple-resource-tokens checkpoint",
    )
    parser.add_argument("--variants", default="all")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1234, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--outdir", type=Path, default=REPO_ROOT / "results/ablate_attention",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="skip evals; only summarise existing CSVs in --outdir",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        failures = 0
        for seed in args.seeds:
            for name, ckpt in (
                ("baseline", args.baseline_ckpt),
                ("ablation", args.ablation_ckpt),
            ):
                try:
                    run_eval(name, ckpt, seed, args)
                except subprocess.CalledProcessError as exc:
                    failures += 1
                    print(f"WARN: {name} seed={seed} failed (exit {exc.returncode});"
                          " continuing", file=sys.stderr)
        if failures:
            print(f"WARN: {failures} eval(s) failed; summary covers the rest",
                  file=sys.stderr)

    return aggregate(args.outdir)


if __name__ == "__main__":
    raise SystemExit(main())
