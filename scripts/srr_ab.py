#!/usr/bin/env python3
"""A/B the search substrate, with no model in the loop.

Route ranking decides which routes SRR perturbs, so a change to it moves the
solution without moving feasibility or the algebra. That is invisible to the
equivalence suite -- both sides of those comparisons go through the same
ranking -- so it needs its own measurement.

Distance-only guidance throughout: no checkpoint, no field, so the only thing
under test is the search. Objectives are paired by (variant, instance, seed),
which removes instance difficulty from the comparison and makes a small mean
shift readable against a large spread.

    python scripts/srr_ab.py --out before.json
    #  ... change the ranking ...
    python scripts/srr_ab.py --out after.json
    python scripts/srr_ab.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import prism_decoder as p  # noqa: E402

from prism_eval.instances import (  # noqa: E402
    GENERATORS, _instance_data, solver_problem,
)
from problem_data import DatasetFinder, load_saved_data  # noqa: E402

DEFAULT = ("cvrp,cvrptw,cvrpl,cvrpb,ocvrp,cvrpbl,cvrpbtw,cvrpltw,cvrpbp,"
           "pdcvrp,cvrpbltw,acvrptw,mdcvrptw,op,pctsp,amdcvrpbpltw")


def neutral(dec):
    md = dec.metadata
    channels, slots = int(md["resource_count"]), int(md["multiplier_count"])
    edges = int(md["edge_count"])
    multipliers = np.zeros(slots, np.float32)
    multipliers[channels] = 1.0
    return dict(
        objective_residual=np.zeros(edges, np.float32),
        edge_field=np.zeros((edges, channels), np.float32),
        edge_additive=np.zeros((edges, channels), np.float32),
        multipliers=multipliers,
        edge_state_field=np.zeros((edges, channels, channels), np.float32),
        coupler_weights=np.zeros((slots, channels), np.float32),
        coupler_bias=np.zeros(slots, np.float32),
    )


def run(args) -> dict:
    finder = DatasetFinder(args.dataset_dir)
    result: dict[str, list] = {}
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for variant in variants:
        try:
            if variant in GENERATORS:
                data = GENERATORS[variant](100, args.instances, seed=1234)
            else:
                paths = finder.get(variant, 100)
                data, _ = load_saved_data(
                    paths["data_path"], variant, args.instances,
                    solution_path=paths["solution_path"],
                    allow_aggregate_reference=False,
                )
        except Exception as error:  # noqa: BLE001
            print(f"{variant}: skipped ({error})", file=sys.stderr)
            continue
        cells = []
        for instance in range(args.instances):
            problem = solver_problem(variant, _instance_data(data, instance))
            for seed in range(args.seeds):
                dec = p.Decoder(
                    problem,
                    candidate_config={"max_candidates": args.candidates},
                    search_config={"use_srr": True},
                    n_rollouts=args.rollouts, beta=2.0,
                )
                dec.seed(1000 * seed + instance)
                best = dec.solve(args.iterations, **neutral(dec))
                cells.append(
                    None if not best["feasible"]
                    else round(float(best["objective"]), 6)
                )
        result[variant] = cells
        done = [c for c in cells if c is not None]
        print(f"{variant}: n={len(done)}/{len(cells)} "
              f"mean={st.mean(done):.4f}" if done else f"{variant}: none",
              flush=True)
    return result


def compare(before: Path, after: Path) -> int:
    a, b = json.loads(before.read_text()), json.loads(after.read_text())
    print(f"{'variant':14s} {'n':>4s} {'before':>9s} {'after':>9s} "
          f"{'delta':>8s} {'delta%':>7s} {'wins':>6s}")
    deltas_all, worse, better = [], 0, 0
    for variant in sorted(set(a) & set(b)):
        pairs = [(x, y) for x, y in zip(a[variant], b[variant])
                 if x is not None and y is not None]
        if not pairs:
            print(f"{variant:14s} {'-':>4s}")
            continue
        xs, ys = zip(*pairs)
        delta = st.mean(ys) - st.mean(xs)
        wins = sum(1 for x, y in pairs if y < x - 1e-9)
        losses = sum(1 for x, y in pairs if y > x + 1e-9)
        better += wins
        worse += losses
        deltas_all += [y - x for x, y in pairs]
        pct = 100.0 * delta / st.mean(xs) if st.mean(xs) else 0.0
        print(f"{variant:14s} {len(pairs):4d} {st.mean(xs):9.4f} "
              f"{st.mean(ys):9.4f} {delta:+8.4f} {pct:+6.2f}% "
              f"{wins:3d}/{losses:<3d}")
    if deltas_all:
        mean = st.mean(deltas_all)
        sd = st.pstdev(deltas_all)
        se = sd / (len(deltas_all) ** 0.5) if len(deltas_all) > 1 else 0.0
        print()
        print(f"paired delta over {len(deltas_all)} cells: "
              f"mean {mean:+.5f}  sd {sd:.5f}  se {se:.5f}  "
              f"{'|t| = %.2f' % (abs(mean) / se) if se else ''}")
        print(f"cells improved {better}, worsened {worse}")
        print("(negative delta = shorter routes = better)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--variants", default=DEFAULT)
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "datasets" / "benchmarks")
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=8)
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        parser.error("--out is required unless --compare is given")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(run(args), indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
