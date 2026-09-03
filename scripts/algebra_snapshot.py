#!/usr/bin/env python3
"""Record the decoder's observable behaviour, so a refactor can be proved inert.

Changes to the resource algebra are supposed to generalize the language without
altering what any existing row does. That is only checkable against a reference,
so this captures one: for each variant, the published declaration of every
active row, its factored program properties, and the complete solution a
fixed-seed distance-only solve
reaches. No model and no checkpoint are involved -- the point is to isolate the
decoder.

    python scripts/algebra_snapshot.py --out before.json
    # ... refactor ...
    python scripts/algebra_snapshot.py --out after.json
    python scripts/algebra_snapshot.py --compare before.json after.json

``--compare`` reports the first difference per variant rather than a diff of the
whole file, because a property layout change touches every row and the useful
question is which *behaviour* moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import prism_decoder  # noqa: E402

from problem_data import DatasetFinder, load_saved_data  # noqa: E402

try:
    from prism_eval.instances import _instance_data, solver_problem  # noqa: E402
except ImportError:  # pre-harness layouts kept these in test.py
    from test import _instance_data, solver_problem  # noqa: E402


DEFAULT_VARIANTS = (
    "tsp,op,pctsp,cvrp,cvrpb,cvrpl,ocvrp,cvrptw,cvrpbl,cvrpbtw,cvrpltw,"
    "cvrpbp,pdcvrp,cvrpbltw,acvrptw,mdcvrptw,amdcvrpbpltw"
)


def neutral_guidance(decoder) -> dict:
    """Zero field, unit objective weight: the energy is the plain edge cost."""
    channels = int(decoder.metadata["resource_count"])
    slots = int(decoder.metadata["multiplier_count"])
    edges = int(decoder.metadata["edge_count"])
    multipliers = np.zeros(slots, dtype=np.float32)
    multipliers[channels] = 1.0
    return {
        "objective_residual": np.zeros(edges, dtype=np.float32),
        "edge_field": np.zeros((edges, channels), dtype=np.float32),
        "edge_additive": np.zeros((edges, channels), dtype=np.float32),
        "multipliers": multipliers,
        "coupler_weights": np.zeros((slots, channels), dtype=np.float32),
        "coupler_bias": np.zeros(slots, dtype=np.float32),
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        # Descriptor and bound values are float32; round so a rebuild with
        # different instruction scheduling does not register as a change.
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def snapshot_variant(variant: str, args) -> dict | None:
    finder = DatasetFinder(args.dataset_dir)
    try:
        paths = finder.get(variant, 100)
        data, _reference = load_saved_data(
            paths["data_path"], variant, args.instances,
            solution_path=paths["solution_path"],
            allow_aggregate_reference=False,
        )
    except Exception as error:  # noqa: BLE001
        return {"skipped": str(error)}

    instances = []
    for instance in range(args.instances):
        problem = solver_problem(variant, _instance_data(data, instance))
        decoder = prism_decoder.Decoder(
            problem,
            candidate_config={"max_candidates": args.candidates},
            search_config={"use_srr": True},
            n_rollouts=args.rollouts,
            beta=2.0,
        )
        decoder.seed(args.seed + instance)
        metadata = decoder.metadata
        active = [
            i for i, flag in enumerate(metadata["field_channel_mask"]) if flag
        ]
        row_properties = np.asarray(decoder.resource_row_properties)
        terms = np.asarray(decoder.resource_term_properties)
        counts = np.asarray(decoder.resource_term_counts, dtype=np.int64)

        guidance = neutral_guidance(decoder)
        best = decoder.solve(args.iterations, **guidance)

        entry = {
            "active": active,
            "objective": round(float(best["objective"]), 6),
            "feasible": bool(best["feasible"]),
            "direction": best["direction"],
            "route": jsonable(np.asarray(best["route"])),
            "route_len": len(best["route"]),
            "optional_reset_duration": jsonable(
                best.get("optional_reset_duration", 0.0)
            ),
            "programs": {},
        }
        for i in active:
            start = int(counts[:i].sum())
            stop = start + int(counts[i])
            entry["programs"][str(i)] = {
                "row": jsonable(row_properties[i]),
                "terms": jsonable(terms[start:stop]),
            }
        declarations = getattr(decoder, "resource_declarations", None)
        if declarations is not None:
            entry["declarations"] = {
                str(i): jsonable(declarations[i]) for i in active
            }
        instances.append(entry)
    return {"instances": instances}


def compare(before: Path, after: Path) -> int:
    left = json.loads(before.read_text())
    right = json.loads(after.read_text())
    names = sorted(set(left) | set(right))
    differences = 0
    behavioral_drift = False
    for name in names:
        if name not in left or name not in right:
            print(f"{name}: present in only one snapshot")
            differences += 1
            continue
        a, b = left[name], right[name]
        if a == b:
            continue
        differences += 1
        # Report the behavioural fields first: a descriptor layout change is
        # expected and uninteresting next to a changed solution.
        moved = []
        for index, (x, y) in enumerate(
            zip(a.get("instances", []), b.get("instances", []))
        ):
            for key in (
                "objective", "feasible", "direction", "route", "route_len",
                "optional_reset_duration", "active",
            ):
                if x.get(key) != y.get(key):
                    moved.append(f"instance {index} {key}: {x.get(key)} -> {y.get(key)}")
        if moved:
            behavioral_drift = True
            print(f"{name}: BEHAVIOUR CHANGED")
            for line in moved[:4]:
                print(f"    {line}")
        else:
            print(f"{name}: program properties/declarations only")
    print()
    print(f"{differences} of {len(names)} variants differ")
    return 1 if behavioral_drift else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("/home/shora/Research/PRISM/datasets/benchmarks"))
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    if args.compare:
        return compare(*args.compare)
    if not args.out:
        parser.error("--out is required unless --compare is given")

    result = {}
    for variant in [n.strip() for n in args.variants.split(",") if n.strip()]:
        result[variant] = snapshot_variant(variant, args)
        state = result[variant]
        note = state.get("skipped") or f"{len(state['instances'])} instances"
        print(f"{variant}: {note}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
