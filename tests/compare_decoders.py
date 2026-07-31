#!/usr/bin/env python3
"""Compare a trained field decoder with the non-neural PRISM decoder."""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import prism_decoder  # noqa: E402
from net import ConstraintFieldNet  # noqa: E402
from problem_data import (  # noqa: E402
    ALL_VARIANTS,
    DEFAULT_DATASET_DIR,
    DatasetFinder,
    SavedProblems,
    load_saved_data,
)
from train import _canonical_cost, infer_instance, setup_seeds  # noqa: E402
from urs_one_each import solver_problem  # noqa: E402
from urs_srr_report import EMBEDDED_REFERENCES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve size-100 instances of each selected PRISM variant "
            "with a checkpoint and the non-neural decoder, then average "
            "the results per variant."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated PRISM names, or 'all' for every variant",
    )
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--ants", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument(
        "--val-size",
        type=int,
        default=8,
        help="number of saved instances to average per variant (default: 8)",
    )
    parser.add_argument(
        "--static-field",
        action="store_true",
        help=(
            "evaluate one frozen neural field per instance; by default the "
            "field is recomputed whenever the decoder graph changes"
        ),
    )
    parser.add_argument(
        "--threads", type=int, default=prism_decoder.get_available_threads()
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args(argv)


def selected_variants(value: str) -> list[str]:
    available = list(ALL_VARIANTS)
    if value == "all":
        return available
    requested = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("unknown PRISM variants: " + ", ".join(unknown))
    if not requested:
        raise ValueError("--variants selected no variants")
    return requested


def _instance_data(data: dict, index: int) -> dict:
    """Retain the batch dimension while selecting one loaded instance."""
    return {
        key: value[index : index + 1]
        if torch.is_tensor(value) and value.ndim > 0
        else value
        for key, value in data.items()
    }


def _mean_reference(
    paths: dict, embedded_reference: float, val_size: int
) -> tuple[bool, float]:
    solution_path = paths["solution_path"]
    if solution_path is None:
        if paths["problem_name"] in EMBEDDED_REFERENCES:
            return True, float(embedded_reference)
        return False, 0.0

    path = Path(solution_path)
    if path.suffix == ".pkl":
        with path.open("rb") as source:
            values = pickle.load(source)[:val_size]
        references = [
            value[0] if isinstance(value, (list, tuple)) else value
            for value in values
        ]
    elif path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=False)
        references = torch.as_tensor(saved["cost"]).reshape(-1)[:val_size].tolist()
    else:
        raise ValueError(f"unsupported reference file: {path}")
    if len(references) != val_size:
        raise ValueError(
            f"requested {val_size} references but {path} contains "
            f"only {len(references)}"
        )
    return True, statistics.fmean(float(value) for value in references)


def _gap_percent(direction: str, objective: float, reference: float) -> float:
    if direction == "maximize":
        return (reference - objective) / abs(reference) * 100.0
    return (objective - reference) / abs(reference) * 100.0


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.val_size < 1:
        raise ValueError("--val-size must be positive")
    setup_seeds(args.seed)
    prism_decoder.set_num_threads(args.threads)
    logging.disable(logging.CRITICAL)

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model = ConstraintFieldNet().to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    finder = DatasetFinder(args.dataset_dir)
    saved_problems = SavedProblems(100, args.dataset_dir)
    variants = selected_variants(args.variants)
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()

    for index, name in enumerate(variants):
        test_started = time.perf_counter()
        try:
            if name == "vrptw":
                generated = [
                    saved_problems.load(name, instance)[0]
                    for instance in range(args.val_size)
                ]
                data = None
                has_reference, reference = False, 0.0
            else:
                paths = finder.get(name, 100)
                if paths is None:
                    raise FileNotFoundError(f"no size-100 dataset for {name}")
                data, embedded_reference = load_saved_data(
                    paths["data_path"],
                    name,
                    args.val_size,
                    solution_path=paths["solution_path"],
                )
                has_reference, reference = _mean_reference(
                    paths, embedded_reference, args.val_size
                )
            baseline_objectives = []
            neural_objectives = []
            baseline_costs = []
            neural_costs = []
            neural_net_evals = []
            neural_model_seconds = 0.0
            neural_decoder_seconds = 0.0
            baseline_seconds = 0.0
            neural_seconds = 0.0
            direction = ""

            for instance_index in range(args.val_size):
                problem = (
                    generated[instance_index]
                    if name == "vrptw"
                    else solver_problem(name, _instance_data(data, instance_index))
                )
                decoder_args = SimpleNamespace(
                    candidates=args.candidates,
                    n_ants=args.ants,
                    beta=2.0,
                    seed=args.seed + index * args.val_size + instance_index,
                    search_iterations=args.iterations,
                    feasibility_lookahead_depth=2,
                    feasibility_risk_penalty=10.0,
                    device=args.device,
                    static_field=args.static_field,
                )

                baseline_started = time.perf_counter()
                _, baseline, _ = infer_instance(None, problem, decoder_args)
                baseline_seconds += time.perf_counter() - baseline_started
                neural_started = time.perf_counter()
                _, neural, neural_metrics = infer_instance(
                    model, problem, decoder_args
                )
                neural_seconds += time.perf_counter() - neural_started
                if not baseline["feasible"] or not neural["feasible"]:
                    raise RuntimeError(
                        f"instance {instance_index} returned an infeasible solution"
                    )
                if baseline["direction"] != neural["direction"]:
                    raise RuntimeError("decoders returned different directions")

                direction = neural["direction"]
                baseline_objectives.append(float(baseline["objective"]))
                neural_objectives.append(float(neural["objective"]))
                baseline_costs.append(_canonical_cost(baseline))
                neural_costs.append(_canonical_cost(neural))
                neural_net_evals.append(neural_metrics["net_evals"])
                neural_model_seconds += neural_metrics["time_neural"]
                neural_decoder_seconds += neural_metrics["time_decoder"]

            baseline_objective = statistics.fmean(baseline_objectives)
            neural_objective = statistics.fmean(neural_objectives)
            baseline_cost = statistics.fmean(baseline_costs)
            neural_cost = statistics.fmean(neural_costs)
            if name == "vrptw" and not has_reference:
                reference = baseline_objective
                has_reference = True
            tolerance = 1e-6 * max(abs(baseline_cost), abs(neural_cost), 1.0)
            outcome = (
                "neural"
                if neural_cost < baseline_cost - tolerance
                else "baseline"
                if baseline_cost < neural_cost - tolerance
                else "tie"
            )
            rows.append(
                {
                    "variant": name,
                    "val_size": args.val_size,
                    "field_mode": "static" if args.static_field else "dynamic",
                    "direction": direction,
                    "reference": reference if has_reference else "",
                    "baseline_objective": baseline_objective,
                    "neural_objective": neural_objective,
                    "baseline_gap_pct": (
                        _gap_percent(direction, baseline_objective, reference)
                        if has_reference
                        else ""
                    ),
                    "neural_gap_pct": (
                        _gap_percent(direction, neural_objective, reference)
                        if has_reference
                        else ""
                    ),
                    "winner": outcome,
                    "baseline_seconds": baseline_seconds,
                    "neural_seconds": neural_seconds,
                    "neural_net_evals_mean": statistics.fmean(
                        neural_net_evals
                    ),
                    "neural_net_evals_total": int(sum(neural_net_evals)),
                    "neural_model_seconds": neural_model_seconds,
                    "neural_decoder_seconds": neural_decoder_seconds,
                }
            )
            baseline_gap = rows[-1]["baseline_gap_pct"]
            neural_gap = rows[-1]["neural_gap_pct"]
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=PASS",
                f"val_size={args.val_size}",
                f"field_mode={rows[-1]['field_mode']}",
                f"winner={outcome}",
                f"baseline_mean={baseline_objective:.6g}",
                f"neural_mean={neural_objective:.6g}",
                f"net_evals_mean={rows[-1]['neural_net_evals_mean']:.3f}",
                "baseline_gap="
                + (
                    f"{baseline_gap:.3f}%"
                    if baseline_gap != ""
                    else "n/a"
                ),
                "neural_gap="
                + (f"{neural_gap:.3f}%" if neural_gap != "" else "n/a"),
                f"seconds={time.perf_counter() - test_started:.3f}",
                flush=True,
            )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            failures.append((name, detail))
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=FAIL",
                f"error={detail}",
                f"seconds={time.perf_counter() - test_started:.3f}",
                flush=True,
            )

    referenced = [row for row in rows if row["reference"] != ""]
    print(
        "DECODER_COMPARE",
        f"checkpoint={args.checkpoint}",
        f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}",
        f"val_size={args.val_size}",
        f"field_mode={'static' if args.static_field else 'dynamic'}",
        f"variants={len(variants)}",
        f"passed={len(rows)}",
        f"failed={len(failures)}",
        f"neural_wins={sum(row['winner'] == 'neural' for row in rows)}",
        f"ties={sum(row['winner'] == 'tie' for row in rows)}",
        f"baseline_wins={sum(row['winner'] == 'baseline' for row in rows)}",
        f"seconds={time.perf_counter() - started:.3f}",
    )
    if referenced:
        print(
            "REFERENCE_GAPS",
            f"n={len(referenced)}",
            "baseline_mean="
            f"{statistics.fmean(row['baseline_gap_pct'] for row in referenced):.3f}%",
            "neural_mean="
            f"{statistics.fmean(row['neural_gap_pct'] for row in referenced):.3f}%",
            "baseline_median="
            f"{statistics.median(row['baseline_gap_pct'] for row in referenced):.3f}%",
            "neural_median="
            f"{statistics.median(row['neural_gap_pct'] for row in referenced):.3f}%",
        )
    for name, error in failures:
        print("FAIL", name, error)

    if args.csv:
        if not rows:
            raise RuntimeError("cannot write comparison CSV without results")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV {args.csv}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
