#!/usr/bin/env python3
"""Compare a trained field decoder with the non-neural decoder on URS."""

from __future__ import annotations

import argparse
import csv
import logging
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
    BENCHMARK_VARIANTS,
    DEFAULT_DATASET_DIR,
    DatasetFinder,
    load_saved_data,
)
from train import _canonical_cost, infer_instance, setup_seeds  # noqa: E402
from urs_one_each import solver_problem  # noqa: E402
from urs_srr_report import first_reference, gap_percent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the first saved size-100 instance of each selected URS "
            "variant with a checkpoint and the non-neural decoder."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated URS names, or 'all' for every variant",
    )
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--ants", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument(
        "--threads", type=int, default=prism_decoder.get_available_threads()
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def selected_variants(value: str) -> list[str]:
    available = list(BENCHMARK_VARIANTS)
    if value == "all":
        return available
    requested = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("unknown URS variants: " + ", ".join(unknown))
    if not requested:
        raise ValueError("--variants selected no variants")
    return requested


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
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
    variants = selected_variants(args.variants)
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()

    for index, name in enumerate(variants):
        test_started = time.perf_counter()
        try:
            paths = finder.get(name, 100)
            data, embedded_reference = load_saved_data(
                paths["data_path"],
                name,
                1,
                solution_path=paths["solution_path"],
            )
            has_reference, reference = first_reference(
                paths, embedded_reference
            )
            problem = solver_problem(name, data)
            decoder_args = SimpleNamespace(
                candidates=args.candidates,
                n_ants=args.ants,
                beta=2.0,
                seed=args.seed + index,
                search_iterations=args.iterations,
                feasibility_lookahead_depth=2,
                feasibility_risk_penalty=10.0,
                device=args.device,
            )

            baseline_started = time.perf_counter()
            _, baseline, _ = infer_instance(None, problem, decoder_args)
            baseline_seconds = time.perf_counter() - baseline_started
            neural_started = time.perf_counter()
            _, neural, _ = infer_instance(model, problem, decoder_args)
            neural_seconds = time.perf_counter() - neural_started
            if not baseline["feasible"] or not neural["feasible"]:
                raise RuntimeError("a decoder returned an infeasible solution")

            baseline_cost = _canonical_cost(baseline)
            neural_cost = _canonical_cost(neural)
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
                    "direction": neural["direction"],
                    "reference": reference if has_reference else "",
                    "baseline_objective": baseline["objective"],
                    "neural_objective": neural["objective"],
                    "baseline_gap_pct": (
                        gap_percent(baseline, reference)
                        if has_reference
                        else ""
                    ),
                    "neural_gap_pct": (
                        gap_percent(neural, reference) if has_reference else ""
                    ),
                    "winner": outcome,
                    "baseline_seconds": baseline_seconds,
                    "neural_seconds": neural_seconds,
                }
            )
            baseline_gap = rows[-1]["baseline_gap_pct"]
            neural_gap = rows[-1]["neural_gap_pct"]
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=PASS",
                f"winner={outcome}",
                f"baseline={baseline['objective']:.6g}",
                f"neural={neural['objective']:.6g}",
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
