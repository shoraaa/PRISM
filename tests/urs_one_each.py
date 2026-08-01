#!/usr/bin/env python3
"""Load and solve the first size-100 instance of every URS variant."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import prism_decoder  # noqa: E402
from problem_data import (  # noqa: E402
    BENCHMARK_VARIANTS,
    DEFAULT_DATASET_DIR,
    DatasetFinder,
    decoder_problem,
    load_saved_data,
)


solver_problem = decoder_problem


def assert_normalized_model_inputs(solver) -> None:
    tensors = {
        "node_features": solver.node_features,
        "edge_features": solver.edge_features,
        "resource_features": solver.resource_features,
        "field_channel_mask": solver.metadata["field_channel_mask"],
    }
    for label, values in tensors.items():
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label} contains a non-finite value")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(f"{label} is not normalized to [0, 1]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--threads", type=int, default=prism_decoder.get_available_threads()
    )
    parser.add_argument(
        "--guidance", choices=("classical", "field"), default="classical"
    )
    parser.add_argument("--verify-incremental-srr", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2 to exercise perturbation")

    logging.disable(logging.CRITICAL)
    finder = DatasetFinder(args.dataset_dir)
    prism_decoder.set_num_threads(args.threads)
    rollouts = args.rollouts if args.rollouts is not None else args.threads
    failures = []
    perturbed = 0
    improved = 0
    srr_active = 0
    unperturbed_names = []
    unimproved_names = []
    inactive_srr_names = []
    accepted_srr_moves = 0
    srr_scope_nodes = 0
    srr_revisits = 0
    srr_evaluations = 0
    srr_incremental_rebuilds = 0
    srr_full_rebuilds = 0
    srr_rebuilt_nodes = 0
    started = time.perf_counter()

    variants = list(BENCHMARK_VARIANTS)
    for index, name in enumerate(variants):
        test_started = time.perf_counter()
        try:
            paths = finder.get(name, 100)
            data, _ = load_saved_data(
                paths["data_path"],
                name,
                1,
                solution_path=paths["solution_path"],
            )
            solver = prism_decoder.Decoder(
                solver_problem(name, data),
                search_config={
                    "classical_behavior": args.guidance == "classical",
                    "verify_incremental_srr": args.verify_incremental_srr,
                },
                n_rollouts=rollouts,
            )
            solver.seed(20260727 + index)
            assert_normalized_model_inputs(solver)

            def guidance() -> dict:
                if args.guidance == "classical":
                    return {}
                field = np.ones(
                    (
                        solver.metadata["edge_count"],
                        len(prism_decoder.FIELD_CHANNEL_NAMES),
                    ),
                    dtype=np.float32,
                )
                multipliers = np.zeros(
                    prism_decoder.MULTIPLIER_COUNT, dtype=np.float32
                )
                # Keep the objective weight (final slot) at 1 so the neutral
                # field reduces to the plain objective edge cost.
                multipliers[prism_decoder.FIELD_CHANNEL_COUNT] = 1.0
                return {"edge_field": field, "multipliers": multipliers}

            bootstrap = solver.solve(1, **guidance())
            assert_normalized_model_inputs(solver)
            result = solver.solve(args.iterations - 1, **guidance())
            assert_normalized_model_inputs(solver)
            replay = solver.evaluate(result["route"])
            if not bootstrap["feasible"] or not result["feasible"]:
                raise RuntimeError(result["error"] or "infeasible solution")
            if not replay["feasible"]:
                raise RuntimeError("route replay failed: " + replay["error"])

            was_perturbed = result["changed_edges"] > 0
            perturbed += int(was_perturbed)
            if not was_perturbed:
                unperturbed_names.append(name)
            accepted_srr_moves += result["srr_moves"]
            srr_scope_nodes += result["srr_scope_nodes"]
            srr_revisits += result["srr_revisits"]
            srr_evaluations += result["srr_evaluations"]
            srr_incremental_rebuilds += result["srr_incremental_rebuilds"]
            srr_full_rebuilds += result["srr_full_rebuilds"]
            srr_rebuilt_nodes += result["srr_rebuilt_nodes"]
            was_srr_active = result["srr_moves"] > 0
            srr_active += int(was_srr_active)
            if not was_srr_active:
                inactive_srr_names.append(name)
            if result["direction"] == "maximize":
                is_better = result["objective"] > bootstrap["objective"] + 1e-5
                is_better |= (
                    abs(result["objective"] - bootstrap["objective"]) <= 1e-5
                    and result["distance"] < bootstrap["distance"] - 1e-5
                )
            else:
                is_better = result["objective"] < bootstrap["objective"] - 1e-5
            improved += int(is_better)
            if not is_better:
                unimproved_names.append(name)
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=PASS",
                f"bootstrap={bootstrap['objective']:.6g}",
                f"result={result['objective']:.6g}",
                f"improved={int(is_better)}",
                f"changed_edges={result['changed_edges']}",
                f"srr_moves={result['srr_moves']}",
                f"incremental_rebuilds={result['srr_incremental_rebuilds']}",
                f"full_rebuilds={result['srr_full_rebuilds']}",
                f"rebuilt_nodes={result['srr_rebuilt_nodes']}",
                f"seconds={time.perf_counter() - test_started:.3f}",
                flush=True,
            )
        except Exception as error:  # The summary must retain every failed variant.
            detail = (name, type(error).__name__, str(error))
            failures.append(detail)
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=FAIL",
                f"error={detail[1]}: {detail[2]}",
                f"seconds={time.perf_counter() - test_started:.3f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(
        "URS_ONE_EACH",
        f"guidance={args.guidance}",
        f"passed={len(variants) - len(failures)}",
        f"failed={len(failures)}",
        f"perturbed={perturbed}",
        f"improved={improved}",
        f"srr_active={srr_active}",
        f"srr_moves={accepted_srr_moves}",
        f"scope_nodes={srr_scope_nodes}",
        f"revisits={srr_revisits}",
        f"evaluations={srr_evaluations}",
        f"incremental_rebuilds={srr_incremental_rebuilds}",
        f"full_rebuilds={srr_full_rebuilds}",
        f"rebuilt_nodes={srr_rebuilt_nodes}",
        f"seconds={elapsed:.3f}",
    )
    for failure in failures:
        print("FAIL", *failure)
    if unperturbed_names:
        print("UNPERTURBED", *unperturbed_names)
    if unimproved_names:
        print("UNIMPROVED", *unimproved_names)
    if inactive_srr_names:
        print("SRR_INACTIVE", *inactive_srr_names)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
