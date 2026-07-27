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
sys.path.insert(0, str(ROOT / "baselines" / "URS"))

import prism_decoder  # noqa: E402
from data.DataFinder import DataFinder  # noqa: E402
from data.DataReader import get_saved_data  # noqa: E402
from problem.ProblemSet import ProblemSet  # noqa: E402


def first(value):
    if torch.is_tensor(value):
        array = value[0].detach().cpu().numpy()
        return array.astype(np.float32) if array.dtype.kind == "f" else array
    return value


def solver_problem(name: str, data: dict) -> dict:
    problem = {"name": name, "capacity": 1.0, "prize_quota": 1.0}
    if "xy" in data:
        coordinates = first(data["xy"])
        problem["coordinates"] = coordinates
        problem["distance"] = np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32)
    else:
        problem["distance"] = first(data["dist"])

    for field in (
        "demand",
        "prize",
        "penalty",
        "tw_start",
        "tw_end",
        "service_time",
    ):
        if field in data:
            problem[field] = first(data[field])
    if "route_limit" in data:
        problem["route_limit"] = float(first(data["route_limit"]))
    if name == "op":
        problem["tour_limit"] = 4.0
    elif name == "aop":
        problem["tour_limit"] = 1.0
    return problem


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
    parser.add_argument("--ants", type=int)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--threads", type=int, default=prism_decoder.get_available_threads()
    )
    parser.add_argument(
        "--guidance", choices=("classical", "field"), default="classical"
    )
    parser.add_argument("--no-pheromone", action="store_true")
    parser.add_argument("--verify-incremental-srr", action="store_true")
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2 to exercise perturbation")

    logging.disable(logging.CRITICAL)
    finder = DataFinder(ROOT / "baselines" / "URS" / "dataset")
    prism_decoder.set_num_threads(args.threads)
    ants = args.ants if args.ants is not None else args.threads
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

    for index, name in enumerate(ProblemSet.get()):
        test_started = time.perf_counter()
        try:
            paths = finder.get(name, 100)
            data, _ = get_saved_data(
                paths["data_path"],
                name,
                1,
                "cpu",
                solution_name=paths["solution_path"],
            )
            solver = prism_decoder.Decoder(
                solver_problem(name, data),
                search_config={
                    "classical_behavior": args.guidance == "classical",
                    "use_pheromone": not args.no_pheromone,
                    "verify_incremental_srr": args.verify_incremental_srr,
                },
                n_ants=ants,
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
                    len(prism_decoder.FIELD_CHANNEL_NAMES), dtype=np.float32
                )
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
                f"{index + 1}/{len(ProblemSet.get())}",
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
                f"{index + 1}/{len(ProblemSet.get())}",
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
        f"pheromone={'off' if args.no_pheromone else 'on'}",
        f"passed={len(ProblemSet.get()) - len(failures)}",
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
