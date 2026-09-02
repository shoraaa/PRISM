"""Compare opt-in Python refinement with the untouched native solve path.

Both methods start from the same deliberately simple feasible incumbent and use
the same checkpoint.  Native ``solve(1)`` includes its production perturbation
before SRR, so this is an end-to-end reference rather than move-for-move parity.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import prism_decoder  # noqa: E402
from net import (  # noqa: E402
    ConstraintFieldNet,
    build_decoder_data,
    decode_python_refinement,
    load_constraint_field_state_dict,
)
from problem_data import generated_problem, problem_schema  # noqa: E402
from search import NeighborhoodConfig, SearchConfig  # noqa: E402


def _model(checkpoint_path: Path, device: str) -> ConstraintFieldNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = ConstraintFieldNet(
        couple_resource_tokens=config.get("couple_resource_tokens", True),
        couple_state_multipliers=config.get("couple_state_multipliers", True),
        gate_multipliers_by_binding=config.get(
            "gate_multipliers_by_binding", True
        ),
        index_embedded_resources=config.get("index_embedded_resources", False),
        monolithic_resource_field=config.get("monolithic_resource_field", False),
        program_blind_resources=config.get("program_blind_resources", False),
        normalize_projections=config.get("normalize_projections", "none"),
        core_interface=config.get("core_interface", "full"),
    ).to(device)
    load_constraint_field_state_dict(
        model,
        checkpoint["model_state_dict"],
        model_schema=checkpoint.get("model_schema"),
        config=config,
    )
    return model.eval()


def _problem(variant: str, size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    problem = problem_schema(variant)
    problem.update(generated_problem(variant, size))
    return problem


def _incumbent(problem: dict) -> np.ndarray:
    count = len(problem.get("coordinates", problem.get("distance")))
    depots = int(problem["depot_count"])
    if depots == 0:
        return np.arange(count, dtype=np.int32)
    route = [0]
    for node in range(depots, count):
        route.extend((node, 0))
    return np.asarray(route, dtype=np.int32)


def _guidance(output: dict) -> dict:
    return {
        "edge_field": output["residual"].detach().cpu().numpy(),
        "multipliers": output["multipliers"][0].detach().cpu().numpy(),
        "coupler_weights": output["coupler_weights"][0].detach().cpu().numpy(),
        "coupler_bias": output["coupler_bias"][0].detach().cpu().numpy(),
    }


def _gain(initial: dict, final: dict) -> float:
    before = float(initial["objective"])
    after = float(final["objective"])
    raw = after - before if initial["direction"] == "maximize" else before - after
    return 100.0 * raw / max(abs(before), 1.0e-12)


def _run_one(args: argparse.Namespace, model, variant: str, size: int, seed: int) -> dict:
    problem = _problem(variant, size, seed)
    route = _incumbent(problem)
    common = {
        "candidate_config": {"max_candidates": args.candidates},
        "search_config": {
            "srr_exploration_budget": 0,
            "min_changed_edges": args.min_changed_edges,
        },
        "n_rollouts": 1,
        "beta": 2.0,
    }
    native = prism_decoder.Decoder(problem, **common)
    python_host = prism_decoder.Decoder(problem, **common)
    initial = native.evaluate(route)
    if not initial["feasible"]:
        raise RuntimeError(f"{variant}-{size} incumbent is infeasible: {initial['error']}")
    native.set_incumbent(route)
    python_host.set_incumbent(route)

    started = time.perf_counter()
    with torch.no_grad():
        native_output = model(build_decoder_data(native, args.device))
    native_model_seconds = time.perf_counter() - started
    started = time.perf_counter()
    native_solution = native.solve(1, **_guidance(native_output))
    native_search_seconds = time.perf_counter() - started

    started = time.perf_counter()
    python_solution, _output, python_result = decode_python_refinement(
        problem,
        python_host,
        model,
        args.device,
        search_config=SearchConfig(
            max_iterations=args.python_iterations,
            max_evaluations=args.python_evaluations,
            max_candidates_per_iteration=args.python_candidates,
            shortlist_size=args.shortlist_size,
            selection=args.selection,
        ),
        neighborhood_config=NeighborhoodConfig(
            nearest_neighbors=args.neighbors,
            max_segment_length=args.segment_length,
        ),
        install=False,
    )
    python_total_seconds = time.perf_counter() - started
    python_search_seconds = float(python_result.elapsed_seconds)
    python_model_seconds = max(python_total_seconds - python_search_seconds, 0.0)
    return {
        "variant": variant,
        "size": size,
        "seed": seed,
        "initial_objective": float(initial["objective"]),
        "native_objective": float(native_solution["objective"]),
        "python_objective": float(python_solution["objective"]),
        "native_gain_percent": _gain(initial, native_solution),
        "python_gain_percent": _gain(initial, python_solution),
        "native_model_seconds": native_model_seconds,
        "native_search_seconds": native_search_seconds,
        "python_model_seconds": python_model_seconds,
        "python_search_seconds": python_search_seconds,
        "native_srr_moves": int(native_solution["srr_moves"]),
        "python_moves": python_result.moves,
        "python_evaluations": python_result.evaluations,
        "python_executed_transitions": python_result.executed_transitions,
        "python_reused_transitions": python_result.reused_transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained/v14/best.pt")
    parser.add_argument("--variants", nargs="+", default=["tsp", "cvrp", "cvrptw"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--min-changed-edges", type=int, default=8)
    parser.add_argument("--python-iterations", type=int, default=1)
    parser.add_argument("--python-evaluations", type=int, default=512)
    parser.add_argument("--python-candidates", type=int, default=512)
    parser.add_argument("--shortlist-size", type=int, default=32)
    parser.add_argument(
        "--selection",
        choices=("first", "objective_first", "objective", "guidance"),
        default="objective",
    )
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--segment-length", type=int, default=2)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    prism_decoder.set_num_threads(args.threads)
    model = _model(args.checkpoint, args.device)
    rows = [
        _run_one(args, model, variant, size, seed)
        for variant in args.variants
        for size in args.sizes
        for seed in args.seeds
    ]
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(rows, indent=2))
    print("summary")
    for variant in args.variants:
        for size in args.sizes:
            selected = [
                row for row in rows
                if row["variant"] == variant and row["size"] == size
            ]
            print(
                variant,
                size,
                "native_gain=", statistics.median(row["native_gain_percent"] for row in selected),
                "python_gain=", statistics.median(row["python_gain_percent"] for row in selected),
                "native_search_s=", statistics.median(row["native_search_seconds"] for row in selected),
                "python_search_s=", statistics.median(row["python_search_seconds"] for row in selected),
            )


if __name__ == "__main__":
    main()
