#!/usr/bin/env python3
"""Report one-round SRR quality against exact first-instance URS references."""

import argparse
import csv
import logging
import pickle
import statistics
import time
from pathlib import Path

import torch

from problem_data import (
    BENCHMARK_VARIANTS,
    DEFAULT_DATASET_DIR,
    DatasetFinder,
    load_saved_data,
)
from urs_one_each import ROOT, prism_decoder, solver_problem


EMBEDDED_REFERENCES = {"tsp", "acvrp"}


def first_reference(paths: dict, embedded_reference: float) -> tuple[bool, float]:
    solution_path = paths["solution_path"]
    if solution_path is None:
        if paths["problem_name"] in EMBEDDED_REFERENCES:
            return True, float(embedded_reference)
        return False, 0.0

    path = Path(solution_path)
    if path.suffix == ".pkl":
        with path.open("rb") as source:
            first_solution = pickle.load(source)[0]
        if isinstance(first_solution, (list, tuple)):
            first_solution = first_solution[0]
        return True, float(first_solution)
    if path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=False)
        cost = saved["cost"]
        if torch.is_tensor(cost):
            cost = cost.reshape(-1)[0].item()
        elif isinstance(cost, (list, tuple)):
            cost = cost[0]
        return True, float(cost)
    raise ValueError(f"unsupported reference file: {path}")


def gap_percent(result: dict, reference: float) -> float:
    if result["direction"] == "maximize":
        return (reference - result["objective"]) / abs(reference) * 100.0
    return (result["objective"] - reference) / abs(reference) * 100.0


def improvement_percent(bootstrap: dict, result: dict) -> float:
    denominator = max(abs(bootstrap["objective"]), 1e-9)
    if result["direction"] == "maximize":
        return (result["objective"] - bootstrap["objective"]) / denominator * 100.0
    return (bootstrap["objective"] - result["objective"]) / denominator * 100.0


def group_rows(rows: list[dict], label: str, predicate) -> dict:
    selected = [row for row in rows if row["has_reference"] and predicate(row)]
    return {
        "group": label,
        "count": len(selected),
        "bootstrap_mean_gap": statistics.fmean(
            row["bootstrap_gap_pct"] for row in selected
        ),
        "srr_mean_gap": statistics.fmean(row["srr_gap_pct"] for row in selected),
        "srr_median_gap": statistics.median(
            row["srr_gap_pct"] for row in selected
        ),
        "mean_improvement": statistics.fmean(
            row["improvement_pct"] for row in selected
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ants", type=int)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--threads", type=int, default=prism_decoder.get_available_threads()
    )
    parser.add_argument("--disable-srr", action="store_true")
    parser.add_argument("--no-pheromone", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")

    logging.disable(logging.CRITICAL)
    finder = DatasetFinder(args.dataset_dir)
    prism_decoder.set_num_threads(args.threads)
    ants = args.ants if args.ants is not None else args.threads
    rows = []
    started = time.perf_counter()
    for index, name in enumerate(BENCHMARK_VARIANTS):
        paths = finder.get(name, 100)
        data, embedded_reference = load_saved_data(
            paths["data_path"],
            name,
            1,
            solution_path=paths["solution_path"],
        )
        has_reference, reference = first_reference(paths, embedded_reference)
        solver = prism_decoder.Decoder(
            solver_problem(name, data),
            search_config={
                "use_srr": not args.disable_srr,
                "use_pheromone": not args.no_pheromone,
            },
            n_ants=ants,
        )
        solver.seed(20260727 + index)
        bootstrap = solver.solve(1)
        result = solver.solve(args.iterations - 1)
        if not result["feasible"] or not solver.evaluate(result["route"])["feasible"]:
            raise RuntimeError(f"{name}: {result['error']}")

        metadata = solver.metadata
        row = {
            "name": name,
            "direction": result["direction"],
            "bootstrap": bootstrap["objective"],
            "srr": result["objective"],
            "reference": reference if has_reference else "",
            "has_reference": has_reference,
            "bootstrap_gap_pct": gap_percent(bootstrap, reference)
            if has_reference
            else "",
            "srr_gap_pct": gap_percent(result, reference)
            if has_reference
            else "",
            "improvement_pct": improvement_percent(bootstrap, result),
            "changed_edges": result["changed_edges"],
            "srr_moves": result["srr_moves"],
            "srr_scope_nodes": result["srr_scope_nodes"],
            "srr_revisits": result["srr_revisits"],
            "srr_evaluations": result["srr_evaluations"],
            "srr_incremental_rebuilds": result["srr_incremental_rebuilds"],
            "srr_full_rebuilds": result["srr_full_rebuilds"],
            "srr_rebuilt_nodes": result["srr_rebuilt_nodes"],
            "asymmetric": name.startswith("a"),
            "time_windows": "tw" in name,
            "multi_depot": "md" in name,
            "strict_backhaul": "bp" in name,
            "open_route": metadata["open_route"],
            "pickup_delivery": "pd" in name,
            "optional": name in {"op", "aop"} or "pctsp" in name,
            "use_pheromone": not args.no_pheromone,
        }
        rows.append(row)

    exact = [row for row in rows if row["has_reference"]]
    groups = [
        group_rows(rows, "all", lambda row: True),
        group_rows(rows, "symmetric", lambda row: not row["asymmetric"]),
        group_rows(rows, "asymmetric", lambda row: row["asymmetric"]),
        group_rows(rows, "time_windows", lambda row: row["time_windows"]),
        group_rows(rows, "multi_depot", lambda row: row["multi_depot"]),
        group_rows(rows, "strict_backhaul", lambda row: row["strict_backhaul"]),
        group_rows(rows, "open_route", lambda row: row["open_route"]),
        group_rows(rows, "pickup_delivery", lambda row: row["pickup_delivery"]),
        group_rows(rows, "optional", lambda row: row["optional"]),
    ]
    print(
        "URS_SRR_REPORT",
        f"mode={'perturb_only' if args.disable_srr else 'srr'}",
        f"pheromone={'off' if args.no_pheromone else 'on'}",
        f"variants={len(rows)}",
        f"references={len(exact)}",
        f"improved={sum(row['improvement_pct'] > 0 for row in rows)}",
        f"scope_nodes={sum(row['srr_scope_nodes'] for row in rows)}",
        f"revisits={sum(row['srr_revisits'] for row in rows)}",
        f"evaluations={sum(row['srr_evaluations'] for row in rows)}",
        "incremental_rebuilds="
        f"{sum(row['srr_incremental_rebuilds'] for row in rows)}",
        f"full_rebuilds={sum(row['srr_full_rebuilds'] for row in rows)}",
        f"rebuilt_nodes={sum(row['srr_rebuilt_nodes'] for row in rows)}",
        f"seconds={time.perf_counter() - started:.3f}",
    )
    for group in groups:
        print(
            f"{group['group']:16s}",
            f"n={group['count']:3d}",
            f"bootstrap_mean={group['bootstrap_mean_gap']:8.3f}%",
            f"srr_mean={group['srr_mean_gap']:8.3f}%",
            f"srr_median={group['srr_median_gap']:8.3f}%",
            f"mean_gain={group['mean_improvement']:7.3f}%",
        )
    print("BEST", *[f"{row['name']}={row['srr_gap_pct']:.3f}%" for row in sorted(exact, key=lambda row: row["srr_gap_pct"])[:5]])
    print("WORST", *[f"{row['name']}={row['srr_gap_pct']:.3f}%" for row in sorted(exact, key=lambda row: row["srr_gap_pct"], reverse=True)[:5]])
    missing = [row["name"] for row in rows if not row["has_reference"]]
    print("NO_REFERENCE", *missing)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
