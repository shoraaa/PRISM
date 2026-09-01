#!/usr/bin/env python3
"""Compare learned, distance-heuristic, and uniform route construction.

This is deliberately a construction-only diagnostic.  Each method receives
the same decoder problem, candidate graph, number of rollouts, and random seed;
the only changed input is the candidate energy used by ``Decoder.sample``:

``learned``
    The complete checkpoint field, including its feasibility-risk term.
``heuristic``
    Normalized geometric/travel distance, with learned terms disabled.
``uniform``
    Identical energy for every legal candidate.

No incumbent is installed and no perturbation, local search, or SRR is run.
Consequently the reported feasibility rates measure construction itself rather
than the exact decoder's ability to repair a constructed route afterwards.

Example::

    PYTHONPATH=src uv run --no-sync python \
      scripts/compare_construction_guidance.py \
      --variants all --val-size 8 \
      --csv results/construction_guidance.csv \
      --json results/construction_guidance.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import prism_decoder  # noqa: E402
from net import (  # noqa: E402
    ConstraintFieldNet,
    load_constraint_field_state_dict,
)
from prism_eval.instances import (  # noqa: E402
    build_instances,
    selected_variants,
    variant_split,
)
from problem_data import DEFAULT_DATASET_DIR, DatasetFinder  # noqa: E402
from train import (  # noqa: E402
    _constant_guidance,
    _distance_guidance,
    _field_guidance,
    _new_decoder,
    setup_seeds,
)


METHODS = ("learned", "heuristic", "uniform")
CSV_FIELDS = (
    "variant",
    "split",
    "n",
    "instance",
    "method",
    "rollout",
    "seed",
    "feasible",
    "direction",
    "objective",
    "error",
    "guidance_seconds",
    "construction_seconds",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure construction-only feasibility under a pretrained field, "
            "a distance heuristic, and uniform legal-candidate sampling."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "pretrained" / "newer" / "best.pt",
        help="checkpoint to evaluate (default: pretrained/newer/best.pt)",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="comma-separated benchmark variants, or 'all' (default: all)",
    )
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-scale", type=int, default=100)
    parser.add_argument(
        "--n-node",
        "--n_node",
        dest="n_node",
        type=int,
        default=None,
        help="generate fresh instances with this many customers",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="constructions per instance (default: checkpoint n_rollouts)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="candidate limit (default: checkpoint candidates)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="construction sampling inverse temperature (default: checkpoint beta)",
    )
    parser.add_argument(
        "--feasibility-lookahead-depth",
        type=int,
        default=None,
        help="hard native lookahead shared by all methods (default: checkpoint value)",
    )
    parser.add_argument(
        "--risk-penalty",
        type=float,
        default=None,
        help="learned risk-energy weight (default: checkpoint value)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="instance and decoder seed (default: checkpoint seed)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--threads", type=int, default=prism_decoder.get_available_threads()
    )
    parser.add_argument("--csv", type=Path, help="optional per-rollout output")
    parser.add_argument("--json", type=Path, help="optional aggregate output")

    # Optional evaluator families use the same instance loader as test.py.
    parser.add_argument("--evrp-size", type=int, default=100)
    parser.add_argument("--vrpdb-size", type=int, default=100)
    parser.add_argument("--tsptw-size", type=int, default=100)
    parser.add_argument(
        "--tsptw-hardness", choices=("hard", "medium", "easy"), default="hard"
    )
    parser.add_argument(
        "--tsptw-source", choices=("dataset", "generator"), default="dataset"
    )
    parser.add_argument(
        "--tsptw-data-dir",
        type=Path,
        default=ROOT / "baselines" / "CaR-constraint" / "data" / "TSPTW",
    )
    parser.add_argument("--tsptw-dataset-seed", type=int, default=2025)

    args = parser.parse_args(argv)
    if args.val_size < 1:
        parser.error("--val-size must be positive")
    if args.n_node is not None and args.n_node < 1:
        parser.error("--n-node must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def model_constructor_kwargs(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Recover every architecture switch understood by the active model.

    Matching the constructor signature makes this future-safe when a new flag
    is added.  ``resource_pooling`` is the training-CLI name of the constructor's
    ``pool_node_resources`` switch, so it needs the one explicit alias.
    """
    parameters = inspect.signature(ConstraintFieldNet.__init__).parameters
    config = checkpoint.get("config") or {}
    stored_kwargs = checkpoint.get("model_kwargs") or {}
    sources = (checkpoint, stored_kwargs, config)
    aliases = {"pool_node_resources": ("pool_node_resources", "resource_pooling")}
    kwargs: dict[str, Any] = {}
    for name in parameters:
        if name == "self":
            continue
        keys = aliases.get(name, (name,))
        for source in sources:
            match = next((key for key in keys if key in source), None)
            if match is not None:
                kwargs[name] = source[match]
                break
    return kwargs


def load_model(
    checkpoint_path: Path, device: str
) -> tuple[ConstraintFieldNet, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    schema = checkpoint.get("model_schema")
    config = checkpoint.get("config", {})
    model = ConstraintFieldNet(**model_constructor_kwargs(checkpoint)).to(device)
    load_constraint_field_state_dict(
        model,
        checkpoint["model_state_dict"],
        model_schema=schema,
        config=config,
    )
    model.eval()
    return model, checkpoint


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_settings(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> SimpleNamespace:
    config = checkpoint.get("config") or {}

    def choose(name: str, fallback: Any) -> Any:
        value = getattr(args, name)
        return config.get(name, fallback) if value is None else value

    settings = SimpleNamespace(
        candidates=int(choose("candidates", 64)),
        n_rollouts=int(choose("rollouts", config.get("n_rollouts", 10))),
        beta=float(choose("beta", 2.0)),
        seed=int(choose("seed", 1234)),
        feasibility_lookahead_depth=int(
            choose("feasibility_lookahead_depth", 2)
        ),
        risk_penalty=float(
            args.risk_penalty
            if args.risk_penalty is not None
            else config.get("feasibility_risk_penalty", 1.0)
        ),
        device=args.device,
        min_changed_edges=1,  # unused because this diagnostic never invokes SRR
        random_escape=False,
        srr_exploration_budget=0,
    )
    if settings.candidates < 1:
        raise ValueError("--candidates must be positive")
    if settings.n_rollouts < 1:
        raise ValueError("--rollouts must be positive")
    if not math.isfinite(settings.beta) or settings.beta <= 0.0:
        raise ValueError("--beta must be finite and positive")
    if settings.feasibility_lookahead_depth < 0:
        raise ValueError("--feasibility-lookahead-depth must be nonnegative")
    if not math.isfinite(settings.risk_penalty) or settings.risk_penalty < 0.0:
        raise ValueError("--risk-penalty must be finite and nonnegative")
    return settings


def _guidance(method: str, model, decoder, problem, settings) -> dict:
    if method == "learned":
        return _field_guidance(
            model,
            decoder,
            settings.device,
            risk_penalty=settings.risk_penalty,
        )
    if method == "heuristic":
        return _distance_guidance(decoder, problem)
    if method == "uniform":
        return _constant_guidance(decoder)
    raise ValueError(f"unknown construction method: {method}")


def construct(
    model: ConstraintFieldNet,
    problem: dict,
    method: str,
    settings: SimpleNamespace,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one matched construction batch and return one record per rollout."""
    decoder_args = SimpleNamespace(**vars(settings))
    decoder_args.seed = seed
    decoder = _new_decoder(
        problem, decoder_args, deterministic=True, use_srr=False
    )
    guidance_started = time.perf_counter()
    guidance = _guidance(method, model, decoder, problem, settings)
    guidance_seconds = time.perf_counter() - guidance_started
    construction_started = time.perf_counter()
    solutions = list(decoder.sample(**guidance))
    construction_seconds = time.perf_counter() - construction_started
    if len(solutions) != settings.n_rollouts:
        raise RuntimeError(
            f"decoder returned {len(solutions)} rollouts, expected "
            f"{settings.n_rollouts}"
        )
    amortized_guidance = guidance_seconds / len(solutions)
    amortized_construction = construction_seconds / len(solutions)
    records = []
    for rollout, solution in enumerate(solutions):
        objective = solution.get("objective")
        feasible = bool(solution.get("feasible", False))
        if objective is None or not math.isfinite(float(objective)):
            feasible = False
            objective = None
        records.append(
            {
                "rollout": rollout,
                "method": method,
                "feasible": int(feasible),
                "direction": str(solution.get("direction", "")),
                "objective": float(objective) if feasible else "",
                "error": "" if feasible else str(solution.get("error", "infeasible")),
                "guidance_seconds": amortized_guidance,
                "construction_seconds": amortized_construction,
            }
        )
    return records


def failure_records(
    method: str, rollouts: int, error: BaseException
) -> list[dict[str, Any]]:
    detail = f"{type(error).__name__}: {error}"
    return [
        {
            "rollout": rollout,
            "method": method,
            "feasible": 0,
            "direction": "",
            "objective": "",
            "error": detail,
            "guidance_seconds": 0.0,
            "construction_seconds": 0.0,
        }
        for rollout in range(rollouts)
    ]


def _best(records: Iterable[dict[str, Any]]) -> tuple[float, str] | None:
    feasible = [row for row in records if row["feasible"]]
    if not feasible:
        return None
    direction = next(
        (str(row["direction"]) for row in feasible if row["direction"]),
        "minimize",
    )
    objectives = [float(row["objective"]) for row in feasible]
    return (
        max(objectives) if direction == "maximize" else min(objectives),
        direction,
    )


def _relative_improvement(
    learned: float, baseline: float, direction: str
) -> float:
    scale = max(abs(baseline), 1.0e-9)
    if direction == "maximize":
        return (learned - baseline) / scale * 100.0
    return (baseline - learned) / scale * 100.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant_method: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    by_instance_method: dict[
        tuple[str, int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in records:
        by_method[row["method"]].append(row)
        by_variant_method[(row["variant"], row["method"])].append(row)
        by_instance_method[
            (row["variant"], int(row["instance"]), row["method"])
        ].append(row)

    method_summary: dict[str, Any] = {}
    variant_summary: dict[str, Any] = {}
    for method in METHODS:
        rows = by_method.get(method, [])
        instance_keys = {
            (row["variant"], int(row["instance"])) for row in rows
        }
        solved = sum(
            _best(by_instance_method[(variant, instance, method)]) is not None
            for variant, instance in instance_keys
        )
        feasible_rollouts = sum(int(row["feasible"]) for row in rows)
        method_summary[method] = {
            "rollouts": len(rows),
            "feasible_rollouts": feasible_rollouts,
            "rollout_feasibility_rate": (
                feasible_rollouts / len(rows) if rows else 0.0
            ),
            "instances": len(instance_keys),
            "feasible_instances": solved,
            "instance_feasibility_rate": (
                solved / len(instance_keys) if instance_keys else 0.0
            ),
            "guidance_seconds": sum(float(row["guidance_seconds"]) for row in rows),
            "construction_seconds": sum(
                float(row["construction_seconds"]) for row in rows
            ),
        }

    variants = sorted({row["variant"] for row in records})
    for variant in variants:
        variant_summary[variant] = {}
        for method in METHODS:
            rows = by_variant_method.get((variant, method), [])
            instances = sorted({int(row["instance"]) for row in rows})
            best = [
                _best(by_instance_method[(variant, instance, method)])
                for instance in instances
            ]
            objectives = [value[0] for value in best if value is not None]
            feasible_rollouts = sum(int(row["feasible"]) for row in rows)
            variant_summary[variant][method] = {
                "rollouts": len(rows),
                "feasible_rollouts": feasible_rollouts,
                "rollout_feasibility_rate": (
                    feasible_rollouts / len(rows) if rows else 0.0
                ),
                "instances": len(instances),
                "feasible_instances": len(objectives),
                "instance_feasibility_rate": (
                    len(objectives) / len(instances) if instances else 0.0
                ),
                "mean_best_objective": (
                    float(np.mean(objectives)) if objectives else None
                ),
            }

    comparisons: dict[str, Any] = {}
    learned_rates = method_summary.get("learned", {})
    for baseline in ("heuristic", "uniform"):
        improvements: list[float] = []
        variant_improvements: dict[str, list[float]] = defaultdict(list)
        learned_only = baseline_only = both_infeasible = wins = ties = losses = 0
        instance_keys = sorted(
            {
                (row["variant"], int(row["instance"]))
                for row in records
                if row["method"] == "learned"
            }
        )
        for variant, instance in instance_keys:
            learned = _best(by_instance_method[(variant, instance, "learned")])
            other = _best(by_instance_method[(variant, instance, baseline)])
            if learned is None and other is None:
                both_infeasible += 1
                continue
            if learned is not None and other is None:
                learned_only += 1
                continue
            if learned is None and other is not None:
                baseline_only += 1
                continue
            assert learned is not None and other is not None
            improvement = _relative_improvement(
                learned[0], other[0], learned[1]
            )
            improvements.append(improvement)
            variant_improvements[variant].append(improvement)
            if improvement > 1.0e-9:
                wins += 1
            elif improvement < -1.0e-9:
                losses += 1
            else:
                ties += 1
        macro = [float(np.mean(values)) for values in variant_improvements.values()]
        base_rates = method_summary.get(baseline, {})
        comparisons[baseline] = {
            "learned_minus_baseline_rollout_feasibility_pp": 100.0
            * (
                learned_rates.get("rollout_feasibility_rate", 0.0)
                - base_rates.get("rollout_feasibility_rate", 0.0)
            ),
            "learned_minus_baseline_instance_feasibility_pp": 100.0
            * (
                learned_rates.get("instance_feasibility_rate", 0.0)
                - base_rates.get("instance_feasibility_rate", 0.0)
            ),
            "learned_only_feasible_instances": learned_only,
            "baseline_only_feasible_instances": baseline_only,
            "both_infeasible_instances": both_infeasible,
            "paired_feasible_instances": len(improvements),
            "learned_wins": wins,
            "ties": ties,
            "learned_losses": losses,
            "mean_paired_best_objective_improvement_percent": (
                float(np.mean(improvements)) if improvements else None
            ),
            "macro_variant_best_objective_improvement_percent": (
                float(np.mean(macro)) if macro else None
            ),
        }
    return {
        "methods": method_summary,
        "variants": variant_summary,
        "comparisons": comparisons,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in CSV_FIELDS}
            for row in records
        )


def _format_rate(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def print_summary(summary: dict[str, Any]) -> None:
    for method in METHODS:
        result = summary["methods"][method]
        print(
            "OVERALL",
            f"method={method}",
            f"rollout_feasible={result['feasible_rollouts']}/{result['rollouts']}",
            f"rollout_rate={_format_rate(result['rollout_feasibility_rate'])}",
            f"instance_feasible={result['feasible_instances']}/{result['instances']}",
            f"instance_rate={_format_rate(result['instance_feasibility_rate'])}",
            f"guidance_seconds={result['guidance_seconds']:.3f}",
            f"construction_seconds={result['construction_seconds']:.3f}",
        )
    for baseline, result in summary["comparisons"].items():
        objective = result["macro_variant_best_objective_improvement_percent"]
        objective_text = "n/a" if objective is None else f"{objective:.3f}%"
        print(
            "COMPARE",
            f"baseline={baseline}",
            "learned_rollout_feasibility_delta="
            f"{result['learned_minus_baseline_rollout_feasibility_pp']:.2f}pp",
            "learned_instance_feasibility_delta="
            f"{result['learned_minus_baseline_instance_feasibility_pp']:.2f}pp",
            f"learned_only={result['learned_only_feasible_instances']}",
            f"baseline_only={result['baseline_only_feasible_instances']}",
            f"paired_wins_ties_losses={result['learned_wins']}/"
            f"{result['ties']}/{result['learned_losses']}",
            f"macro_best_objective_improvement={objective_text}",
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, checkpoint = load_model(args.checkpoint, args.device)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    settings = resolve_settings(args, checkpoint)
    args.seed = settings.seed
    setup_seeds(settings.seed)
    prism_decoder.set_num_threads(args.threads)
    variants = selected_variants(args.variants)
    finder = DatasetFinder(args.dataset_dir)
    records: list[dict[str, Any]] = []

    print(
        "CHECKPOINT",
        f"path={args.checkpoint.resolve()}",
        f"sha256={checkpoint_sha256}",
        f"schema={checkpoint.get('model_schema')}",
        f"epoch={checkpoint.get('epoch', 'unknown')}",
    )
    print(
        "CONFIG",
        f"variants={len(variants)}",
        f"val_size={args.val_size}",
        f"rollouts={settings.n_rollouts}",
        f"candidates={settings.candidates}",
        f"beta={settings.beta}",
        f"lookahead={settings.feasibility_lookahead_depth}",
        f"risk_penalty={settings.risk_penalty}",
        f"seed={settings.seed}",
        "search=disabled",
    )

    for variant_index, variant in enumerate(variants):
        batch = build_instances(variant, variant_index, args, finder)
        for instance in range(len(batch)):
            problem, _source_initial_route = batch.problem(instance)
            seed = settings.seed + variant_index * len(batch) + instance
            for method in METHODS:
                try:
                    batch_records = construct(
                        model, problem, method, settings, seed
                    )
                except Exception as error:  # preserve paired rows on failure
                    batch_records = failure_records(
                        method, settings.n_rollouts, error
                    )
                for row in batch_records:
                    row.update(
                        {
                            "variant": variant,
                            "split": variant_split(variant),
                            "n": batch.n,
                            "instance": instance,
                            "seed": seed,
                        }
                    )
                    records.append(row)
            instance_records = [
                row
                for row in records
                if row["variant"] == variant and row["instance"] == instance
            ]
            status = []
            for method in METHODS:
                rows = [row for row in instance_records if row["method"] == method]
                status.append(
                    f"{method}={sum(int(row['feasible']) for row in rows)}/{len(rows)}"
                )
            print(
                "CONSTRUCT",
                f"variant={variant}",
                f"instance={instance + 1}/{len(batch)}",
                "feasible_rollouts=[" + ",".join(status) + "]",
                flush=True,
            )

    summary = summarize(records)
    print_summary(summary)
    if args.csv is not None:
        write_csv(args.csv, records)
        print(f"CSV {args.csv}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": checkpoint_sha256,
                "schema": checkpoint.get("model_schema"),
                "epoch": checkpoint.get("epoch"),
            },
            "config": {
                "variants": variants,
                "val_size": args.val_size,
                "rollouts": settings.n_rollouts,
                "candidates": settings.candidates,
                "beta": settings.beta,
                "feasibility_lookahead_depth": settings.feasibility_lookahead_depth,
                "risk_penalty": settings.risk_penalty,
                "seed": settings.seed,
                "dataset_scale": args.dataset_scale,
                "n_node": args.n_node,
                "search": False,
            },
            "summary": summary,
        }
        with args.json.open("w") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
        print(f"JSON {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
