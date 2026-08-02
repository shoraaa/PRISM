#!/usr/bin/env python3
"""Evaluate a trained field decoder on all 110 routing benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "tests"))

import prism_decoder  # noqa: E402
from net import (  # noqa: E402
    ConstraintFieldNet,
    MODEL_SCHEMA,
    load_constraint_field_state_dict,
)
from problem_data import (  # noqa: E402
    BENCHMARK_VARIANTS,
    DEFAULT_DATASET_DIR,
    DatasetFinder,
    TRAIN_VARIANTS,
    load_saved_data,
)
from train import _canonical_cost, infer_instance, setup_seeds  # noqa: E402
from urs_one_each import solver_problem  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a checkpoint with the non-neural decoder on all 110 "
            "saved size-100 benchmark variants by default, with separate "
            "SEEN and HELDOUT summaries."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="all",
        help="comma-separated names, or 'all' for all 110 variants (default)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=16,
        help="post-bootstrap perturbation/SRR iterations (default: 16)",
    )
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument(
        "--candidate-mode",
        choices=["schema", "geometric"],
        default="schema",
        help=(
            "Candidate-graph construction shared by PRISM and the fields-off "
            "baseline. 'schema' (default) admits resource candidates by the "
            "schema-derived relevance (uniform equal-share prior when no learned "
            "quota is installed); 'geometric' keeps only the distance "
            "neighborhood."
        ),
    )
    parser.add_argument(
        "--learned-candidate-quotas",
        action="store_true",
        help="Let the learned quota policy reweight the schema allocation.",
    )
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
    parser.add_argument(
        "--baseline",
        choices=["fields-off", "classical"],
        default="fields-off",
        help=(
            "Reference the neural field is scored against. 'fields-off' (default)"
            " is the same decoder with the field ablated to pure distance"
            " (E = c(e)); 'classical' is the hand-tuned proximity ranking."
        ),
    )
    parser.add_argument(
        "--urs-checkpoint",
        dest="urs_checkpoint",
        type=Path,
        default=None,
        help=(
            "If set, also run the URS unified checkpoint (baselines/URS) on the"
            " same instances in an isolated subprocess and report a paired"
            " PRISM-vs-URS comparison. Point at e.g."
            " baselines/URS/pretrained/unified_checkpoint_500.pt."
        ),
    )
    parser.add_argument(
        "--urs-cuda",
        dest="urs_cuda",
        type=int,
        default=-1,
        help="CUDA device for the URS subprocess (-1 = CPU).",
    )
    parser.add_argument(
        "--urs-batch-size", dest="urs_batch_size", type=int, default=50
    )
    parser.add_argument(
        "--urs-no-aug",
        dest="urs_no_aug",
        action="store_true",
        help="Disable URS instance augmentation (faster, weaker URS).",
    )
    parser.add_argument(
        "--urs-cache",
        dest="urs_cache",
        type=Path,
        default=ROOT / "results" / "urs_cache.json",
        help=(
            "JSON cache of URS per-instance objectives, keyed by variant +"
            " checkpoint + augmentation + data file. Reused across runs so URS"
            " is not re-evaluated; a changed checkpoint invalidates its entries."
        ),
    )
    parser.add_argument(
        "--urs-refresh",
        dest="urs_refresh",
        action="store_true",
        help="Ignore any cached URS results and recompute (and overwrite them).",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args(argv)


URS_RUNNER = ROOT / "baselines" / "URS" / "run_instances.py"


def _urs_cache_key(name: str, data_path: str, args: argparse.Namespace) -> str:
    """Identity of a URS result: variant, checkpoint (path+mtime), aug, data."""
    checkpoint = Path(args.urs_checkpoint).resolve()
    try:
        stamp = checkpoint.stat().st_mtime_ns
    except OSError:
        stamp = 0
    aug = "noaug" if args.urs_no_aug else "aug"
    return f"{name}|{checkpoint}|{stamp}|{aug}|{Path(data_path).resolve()}"


def load_urs_cache(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with path.open() as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_urs_cache(path: Path | None, cache: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(cache, handle)
    temporary.replace(path)


def run_urs_variant(
    name: str,
    data_path: str,
    episodes: int,
    args: argparse.Namespace,
    cache: dict,
) -> dict | None:
    """Run URS on ``episodes`` instances of ``name`` via an isolated subprocess.

    Returns ``{"objectives": [...], "direction": ...}`` with per-instance
    objectives in dataset order (aligned with PRISM, which reads the same
    ``data_path``), or ``None`` if URS does not cover the variant or the run
    fails. Results are cached to ``args.urs_cache`` keyed by variant, checkpoint,
    augmentation and data file; a cached run with at least ``episodes`` instances
    is reused (sliced) instead of re-invoking URS.
    """
    key = _urs_cache_key(name, data_path, args)
    cached = cache.get(key)
    if (
        not args.urs_refresh
        and cached is not None
        and len(cached.get("objectives", [])) >= episodes
    ):
        return {
            "objectives": [float(v) for v in cached["objectives"][:episodes]],
            "direction": cached.get("direction", ""),
            "cached": True,
        }
    command = [
        sys.executable,
        str(URS_RUNNER),
        "--problem", name,
        "--data-path", str(data_path),
        "--episodes", str(episodes),
        "--model-load", str(args.urs_checkpoint),
        "--cuda", str(args.urs_cuda),
        "--batch-size", str(args.urs_batch_size),
    ]
    if args.urs_no_aug:
        command.append("--disable-aug")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith("URS_RESULT_JSON "):
            payload = json.loads(line[len("URS_RESULT_JSON "):])
    if payload is None or payload.get("error"):
        return None
    objectives = payload.get("objectives")
    if not objectives or len(objectives) < episodes:
        return None
    result = {
        "objectives": [float(value) for value in objectives[:episodes]],
        "direction": payload.get("direction", ""),
    }
    cache[key] = result
    save_urs_cache(args.urs_cache, cache)
    return {**result, "cached": False}


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


SEEN_VARIANTS = frozenset(BENCHMARK_VARIANTS) & frozenset(TRAIN_VARIANTS)


def variant_split(name: str) -> str:
    """Match training's seen/held-out boundary over the 110 benchmarks."""
    if name not in BENCHMARK_VARIANTS:
        raise ValueError(f"unknown benchmark variant: {name}")
    return "seen" if name in SEEN_VARIANTS else "heldout"


def _instance_data(data: dict, index: int) -> dict:
    """Retain the batch dimension while selecting one loaded instance."""
    return {
        key: value[index : index + 1]
        if torch.is_tensor(value) and value.ndim > 0
        else value
        for key, value in data.items()
    }


def _gap_percent(direction: str, objective: float, reference: float) -> float:
    denominator = max(abs(reference), 1e-9)
    if direction == "maximize":
        return (reference - objective) / denominator * 100.0
    return (objective - reference) / denominator * 100.0


def _split_summary(
    split: str,
    variants: list[str],
    rows: list[dict],
    failures: list[dict],
) -> dict:
    selected = {name for name in variants if variant_split(name) == split}
    split_rows = [row for row in rows if row["variant"] in selected]
    split_failures = [item for item in failures if item["variant"] in selected]
    referenced = [row for row in split_rows if row["reference"] != ""]
    summary = {
        "split": split,
        "variants": len(selected),
        "passed": len(split_rows),
        "failed": len(split_failures),
        "neural_wins": sum(row["winner"] == "neural" for row in split_rows),
        "ties": sum(row["winner"] == "tie" for row in split_rows),
        "baseline_wins": sum(
            row["winner"] == "baseline" for row in split_rows
        ),
        "reference_variants": len(referenced),
    }
    if split_rows:
        improvements = [row["neural_improvement_pct"] for row in split_rows]
        summary["neural_improvement_mean"] = statistics.fmean(improvements)
        summary["neural_improvement_median"] = statistics.median(improvements)
    if referenced:
        summary["baseline_gap_mean"] = statistics.fmean(
            row["baseline_gap_pct"] for row in referenced
        )
        summary["neural_gap_mean"] = statistics.fmean(
            row["neural_gap_pct"] for row in referenced
        )
    return summary


def _print_split_summary(summary: dict) -> None:
    fields = [
        "SPLIT_RESULTS",
        f"split={summary['split'].upper()}",
        f"variants={summary['variants']}",
        f"passed={summary['passed']}",
        f"failed={summary['failed']}",
        f"neural_wins={summary['neural_wins']}",
        f"ties={summary['ties']}",
        f"baseline_wins={summary['baseline_wins']}",
    ]
    if "neural_improvement_mean" in summary:
        fields.extend(
            (
                "neural_improvement_mean="
                f"{summary['neural_improvement_mean']:.3f}%",
                "neural_improvement_median="
                f"{summary['neural_improvement_median']:.3f}%",
            )
        )
    print(*fields)
    if summary["reference_variants"]:
        print(
            "SPLIT_REFERENCE_GAPS",
            f"split={summary['split'].upper()}",
            f"n={summary['reference_variants']}",
            f"baseline_mean={summary['baseline_gap_mean']:.3f}%",
            f"neural_mean={summary['neural_gap_mean']:.3f}%",
        )


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
    if checkpoint.get("model_schema") != MODEL_SCHEMA:
        raise RuntimeError(
            "checkpoint is not a typed-resource v2 checkpoint"
        )
    model = ConstraintFieldNet().to(args.device)
    load_constraint_field_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    finder = DatasetFinder(args.dataset_dir)
    variants = selected_variants(args.variants)
    urs_cache = (
        load_urs_cache(args.urs_cache) if args.urs_checkpoint is not None else {}
    )
    rows: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()

    for index, name in enumerate(variants):
        test_started = time.perf_counter()
        try:
            paths = finder.get(name, 100)
            data, reference = load_saved_data(
                paths["data_path"],
                name,
                args.val_size,
                solution_path=paths["solution_path"],
                allow_aggregate_reference=False,
            )
            has_reference = reference is not None
            urs_result = (
                run_urs_variant(
                    name, paths["data_path"], args.val_size, args, urs_cache
                )
                if args.urs_checkpoint is not None
                else None
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
                problem = solver_problem(
                    name, _instance_data(data, instance_index)
                )
                decoder_args = SimpleNamespace(
                    candidates=args.candidates,
                    n_rollouts=args.rollouts,
                    beta=2.0,
                    seed=args.seed + index * args.val_size + instance_index,
                    search_iterations=args.iterations,
                    feasibility_lookahead_depth=2,
                    feasibility_risk_penalty=10.0,
                    device=args.device,
                    static_field=args.static_field,
                    baseline=args.baseline,
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
            tolerance = 1e-6 * max(abs(baseline_cost), abs(neural_cost), 1.0)
            outcome = (
                "neural"
                if neural_cost < baseline_cost - tolerance
                else "baseline"
                if baseline_cost < neural_cost - tolerance
                else "tie"
            )

            # Paired PRISM-vs-URS comparison over the same instances (dataset
            # order). URS is only compared when it covers the variant and agrees
            # on the objective direction; otherwise the fields stay blank.
            def _better(lhs: float, rhs: float) -> bool:
                tol = 1e-6 * max(abs(lhs), abs(rhs), 1.0)
                return (lhs > rhs + tol) if direction == "maximize" else (
                    lhs < rhs - tol
                )

            urs_objective: float | str = ""
            urs_gap: float | str = ""
            neural_vs_urs = ""
            neural_beats_urs: int | str = ""
            urs_neural_ties: int | str = ""
            urs_beats_neural: int | str = ""
            if urs_result is not None and urs_result["direction"] == direction:
                urs_objs = urs_result["objectives"]
                urs_objective = statistics.fmean(urs_objs)
                if has_reference:
                    urs_gap = _gap_percent(direction, urs_objective, reference)
                nb = nt = ub = 0
                for n_obj, u_obj in zip(neural_objectives, urs_objs):
                    if _better(n_obj, u_obj):
                        nb += 1
                    elif _better(u_obj, n_obj):
                        ub += 1
                    else:
                        nt += 1
                neural_beats_urs, urs_neural_ties, urs_beats_neural = nb, nt, ub
                neural_vs_urs = (
                    "neural"
                    if _better(neural_objective, urs_objective)
                    else "urs"
                    if _better(urs_objective, neural_objective)
                    else "tie"
                )
            rows.append(
                {
                    "variant": name,
                    "split": variant_split(name),
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
                    "urs_objective": urs_objective,
                    "urs_gap_pct": urs_gap,
                    "neural_vs_urs": neural_vs_urs,
                    "neural_beats_urs": neural_beats_urs,
                    "urs_neural_ties": urs_neural_ties,
                    "urs_beats_neural": urs_beats_neural,
                    "neural_improvement_pct": -_gap_percent(
                        direction, neural_objective, baseline_objective
                    ),
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
                f"split={rows[-1]['split'].upper()}",
                f"val_size={args.val_size}",
                f"field_mode={rows[-1]['field_mode']}",
                f"winner={outcome}",
                f"baseline_mean={baseline_objective:.6g}",
                f"neural_mean={neural_objective:.6g}",
                "baseline_gap="
                + (
                    f"{baseline_gap:.3f}%"
                    if baseline_gap != ""
                    else "n/a"
                ),
                "neural_gap="
                + (f"{neural_gap:.3f}%" if neural_gap != "" else "n/a"),
                *(
                    (
                        f"urs_mean={urs_objective:.6g}",
                        "urs_gap="
                        + (f"{urs_gap:.3f}%" if urs_gap != "" else "n/a"),
                        f"neural_vs_urs={neural_vs_urs}",
                        f"neural_beats_urs={neural_beats_urs}/{args.val_size}",
                        f"urs_cached={'yes' if urs_result.get('cached') else 'no'}",
                    )
                    if urs_result is not None and neural_vs_urs != ""
                    else ()
                ),
                f"net_evals_mean={rows[-1]['neural_net_evals_mean']:.3f}",
                f"seconds={time.perf_counter() - test_started:.3f}",
                flush=True,
            )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            failures.append(
                {
                    "variant": name,
                    "split": variant_split(name),
                    "error": detail,
                }
            )
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=FAIL",
                f"split={variant_split(name).upper()}",
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
    urs_rows = [row for row in rows if row.get("neural_vs_urs")]
    if urs_rows:
        urs_referenced = [row for row in urs_rows if row["urs_gap_pct"] != ""]
        instance_neural_wins = sum(row["neural_beats_urs"] for row in urs_rows)
        instance_urs_wins = sum(row["urs_beats_neural"] for row in urs_rows)
        instance_ties = sum(row["urs_neural_ties"] for row in urs_rows)
        print(
            "URS_COMPARE",
            f"checkpoint={args.urs_checkpoint}",
            f"aug={'off' if args.urs_no_aug else 'on'}",
            f"variants_covered={len(urs_rows)}",
            "variant_neural_wins="
            f"{sum(row['neural_vs_urs'] == 'neural' for row in urs_rows)}",
            f"variant_ties={sum(row['neural_vs_urs'] == 'tie' for row in urs_rows)}",
            "variant_urs_wins="
            f"{sum(row['neural_vs_urs'] == 'urs' for row in urs_rows)}",
            "instances="
            f"{instance_neural_wins + instance_urs_wins + instance_ties}",
            f"instance_neural_wins={instance_neural_wins}",
            f"instance_ties={instance_ties}",
            f"instance_urs_wins={instance_urs_wins}",
        )
        if urs_referenced:
            print(
                "URS_REFERENCE_GAPS",
                f"n={len(urs_referenced)}",
                "neural_mean="
                f"{statistics.fmean(row['neural_gap_pct'] for row in urs_referenced):.3f}%",
                "urs_mean="
                f"{statistics.fmean(row['urs_gap_pct'] for row in urs_referenced):.3f}%",
                "neural_median="
                f"{statistics.median(row['neural_gap_pct'] for row in urs_referenced):.3f}%",
                "urs_median="
                f"{statistics.median(row['urs_gap_pct'] for row in urs_referenced):.3f}%",
            )
    for split in ("seen", "heldout"):
        _print_split_summary(_split_summary(split, variants, rows, failures))
    for failure in failures:
        print("FAIL", failure["variant"], failure["error"])

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
