#!/usr/bin/env python3
"""Evaluate a trained field decoder on all 110 routing benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import statistics
import subprocess
import sys
import tempfile
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
    generate_evrp_data,
    load_saved_data,
)
from train import (  # noqa: E402
    _canonical_cost,
    _constant_guidance,
    _distance_guidance,
    _new_decoder,
    _random_guidance,
    infer_instance,
    setup_seeds,
)
from urs_one_each import solver_problem  # noqa: E402


NATIVE_BASELINE_NAMES = ("constant", "distance", "random")
BASELINE_NAMES = (*NATIVE_BASELINE_NAMES, "urs")


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
        "--min-changed-edges",
        type=int,
        default=8,
        help=(
            "Minimum number of route edges each perturbation tries to change "
            "before SRR refinement (default: 8)"
        ),
    )
    parser.add_argument(
        "--max-perturb-attempts",
        type=int,
        default=64,
        help=(
            "Maximum candidate-move attempts within each perturbation rollout "
            "(default: 64)"
        ),
    )
    parser.add_argument(
        "--srr-exploration-budget",
        type=int,
        default=0,
        help=(
            "Energy-guided SRR exploration budget (bounded uphill escapes the "
            "guidance can take; inert for the constant-energy baseline). "
            "Must match the value used at training time. 0 disables (default)."
        ),
    )
    parser.add_argument(
        "--unified-srr-policy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the unified feasible-move-plus-STOP SRR policy. By default "
            "this is recovered from the checkpoint training configuration."
        ),
    )
    parser.add_argument(
        "--srr-policy-horizon",
        type=int,
        default=None,
        help=(
            "Maximum accepted moves per unified SRR invocation. By default "
            "this is recovered from the checkpoint, falling back to 64."
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["schema", "geometric"],
        default="geometric",
        help=(
            "Candidate-graph construction shared by PRISM and native controls. "
            "'schema' admits resource candidates by the "
            "schema-derived relevance (uniform equal-share prior when no learned "
            "quota is installed); 'geometric' keeps only the distance "
            "neighborhood."
        ),
    )
    parser.add_argument(
        "--learned-candidate-quotas",
        action="store_true",
        help="EXPERIMENTAL (off by default; may be removed or evolved in the "
        "future). Let the learned quota policy reweight the schema allocation.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=8,
        help="number of saved instances to average per variant (default: 8)",
    )
    parser.add_argument(
        "--evrp-size",
        type=int,
        default=100,
        help="customer count for generated 'evrp' zero-shot instances (default: 100)",
    )
    parser.add_argument(
        "--tsptw-size",
        type=int,
        default=100,
        help="node count for the optional CaR TSPTW evaluation (default: 100)",
    )
    parser.add_argument(
        "--tsptw-hardness",
        choices=("hard", "medium", "easy"),
        default="hard",
        help="CaR TSPTW instance hardness (default: hard)",
    )
    parser.add_argument(
        "--tsptw-source",
        choices=("dataset", "generator"),
        default="dataset",
        help=(
            "Use CaR's saved TSPTW dataset and LKH references (default), or "
            "generate fresh instances with CaR's generator"
        ),
    )
    parser.add_argument(
        "--tsptw-data-dir",
        type=Path,
        default=ROOT / "baselines" / "CaR-constraint" / "data" / "TSPTW",
        help="directory containing CaR tsptw*_*.pkl datasets",
    )
    parser.add_argument(
        "--tsptw-dataset-seed",
        type=int,
        default=2025,
        help=(
            "seed used to create a saved hard CaR TSPTW dataset; needed to "
            "recover its generator-guaranteed feasible starting tours "
            "(default: CaR's 2025)"
        ),
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--baselines",
        "--baseline",
        dest="baselines",
        choices=(*BASELINE_NAMES, "none", "all"),
        default=None,
        help=(
            "Controls to compare with PRISM: 'constant' assigns identical "
            "energy to every candidate; 'distance' uses normalized edge "
            "distance as energy; 'random' uses stable edge-keyed random energy "
            "with the same SRR exploration budget; 'urs' runs the supplied URS "
            "baseline; 'none' evaluates PRISM without running a baseline; 'all' "
            "evaluates the three native controls and excludes URS. The default "
            "is constant, or every baseline present in --cached when supplied."
        ),
    )
    parser.add_argument(
        "--shared-greedy-bootstrap",
        action="store_true",
        help=(
            "Construct one deterministic greedy route with the selected live "
            "native baseline's guidance and install that exact route as the "
            "bootstrap incumbent for both PRISM and the baseline. Requires "
            "exactly one of: constant, distance, random."
        ),
    )
    parser.add_argument(
        "--urs-baseline-id",
        "--urs-checkpoint",
        dest="urs_checkpoint",
        type=Path,
        default=None,
        help=(
            "URS checkpoint identifier required by '--baselines urs'. The "
            "checkpoint runs on the same instances in an isolated subprocess. "
            "Point at e.g."
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
    parser.add_argument(
        "--cached",
        type=Path,
        help=(
            "Reuse per-variant baseline objectives from a prior test.py CSV. "
            "Rows must match variant, baseline, and --val-size; missing rows "
            "fall back to normal baseline evaluation. New CSVs also retain "
            "baseline_construction_objective for construction/final logging."
        ),
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args(argv)
    if args.baselines is None:
        args.baselines = "cached" if args.cached is not None else "constant"
    if (
        args.baselines == "urs"
        and args.urs_checkpoint is None
        and args.cached is None
    ):
        parser.error("--baselines urs requires --urs-baseline-id")
    if args.min_changed_edges < 1:
        parser.error("--min-changed-edges must be positive")
    if args.max_perturb_attempts < 1:
        parser.error("--max-perturb-attempts must be positive")
    return args


def selected_baselines(value: str) -> tuple[str, ...]:
    if value == "all":
        return NATIVE_BASELINE_NAMES
    if value == "none":
        return ("none",)
    if value not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline: {value}")
    return (value,)


def _shared_greedy_baseline(
    enabled: bool,
    baselines: tuple[str, ...],
    cached: Path | None,
) -> str | None:
    if not enabled:
        return None
    if cached is not None:
        raise ValueError(
            "--shared-greedy-bootstrap cannot verify cached baseline starts"
        )
    if len(baselines) != 1 or baselines[0] not in NATIVE_BASELINE_NAMES:
        raise ValueError(
            "--shared-greedy-bootstrap requires exactly one live native "
            "baseline: constant, distance, or random"
        )
    return baselines[0]


def _greedy_baseline_route(
    problem: dict,
    decoder_args: argparse.Namespace,
    baseline: str,
) -> list[int]:
    """Build the deterministic baseline route shared by both evaluator arms."""
    decoder = _new_decoder(problem, decoder_args, deterministic=True)
    if baseline == "constant":
        guidance = _constant_guidance(decoder)
    elif baseline == "distance":
        guidance = _distance_guidance(decoder, problem)
    elif baseline == "random":
        guidance = _random_guidance(decoder, decoder_args.seed)
    else:
        raise ValueError(f"unknown native baseline: {baseline}")
    incumbent = decoder.sample_greedy(**guidance)
    if not incumbent["feasible"]:
        raise RuntimeError(
            f"{baseline} greedy bootstrap is infeasible: "
            + incumbent.get("error", "unknown route error")
        )
    return [int(node) for node in incumbent["route"]]


def load_cached_baseline_rows(
    path: Path,
) -> tuple[dict[tuple[str, str, int], dict], tuple[str, ...]]:
    """Load reusable per-variant baseline averages from a prior evaluator CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"cached baseline CSV does not exist: {path}")
    required = {
        "variant",
        "baseline",
        "val_size",
        "direction",
        "baseline_objective",
    }
    cached: dict[tuple[str, str, int], dict] = {}
    baseline_order: list[str] = []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"cached baseline CSV is missing columns: {', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            variant = row["variant"].strip()
            baseline = row["baseline"].strip()
            objective_text = row["baseline_objective"].strip()
            construction_text = (
                row.get("baseline_construction_objective", "") or ""
            ).strip()
            # Neural-only rows intentionally contain no reusable comparator.
            if baseline == "none" or not objective_text:
                continue
            if not variant or baseline not in BASELINE_NAMES:
                raise ValueError(
                    f"invalid cached variant/baseline on line {line_number}"
                )
            try:
                val_size = int(row["val_size"])
                objective = float(objective_text)
                construction_objective: float | str = (
                    float(construction_text) if construction_text else ""
                )
            except ValueError as error:
                raise ValueError(
                    f"invalid cached numeric value on line {line_number}"
                ) from error
            direction = row["direction"].strip()
            if val_size < 1 or not math.isfinite(objective):
                raise ValueError(
                    f"invalid cached baseline value on line {line_number}"
                )
            if construction_objective != "" and not math.isfinite(
                construction_objective
            ):
                raise ValueError(
                    f"invalid cached baseline value on line {line_number}"
                )
            if direction not in {"minimize", "maximize"}:
                raise ValueError(
                    f"invalid cached direction on line {line_number}"
                )
            key = (variant, baseline, val_size)
            if key in cached:
                raise ValueError(
                    "duplicate cached baseline row for "
                    f"variant={variant}, baseline={baseline}, val_size={val_size}"
                )
            cached[key] = {
                "baseline_objective": objective,
                "baseline_construction_objective": construction_objective,
                "direction": direction,
            }
            if baseline not in baseline_order:
                baseline_order.append(baseline)
    return cached, tuple(baseline_order)


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


# Variants with no saved dataset: instances are generated on the fly. "evrp" is
# the zero-shot unseen-resource probe (battery via the resource algebra) and is
# never in the 110 benchmarks, so it must be requested explicitly.
GENERATED_VARIANTS = ("evrp",)
OPTIONAL_VARIANTS = GENERATED_VARIANTS + ("tsptw",)
CAR_ROOT = ROOT / "baselines" / "CaR-constraint"


def selected_variants(value: str) -> list[str]:
    available = list(BENCHMARK_VARIANTS)
    if value == "all":
        return available
    requested = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(available) - set(OPTIONAL_VARIANTS))
    if unknown:
        raise ValueError("unknown evaluator variants: " + ", ".join(unknown))
    if not requested:
        raise ValueError("--variants selected no variants")
    return requested


SEEN_VARIANTS = frozenset(BENCHMARK_VARIANTS) & frozenset(TRAIN_VARIANTS)


def variant_split(name: str) -> str:
    """Match training's seen/held-out boundary over the 110 benchmarks."""
    if name in OPTIONAL_VARIANTS:
        return "heldout"
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


def _load_car_tsptw_rows(path: Path, count: int) -> tuple[dict, int]:
    """Load CaR's native ``(xy, service, tw_start, tw_end)`` pickle format."""
    if not path.is_file():
        raise FileNotFoundError(f"CaR TSPTW dataset does not exist: {path}")
    with path.open("rb") as source:
        saved = pickle.load(source)
    rows = saved[:count]
    if len(rows) != count:
        raise ValueError(
            f"requested {count} TSPTW instances, but {path} contains "
            f"only {len(rows)}"
        )
    if any(not isinstance(row, (tuple, list)) or len(row) != 4 for row in rows):
        raise ValueError(f"unexpected CaR TSPTW row format in {path}")
    return (
        {
            "xy": torch.tensor([row[0] for row in rows], dtype=torch.float32),
            "service_time": torch.tensor(
                [row[1] for row in rows], dtype=torch.float32
            ),
            "tw_start": torch.tensor(
                [row[2] for row in rows], dtype=torch.float32
            ),
            "tw_end": torch.tensor(
                [row[3] for row in rows], dtype=torch.float32
            ),
        },
        len(saved),
    )


def _car_hard_feasible_routes(
    xy: torch.Tensor,
    *,
    seed: int,
    generated_count: int,
) -> torch.Tensor:
    """Recover the feasible permutations embedded by CaR's hard generator.

    CaR samples all coordinates, then one permutation per instance and builds
    each time-window sequence around that permutation. Replaying those two RNG
    operations recovers the generator witness without using the LKH solutions.
    Coordinate equality guards against applying a wrong seed or data provenance.
    """
    count, size, coordinate_dim = xy.shape
    if coordinate_dim != 2 or generated_count < count:
        raise ValueError("unexpected CaR hard TSPTW coordinate shape")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected_xy = torch.rand(generated_count, size, 2) * 100.0
        if not torch.equal(expected_xy[:count], xy):
            raise ValueError(
                "CaR hard TSPTW coordinates do not match --tsptw-dataset-seed; "
                "pass the seed used to generate this dataset"
            )
        routes = []
        customers = torch.arange(1, size, dtype=torch.long)
        for index in range(generated_count):
            permutation = customers[torch.randperm(size - 1)]
            if index < count:
                routes.append(
                    torch.cat((torch.zeros(1, dtype=torch.long), permutation))
                )
    return torch.stack(routes)


def _load_car_tsptw_reference(path: Path, count: int) -> float | None:
    """Average the paired CaR LKH costs when the reference file is available."""
    if not path.is_file():
        return None
    with path.open("rb") as source:
        rows = pickle.load(source)[:count]
    if len(rows) != count:
        raise ValueError(
            f"requested {count} TSPTW references, but {path} contains "
            f"only {len(rows)}"
        )
    costs = [row[0] if isinstance(row, (tuple, list)) else row for row in rows]
    values = torch.tensor(costs, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError(f"non-finite TSPTW reference in {path}")
    return float(values.mean())


def load_car_tsptw_data(
    data_dir: Path,
    size: int,
    hardness: str,
    count: int,
    dataset_seed: int = 2025,
) -> tuple[dict, float | None]:
    """Load paired instances/references distributed by CaR-constraint."""
    stem = f"tsptw{size}_{hardness}.pkl"
    data, saved_count = _load_car_tsptw_rows(data_dir / stem, count)
    if hardness == "hard":
        data["initial_route"] = _car_hard_feasible_routes(
            data["xy"], seed=dataset_seed, generated_count=saved_count
        )
    reference = _load_car_tsptw_reference(data_dir / f"lkh_{stem}", count)
    return data, reference


def generate_car_tsptw_data(
    size: int,
    hardness: str,
    count: int,
    seed: int,
) -> dict:
    """Generate fresh data through CaR's own TSPTW generator entry point."""
    script = CAR_ROOT / "generate_data.py"
    if not script.is_file():
        raise FileNotFoundError(f"CaR TSPTW generator does not exist: {script}")
    with tempfile.TemporaryDirectory(prefix="prism-car-tsptw-") as temporary:
        output_root = Path(temporary)
        command = [
            sys.executable,
            str(script),
            "--problem",
            "TSPTW",
            "--problem_size",
            str(size),
            "--pomo_size",
            str(size),
            "--hardness",
            hardness,
            "--num_samples",
            str(count),
            "--seed",
            str(seed),
            "--dir",
            str(output_root),
            "--no_cuda",
        ]
        result = subprocess.run(
            command,
            cwd=CAR_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"CaR TSPTW generator failed: {detail}")
        path = output_root / "TSPTW" / f"tsptw{size}_{hardness}.pkl"
        data, _ = _load_car_tsptw_rows(path, count)
        if hardness == "hard":
            data["initial_route"] = _car_hard_feasible_routes(
                data["xy"], seed=seed, generated_count=count
            )
        return data


def _gap_percent(direction: str, objective: float, reference: float) -> float:
    denominator = max(abs(reference), 1e-9)
    if direction == "maximize":
        return (reference - objective) / denominator * 100.0
    return (objective - reference) / denominator * 100.0


def _format_construction_final(
    construction: float | str,
    final: float | str,
    *,
    format_spec: str,
    suffix: str = "",
) -> str:
    """Format evaluator metrics in construction/final order."""
    values = []
    for value in (construction, final):
        values.append(
            "n/a"
            if value == ""
            else f"{format(float(value), format_spec)}{suffix}"
        )
    return "/".join(values)


def _split_summary(
    split: str,
    variants: list[str],
    rows: list[dict],
    failures: list[dict],
    baseline: str = "constant",
) -> dict:
    selected = {name for name in variants if variant_split(name) == split}
    split_rows = [
        row
        for row in rows
        if row["variant"] in selected
        and row.get("baseline", "constant") == baseline
    ]
    split_failures = [
        item
        for item in failures
        if item["variant"] in selected
        and item.get("baseline", "constant") == baseline
    ]
    referenced = [row for row in split_rows if row["reference"] != ""]
    summary = {
        "split": split,
        "baseline": baseline,
        "variants": len(selected),
        "passed": len(split_rows),
        "failed": len(split_failures),
        "reference_variants": len(referenced),
    }
    if baseline != "none":
        summary.update(
            neural_wins=sum(
                row["winner"] == "neural" for row in split_rows
            ),
            ties=sum(row["winner"] == "tie" for row in split_rows),
            baseline_wins=sum(
                row["winner"] == "baseline" for row in split_rows
            ),
        )
    if split_rows and baseline != "none":
        improvements = [row["neural_improvement_pct"] for row in split_rows]
        summary["neural_improvement_mean"] = statistics.fmean(improvements)
        summary["neural_improvement_median"] = statistics.median(improvements)
    if referenced:
        if baseline != "none":
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
        f"baseline={summary['baseline']}",
        f"variants={summary['variants']}",
        f"passed={summary['passed']}",
        f"failed={summary['failed']}",
    ]
    if summary["baseline"] != "none":
        fields.extend(
            (
                f"neural_wins={summary['neural_wins']}",
                f"ties={summary['ties']}",
                f"baseline_wins={summary['baseline_wins']}",
            )
        )
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
        reference_fields = [
            "SPLIT_REFERENCE_GAPS",
            f"split={summary['split'].upper()}",
            f"baseline={summary['baseline']}",
            f"n={summary['reference_variants']}",
        ]
        if summary["baseline"] != "none":
            reference_fields.append(
                f"baseline_mean={summary['baseline_gap_mean']:.3f}%"
            )
        reference_fields.append(
            f"neural_mean={summary['neural_gap_mean']:.3f}%"
        )
        print(*reference_fields)


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
            f"checkpoint schema does not match {MODEL_SCHEMA}"
        )
    model = ConstraintFieldNet().to(args.device)
    load_constraint_field_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    checkpoint_config = checkpoint.get("config", {})
    if args.unified_srr_policy is None:
        args.srr_policy_enabled = bool(
            checkpoint_config.get("srr_policy_enabled", False)
            or float(checkpoint_config.get("srr_loss_weight", 0.0)) > 0.0
        )
    else:
        args.srr_policy_enabled = args.unified_srr_policy
    if args.srr_policy_horizon is None:
        args.srr_policy_horizon = int(
            checkpoint_config.get("srr_policy_horizon", 64)
        )
    if args.srr_policy_horizon <= 0:
        raise ValueError("--srr-policy-horizon must be positive")

    finder = DatasetFinder(args.dataset_dir)
    variants = selected_variants(args.variants)
    cached_rows: dict[tuple[str, str, int], dict] = {}
    cached_baseline_names: tuple[str, ...] = ()
    if args.cached is not None:
        cached_rows, cached_baseline_names = load_cached_baseline_rows(
            args.cached
        )
    if args.baselines == "cached":
        baselines = tuple(
            baseline
            for baseline in cached_baseline_names
            if any(
                key_baseline == baseline and val_size == args.val_size
                for _, key_baseline, val_size in cached_rows
            )
        )
        if not baselines:
            raise ValueError(
                f"{args.cached} has no baseline rows for --val-size {args.val_size}"
            )
    else:
        baselines = selected_baselines(args.baselines)
    native_baselines = tuple(
        baseline for baseline in baselines
        if baseline in NATIVE_BASELINE_NAMES
    )
    shared_greedy_baseline = _shared_greedy_baseline(
        args.shared_greedy_bootstrap, baselines, args.cached
    )
    include_urs = "urs" in baselines
    urs_cache = (
        load_urs_cache(args.urs_cache) if include_urs else {}
    )
    rows: list[dict] = []
    failures: list[dict] = []
    cached_baseline_hits = 0
    started = time.perf_counter()

    for index, name in enumerate(variants):
        test_started = time.perf_counter()
        cached_for_variant = {
            baseline: cached_rows[(name, baseline, args.val_size)]
            for baseline in baselines
            if (name, baseline, args.val_size) in cached_rows
        }
        try:
            if name in GENERATED_VARIANTS:
                # Zero-shot unseen-resource probe: no saved dataset and no
                # reference oracle yet, so score PRISM's learned field only
                # against the selected native controls on the same generated
                # instances (feasibility is guaranteed by the exact decoder).
                data = generate_evrp_data(
                    args.evrp_size, args.val_size, seed=args.seed
                )
                reference = None
                has_reference = False
                urs_result = None
            elif name == "tsptw":
                # External single-tour time-window probe. It is intentionally
                # absent from the 110-variant registry and training curriculum.
                if args.tsptw_source == "dataset":
                    data, reference = load_car_tsptw_data(
                        args.tsptw_data_dir,
                        args.tsptw_size,
                        args.tsptw_hardness,
                        args.val_size,
                        args.tsptw_dataset_seed,
                    )
                else:
                    data = generate_car_tsptw_data(
                        args.tsptw_size,
                        args.tsptw_hardness,
                        args.val_size,
                        args.seed,
                    )
                    reference = None
                has_reference = reference is not None
                # URS does not expose this external dataset through its runner.
                urs_result = None
                print(
                    "TSPTW_DATA",
                    f"source={args.tsptw_source}",
                    f"size={args.tsptw_size}",
                    f"hardness={args.tsptw_hardness}",
                    "bootstrap="
                    + (
                        f"shared_greedy_{shared_greedy_baseline}"
                        if shared_greedy_baseline is not None
                        else (
                            "car_generator_witness"
                            if "initial_route" in data
                            else "field_construction"
                        )
                    ),
                    "reference=" + ("lkh" if has_reference else "none"),
                    flush=True,
                )
            else:
                paths = finder.get(name, 100)
                data, reference = load_saved_data(
                    paths["data_path"],
                    name,
                    args.val_size,
                    solution_path=paths["solution_path"],
                    allow_aggregate_reference=False,
                )
                has_reference = reference is not None
                if (
                    include_urs
                    and "urs" not in cached_for_variant
                    and args.urs_checkpoint is None
                ):
                    raise RuntimeError(
                        f"URS cache miss for {name}; supply --urs-baseline-id"
                    )
                urs_result = (
                    run_urs_variant(
                        name, paths["data_path"], args.val_size, args, urs_cache
                    )
                    if include_urs and "urs" not in cached_for_variant
                    else None
                )
            neural_objectives = []
            neural_construction_objectives = []
            neural_costs = []
            baseline_objectives = {baseline: [] for baseline in baselines}
            baseline_construction_objectives = {
                baseline: [] for baseline in baselines
            }
            baseline_costs = {baseline: [] for baseline in baselines}
            baseline_seconds = {baseline: 0.0 for baseline in baselines}
            neural_net_evals = []
            neural_model_seconds = 0.0
            neural_decoder_seconds = 0.0
            neural_seconds = 0.0
            direction = ""

            for instance_index in range(args.val_size):
                problem = solver_problem(
                    name, _instance_data(data, instance_index)
                )
                initial_route = data.get("initial_route")
                if torch.is_tensor(initial_route):
                    initial_route = (
                        initial_route[instance_index].detach().cpu().numpy()
                    )
                decoder_args = SimpleNamespace(
                    candidates=args.candidates,
                    n_rollouts=args.rollouts,
                    beta=2.0,
                    min_changed_edges=args.min_changed_edges,
                    max_perturb_attempts=args.max_perturb_attempts,
                    seed=args.seed + index * args.val_size + instance_index,
                    search_iterations=args.iterations,
                    feasibility_lookahead_depth=2,
                    feasibility_risk_penalty=1.0,
                    device=args.device,
                    static_field=args.static_field,
                    candidate_mode=args.candidate_mode,
                    srr_exploration_budget=args.srr_exploration_budget,
                    srr_policy_enabled=args.srr_policy_enabled,
                    srr_policy_horizon=args.srr_policy_horizon,
                )
                if shared_greedy_baseline is not None:
                    initial_route = _greedy_baseline_route(
                        problem, decoder_args, shared_greedy_baseline
                    )
                    bootstrap_mode = (
                        f"shared_greedy_{shared_greedy_baseline}"
                    )
                elif initial_route is not None:
                    bootstrap_mode = "external"
                else:
                    bootstrap_mode = "method_specific"

                neural_started = time.perf_counter()
                _, neural, neural_metrics = infer_instance(
                    model, problem, decoder_args, initial_route=initial_route
                )
                neural_seconds += time.perf_counter() - neural_started
                if not neural["feasible"]:
                    raise RuntimeError(
                        f"instance {instance_index} returned an infeasible neural solution"
                    )
                direction = neural["direction"]
                neural_objectives.append(float(neural["objective"]))
                neural_construction_objectives.append(
                    float(neural_metrics["construction_objective"])
                )
                neural_costs.append(_canonical_cost(neural))
                neural_net_evals.append(neural_metrics["net_evals"])
                neural_model_seconds += neural_metrics["time_neural"]
                neural_decoder_seconds += neural_metrics["time_decoder"]

                for baseline_name in native_baselines:
                    if baseline_name in cached_for_variant:
                        continue
                    baseline_started = time.perf_counter()
                    _, baseline_solution, baseline_metrics = infer_instance(
                        None,
                        problem,
                        decoder_args,
                        initial_route=initial_route,
                        baseline=baseline_name,
                    )
                    baseline_seconds[baseline_name] += (
                        time.perf_counter() - baseline_started
                    )
                    if not baseline_solution["feasible"]:
                        raise RuntimeError(
                            f"instance {instance_index} returned an infeasible "
                            f"{baseline_name} baseline solution"
                        )
                    if baseline_solution["direction"] != direction:
                        raise RuntimeError(
                            f"{baseline_name} baseline returned a different direction"
                        )
                    baseline_objectives[baseline_name].append(
                        float(baseline_solution["objective"])
                    )
                    baseline_construction_objectives[baseline_name].append(
                        float(baseline_metrics["construction_objective"])
                    )
                    baseline_costs[baseline_name].append(
                        _canonical_cost(baseline_solution)
                    )

            for baseline_name, cached_row in cached_for_variant.items():
                if cached_row["direction"] != direction:
                    raise RuntimeError(
                        f"cached {baseline_name} baseline returned a different "
                        f"direction for {name}"
                    )
                cached_objective = float(cached_row["baseline_objective"])
                baseline_objectives[baseline_name] = [cached_objective]
                cached_construction_objective = cached_row.get(
                    "baseline_construction_objective", ""
                )
                if cached_construction_objective != "":
                    baseline_construction_objectives[baseline_name] = [
                        float(cached_construction_objective)
                    ]
                baseline_costs[baseline_name] = [
                    -cached_objective
                    if direction == "maximize"
                    else cached_objective
                ]
                cached_baseline_hits += 1

            if include_urs:
                if baseline_objectives["urs"]:
                    pass
                elif urs_result is None:
                    raise RuntimeError(
                        f"URS does not provide a cached result for variant {name}; "
                        "supply --urs-baseline-id to evaluate the missing row"
                    )
                elif urs_result["direction"] != direction:
                    raise RuntimeError("URS returned a different direction")
                else:
                    urs_values = [
                        float(value) for value in urs_result["objectives"]
                    ]
                    baseline_objectives["urs"] = urs_values
                    baseline_costs["urs"] = [
                        -value if direction == "maximize" else value
                        for value in urs_values
                    ]

            neural_objective = statistics.fmean(neural_objectives)
            neural_construction_objective = statistics.fmean(
                neural_construction_objectives
            )
            neural_cost = statistics.fmean(neural_costs)

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
            neural_gap: float | str = (
                _gap_percent(direction, neural_objective, reference)
                if has_reference
                else ""
            )
            neural_construction_gap: float | str = (
                _gap_percent(
                    direction, neural_construction_objective, reference
                )
                if has_reference
                else ""
            )
            result_reports = [
                "prism(objective="
                + _format_construction_final(
                    neural_construction_objective,
                    neural_objective,
                    format_spec=".6g",
                )
                + ",gap="
                + _format_construction_final(
                    neural_construction_gap,
                    neural_gap,
                    format_spec=".3f",
                    suffix="%",
                )
                + ")"
            ]
            for baseline_name in baselines:
                baseline_objective: float | str = ""
                baseline_construction_objective: float | str = ""
                baseline_gap: float | str = ""
                baseline_construction_gap: float | str = ""
                outcome = ""
                neural_improvement: float | str = ""
                if baseline_name != "none":
                    baseline_objective = statistics.fmean(
                        baseline_objectives[baseline_name]
                    )
                    if baseline_construction_objectives[baseline_name]:
                        baseline_construction_objective = statistics.fmean(
                            baseline_construction_objectives[baseline_name]
                        )
                    baseline_cost = statistics.fmean(
                        baseline_costs[baseline_name]
                    )
                    tolerance = 1e-6 * max(
                        abs(baseline_cost), abs(neural_cost), 1.0
                    )
                    outcome = (
                        "neural"
                        if neural_cost < baseline_cost - tolerance
                        else "baseline"
                        if baseline_cost < neural_cost - tolerance
                        else "tie"
                    )
                    baseline_gap = (
                        _gap_percent(
                            direction, baseline_objective, reference
                        )
                        if has_reference
                        else ""
                    )
                    baseline_construction_gap = (
                        _gap_percent(
                            direction,
                            baseline_construction_objective,
                            reference,
                        )
                        if has_reference
                        and baseline_construction_objective != ""
                        else ""
                    )
                    neural_improvement = -_gap_percent(
                        direction, neural_objective, baseline_objective
                    )
                row = {
                    "variant": name,
                    "split": variant_split(name),
                    "baseline": baseline_name,
                    "baseline_source": (
                        "none"
                        if baseline_name == "none"
                        else "cached"
                        if baseline_name in cached_for_variant
                        else "evaluated"
                    ),
                    "val_size": args.val_size,
                    "bootstrap_mode": bootstrap_mode,
                    "field_mode": "static" if args.static_field else "dynamic",
                    "direction": direction,
                    "reference": reference if has_reference else "",
                    "baseline_construction_objective": (
                        baseline_construction_objective
                    ),
                    "baseline_objective": baseline_objective,
                    "neural_construction_objective": (
                        neural_construction_objective
                    ),
                    "neural_objective": neural_objective,
                    "baseline_construction_gap_pct": (
                        baseline_construction_gap
                    ),
                    "baseline_gap_pct": baseline_gap,
                    "neural_construction_gap_pct": (
                        neural_construction_gap
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
                    "neural_improvement_pct": neural_improvement,
                    "baseline_seconds": baseline_seconds[baseline_name],
                    "neural_seconds": neural_seconds,
                    "neural_net_evals_mean": statistics.fmean(
                        neural_net_evals
                    ),
                    "neural_net_evals_total": int(sum(neural_net_evals)),
                    "neural_model_seconds": neural_model_seconds,
                    "neural_decoder_seconds": neural_decoder_seconds,
                }
                rows.append(row)
                if baseline_name != "none":
                    source = (
                        "cached"
                        if baseline_name in cached_for_variant
                        else "evaluated"
                    )
                    result_reports.append(
                        f"{baseline_name}[{source}]"
                        "(objective="
                        + _format_construction_final(
                            baseline_construction_objective,
                            baseline_objective,
                            format_spec=".6g",
                        )
                        + ",gap="
                        + _format_construction_final(
                            baseline_construction_gap,
                            baseline_gap,
                            format_spec=".3f",
                            suffix="%",
                        )
                        + ")"
                    )
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                f"bootstrap={bootstrap_mode}",
                f"results=[{'; '.join(result_reports)}]",
                flush=True,
            )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            for baseline_name in baselines:
                failures.append(
                    {
                        "variant": name,
                        "split": variant_split(name),
                        "baseline": baseline_name,
                        "error": detail,
                    }
                )
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=FAIL",
                f"split={variant_split(name).upper()}",
                f"baselines=[{'; '.join(baselines)}]",
                f"error={detail}",
                f"seconds={time.perf_counter() - test_started:.3f}",
                flush=True,
            )

    if args.cached is not None:
        print(
            "CACHED_BASELINES",
            f"path={args.cached}",
            f"rows={len(cached_rows)}",
            f"hits={cached_baseline_hits}",
        )

    for baseline_name in baselines:
        baseline_rows = [
            row for row in rows if row["baseline"] == baseline_name
        ]
        baseline_failures = [
            failure
            for failure in failures
            if failure["baseline"] == baseline_name
        ]
        referenced = [
            row for row in baseline_rows if row["reference"] != ""
        ]
        comparison_fields = [
            "DECODER_COMPARE",
            f"checkpoint={args.checkpoint}",
            f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}",
            f"baseline={baseline_name}",
            f"val_size={args.val_size}",
            "bootstrap="
            + (
                f"shared_greedy_{shared_greedy_baseline}"
                if shared_greedy_baseline is not None
                else "method_specific_or_external"
            ),
            f"field_mode={'static' if args.static_field else 'dynamic'}",
            f"variants={len(variants)}",
            f"passed={len(baseline_rows)}",
            f"failed={len(baseline_failures)}",
            f"seconds={time.perf_counter() - started:.3f}",
        ]
        if baseline_name != "none":
            comparison_fields.extend(
                (
                    "neural_wins="
                    f"{sum(row['winner'] == 'neural' for row in baseline_rows)}",
                    f"ties={sum(row['winner'] == 'tie' for row in baseline_rows)}",
                    "baseline_wins="
                    f"{sum(row['winner'] == 'baseline' for row in baseline_rows)}",
                )
            )
        print(*comparison_fields)
        if referenced:
            reference_fields = [
                "REFERENCE_GAPS",
                f"baseline={baseline_name}",
                f"n={len(referenced)}",
            ]
            if baseline_name != "none":
                reference_fields.extend(
                    (
                        "baseline_mean="
                        f"{statistics.fmean(row['baseline_gap_pct'] for row in referenced):.3f}%",
                        "baseline_median="
                        f"{statistics.median(row['baseline_gap_pct'] for row in referenced):.3f}%",
                    )
                )
            reference_fields.extend(
                (
                    "neural_mean="
                    f"{statistics.fmean(row['neural_gap_pct'] for row in referenced):.3f}%",
                    "neural_median="
                    f"{statistics.median(row['neural_gap_pct'] for row in referenced):.3f}%",
                )
            )
            print(*reference_fields)

    # Neural-vs-URS is independent of the native control. Keep one row per
    # variant when all controls are requested.
    urs_rows = [
        row
        for row in rows
        if row["baseline"] == baselines[0] and row.get("neural_vs_urs")
    ]
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
    for baseline_name in baselines:
        for split in ("seen", "heldout"):
            _print_split_summary(
                _split_summary(
                    split, variants, rows, failures, baseline=baseline_name
                )
            )
    for failure in failures:
        print(
            "FAIL",
            failure["variant"],
            f"baseline={failure['baseline']}",
            failure["error"],
        )

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
