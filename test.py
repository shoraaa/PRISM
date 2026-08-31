#!/usr/bin/env python3
"""Evaluate a trained field decoder on all 110 routing benchmarks.

This file owns PRISM's side of a run and nothing else: the command line, the
checkpoint, and the model rebuilt under the flags it was trained with. The
harness around it lives in ``prism_eval`` -- where instances come from, how a
measurement is recorded, how methods are registered and run, and how the report
is aggregated -- so a new baseline is a module under ``prism_eval.methods``
plus a registry entry, and never an edit here.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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
    DEFAULT_DATASET_DIR,
    DatasetFinder,
)
from train import setup_seeds  # noqa: E402

from prism_eval import methods, runner, summary  # noqa: E402
from prism_eval.instances import InstanceBatch, selected_variants  # noqa: E402
from prism_eval.methods import oracle  # noqa: E402
from prism_eval.results import RowWriter, load_cached_rows, read_rows  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a checkpoint against any registered method on the same "
            "instances, over all 110 saved size-100 benchmark variants by "
            "default, with separate SEEN and HELDOUT summaries."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "comma-separated names, 'all' for all 110 variants (default), or "
            "'ccl' for the 48 symmetric single-/multi-depot VRP variants that "
            "CCL-MTLVRP releases data for"
        ),
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
        "--srr-exploration-budget",
        type=int,
        default=0,
        help=(
            "Bounded uphill SRR exploration budget for PRISM. Native baselines "
            "receive zero unless --random-escape is enabled. Must match the "
            "value used at training time. 0 disables (default)."
        ),
    )
    parser.add_argument(
        "--random-escape",
        action="store_true",
        help=(
            "Allow native constant, distance, and random baselines to spend "
            "--srr-exploration-budget on seeded-random non-improving SRR "
            "moves. Without this flag, every native baseline uses budget 0."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=8,
        help="number of saved instances to average per variant (default: 8)",
    )
    parser.add_argument(
        "--n-node",
        "--n_node",
        dest="n_node",
        type=int,
        default=None,
        help=(
            "generate fresh benchmark instances with this many customers "
            "instead of loading --dataset-dir (disabled by default)"
        ),
    )
    parser.add_argument(
        "--evrp-size",
        type=int,
        default=100,
        help="customer count for generated 'evrp' zero-shot instances (default: 100)",
    )
    parser.add_argument(
        "--vrpdb-size",
        type=int,
        default=100,
        help=(
            "customer count for generated 'vrpdb' driver-break semantic-OOD "
            "instances (default: 100)"
        ),
    )
    parser.add_argument(
        "--evrp-oracle",
        choices=("none", "ortools"),
        default="none",
        help=(
            "Reference solver for generated semantic-OOD probes. 'ortools'"
            " builds the exact capacity/time-window/resource model, validates its"
            " route through the decoder, and reports the decoder-measured cost"
            " as a gap reference (default: none)."
        ),
    )
    parser.add_argument(
        "--evrp-oracle-time",
        type=float,
        default=20.0,
        help=(
            "per-instance OR-Tools time budget in seconds (default: 20.0). At"
            " n=100 OR-Tools surpasses PRISM by ~20s and keeps improving to 60s+;"
            " scale this up with instance size for a tighter reference bound."
        ),
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
        "--feasibility-risk-penalty",
        type=float,
        default=None,
        help=(
            "Weight of the checkpoint's feasibility-risk energy. By default "
            "this is recovered from the checkpoint config; use 0 for the "
            "matched risk-guidance ablation."
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
        "--methods",
        "--baselines",
        "--baseline",
        dest="methods",
        default=None,
        help=(
            "Comma-separated methods to run beside PRISM (PRISM is always "
            "measured): "
            + "; ".join(
                f"'{spec.name}' {spec.help}"
                for spec in methods.REGISTRY.values()
                if spec.name != "prism"
            )
            + ". 'all' selects the native controls, 'none' runs PRISM alone. "
            "The default is constant, or every method present in --cached."
        ),
    )
    parser.add_argument(
        "--cached",
        type=Path,
        help=(
            "Reuse per-instance baseline objectives from a prior test.py CSV. "
            "Rows must match variant, baseline, and --val-size; missing rows "
            "fall back to normal baseline evaluation."
        ),
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run from --csv. Completed method/variant "
            "batches are reused; incomplete batches are rerun and only missing "
            "rows are appended. Use the same checkpoint and evaluator options."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--dataset-scale",
        type=int,
        default=100,
        help=(
            "customer count to select inside --dataset-dir (default: 100). A "
            "variant directory holds every scale generated or converted for "
            "it, so this is what picks between them; the n=100 families carry "
            "per-instance oracle costs, and larger scales get their reference "
            "from whatever oracle the run solves with."
        ),
    )
    # Every registered method contributes its own flags, so adding a baseline
    # never edits this function.
    methods.add_method_arguments(parser)
    args = parser.parse_args(argv)
    if args.min_changed_edges < 1:
        parser.error("--min-changed-edges must be positive")
    if args.n_node is not None and args.n_node < 1:
        parser.error("--n-node must be positive")
    if args.evrp_size < 1:
        parser.error("--evrp-size must be positive")
    if args.vrpdb_size < 1:
        parser.error("--vrpdb-size must be positive")
    methods.validate_method_arguments(args, parser.error)
    if args.methods is None:
        args.methods = "cached" if args.cached is not None else "constant"
    if args.n_node is not None and args.cached is not None:
        parser.error("--n-node cannot reuse results from --cached")
    if args.resume and args.csv is None:
        parser.error("--resume requires --csv")
    if args.resume and not args.csv.is_file():
        parser.error(f"--resume CSV does not exist: {args.csv}")
    if (
        args.resume
        and args.cached is not None
        and args.csv.resolve() == args.cached.resolve()
    ):
        parser.error("--resume and --cached cannot use the same CSV")
    if args.methods != "cached":
        try:
            selected = methods.resolve(args.methods)
        except ValueError as error:
            parser.error(str(error))
        if (
            "urs" in selected
            and args.urs_checkpoint is None
            and args.cached is None
        ):
            parser.error("--methods urs requires --urs-baseline-id")
    return args


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
    # Reconstruct architecture-shaping flags from the training config so an
    # ablated checkpoint evaluates with the same architecture it was trained
    # under. The attention parameters are always present in the state dict, so a
    # missing flag would silently evaluate an ablated model *with* coupling.
    train_config = checkpoint.get("config", {})
    if args.feasibility_risk_penalty is None:
        args.feasibility_risk_penalty = float(
            train_config.get("feasibility_risk_penalty", 1.0)
        )
    if args.feasibility_risk_penalty < 0.0:
        raise ValueError("--feasibility-risk-penalty must be nonnegative")
    model = ConstraintFieldNet(
        couple_resource_tokens=train_config.get(
            "couple_resource_tokens", True
        ),
        # These reshape the objective-residual head, so the state dict will not
        # load unless the architecture is rebuilt exactly as trained.
        linear_objective_residual_head=train_config.get(
            "linear_objective_residual_head", False
        ),
        unconditioned_objective_residual_head=train_config.get(
            "unconditioned_objective_residual_head", False
        ),
        # Forward-time only (coupler params always present), but a missing flag
        # would silently evaluate a static-coupler ablation *with* live coupling.
        couple_state_multipliers=train_config.get(
            "couple_state_multipliers", True
        ),
        # Also forward-time only: the ungated ablation would otherwise be
        # evaluated with the binding gate it was never trained under.
        gate_multipliers_by_binding=train_config.get(
            "gate_multipliers_by_binding", True
        ),
        # Forward-time only (the embedding table is always in the state dict),
        # so a missing flag would silently evaluate an index-embedded model on
        # the semantic descriptors it never saw.
        index_embedded_resources=train_config.get(
            "index_embedded_resources", False
        ),
        # Forward-time only (pooling adds no parameters), so a missing flag
        # would silently evaluate a monolithic model with the per-resource
        # factorization it was never trained under.
        monolithic_resource_field=train_config.get(
            "monolithic_resource_field", False
        ),
        # Forward-time only (the program encoder is always in the state dict),
        # so a missing flag would silently evaluate a program-blind model on the
        # descriptors it was never trained to read.
        program_blind_resources=train_config.get(
            "program_blind_resources", False
        ),
        # Forward-time only (layer_norm adds no parameters), so an old
        # checkpoint loads either way. It must default to False: every
        # checkpoint predating the fix was trained against saturated
        # projections, and evaluating one with them normalized feeds heads an
        # input distribution they never saw.
        normalize_projections=train_config.get("normalize_projections", False),
    ).to(args.device)
    load_constraint_field_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    finder = DatasetFinder(args.dataset_dir)
    variants = selected_variants(args.variants)

    cached_rows: dict[tuple[str, str], dict] = {}
    if args.cached is not None:
        cached_rows, cached_methods = load_cached_rows(args.cached)
    if args.methods == "cached":
        selected = methods.resolve(",".join(cached_methods))
    else:
        selected = methods.resolve(args.methods)
    active = methods.build(selected, args, model)
    resume_rows = read_rows(args.csv) if args.resume else []
    if args.resume:
        resume_keys = [
            (row.variant, row.method, row.instance) for row in resume_rows
        ]
        if len(resume_keys) != len(set(resume_keys)):
            raise ValueError(
                "--resume CSV contains duplicate variant/method/instance rows"
            )
        allowed_variants = set(variants)
        allowed_methods = {method.name for method in active}
        unexpected = sorted(
            {
                (row.variant, row.method)
                for row in resume_rows
                if row.variant not in allowed_variants
                or row.method not in allowed_methods
            }
        )
        if unexpected:
            detail = ", ".join(f"{variant}/{method}" for variant, method in unexpected)
            raise ValueError(
                "--resume CSV contains rows outside this run: " + detail
            )
    print(
        "METHODS",
        f"selected={','.join(selected)}",
        flush=True,
    )

    def reference_hook(name: str, batch: InstanceBatch) -> float | None:
        """Generated probes have no saved reference; solve one on request."""
        if args.evrp_oracle != "ortools":
            return None
        return oracle.oracle_reference(
            batch,
            time_limit=args.evrp_oracle_time,
            candidates=args.candidates,
        )

    started = time.perf_counter()
    with RowWriter(args.csv, append=args.resume) as writer:
        rows = runner.run(
            active,
            variants,
            args,
            finder,
            writer,
            reference_hook=reference_hook,
            cached=cached_rows,
            resume_rows=resume_rows,
        )

    print(
        "RUN",
        f"checkpoint={args.checkpoint}",
        f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}",
        f"variants={len(variants)}",
        f"val_size={args.val_size}",
        f"seconds={time.perf_counter() - started:.3f}",
    )
    summary.print_report(rows)
    if args.csv:
        print(f"CSV {args.csv}")
        if not any(isinstance(row.objective, (int, float)) for row in rows):
            # The file still holds one status row per failed method, which is
            # the diagnostic, so it is kept rather than deleted.
            print(
                "NO_MEASUREMENTS every method failed or was unsupported",
                file=sys.stderr,
                flush=True,
            )
    return int(any(row.status.startswith("failed") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
