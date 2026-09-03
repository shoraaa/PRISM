#!/usr/bin/env python3
"""Static probe of the learned field: dump per-(state, candidate-edge) rows.

This is the harness the diagnostic experiments in ``experiments.md`` read from.
It answers "what does the model know?" rather than "what gap does it reach", so
it runs **no search**: it installs an incumbent, evaluates the network once per
instance, then replays the incumbent as a sequence of prefixes and records, at
every decision state, the full energy decomposition over that state's candidate
edges together with the ground-truth one-step feasibility the hard mask would
have applied.

One row per (variant, instance, state, candidate edge). Downstream analyses --
E4a feasibility AUROC, E4c incremental information, E4f multiplier audit, E7b
top-k heatmap metrics -- are then pandas queries over the emitted CSV rather
than separate evaluation runs.

The energy reproduces ``RoutingDecoder::edge_energy`` exactly:

    E(e | s) = w_obj(s) * (objective_cost(e)/objective_energy_scale + obj_residual(e))
             + sum_{r active} lambda_r(s) * (field_r(e) + additive_r(e))
             + risk_penalty * risk(e)

with lambda_r(s) = multiplier_r * 2*sigmoid(bias_r + weights_r . live_state(s)),
the same state modulation ``ConstraintFieldNet.couple`` applies.

Because the checkpoint under study predates the current working tree, the model
code is imported from ``--code-root`` rather than from this repository. Point
``--code-root`` at a checkout matching the checkpoint's ``model_schema``; the
compiled ``prism_decoder`` extension must be built inside that checkout, since
the descriptor width and feature layout are compiled in.

Example:
    python scripts/probe_field.py \
        --checkpoint ../epoch235_015.pt \
        --code-root ../PRISM-v6 \
        --dataset-dir datasets/benchmarks \
        --variants cvrp,cvrptw,cvrpbltw \
        --instances 4 --states 12 \
        --out results/probe/v6.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# --------------------------------------------------------------------------- #
# Code-root import
# --------------------------------------------------------------------------- #
def load_code_root(code_root: Path) -> SimpleNamespace:
    """Import net/problem_data/test/prism_decoder from a specific checkout.

    Front-inserted so an editable install of this repository cannot shadow the
    checkout under study -- the two ship different compiled extensions with
    different descriptor widths, and importing the wrong one fails silently in
    the sense that the state dict still loads.
    """
    code_root = code_root.resolve()
    if not code_root.is_dir():
        raise SystemExit(f"--code-root does not exist: {code_root}")
    for entry in (code_root / "src", code_root):
        path = str(entry)
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    import prism_decoder  # noqa: E402

    extension = Path(prism_decoder.__file__).resolve()
    if code_root not in extension.parents:
        raise SystemExit(
            f"prism_decoder resolved to {extension}, outside --code-root "
            f"{code_root}. Build the extension in that checkout first:\n"
            f"  cd {code_root} && python setup.py build_ext --inplace"
        )

    import torch  # noqa: E402
    import net as net_module  # noqa: E402
    import problem_data  # noqa: E402

    # Instance construction lived in test.py through v6 and moved into the
    # prism_eval harness afterwards. Accept either so one probe serves both.
    solver_problem = instance_data = test_module = None
    try:
        import test as test_module  # noqa: E402

        solver_problem = getattr(test_module, "solver_problem", None)
        instance_data = getattr(test_module, "_instance_data", None)
    except Exception:  # noqa: BLE001 - the harness may not import standalone
        pass
    if solver_problem is None:
        from prism_eval.instances import (  # noqa: E402
            _instance_data as instance_data,
            solver_problem,
        )

    return SimpleNamespace(
        torch=torch,
        prism_decoder=prism_decoder,
        net=net_module,
        problem_data=problem_data,
        test=test_module,
        solver_problem=solver_problem,
        instance_data=instance_data,
        root=code_root,
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(code, checkpoint_path: Path, device: str):
    """Rebuild the trained architecture from the checkpoint's own config."""
    checkpoint = code.torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    schema = checkpoint.get("model_schema")
    supported = {
        code.net.MODEL_SCHEMA,
        getattr(code.net, "LEGACY_NO_OBJECTIVE_RESIDUAL_SCHEMA", None),
    }
    if schema not in supported:
        raise SystemExit(
            f"checkpoint schema {schema!r} is not supported by --code-root; "
            f"point --code-root at a matching checkout"
        )
    config = checkpoint.get("config", {})
    # Rebuild under the flags the checkpoint was trained with. Several of them
    # reshape heads, so the state dict will not load otherwise, and the purely
    # forward-time ones would silently evaluate an ablation as the full model.
    # Read the constructor's own signature rather than a hardcoded list, so a
    # schema that adds a flag does not need this function edited.
    import inspect

    parameters = inspect.signature(code.net.ConstraintFieldNet.__init__).parameters
    kwargs = {
        name: config[name]
        for name in parameters
        if name not in ("self",) and name in config
    }
    model = code.net.ConstraintFieldNet(**kwargs).to(device)
    if schema == code.net.MODEL_SCHEMA:
        code.net.load_constraint_field_state_dict(
            model, checkpoint["model_state_dict"]
        )
    else:
        code.net.load_constraint_field_state_dict(
            model,
            checkpoint["model_state_dict"],
            model_schema=schema,
            config=config,
        )
    model.eval()
    return model, checkpoint, config


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
FIELDS = (
    "variant",
    "instance",
    "state",
    "step_frac",
    "source",
    "target",
    "edge",
    "n_active_resources",
    # ground truth. ``mask_feasible`` is the raw hard mask, which rejects
    # already-visited nodes as well as constraint violations; ``visited``
    # separates the two so an analysis can isolate the constraint-driven
    # rejections, which are the only ones that carry resource semantics.
    "mask_feasible",
    "visited",
    "is_depot",
    "constraint_infeasible",
    "in_incumbent",
    # inputs the network is given
    "distance",
    "objective_cost",
    # model output, decomposed
    "energy",
    "energy_objective",
    "energy_resource",
    "energy_risk",
    "objective_residual",
    "feasibility_logit",
    "risk",
)


def resource_columns(count: int) -> tuple[str, ...]:
    """Per-resource columns, one block per registry channel."""
    columns: list[str] = []
    for channel in range(count):
        columns += [
            f"active_{channel}",
            f"field_{channel}",
            f"additive_{channel}",
            f"lambda_{channel}",
            f"pressure_{channel}",
            f"live_{channel}",
            f"binding_logit_{channel}",
        ]
    return tuple(columns)


def neutral_guidance(decoder) -> dict:
    """Zero field with unit objective weight: energy collapses to plain cost.

    Byte-identical to the ``field-off`` control the paper reports, so an
    incumbent built this way owes nothing to the model.
    """
    channels = int(decoder.metadata["resource_count"])
    slots = int(decoder.metadata["multiplier_count"])
    edges = int(decoder.metadata["edge_count"])
    multipliers = np.zeros(slots, dtype=np.float32)
    multipliers[channels] = 1.0
    return {
        "objective_residual": np.zeros(edges, dtype=np.float32),
        "edge_field": np.zeros((edges, channels), dtype=np.float32),
        "edge_additive": np.zeros((edges, channels), dtype=np.float32),
        "multipliers": multipliers,
        "coupler_weights": np.zeros((edges, slots, channels), dtype=np.float32),
        "coupler_bias": np.zeros((edges, slots), dtype=np.float32),
        "edge_risk": np.zeros(edges, dtype=np.float32),
        "risk_penalty": 0.0,
    }


def edge_distance(code, decoder, problem: dict) -> np.ndarray:
    """Raw (unnormalised) distance for each candidate edge."""
    edge_index = np.asarray(decoder.edge_index, dtype=np.int64)
    if "distance" in problem:
        matrix = np.asarray(problem["distance"], dtype=np.float64)
        return matrix[edge_index[0], edge_index[1]]
    coordinates = np.asarray(problem["coordinates"], dtype=np.float64)
    return np.linalg.norm(
        coordinates[edge_index[0]] - coordinates[edge_index[1]], axis=1
    )


def coupled_multipliers(code, model, output, live_state: np.ndarray):
    """lambda(s) for every multiplier slot, matching coupled_multiplier() in C++.

    ``ConstraintFieldNet.couple`` is the authority; calling it rather than
    reimplementing the sigmoid keeps the probe honest if the modulation changes.
    """
    torch = code.torch
    state = torch.as_tensor(
        live_state, dtype=torch.float32, device=output["multipliers"].device
    ).view(1, -1)
    with torch.no_grad():
        return model.couple(output, state)[0].cpu().numpy()


def build_decoder(code, problem: dict, args, instance: int):
    """Decoder configured exactly as inference configures it."""
    decoder = code.prism_decoder.Decoder(
        problem,
        candidate_config={
            "max_candidates": args.candidates,
            "candidate_mode": args.candidate_mode,
        },
        search_config={
            "use_srr": True,
            "min_changed_edges": args.min_changed_edges,
            "feasibility_lookahead_depth": 2,
            "srr_exploration_budget": 0,
        },
        n_rollouts=args.rollouts,
        beta=2.0,
    )
    decoder.seed(args.seed + instance)
    return decoder


def field_guidance(code, model, decoder, args) -> tuple[dict, object]:
    """One no-grad model evaluation of the current graph, as guidance."""
    torch = code.torch
    graph = code.net.build_decoder_data(decoder, args.device)
    with torch.no_grad():
        output = model(graph)
    guidance = {
        "edge_field": output["residual"].detach().cpu().numpy(),
        "edge_additive": output["additive"].detach().cpu().numpy(),
        "multipliers": output["multipliers"][0].detach().cpu().numpy(),
        "coupler_weights": output["coupler_weights"].detach().cpu().numpy(),
        "coupler_bias": output["coupler_bias"].detach().cpu().numpy(),
        "edge_risk": output["feasibility_risk"].detach().cpu().numpy(),
        "risk_penalty": float(args.risk_penalty),
    }
    return guidance, output


def install_incumbent(code, model, decoder, args):
    """Bootstrap a feasible incumbent and install it. Returns it, or None.

    ``--incumbent field`` reproduces inference exactly, so the probed states are
    the ones the model actually visits. ``--incumbent distance`` bootstraps from
    the zero-field control instead, which is what any comparison between the
    energy and the incumbent's own successor requires: against a
    self-constructed incumbent that agreement is circular.
    """
    if args.incumbent == "distance":
        guidance = neutral_guidance(decoder)
    else:
        guidance, _ = field_guidance(code, model, decoder, args)
    incumbent = decoder.sample_greedy(**guidance)
    if not incumbent["feasible"]:
        feasible = [
            item for item in decoder.sample(**guidance) if item["feasible"]
        ]
        if not feasible:
            return None
        incumbent = min(feasible, key=lambda item: float(item["objective"]))
    decoder.set_incumbent(incumbent["route"])
    return incumbent


def sampled_positions(route: np.ndarray, states: int) -> np.ndarray:
    """Decision positions along the incumbent.

    Position 0 is the start depot and the last has no successor, so both are
    excluded.
    """
    positions = np.arange(1, max(len(route) - 1, 1))
    if states and len(positions) > states:
        picked = np.unique(
            np.linspace(0, len(positions) - 1, states).round().astype(int)
        )
        positions = positions[picked]
    return positions


def state_records(
    code, model, decoder, problem, output, incumbent, args
) -> list[dict]:
    """Per-(state, candidate edge) records for one installed incumbent.

    Reproduces ``RoutingDecoder::edge_energy`` exactly, so a record's ``energy``
    is the quantity the decoder ranks by, decomposed into its objective,
    resource, and risk parts.
    """
    metadata = decoder.metadata
    resource_count = int(metadata["resource_count"])
    energy_scale = float(metadata["objective_energy_scale"])
    depot_count = int(metadata["depot_count"])
    objective_slot = resource_count  # objective_multiplier() in the decoder

    offsets = np.asarray(decoder.edge_offsets, dtype=np.int64)
    edge_index = np.asarray(decoder.edge_index, dtype=np.int64)
    objective_cost = np.asarray(decoder.objective_edge_costs, dtype=np.float64)
    pressure = np.asarray(decoder.resource_pressure, dtype=np.float64)
    live_states = np.asarray(decoder.incumbent_live_state, dtype=np.float32)
    distances = edge_distance(code, decoder, problem)

    active = output["active_channels"][0].detach().cpu().numpy()
    field = output["residual"].detach().cpu().numpy()
    additive = output["additive"].detach().cpu().numpy()
    obj_residual = np.zeros(metadata["edge_count"], dtype=np.float32)
    risk = output["feasibility_risk"].detach().cpu().numpy()
    feas_logit = output["feasibility_logits"].detach().cpu().numpy()
    binding_logit = output["binding_logits"][0].detach().cpu().numpy()

    route = np.asarray(incumbent["route"], dtype=np.int32)
    incumbent_edges = {
        (int(route[i]), int(route[i + 1])) for i in range(len(route) - 1)
    }

    records: list[dict] = []
    for state_id, position in enumerate(sampled_positions(route, args.states)):
        prefix = route[: position + 1]
        legal = np.asarray(decoder.mask(prefix), dtype=np.uint8)
        # A depot may legitimately recur, so prefix membership only marks a
        # customer as spent.
        seen = {int(node) for node in prefix[depot_count:]}
        source = int(route[position])
        live = live_states[source]
        multipliers = coupled_multipliers(code, model, output, live)

        lo, hi = int(offsets[source]), int(offsets[source + 1])
        for edge in range(lo, hi):
            target = int(edge_index[1, edge])
            is_depot = target < depot_count
            visited = int(not is_depot and target in seen)
            resource_energy = float(
                (
                    multipliers[:resource_count]
                    * active
                    * (field[edge] + additive[edge])
                ).sum()
            )
            objective_energy = float(
                multipliers[objective_slot]
                * (objective_cost[edge] / energy_scale + obj_residual[edge])
            )
            risk_energy = float(args.risk_penalty * risk[edge])
            record = {
                "state": state_id,
                "step_frac": round(position / max(len(route) - 1, 1), 6),
                "source": source,
                "target": target,
                "edge": edge,
                "n_active_resources": int(active.sum()),
                "mask_feasible": int(legal[target]),
                "visited": visited,
                "is_depot": int(is_depot),
                "constraint_infeasible": int(not visited and not legal[target]),
                "in_incumbent": int((source, target) in incumbent_edges),
                "distance": float(distances[edge]),
                "objective_cost": float(objective_cost[edge]),
                "energy": objective_energy + resource_energy + risk_energy,
                "energy_objective": objective_energy,
                "energy_resource": resource_energy,
                "energy_risk": risk_energy,
                "objective_residual": float(obj_residual[edge]),
                "feasibility_logit": float(feas_logit[edge]),
                "risk": float(risk[edge]),
            }
            for channel in range(resource_count):
                record[f"active_{channel}"] = float(active[channel])
                record[f"field_{channel}"] = float(field[edge, channel])
                record[f"additive_{channel}"] = float(additive[edge, channel])
                record[f"lambda_{channel}"] = float(multipliers[channel])
                record[f"pressure_{channel}"] = float(pressure[edge, channel])
                record[f"live_{channel}"] = float(live[channel])
                record[f"binding_logit_{channel}"] = float(
                    binding_logit[channel]
                )
            records.append(record)
    return records


def probe_instance(
    code,
    model,
    variant: str,
    problem: dict,
    args,
    instance: int,
    writer: csv.DictWriter,
) -> int:
    """Emit rows for one instance. Returns the number of rows written."""
    decoder = build_decoder(code, problem, args, instance)
    incumbent = install_incumbent(code, model, decoder, args)
    if incumbent is None:
        return 0

    # The candidate graph is rebuilt around the installed incumbent, so the
    # network must be re-evaluated on the graph the probe will actually read.
    graph = code.net.build_decoder_data(decoder, args.device)
    with code.torch.no_grad():
        output = model(graph)

    records = state_records(
        code, model, decoder, problem, output, incumbent, args
    )
    for record in records:
        writer.writerow({"variant": variant, "instance": instance, **record})
    return len(records)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--code-root",
        type=Path,
        required=True,
        help=(
            "checkout whose net.py and compiled prism_decoder match the "
            "checkpoint's model_schema"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="benchmark root (default: the code-root's own default)",
    )
    parser.add_argument(
        "--variants",
        default="cvrp,cvrptw,cvrpb,cvrpl,cvrpbltw",
        help="comma-separated variant names, or 'all'",
    )
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument(
        "--states",
        type=int,
        default=16,
        help="decision states sampled per instance (0 = every position)",
    )
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--min-changed-edges", type=int, default=8)
    parser.add_argument("--candidate-mode", default="geometric")
    parser.add_argument("--risk-penalty", type=float, default=1.0)
    parser.add_argument(
        "--incumbent",
        choices=("field", "distance"),
        default="field",
        help=(
            "solution to probe around: 'field' reproduces inference; "
            "'distance' uses the zero-field control, needed for any comparison "
            "between the energy and the incumbent's own successor"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def registry_width(code, args, finder, variants: list[str]) -> tuple[int, dict]:
    """Widest resource registry over the selected variants, and their data.

    Under schemas that hold every field channel whether active or not this is
    constant, but v11 onward the registry carries only the resources a variant
    activates, so the CSV has to be sized for the widest one and narrower rows
    left blank -- blank, not zero, because an absent channel is not an inactive
    channel.
    """
    width = 0
    loaded: dict = {}
    for variant in variants:
        try:
            paths = finder.get(variant, 100)
            data, _reference = code.problem_data.load_saved_data(
                paths["data_path"],
                variant,
                args.instances,
                solution_path=paths["solution_path"],
                allow_aggregate_reference=False,
            )
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"{variant}: skipped ({error})", flush=True)
            continue
        problem = code.solver_problem(variant, code.instance_data(data, 0))
        decoder = code.prism_decoder.Decoder(
            problem, candidate_config={"max_candidates": args.candidates},
            n_rollouts=1, beta=2.0,
        )
        width = max(width, int(decoder.metadata["resource_count"]))
        loaded[variant] = data
    return width, loaded


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code = load_code_root(args.code_root)
    model, checkpoint, config = build_model(
        code, args.checkpoint, args.device
    )
    print(
        f"checkpoint  epoch={checkpoint.get('epoch')} "
        f"val_gap={checkpoint.get('val_gap')} schema={code.net.MODEL_SCHEMA}",
        flush=True,
    )

    dataset_dir = args.dataset_dir or code.problem_data.DEFAULT_DATASET_DIR
    finder = code.problem_data.DatasetFinder(dataset_dir)
    if args.variants == "all":
        selector = getattr(code.test, "selected_variants", None)
        if selector is None:
            from prism_eval.instances import selected_variants as selector
        variants = selector("all")
    else:
        variants = [name.strip() for name in args.variants.split(",") if name.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    width, loaded = registry_width(code, args, finder, variants)
    if not loaded:
        raise SystemExit("no variants loaded")
    print(f"registry width={width} over {len(loaded)} variants", flush=True)

    total = 0
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FIELDS + resource_columns(width), restval=""
        )
        writer.writeheader()
        for variant, data in loaded.items():
            variant_rows = 0
            for instance in range(args.instances):
                problem = code.solver_problem(
                    variant, code.instance_data(data, instance)
                )
                variant_rows += probe_instance(
                    code, model, variant, problem, args, instance, writer
                )
            stream.flush()
            total += variant_rows
            print(f"{variant}: {variant_rows} rows", flush=True)

    print(f"wrote {total} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
