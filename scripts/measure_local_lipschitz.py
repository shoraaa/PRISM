#!/usr/bin/env python3
"""Measure checkpoint-local semantic sensitivity of the action energy.

This estimates, rather than certifies, the Lipschitz factor in the semantic
stability argument.  The discrete graph, active-resource mask, transition
validity mask, and declaration structure are held fixed.  Autograd differentiates
the constraint-conditioned action energy through every continuous executable
quantity that the deployed v15 scorer consumes:

* normalized and raw/scale edge pressure, and resource events;
* per-node resource attributes, live state, and suffix state/features;
* candidate next state and signed admissibility margin.

For each sampled action, the gradient norm is the exact local operator norm of
that scalar energy under the stated input norm.  The maximum over sampled
actions estimates the local L2->Linf or Linf->Linf constant for the energy
vector.  It is not a global certificate between finite semantic systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import net  # noqa: E402
from problem_data import DEFAULT_DATASET_DIR, DatasetFinder  # noqa: E402
from prism_eval.instances import _instance_data, solver_problem  # noqa: E402
from scripts.probe_feasibility import (  # noqa: E402
    build_decoder,
    build_model,
    distance_incumbent,
    load_variant,
    sampled_positions,
)


SEMANTIC_INPUTS = (
    "resource_features",
    "raw_resource_pressure",
    "resource_events",
    "node_resource",
    "node_live_state",
    "node_suffix_state",
    "node_suffix_features",
    "resource_transition_features",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    indices = np.unique(
        np.linspace(0, len(values) - 1, count).round().astype(np.int64)
    )
    return [values[int(index)] for index in indices]


def install_semantic_variables(graph) -> list[torch.Tensor]:
    variables = []
    for name in SEMANTIC_INPUTS:
        value = getattr(graph, name).detach().clone().requires_grad_(True)
        setattr(graph, name, value)
        variables.append(value)
    return variables


def normalized_gradient(gradient, name: str, graph):
    """Return dE/dU when raw pressure is parameterized as raw / row scale."""
    if gradient is None:
        return None
    if name == "raw_resource_pressure":
        return gradient * graph.resource_scales.view(1, -1)
    return gradient


def probe_graph(model, graph, decoder, route, args, variant: str, instance: int):
    variables = install_semantic_variables(graph)
    output = model(graph)
    resource_count = output["residual"].shape[1]
    offsets = np.asarray(decoder.edge_offsets, dtype=np.int64)
    targets = np.asarray(decoder.edge_index, dtype=np.int64).reshape(2, -1)[1]
    depot_count = int(decoder.metadata["depot_count"])
    route = np.asarray(route, dtype=np.int64)

    records = []
    for state_id, position in enumerate(sampled_positions(route, args.states)):
        source = int(route[position])
        seen = {int(node) for node in route[: position + 1][depot_count:]}
        candidates = [
            edge
            for edge in range(int(offsets[source]), int(offsets[source + 1]))
            if int(targets[edge]) >= depot_count and int(targets[edge]) not in seen
        ]
        for edge in evenly_spaced(candidates, args.actions):
            live_state = graph.node_live_state[source].view(1, -1)
            multipliers = model.couple(
                output,
                live_state,
                torch.tensor([edge], device=live_state.device),
            )[0, :resource_count]
            energy = (multipliers * output["residual"][edge]).sum()
            gradients = torch.autograd.grad(
                energy,
                variables,
                retain_graph=True,
                allow_unused=True,
            )
            component_l2_sq = {}
            component_l1 = {}
            for name, gradient in zip(SEMANTIC_INPUTS, gradients):
                gradient = normalized_gradient(gradient, name, graph)
                if gradient is None:
                    component_l2_sq[name] = 0.0
                    component_l1[name] = 0.0
                else:
                    component_l2_sq[name] = float(gradient.square().sum())
                    component_l1[name] = float(gradient.abs().sum())
            records.append(
                {
                    "variant": variant,
                    "instance": instance,
                    "state": state_id,
                    "source": source,
                    "edge": int(edge),
                    "target": int(targets[edge]),
                    "energy": float(energy.detach()),
                    "l2_to_linf": float(sum(component_l2_sq.values()) ** 0.5),
                    "linf_to_linf": float(sum(component_l1.values())),
                    "component_l2": {
                        name: float(value**0.5)
                        for name, value in component_l2_sq.items()
                    },
                }
            )
    return records


def quantiles(values):
    levels = (0.0, 0.5, 0.95, 1.0)
    result = np.quantile(np.asarray(values, dtype=np.float64), levels)
    labels = ("min", "median", "p95", "max")
    return {label: float(value) for label, value in zip(labels, result)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument(
        "--variants",
        default="cvrpbpltw,acvrpbpltw,mdcvrpbpltw,amdocvrpbpltw",
    )
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--actions", type=int, default=4)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model, checkpoint = build_model(args.checkpoint, args.device)
    finder = DatasetFinder(args.dataset_dir or DEFAULT_DATASET_DIR)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    decoder_args = SimpleNamespace(
        candidates=args.candidates,
        rollouts=args.rollouts,
        seed=args.seed,
        device=args.device,
    )

    records = []
    for variant in variants:
        data = load_variant(variant, finder, args.instances, args.size)
        before = len(records)
        for instance in range(args.instances):
            problem = solver_problem(variant, _instance_data(data, instance))
            decoder = build_decoder(problem, decoder_args, instance)
            incumbent = distance_incumbent(decoder)
            if incumbent is None:
                continue
            graph = net.build_decoder_data(decoder, args.device)
            records.extend(
                probe_graph(
                    model,
                    graph,
                    decoder,
                    incumbent["route"],
                    args,
                    variant,
                    instance,
                )
            )
        print(f"{variant}: {len(records) - before} sampled actions", flush=True)

    if not records:
        raise SystemExit("no actions sampled")
    summary = {
        "checkpoint": str(args.checkpoint),
        "sha256": sha256(args.checkpoint),
        "schema": checkpoint.get("model_schema"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "val_gap": checkpoint.get("val_gap"),
        "domain": {
            "variants": variants,
            "instances": args.instances,
            "size": args.size,
            "states_per_instance": args.states,
            "actions_per_state": args.actions,
            "discrete_structure": "fixed",
            "raw_pressure_coordinate": "raw_resource_pressure / resource_scale",
        },
        "samples": len(records),
        "local_l2_to_linf": quantiles([row["l2_to_linf"] for row in records]),
        "local_linf_to_linf": quantiles(
            [row["linf_to_linf"] for row in records]
        ),
        "records": records,
    }
    short_summary = {
        key: value for key, value in summary.items() if key != "records"
    }
    print(json.dumps(short_summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {len(records)} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
