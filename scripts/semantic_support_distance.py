#!/usr/bin/env python3
"""Empirical support distance for executable resource observations.

This is deliberately not the supremum ``d_sem`` from the stability theorem.
That distance requires paired states and actions in two semantic systems.  Here
we build a fixed-width, parameter-free signature from marginal quantiles of all
per-resource executable observations consumed by the program-blind v15 scorer,
after installing an objective-only incumbent.  Nearest-neighbour distance to
the training-variant bank is therefore an empirical out-of-support diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import net  # noqa: E402
from problem_data import TRAIN_VARIANTS, DatasetFinder  # noqa: E402
from scripts.probe_feasibility import distance_incumbent  # noqa: E402
from scripts.resource_geometry import build_decoder  # noqa: E402


QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_columns(graph, channel: int) -> dict[str, np.ndarray]:
    scale = float(graph.resource_scales[channel])
    columns = {
        "edge.pressure": graph.resource_features[:, channel],
        "edge.raw_pressure_over_scale": (
            graph.raw_resource_pressure[:, channel] / scale
        ),
        "edge.event": graph.resource_events[:, channel],
        "edge.next_state": graph.resource_transition_features[:, channel, 0],
        "edge.signed_margin": graph.resource_transition_features[:, channel, 1],
        "edge.transition_valid": graph.resource_transition_mask[:, channel].float(),
        "node.live_state": graph.node_live_state[:, channel],
        "node.suffix_state": graph.node_suffix_state[:, channel],
    }
    for index in range(graph.node_resource.shape[-1]):
        columns[f"node.attribute[{index}]"] = graph.node_resource[:, channel, index]
    for index in range(graph.node_suffix_features.shape[-1]):
        columns[f"node.suffix_feature[{index}]"] = (
            graph.node_suffix_features[:, channel, index]
        )
    return {
        name: value.detach().cpu().numpy().astype(np.float64, copy=False)
        for name, value in columns.items()
    }


def signature(graph, channel: int) -> tuple[np.ndarray, list[str]]:
    values = []
    labels = []
    for name, column in observation_columns(graph, channel).items():
        values.extend(np.quantile(column, QUANTILES))
        labels.extend(f"{name}.q{quantile:g}" for quantile in QUANTILES)
    return np.asarray(values, dtype=np.float64), labels


def decoder_records(variant, finder, size, seed):
    decoder = build_decoder(variant, finder, size, seed)
    incumbent = distance_incumbent(decoder)
    if incumbent is None:
        raise RuntimeError("objective-only construction produced no feasible incumbent")
    graph = net.build_decoder_data(decoder, "cpu")
    records = []
    for channel, declaration in enumerate(decoder.resource_declarations):
        vector, labels = signature(graph, channel)
        records.append(
            {
                "variant": variant,
                "resource": declaration.get("name", f"row{channel}"),
                "signature": vector,
                "labels": labels,
            }
        )
    return records


def rms_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / left.size**0.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", default="datasets/benchmarks")
    parser.add_argument("--unseen", default="evrp,evrptw,vrpdb")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    if config.get("program_blind_resources") is not None:
        raise SystemExit(
            "this diagnostic is scoped to a program-blind checkpoint; this one "
            "uses declaration descriptors"
        )
    finder = DatasetFinder(args.dataset_dir)

    bank = []
    for variant in TRAIN_VARIANTS:
        try:
            bank.extend(decoder_records(variant, finder, args.size, args.seed))
        except Exception as error:  # noqa: BLE001
            print(f"{variant}: skipped ({type(error).__name__}: {error})", flush=True)
    if not bank:
        raise SystemExit("training observation bank is empty")
    matrix = np.stack([record["signature"] for record in bank])
    dimensions = matrix.shape[1]
    pairwise = []
    for left in range(len(bank)):
        for right in range(left + 1, len(bank)):
            pairwise.append(rms_l2(matrix[left], matrix[right]))
    pairwise = np.asarray(pairwise, dtype=np.float64)

    trained_names = {record["resource"] for record in bank}
    results = []
    for variant in [item.strip() for item in args.unseen.split(",") if item.strip()]:
        for query in decoder_records(variant, finder, args.size, args.seed):
            if query["resource"] in trained_names:
                continue
            distances = np.asarray(
                [rms_l2(query["signature"], row["signature"]) for row in bank]
            )
            order = np.argsort(distances)
            neighbours = [
                {
                    "resource": bank[int(index)]["resource"],
                    "variant": bank[int(index)]["variant"],
                    "distance": float(distances[int(index)]),
                }
                for index in order[: args.top]
            ]
            nearest = float(distances[int(order[0])])
            percentile = float(100.0 * np.mean(pairwise < nearest))
            result = {
                "variant": variant,
                "resource": query["resource"],
                "nearest_distance": nearest,
                "within_bank_percentile": percentile,
                "neighbours": neighbours,
            }
            results.append(result)
            print(
                f"{variant}:{query['resource']} nearest={nearest:.6f} "
                f"bank_percentile={percentile:.1f}%"
            )
            for neighbour in neighbours:
                print(
                    f"  {neighbour['resource']}@{neighbour['variant']} "
                    f"{neighbour['distance']:.6f}"
                )

    report = {
        "checkpoint": str(args.checkpoint),
        "sha256": sha256(args.checkpoint),
        "schema": checkpoint.get("model_schema"),
        "epoch": checkpoint.get("epoch"),
        "metric": "RMS L2 between marginal-quantile signatures",
        "scope": "empirical support diagnostic, not theorem d_sem",
        "quantiles": QUANTILES,
        "dimensions": dimensions,
        "training_points": len(bank),
        "training_pairwise": {
            "median": float(np.median(pairwise)),
            "p95": float(np.quantile(pairwise, 0.95)),
            "max": float(pairwise.max()),
        },
        "results": results,
    }
    print(
        f"bank={len(bank)} points dimensions={dimensions} "
        f"median_pairwise={np.median(pairwise):.6f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
