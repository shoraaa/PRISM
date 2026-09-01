#!/usr/bin/env python3
"""Where do unseen resources land relative to the trained ones?

The descriptor is defended on the grounds that it supplies a *generic
similarity coordinate system*: battery should land near capacity before the
model has seen a single battery transition, because both declare a bounded
accumulator with a reset. That is a claim about geometry, and it is testable
without running any search.

For every resource declared by the training variants and for each unseen
resource, this script encodes the declaration into the model's resource type
space and reports the unseen resource's nearest trained neighbours.

Three geometries are compared, because a neighbour ordering alone does not say
where the structure comes from:

    raw       distance between the parameter-free property maps themselves
              (row properties + pooled term properties). This is the input
              geometry, present before any training.
    random    the same encoder at its initialization. Structure that survives
              here is inherited from the input geometry, not learned.
    trained   the checkpoint's encoder.

If ``trained`` orders neighbours no better than ``random``, the learned encoder
is passing the input geometry through rather than organizing it -- the
similarity claim then rests on the property maps, not on training. If
``trained`` and ``raw`` disagree, training has reorganized the space, and the
ordering it produces is the thing that has to predict transfer.

Usage:
    python scripts/resource_geometry.py --checkpoint pretrained/newer/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import net as net_module  # noqa: E402
import prism_decoder  # noqa: E402
from net import (  # noqa: E402
    ConstraintFieldNet,
    load_constraint_field_state_dict,
)
from prism_eval import instances as pe_instances  # noqa: E402
from problem_data import (  # noqa: E402
    TRAIN_VARIANTS,
    DatasetFinder,
    load_saved_data,
)

ROW_DIM = prism_decoder.RESOURCE_ROW_PROPERTY_DIM
TERM_DIM = prism_decoder.RESOURCE_TERM_PROPERTY_DIM


def build_decoder(variant: str, finder: DatasetFinder, size: int, seed: int):
    if variant in pe_instances.GENERATORS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            data = pe_instances.GENERATORS[variant](size, 1)
    else:
        paths = finder.get(variant, size)
        data, _reference = load_saved_data(
            paths["data_path"],
            variant,
            1,
            solution_path=paths["solution_path"],
            allow_aggregate_reference=False,
        )
    problem = pe_instances.solver_problem(
        variant, pe_instances._instance_data(data, 0)
    )
    return prism_decoder.Decoder(problem)


def declarations(decoder) -> list[dict]:
    """One record per declared resource row, with its property tensors."""
    rows = np.asarray(
        decoder.resource_row_properties, dtype=np.float32
    ).reshape(-1, ROW_DIM)
    terms = np.asarray(
        decoder.resource_term_properties, dtype=np.float32
    ).reshape(-1, TERM_DIM)
    counts = np.asarray(decoder.resource_term_counts, dtype=np.int64).ravel()
    bounds = np.cumsum(np.concatenate(([0], counts)))
    names = [
        row.get("name", f"row{i}")
        for i, row in enumerate(decoder.resource_declarations)
    ]
    return [
        {
            "name": names[i],
            "row": rows[i],
            "terms": terms[bounds[i] : bounds[i + 1]],
            "count": int(counts[i]),
            "signature": signature(decoder, i),
        }
        for i in range(len(counts))
    ]


def encode(encoder, record: dict) -> np.ndarray:
    with torch.no_grad():
        vector = encoder(
            torch.from_numpy(record["row"]).unsqueeze(0),
            torch.from_numpy(record["terms"]),
            torch.tensor([record["count"]], dtype=torch.long),
        )
    return vector.squeeze(0).numpy()


def raw_vector(record: dict) -> np.ndarray:
    pooled = (
        record["terms"].sum(axis=0)
        if record["count"]
        else np.zeros(TERM_DIM, dtype=np.float32)
    )
    scale = max(record["count"], 1) ** 0.5
    return np.concatenate((record["row"], pooled / scale))


SIGNATURE_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def signature(decoder, index: int) -> np.ndarray:
    """Behavioral signature of one resource, from executed quantities only.

    The syntactic property maps describe how a declaration is *written*. This
    describes what it *does* to the instance: the distribution of resource
    pressure it induces over candidate edges, and the per-node attributes its
    algebra resolves. Both are published by the decoder for every row -- the
    compiled kernels and the interpreted ones alike -- so no coordinate has to
    be enumerated in advance and a new language primitive changes the response
    rather than the width.

    This is the no-C++ prototype of the probe battery. It samples the response
    on the instance's own edges and nodes instead of on a fixed synthetic grid,
    so it is instance-conditioned; a fixed battery needs a transition-level
    binding (see the module docstring).
    """
    pressure = np.asarray(decoder.resource_pressure, dtype=np.float64)
    pressure = pressure.reshape(-1, pressure.shape[-1])[:, index]
    node = np.asarray(decoder.node_resource_features, dtype=np.float64)
    node = node.reshape(node.shape[0], -1, node.shape[-1])[:, index, :]
    scale = float(np.asarray(decoder.resource_scales, dtype=np.float64).ravel()[index])
    # Pressure is a ratio to the declared bound and unbounded above, so a row
    # that is merely tighter than a trained one would be pushed out of the space
    # by magnitude alone. Compress it; the shape of the profile is the signal.
    quantiles = np.quantile(pressure, SIGNATURE_QUANTILES)
    quantiles = quantiles / (1.0 + np.abs(quantiles))
    return np.concatenate(
        (
            quantiles,
            node.mean(axis=0),
            node.std(axis=0),
            [scale / (1.0 + abs(scale))],
        )
    ).astype(np.float32)


def spread(bank: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Pairwise L2 distances within the bank, and their median."""
    matrix = np.stack([bank[name] for name in sorted(bank)])
    diff = matrix[:, None, :] - matrix[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    upper = distances[np.triu_indices(len(matrix), k=1)]
    return upper, float(np.median(upper)) if upper.size else 0.0


def neighbours(query: np.ndarray, bank: dict[str, np.ndarray]) -> list[tuple]:
    """Rank the bank by distance to the query, in L2 and centred cosine."""
    names = sorted(bank)
    matrix = np.stack([bank[name] for name in names])
    centre = matrix.mean(axis=0)
    centred = matrix - centre
    query_centred = query - centre
    norms = np.linalg.norm(centred, axis=1) * np.linalg.norm(query_centred)
    cosine = np.where(
        norms > 1e-9, (centred @ query_centred) / np.maximum(norms, 1e-9), 0.0
    )
    distance = np.linalg.norm(matrix - query, axis=1)
    order = np.argsort(distance)
    return [(names[i], float(distance[i]), float(cosine[i])) for i in order]


def training_reference(bank: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mean row and term property vectors over the trained declarations.

    For a coordinate that is constant across training, this mean IS that
    constant, so substituting it moves the query onto the trained value of that
    axis without disturbing any other coordinate.
    """
    rows = np.stack([record["row"] for record in bank.values()])
    terms = np.vstack([record["terms"] for record in bank.values()])
    return rows.mean(axis=0), terms.mean(axis=0)


def substituted(record: dict, reference, where: str, index: int) -> dict:
    """Copy of ``record`` with one coordinate set to its training reference."""
    row_reference, term_reference = reference
    clone = {
        "name": record["name"],
        "row": record["row"].copy(),
        "terms": record["terms"].copy(),
        "count": record["count"],
    }
    if where == "row":
        clone["row"][index] = row_reference[index]
    else:
        clone["terms"][:, index] = term_reference[index]
    return clone


def report_attribution(
    record: dict,
    label: str,
    bank: dict[str, dict],
    spaces: dict[str, dict[str, np.ndarray]],
    encoders: dict,
    untrained: dict[str, list[int]],
    top: int,
) -> None:
    """Which coordinates hold the query where it is?"""
    reference = training_reference(bank)
    trained_space = spaces["trained"]
    base_vector = encode(encoders["trained"].resource_program_encoder, record)
    base_neighbour = neighbours(base_vector, trained_space)[0]
    print(f"\n=== attribution: {label} ===")
    print(f"unperturbed nearest: {base_neighbour[0]} (L2 {base_neighbour[1]:.3f})")
    print(
        f"\n{'coordinate':<14}{'trained?':<10}{'displacement':>13}"
        f"  nearest after substitution"
    )

    scored = []
    for where, dim in (("row", ROW_DIM), ("term", TERM_DIM)):
        for index in range(dim):
            clone = substituted(record, reference, where, index)
            if np.array_equal(clone["row"], record["row"]) and np.array_equal(
                clone["terms"], record["terms"]
            ):
                continue  # already at the training reference; a no-op
            vector = encode(encoders["trained"].resource_program_encoder, clone)
            moved = float(np.linalg.norm(vector - base_vector))
            neighbour = neighbours(vector, trained_space)[0]
            scored.append((moved, where, index, neighbour))
    scored.sort(reverse=True)
    for moved, where, index, neighbour in scored[:top]:
        flag = "no" if index in untrained[where] else "yes"
        print(
            f"{where + '[' + str(index) + ']':<14}{flag:<10}{moved:>13.3f}"
            f"  {neighbour[0]} ({neighbour[1]:.3f})"
        )

    # Headline: neutralize every untrained coordinate at once.
    clone = {
        "name": record["name"],
        "row": record["row"].copy(),
        "terms": record["terms"].copy(),
        "count": record["count"],
    }
    for index in untrained["row"]:
        clone["row"][index] = reference[0][index]
    for index in untrained["term"]:
        clone["terms"][:, index] = reference[1][index]
    changed = not (
        np.array_equal(clone["row"], record["row"])
        and np.array_equal(clone["terms"], record["terms"])
    )
    vector = encode(encoders["trained"].resource_program_encoder, clone)
    print("\n  all untrained coordinates set to their training value"
          f" ({'input changed' if changed else 'INPUT UNCHANGED - no-op'}):")
    for rank, (name, distance, _cos) in enumerate(
        neighbours(vector, trained_space)[:3], start=1
    ):
        print(f"    {rank}. {name:<28}{distance:>8.3f}")
    print(
        f"    displacement from unperturbed: "
        f"{float(np.linalg.norm(vector - base_vector)):.3f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="datasets/benchmarks")
    parser.add_argument(
        "--unseen",
        default="evrp,evrptw,vrpdb,vrpdbtw",
        help="variants whose novel resources are scored against the bank",
    )
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--attribution",
        action="store_true",
        help=(
            "For each unseen resource, substitute one coordinate at a time with"
            " its training-set value and report which coordinates hold the"
            " resource at its learned position. Coordinates that never varied in"
            " training are flagged: if those are what place the resource, the"
            " placement is produced by weights that received no gradient."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    finder = DatasetFinder(args.dataset_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = checkpoint.get("model_schema")
    config = checkpoint.get("config", {})
    config = config if isinstance(config, dict) else vars(config)
    trained_model = ConstraintFieldNet(
        index_embedded_resources=config.get("index_embedded_resources", False),
        monolithic_resource_field=config.get("monolithic_resource_field", False),
        program_blind_resources=config.get("program_blind_resources", False),
        core_interface=config.get("core_interface", "full"),
    )
    load_constraint_field_state_dict(
        trained_model,
        checkpoint["model_state_dict"],
        model_schema=schema,
        config=config,
    )
    trained_model.eval()
    torch.manual_seed(0)
    random_model = ConstraintFieldNet().eval()
    print(
        f"checkpoint {args.checkpoint} | schema {schema} | "
        f"epoch {checkpoint.get('epoch')} | val {checkpoint.get('val_gap'):.4f}"
    )

    # ---- bank of trained resource declarations ----
    bank: dict[str, dict] = {}
    for variant in TRAIN_VARIANTS:
        try:
            decoder = build_decoder(variant, finder, args.size, args.seed)
        except Exception as error:  # noqa: BLE001
            print(f"  {variant}: skipped ({type(error).__name__}: {error})")
            continue
        for record in declarations(decoder):
            key = record["name"]
            fingerprint = (
                record["row"].tobytes(),
                np.sort(record["terms"], axis=0).tobytes(),
            )
            existing = bank.get(key)
            if existing is None:
                record["variants"] = [variant]
                record["fingerprint"] = fingerprint
                bank[key] = record
            elif existing["fingerprint"] != fingerprint:
                # Same constraint, different declared form (e.g. capacity with
                # and without the backhaul guard). Keep both as distinct points.
                alt = f"{key}@{variant}"
                if alt not in bank:
                    record["variants"] = [variant]
                    record["fingerprint"] = fingerprint
                    bank[alt] = record
            else:
                existing["variants"].append(variant)

    print(f"\ntrained resource bank: {len(bank)} distinct declarations")
    for name, record in sorted(bank.items()):
        print(
            f"  {name:<28} {record['count']} terms, "
            f"in {len(record['variants'])} training variants"
        )

    spaces = {
        "raw": {name: raw_vector(r) for name, r in bank.items()},
        "random": {
            name: encode(random_model.resource_program_encoder, r)
            for name, r in bank.items()
        },
        "trained": {
            name: encode(trained_model.resource_program_encoder, r)
            for name, r in bank.items()
        },
        "signature": {name: r["signature"] for name, r in bank.items()},
    }

    # Coordinates that never varied across the trained declarations. Weights on
    # these input directions received no gradient, so anything they contribute
    # to a placement comes from initialization.
    bank_rows = np.stack([record["row"] for record in bank.values()])
    bank_terms = np.vstack([record["terms"] for record in bank.values()])
    untrained = {
        "row": [
            i for i in range(ROW_DIM)
            if len(np.unique(np.round(bank_rows[:, i], 4))) == 1
        ],
        "term": [
            j for j in range(TERM_DIM)
            if len(np.unique(np.round(bank_terms[:, j], 4))) == 1
        ],
    }
    print(
        f"\nconstant across the trained bank: row {untrained['row']}, "
        f"term {untrained['term']}"
    )
    encoders = {"trained": trained_model, "random": random_model}

    # ---- unseen resources ----
    unseen = [name.strip() for name in args.unseen.split(",") if name.strip()]
    known = set(bank)
    for variant in unseen:
        try:
            decoder = build_decoder(variant, finder, args.size, args.seed)
        except Exception as error:  # noqa: BLE001
            print(f"\n{variant}: skipped ({type(error).__name__}: {error})")
            continue
        for record in declarations(decoder):
            if record["name"] in known:
                continue
            print(f"\n=== {variant}: {record['name']} ({record['count']} terms) ===")
            print(
                f"{'space':<9}{'rank':<6}{'nearest trained resource':<30}"
                f"{'L2':>8}{'cos':>8}{'pct':>7}"
            )
            for space, vectors in spaces.items():
                if space == "signature":
                    query = record["signature"]
                else:
                    query = (
                        raw_vector(record)
                        if space == "raw"
                        else encode(
                            (random_model if space == "random" else trained_model)
                            .resource_program_encoder,
                            record,
                        )
                    )
                ranked = neighbours(query, vectors)
                pairwise, median = spread(vectors)
                for rank, (name, distance, cosine) in enumerate(
                    ranked[: args.top], start=1
                ):
                    label = space if rank == 1 else ""
                    # Where the query sits inside the bank's own spread: 0%
                    # means closer than every trained pair, 100% means further.
                    percentile = (
                        100.0 * float((pairwise < distance).mean())
                        if pairwise.size
                        else float("nan")
                    )
                    print(
                        f"{label:<9}{rank:<6}{name:<30}{distance:>8.3f}"
                        f"{cosine:>8.3f}{percentile:>6.0f}%"
                    )
                anchors = [
                    name
                    for name in ("capacity", "route_limit", "tour_limit",
                                 "time_window", "prize_quota", "backhaul_order")
                    if name in vectors
                ]
                by_name = {name: (d, c) for name, d, c in ranked}
                detail = "  ".join(
                    f"{name.split('_')[0]}={by_name[name][0]:.2f}"
                    for name in anchors
                )
                print(f"{'':<9}{'':<6}{'anchors: ' + detail}")
                print(
                    f"{'':<9}{'':<6}bank median pairwise L2 = {median:.3f}"
                )
            if args.attribution:
                report_attribution(
                    record,
                    f"{variant}: {record['name']}",
                    bank,
                    spaces,
                    encoders,
                    untrained,
                    args.top,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
