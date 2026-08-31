#!/usr/bin/env python3
"""Persist a deterministic full PRISM benchmark suite at an arbitrary scale.

The suite is written directly in PRISM's neutral batched tensor schema as
``<variant><scale>_prism.pt``. Related variants share geometry, demands, and
time-window draws so adding a constraint does not silently change the underlying
instances. Generated artifacts declare ``solution_file: null`` in an adjacent
JSON sidecar: old oracle files for another dataset must never become references
for these fresh instances.

Asymmetric instances use PRISM's existing random directed metric-closure
distribution. Geometry banks are shared across related variants to avoid
resampling that cubic construction for every constraint composition. Samples
are never filtered or rescaled for feasibility; certifiable infeasibility is
recorded per variant in metadata and printed by the command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from problem_data import (  # noqa: E402
    BENCHMARK_VARIANTS,
    DEFAULT_DATASET_DIR,
    _metric_distance,
    load_saved_data,
)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _geometry_bank(
    size: int, count: int, seed: int
) -> dict[tuple[bool, int], dict[str, torch.Tensor]]:
    banks = {}
    for asymmetric in (False, True):
        for depot_count in (0, 1, 3):
            node_count = size + depot_count
            key_seed = seed + 100_000 * asymmetric + 10_000 * depot_count
            if asymmetric:
                # Use the exact PRISM training distribution. It is cubic, but
                # each bank is generated only once and shared by every related
                # variant rather than being resampled hundreds of times.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(key_seed)
                    distance = torch.stack(
                        [_metric_distance(node_count) for _ in range(count)]
                    )
                banks[(asymmetric, depot_count)] = {"dist": distance}
            else:
                generator = _generator(key_seed)
                banks[(asymmetric, depot_count)] = {
                    "xy": torch.rand(count, node_count, 2, generator=generator)
                }
    return banks


def _travel_to_depots(
    geometry: dict[str, torch.Tensor], depot_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if "dist" in geometry:
        matrix = geometry["dist"]
        outgoing = matrix[:, :depot_count, depot_count:].amin(dim=1)
        incoming = matrix[:, depot_count:, :depot_count].amin(dim=2)
        return outgoing, incoming
    xy = geometry["xy"]
    radial = torch.linalg.vector_norm(
        xy[:, depot_count:, None, :] - xy[:, None, :depot_count, :], dim=-1
    )
    nearest = radial.amin(dim=2)
    return nearest, nearest


def _feature_bank(
    geometry: dict[str, torch.Tensor],
    *,
    size: int,
    count: int,
    depot_count: int,
    capacity: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = _generator(seed)
    node_count = size + depot_count
    demand = torch.randint(
        1, 10, (count, node_count), generator=generator
    ).float() / float(capacity)
    demand[:, :depot_count] = 0.0
    backhaul = demand.clone()
    backhaul_count = max(1, int(size * 0.2))
    for row in range(count):
        selected = torch.randperm(size, generator=generator)[:backhaul_count]
        backhaul[row, selected + depot_count] *= -1.0

    bank = {"demand": demand, "backhaul_demand": backhaul}
    if depot_count == 1 and size % 2 == 0:
        pickup = torch.randint(
            1, 10, (count, size // 2), generator=generator
        ).float() / float(capacity)
        bank["pickup_demand"] = torch.cat(
            (torch.zeros(count, 1), pickup, -pickup), dim=1
        )

    outgoing, incoming = _travel_to_depots(geometry, depot_count)
    service = torch.full((count, size), 0.2)
    horizon = 1.0 if "dist" in geometry else 3.0
    earliest = outgoing
    latest = horizon - incoming - service
    center = latest + (earliest - latest) * torch.rand(
        count, size, generator=generator
    )
    half_width = horizon / 3 + (service / 2 - horizon / 3) * torch.rand(
        count, size, generator=generator
    )
    depot_zeros = torch.zeros(count, depot_count)
    bank["service_time"] = torch.cat((depot_zeros, service), dim=1)
    bank["tw_start"] = torch.cat(
        (depot_zeros, torch.clamp(center - half_width, 0.0, horizon)), dim=1
    )
    bank["tw_end"] = torch.cat(
        (
            torch.full((count, depot_count), horizon),
            torch.clamp(center + half_width, 0.0, horizon),
        ),
        dim=1,
    )
    if "dist" in geometry:
        matrix = geometry["dist"]
        same_depot_round_trip = (
            matrix[:, :depot_count, depot_count:].transpose(1, 2)
            + matrix[:, depot_count:, :depot_count]
        )
        bank["route_limit"] = 1.1 * same_depot_round_trip.amin(dim=2).amax(dim=1)
    else:
        bank["route_limit"] = torch.full((count,), 3.0)
    return bank


def generate_suite(
    size: int,
    count: int,
    seed: int,
    capacity: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Generate all 110 variants as neutral, batched tensor dictionaries."""
    if size < 1 or count < 1 or capacity < 1:
        raise ValueError("size, count, and capacity must be positive")
    if size % 2:
        raise ValueError("full-suite pickup-delivery generation requires even size")
    geometries = _geometry_bank(size, count, seed)
    features = {
        key: _feature_bank(
            geometry,
            size=size,
            count=count,
            depot_count=key[1],
            capacity=capacity,
            seed=seed + 1_000_000 + 100_000 * key[0] + 10_000 * key[1],
        )
        for key, geometry in geometries.items()
        if key[1] > 0
    }
    prize_generators = {
        key: _generator(seed + 2_000_000 + 100_000 * key[0] + 10_000 * key[1])
        for key in geometries
    }

    suite = {}
    for name in BENCHMARK_VARIANTS:
        asymmetric = name.startswith("a")
        depot_count = 0 if name in {"tsp", "atsp"} else 3 if "md" in name else 1
        key = (asymmetric, depot_count)
        geometry = geometries[key]
        data = {field: value for field, value in geometry.items()}
        bank = features.get(key)

        if "cvrp" in name:
            if "pd" in name:
                data["demand"] = bank["pickup_demand"]
            elif "b" in name:
                data["demand"] = bank["backhaul_demand"]
            else:
                data["demand"] = bank["demand"]
            if "tw" in name:
                for field in ("service_time", "tw_start", "tw_end"):
                    data[field] = bank[field]
            if "l" in name:
                data["route_limit"] = bank["route_limit"]

        if name in {"op", "aop"}:
            if "dist" in geometry:
                radius = geometry["dist"][:, 0, :]
            else:
                xy = geometry["xy"]
                radius = torch.linalg.vector_norm(xy[:, :1] - xy, dim=-1)
            maximum = radius.amax(dim=1, keepdim=True).clamp_min(1e-9)
            prize = (1 + (radius / maximum * 99).int()).float() / 100.0
            prize[:, 0] = 0.0
            data["prize"] = prize
        elif "pctsp" in name:
            generator = prize_generators[key]
            prize = torch.rand(count, size, generator=generator) * 4.0 / size
            scale = {20: 2, 50: 3, 100: 4, 500: 9, 1000: 12}.get(
                size, max(2, round(size**0.4))
            )
            penalty = (
                torch.rand(count, size, generator=generator) * 3.0 * scale / size
            )
            data["prize"] = torch.cat((torch.zeros(count, 1), prize), dim=1)
            data["penalty"] = torch.cat((torch.zeros(count, 1), penalty), dim=1)
        suite[name] = data
    return suite


def infeasibility_report(
    name: str, data: dict[str, torch.Tensor]
) -> dict[str, int]:
    """Report generator-certifiable infeasibility without changing samples."""
    anchor = data.get("xy", data.get("dist"))
    count = len(anchor)
    failed = torch.zeros(count, dtype=torch.bool)
    reasons: dict[str, int] = {}
    depots = 0 if name in {"tsp", "atsp"} else 3 if "md" in name else 1

    if depots and ("tw" in name or "route_limit" in data):
        if "dist" in data:
            outgoing = data["dist"][:, :depots, depots:]
            incoming = data["dist"][:, depots:, :depots].transpose(1, 2)
        else:
            outgoing = torch.linalg.vector_norm(
                data["xy"][:, :depots, None, :]
                - data["xy"][:, None, depots:, :],
                dim=-1,
            )
            incoming = outgoing
        route_feasible = torch.ones(
            count, depots, outgoing.shape[2], dtype=torch.bool
        )
        if "route_limit" in data:
            route_feasible &= (
                outgoing + incoming
                <= data["route_limit"][:, None, None] + 1e-6
            )
            bad = ~route_feasible.any(dim=1).all(dim=1)
            reasons["route_limit_singleton"] = int(bad.sum())
            failed |= bad
        if "tw" in name:
            start = data["tw_start"][:, None, depots:]
            end = data["tw_end"][:, None, depots:]
            service = data["service_time"][:, None, depots:]
            depot_end = data["tw_end"][:, :depots, None]
            finish = torch.maximum(outgoing, start) + service + incoming
            tw_feasible = (outgoing <= end + 1e-6) & (finish <= depot_end + 1e-6)
            bad = ~tw_feasible.any(dim=1).all(dim=1)
            reasons["time_window_singleton"] = int(bad.sum())
            failed |= bad
    if "pctsp" in name:
        bad = data["prize"].sum(dim=1) < 1.0
        reasons["prize_quota"] = int(bad.sum())
        failed |= bad
    return {"instances": int(failed.sum()), **reasons}


def _write_variant(
    root: Path,
    name: str,
    size: int,
    data: dict[str, torch.Tensor],
    *,
    count: int,
    seed: int,
    capacity: int,
    infeasibility: dict[str, int],
    force: bool,
) -> str:
    directory = root / name
    target = directory / f"{name}{size}_prism.pt"
    metadata_path = target.with_suffix(".json")
    if target.exists() and not force:
        load_saved_data(target, name, min(count, 8), allow_aggregate_reference=False)
        return "verified"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(target)
    metadata = {
        "format": "prism_tensor_v1",
        "variant": name,
        "scale": size,
        "instances": count,
        "seed": seed,
        "capacity": capacity,
        "distribution": "shared_prism_generated_v1",
        "infeasibility": infeasibility,
        "solution_file": None,
    }
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with metadata_temporary.open("w") as destination:
        json.dump(metadata, destination, indent=2, sort_keys=True)
        destination.write("\n")
    metadata_temporary.replace(metadata_path)
    load_saved_data(target, name, min(count, 8), allow_aggregate_reference=False)
    return "generated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.size < 1 or args.count < 1 or args.capacity < 1:
        parser.error("--size, --count, and --capacity must be positive")
    if args.size % 2:
        parser.error("--size must be even for the pickup-delivery variants")

    suite = generate_suite(args.size, args.count, args.seed, args.capacity)
    generated = verified = 0
    for index, name in enumerate(BENCHMARK_VARIANTS, 1):
        infeasibility = infeasibility_report(name, suite[name])
        status = _write_variant(
            args.out,
            name,
            args.size,
            suite[name],
            count=args.count,
            seed=args.seed,
            capacity=args.capacity,
            infeasibility=infeasibility,
            force=args.force,
        )
        generated += status == "generated"
        verified += status == "verified"
        print(
            "GENERATE",
            f"{index}/{len(BENCHMARK_VARIANTS)}",
            f"variant={name}",
            f"status={status}",
            f"infeasible={infeasibility['instances']}/{args.count}",
            flush=True,
        )
    print(
        "GENERATE_SUMMARY",
        f"variants={len(suite)}",
        f"size={args.size}",
        f"count={args.count}",
        f"generated={generated}",
        f"verified={verified}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
