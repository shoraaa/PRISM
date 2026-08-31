#!/usr/bin/env python3
"""Convert CCL-MTLVRP .npz benchmarks into PRISM's saved-dataset layout.

CCL ships 48 MTVRP variants as flat .npz files with a RouteFinder-style feature
schema (separate linehaul/backhaul demand, a backhaul_class flag, an open_route
flag, depot-inclusive time windows).  PRISM's DatasetFinder expects one
directory per variant holding ``<variant><scale>.pt`` in the neutral tensor
schema of ``problem_data._load_tensor_data``.  This script performs that
translation once so ``test.py`` can evaluate on CCL data unmodified.

Name mapping (CCL -> PRISM):
    cvrp/ovrp/vrp  -> cvrp, with a leading ``o`` when the route is open
    b   (class 1)  -> bp     classic VRPB: linehauls precede backhauls
    mb  (class 2)  -> b      mixed backhaul, no ordering requirement
    l, tw, md      -> unchanged

Value mapping:
    locs                             -> xy            (depots stay first)
    demand_linehaul - demand_backhaul-> demand        (backhaul = negative)
    time_windows[..., 0] / [..., 1]  -> tw_start / tw_end
    service_time                     -> service_time
    distance_limit                   -> route_limit
    <size>_sol_<solver>.npz costs    -> optimal       (per-instance reference)

``speed`` and ``vehicle_capacity`` are 1.0 throughout CCL's released data and
are asserted rather than carried, since PRISM's schema fixes both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from problem_data import BENCHMARK_VARIANTS  # noqa: E402


def prism_name(ccl: str) -> str:
    """Translate one CCL variant directory name into PRISM's variant name."""
    rest = ccl
    multi_depot = rest.startswith("md")
    if multi_depot:
        rest = rest[2:]
    if rest.startswith("cvrp"):
        open_route, rest = False, rest[4:]
    elif rest.startswith("ovrp"):
        open_route, rest = True, rest[4:]
    elif rest.startswith("vrp"):
        open_route, rest = False, rest[3:]
    else:
        raise ValueError(f"unrecognised CCL variant: {ccl}")
    if rest.startswith("mb"):
        exclusive, rest = "b", rest[2:]
    elif rest.startswith("b"):
        exclusive, rest = "bp", rest[1:]
    else:
        exclusive = ""
    limited = rest.startswith("l")
    if limited:
        rest = rest[1:]
    windows = rest == "tw"
    if rest not in {"", "tw"}:
        raise ValueError(f"unparsed tail {rest!r} in CCL variant: {ccl}")
    return (
        ("md" if multi_depot else "")
        + ("o" if open_route else "")
        + "cvrp"
        + exclusive
        + ("l" if limited else "")
        + ("tw" if windows else "")
    )


def convert_instances(source: Path, name: str) -> dict[str, torch.Tensor]:
    """Read one CCL .npz into PRISM's neutral batched tensor schema."""
    raw = np.load(source)
    keys = set(raw.files)

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(raw[key])).float()

    if not np.allclose(raw["speed"], 1.0):
        raise ValueError(f"{source}: PRISM assumes unit speed")
    if not np.allclose(raw["vehicle_capacity"], 1.0):
        raise ValueError(f"{source}: PRISM assumes capacity-normalised demand")

    locations = tensor("locs")
    count, node_count, _ = locations.shape
    depot_count = int(np.asarray(raw["num_depots"]).reshape(-1)[0])
    expected_depots = 3 if name.startswith("md") else 1
    if depot_count != expected_depots:
        raise ValueError(
            f"{source}: {depot_count} depots but {name} expects {expected_depots}"
        )

    demand = tensor("demand_linehaul")
    if "demand_backhaul" in keys:
        # PRISM inherits URS's single signed demand vector: backhaul is negative.
        demand = demand - tensor("demand_backhaul")
    data = {
        "xy": locations,
        "demand": torch.cat((torch.zeros(count, depot_count), demand), dim=1),
    }

    if "time_windows" in keys:
        windows = tensor("time_windows")
        data["tw_start"] = windows[..., 0]
        data["tw_end"] = windows[..., 1]
        data["service_time"] = tensor("service_time")
    if "distance_limit" in keys:
        data["route_limit"] = tensor("distance_limit").reshape(count)

    for key, value in data.items():
        if value.shape[0] != count or (value.ndim > 1 and value.shape[1] != node_count):
            raise ValueError(f"{source}: field {key} has shape {tuple(value.shape)}")
    return data


def load_reference(directory: Path, scale: int, solver: str) -> torch.Tensor | None:
    """Read CCL's per-instance solver costs, stored there as negative rewards."""
    solvers = ("pyvrp", "ortools") if solver == "best" else (solver,)
    costs = []
    for candidate in solvers:
        path = directory / f"{scale}_sol_{candidate}.npz"
        if path.is_file():
            costs.append(torch.as_tensor(np.load(path)["costs"]).float().neg())
    if not costs:
        return None
    return torch.stack(costs).min(dim=0).values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ccl-data",
        type=Path,
        default=ROOT / "baselines" / "CCL-MTLVRP" / "CCL" / "data",
        help="CCL data root holding one directory per variant",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "datasets" / "ccl",
        help="destination dataset directory for --dataset-dir",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help=(
            "customer count encoded in the CCL filename (default: 100); "
            "generated CCL datasets may use arbitrary positive sizes"
        ),
    )
    parser.add_argument("--split", default="test", choices=("test", "val"))
    parser.add_argument(
        "--reference",
        default="pyvrp",
        choices=("pyvrp", "ortools", "best", "none"),
        help="solver costs to embed as the gap reference (CCL's BKS is pyvrp)",
    )
    args = parser.parse_args(argv)

    if args.scale < 1:
        parser.error("--scale must be positive")

    if not args.ccl_data.is_dir():
        parser.error(f"CCL data root does not exist: {args.ccl_data}")
    variants = sorted(path.name for path in args.ccl_data.iterdir() if path.is_dir())
    converted = []
    for ccl in variants:
        directory = args.ccl_data / ccl / args.split
        source = directory / f"{args.scale}.npz"
        if not source.is_file():
            print(f"skip {ccl}: no {source}")
            continue
        name = prism_name(ccl)
        if name not in BENCHMARK_VARIANTS:
            raise ValueError(f"{ccl} maps to {name}, which is not a PRISM benchmark")
        saved = convert_instances(source, name)
        if args.reference != "none":
            reference = load_reference(directory, args.scale, args.reference)
            if reference is not None:
                # problem_data._load_tensor_data reads "optimal" as the
                # per-instance reference, so no sidecar solution file is needed.
                saved["optimal"] = reference
        target = args.out / name
        target.mkdir(parents=True, exist_ok=True)
        torch.save(saved, target / f"{name}{args.scale}.pt")
        converted.append((ccl, name, saved["xy"].shape[0]))

    print(f"converted {len(converted)} variants into {args.out}")
    print(",".join(name for _, name, _ in converted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
