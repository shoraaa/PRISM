#!/usr/bin/env python3
"""Convert PRISM's saved benchmarks into CCL-MTLVRP's .npz release format.

This is the inverse of scripts/ccl_to_prism.py, so CCL's pretrained RouteFinder
models can be evaluated on exactly the instances PRISM's test.py evaluates.
Instances are read through ``problem_data.load_saved_data`` -- the same call
test.py makes -- so both models provably see the same tensors.

Only the 48 variants in ``problem_data.CCL_VARIANTS`` are convertible.  CCL's
feature space is capacity x open x backhaul x distance-limit x time-window over
Euclidean coordinates, which cannot express PRISM's asymmetric (a*) families
(no coordinates), nor pickup-delivery, TSP, OP, or PCTSP.

Per-instance oracle costs from the dataset's solution files are written
alongside as ``<scale>_sol_pyvrp.npz``, the filename MTVRPEnv/MTDVRPEnv look
for, so CCL reports a gap against PRISM's reference rather than its dummy
69420.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from problem_data import (  # noqa: E402
    CCL_VARIANTS,
    DEFAULT_DATASET_DIR,
    DatasetFinder,
    load_saved_data,
)


def ccl_name(name: str) -> str:
    """Translate a PRISM variant name into CCL's dataset directory name."""
    rest = name
    multi_depot = rest.startswith("md")
    if multi_depot:
        rest = rest[2:]
    open_route = rest.startswith("o")
    if open_route:
        rest = rest[1:]
    if not rest.startswith("cvrp"):
        raise ValueError(f"{name} is not a CCL-expressible VRP variant")
    rest = rest[4:]
    if rest.startswith("bp"):
        backhaul, rest = "b", rest[2:]
    elif rest.startswith("b"):
        backhaul, rest = "mb", rest[1:]
    else:
        backhaul = ""
    limited = rest.startswith("l")
    if limited:
        rest = rest[1:]
    windows = rest == "tw"
    if rest not in {"", "tw"}:
        raise ValueError(f"unparsed tail {rest!r} in PRISM variant: {name}")
    tokens = backhaul + ("l" if limited else "") + ("tw" if windows else "")
    stem = "ovrp" if open_route else ("vrp" if tokens else "cvrp")
    return ("md" if multi_depot else "") + stem + tokens


def oracle_costs(path: Path, count: int) -> np.ndarray:
    """Read the dataset's per-instance reference objectives, as _reference() does."""
    if path.suffix == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=False)
        values = torch.as_tensor(saved["cost"]).reshape(-1)[:count]
    elif path.suffix == ".pkl":
        with path.open("rb") as source:
            rows = pickle.load(source)[:count]
        values = torch.tensor(
            [row[0] if isinstance(row, (tuple, list)) else row for row in rows],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"unsupported reference file: {path}")
    if values.numel() != count:
        raise ValueError(f"{path}: expected {count} references, got {values.numel()}")
    return values.float().numpy()


def to_ccl_arrays(name: str, data: dict, count: int) -> dict[str, np.ndarray]:
    """Convert PRISM's neutral tensor schema into CCL's .npz key set."""
    depot_count = 3 if name.startswith("md") else 1
    locations = data["xy"].numpy().astype(np.float32)
    node_count = locations.shape[1]

    demand = data["demand"][:, depot_count:].numpy().astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        "locs": locations,
        # CCL splits PRISM/URS's single signed demand vector back into two.
        "demand_linehaul": np.clip(demand, 0.0, None),
        "vehicle_capacity": np.ones((count, 1), dtype=np.float32),
        "speed": np.ones((count, 1), dtype=np.float32),
        "num_depots": np.full((count, 1), depot_count, dtype=np.int32),
    }

    tail = name[2:] if name.startswith("md") else name
    tail = tail[1:] if tail.startswith("o") else tail
    tail = tail[4:]
    if tail.startswith("b"):
        arrays["demand_backhaul"] = np.clip(-demand, 0.0, None)
        # bp is classic VRPB (class 1); bare b is URS's mixed backhaul (class 2).
        arrays["backhaul_class"] = np.full(
            (count, 1), 1.0 if tail.startswith("bp") else 2.0, dtype=np.float32
        )
    elif (demand < 0).any():
        raise ValueError(f"{name}: negative demand in a variant without backhaul")

    if name.startswith("o") or name.startswith("mdo"):
        arrays["open_route"] = np.ones((count, 1), dtype=bool)

    if "tw" in name:
        windows = np.stack(
            (data["tw_start"].numpy(), data["tw_end"].numpy()), axis=-1
        ).astype(np.float32)
        arrays["time_windows"] = windows
        arrays["service_time"] = data["service_time"].numpy().astype(np.float32)

    if "l" in name:
        limits = data["route_limit"].reshape(count, 1).numpy().astype(np.float32)
        if not np.isfinite(limits).all():
            raise ValueError(f"{name}: non-finite route limit")
        arrays["distance_limit"] = limits

    for key, value in arrays.items():
        if value.shape[0] != count or (value.ndim > 1 and value.shape[1] not in
                                       (1, 2, node_count, node_count - depot_count)):
            raise ValueError(f"{name}: field {key} has shape {value.shape}")
    return arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="PRISM dataset directory to read (default: datasets/benchmarks)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "baselines" / "CCL-MTLVRP" / "urs-data",
        help="destination laid out as <variant>/test/<scale>.npz",
    )
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args(argv)

    finder = DatasetFinder(args.dataset_dir)
    single_depot, multi_depot = [], []
    for name in CCL_VARIANTS:
        paths = finder.get(name, args.scale)
        if paths is None:
            print(f"skip {name}: no size-{args.scale} data")
            continue
        data, _ = load_saved_data(
            paths["data_path"],
            name,
            args.count,
            solution_path=None,
            allow_aggregate_reference=False,
        )
        arrays = to_ccl_arrays(name, data, args.count)
        target = args.out / ccl_name(name) / "test"
        target.mkdir(parents=True, exist_ok=True)
        source = target / f"{args.scale}.npz"
        np.savez(source, **arrays)
        if paths["solution_path"] is not None:
            costs = oracle_costs(paths["solution_path"], args.count)
            np.savez(
                target / f"{args.scale}_sol_pyvrp.npz",
                costs=-np.abs(costs).astype(np.float32),
            )
        # test.py routes "m"-free paths, test_m.py takes the rest.
        bucket = multi_depot if "m" in ccl_name(name) else single_depot
        bucket.append(str(source))

    for label, paths in (("single_depot", single_depot), ("multi_depot", multi_depot)):
        listing = args.out / f"{label}_{args.scale}.txt"
        listing.write_text(",".join(sorted(paths, key=len)) + "\n")
        print(f"{label}: {len(paths)} variants -> {listing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
