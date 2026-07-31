"""PRISM-owned routing problem definitions, generators, and dataset readers.

The native decoder consumes a small, explicit dictionary schema.  This module
owns the conversion to that schema; benchmark repositories are data sources,
not Python dependencies.
"""

from __future__ import annotations

import os
import pickle
import random
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = Path(
    os.environ.get("PRISM_DATASET_DIR", ROOT / "baselines" / "URS" / "dataset")
)


def _sort_variants(values: Iterable[str]) -> list[str]:
    return sorted(values, key=len)


def _vrp_mix() -> list[str]:
    variants = []
    for prefix in ("", "o"):
        for exclusive in ("", "b", "bp"):
            for enabled in product((False, True), repeat=2):
                optional = "".join(
                    token
                    for token, active in zip(("l", "tw"), enabled)
                    if active
                )
                variants.append(f"{prefix}cvrp{exclusive}{optional}")
    return _sort_variants(variants)


VRP_VARIANTS = _vrp_mix()
ASYMMETRIC_VRP_VARIANTS = [f"a{name}" for name in VRP_VARIANTS]
MULTI_DEPOT_VRP_VARIANTS = [f"md{name}" for name in VRP_VARIANTS]
ASYMMETRIC_MULTI_DEPOT_VRP_VARIANTS = [
    f"amd{name}" for name in VRP_VARIANTS
]
PICKUP_DELIVERY_VRP_VARIANTS = _sort_variants(
    ("pdcvrp", "apdcvrp", "opdcvrp", "aopdcvrp")
)
OTHER_BENCHMARK_VARIANTS = _sort_variants(
    (
        "tsp",
        "pctsp",
        "spctsp",
        "op",
        "pdtsp",
        "atsp",
        "apctsp",
        "aspctsp",
        "aop",
        "apdtsp",
    )
)
BENCHMARK_VARIANTS = _sort_variants(
    VRP_VARIANTS
    + ASYMMETRIC_VRP_VARIANTS
    + MULTI_DEPOT_VRP_VARIANTS
    + ASYMMETRIC_MULTI_DEPOT_VRP_VARIANTS
    + PICKUP_DELIVERY_VRP_VARIANTS
    + OTHER_BENCHMARK_VARIANTS
)
assert len(BENCHMARK_VARIANTS) == 110

TRAIN_VARIANTS = _sort_variants(
    (
        "atsp",
        "acvrp",
        "tsp",
        "op",
        "pctsp",
        "cvrp",
        "cvrpb",
        "cvrptw",
        "ocvrp",
        "ocvrptw",
        "pdtsp",
    )
)
ALL_VARIANTS = list(BENCHMARK_VARIANTS)


def problem_variants(
    collection: str = "all",
    *,
    included: str | Iterable[str] | None = None,
    excluded: str | Iterable[str] | None = None,
) -> list[str]:
    """Return a stable PRISM problem collection or a substring-filtered list."""
    collections = {
        "all": ALL_VARIANTS,
        "benchmark": BENCHMARK_VARIANTS,
        "train": TRAIN_VARIANTS,
        "vrp": VRP_VARIANTS,
        "asymmetric_vrp": ASYMMETRIC_VRP_VARIANTS,
        "multi_depot_vrp": MULTI_DEPOT_VRP_VARIANTS,
    }
    if collection not in collections:
        raise ValueError(
            f"unknown problem collection {collection!r}; choose from "
            + ", ".join(collections)
        )
    values = list(collections[collection])
    include = [included] if isinstance(included, str) else list(included or ())
    exclude = [excluded] if isinstance(excluded, str) else list(excluded or ())
    return [
        name
        for name in values
        if all(token in name for token in include)
        and not any(token in name for token in exclude)
    ]


def problem_schema(name: str) -> dict:
    """Describe routing semantics explicitly instead of relying on name parsing."""
    name = name.lower()
    is_pctsp = "pctsp" in name
    is_op = name in {"op", "aop"}
    has_capacity = "cvrp" in name
    is_vrp = has_capacity
    constraints = []
    if not is_pctsp and not is_op:
        constraints.append("visit_all")
    if has_capacity:
        constraints.append("capacity")
    if "bp" in name:
        constraints.append("backhaul_order")
    if "pd" in name:
        constraints.append("pickup_delivery")
    if "l" in name:
        constraints.append("route_limit")
    if "tw" in name:
        constraints.append("time_windows")
    if is_op:
        constraints.append("tour_limit")
    if is_pctsp:
        constraints.append("prize_quota")

    no_depot = name in {"tsp", "atsp"}
    return {
        "name": name,
        "constraints": constraints,
        "objective": (
            "prize"
            if is_op
            else "distance_plus_penalty"
            if is_pctsp
            else "distance"
        ),
        "depot_count": 0 if no_depot else 3 if "md" in name else 1,
        "multi_route": is_vrp,
        "open_route": "ocvrp" in name or "opdcvrp" in name,
        "capacity": 1.0,
        "prize_quota": 1.0,
    }


def _first(value):
    if torch.is_tensor(value):
        value = value[0].detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return value.astype(np.float32, copy=False)
    return value


def decoder_problem(name: str, data: dict) -> dict:
    """Convert one batched tensor dictionary to the native decoder schema."""
    problem = problem_schema(name)
    if "xy" in data:
        coordinates = _first(data["xy"])
        problem["coordinates"] = coordinates
        if coordinates.shape[0] <= 512:
            problem["distance"] = np.linalg.norm(
                coordinates[:, None] - coordinates[None, :], axis=-1
            ).astype(np.float32)
        node_count = coordinates.shape[0]
    else:
        distance = data.get("dist", data.get("distance"))
        if distance is None:
            raise ValueError("problem data requires xy or dist")
        problem["distance"] = _first(distance)
        node_count = problem["distance"].shape[0]
    for field in (
        "demand",
        "prize",
        "penalty",
        "tw_start",
        "tw_end",
        "service_time",
    ):
        if field in data:
            values = _first(data[field])
            if len(values) == node_count:
                problem[field] = values
    if "route_limit" in data:
        problem["route_limit"] = float(_first(data["route_limit"]))
    if name == "op":
        problem["tour_limit"] = 4.0
    elif name == "aop":
        problem["tour_limit"] = 1.0
    return problem


def _metric_distance(node_count: int) -> torch.Tensor:
    values = torch.randint(0, 1_000_000, (node_count, node_count))
    values[torch.arange(node_count), torch.arange(node_count)] = 0
    # Floyd-Warshall produces the same directed metric-closure distribution
    # expected by the asymmetric benchmark tasks without importing its code.
    for pivot in range(node_count):
        values = torch.minimum(
            values, values[:, pivot : pivot + 1] + values[pivot : pivot + 1, :]
        )
    return values.float() / 1_000_000


def generated_problem(name: str, size: int, capacity: int = 50) -> dict:
    """Generate one training problem using PRISM-owned distributions."""
    name = name.lower()
    supported = set(TRAIN_VARIANTS) | {
        variant
        for variant in BENCHMARK_VARIANTS
        if "cvrp" in variant and "md" not in variant
    }
    if name not in supported:
        raise NotImplementedError(f"no random generator for {name}")
    if name in {"pdtsp", "apdtsp"} and size % 2:
        size += 1

    depot_count = 0 if name in {"tsp", "atsp"} else 1
    node_count = size + depot_count
    data: dict[str, torch.Tensor] = {}
    if name.startswith("a"):
        data["dist"] = _metric_distance(node_count).unsqueeze(0)
    else:
        data["xy"] = torch.rand(1, node_count, 2)

    if "cvrp" in name:
        demand = torch.randint(1, 10, (1, node_count)).float() / float(capacity)
        demand[:, 0] = 0
        if "pd" in name:
            customer_count = node_count - 1
            if customer_count % 2:
                raise ValueError("pickup-delivery generation requires an even size")
            pickup = torch.randint(1, 10, (1, customer_count // 2)).float()
            demand = torch.cat(
                (torch.zeros(1, 1), pickup, -pickup), dim=1
            ) / float(capacity)
        elif "b" in name:
            count = max(1, int(size * 0.2))
            backhaul = torch.randperm(size)[:count] + 1
            demand[:, backhaul] *= -1
        data["demand"] = demand

        if "tw" in name:
            if "dist" in data:
                travel_out = data["dist"][:, 0, 1:]
                travel_back = data["dist"][:, 1:, 0]
            else:
                coordinates = data["xy"]
                travel_out = travel_back = torch.linalg.vector_norm(
                    coordinates[:, 1:] - coordinates[:, :1], dim=-1
                )
            service = torch.full((1, size), 0.2)
            horizon = 1.0 if name.startswith("a") else 3.0
            earliest = travel_out
            latest = horizon - travel_back - service
            center = latest + (earliest - latest) * torch.rand(1, size)
            half_width = horizon / 3 + (service / 2 - horizon / 3) * torch.rand(1, size)
            start = torch.clamp(center - half_width, 0.0, horizon)
            end = torch.clamp(center + half_width, 0.0, horizon)
            data["service_time"] = torch.cat((torch.zeros(1, 1), service), dim=1)
            data["tw_start"] = torch.cat((torch.zeros(1, 1), start), dim=1)
            data["tw_end"] = torch.cat((torch.full((1, 1), horizon), end), dim=1)
        if "l" in name:
            limit = 0.6 if name.startswith("a") else 3.0
            data["route_limit"] = torch.full((1,), limit)

    if name == "op":
        xy = data["xy"]
        radius = torch.linalg.vector_norm(xy[:, :1] - xy, dim=-1)
        prize = (1 + (radius / radius.max(dim=-1, keepdim=True).values * 99).int()).float() / 100
        prize[:, 0] = 0
        data["prize"] = prize
    elif name == "pctsp":
        prize = torch.cat((torch.zeros(1, 1), torch.rand(1, size) * 4 / size), dim=1)
        scale = {20: 2, 50: 3, 100: 4, 500: 9, 1000: 12}.get(size, max(2, round(size ** 0.4)))
        penalty = torch.cat(
            (torch.zeros(1, 1), torch.rand(1, size) * 3 * scale / size), dim=1
        )
        data.update(prize=prize, penalty=penalty)
    return decoder_problem(name, data)


def resource_count(name: str) -> int:
    schema = problem_schema(name)
    resource_constraints = {
        "capacity",
        "route_limit",
        "time_windows",
        "backhaul_order",
        "pickup_delivery",
        "tour_limit",
        "prize_quota",
    }
    return sum(value in resource_constraints for value in schema["constraints"])


@dataclass
class VariantCurriculum:
    variants: list[str]
    rng: random.Random

    @classmethod
    def default(cls, seed: int) -> "VariantCurriculum":
        return cls(list(TRAIN_VARIANTS), random.Random(seed))

    @property
    def held_out(self) -> list[str]:
        selected = set(self.variants)
        return [name for name in BENCHMARK_VARIANTS if name not in selected]

    def eligible(self, epoch: int, epochs: int) -> list[str]:
        progress = (epoch + 1) / max(epochs, 1)
        maximum = 1 if progress <= 1 / 3 else 2 if progress <= 2 / 3 else 7
        return [name for name in self.variants if resource_count(name) <= maximum]

    def sample(self, epoch: int, epochs: int) -> str:
        eligible = self.eligible(epoch, epochs)
        return self.rng.choice(eligible or self.variants)


_ORACLE_KEYWORDS = ("compass", "hgs", "ils", "lkh", "ortools", "pyvrp")


def _oracle_for(name: str) -> str | None:
    if name.startswith("a"):
        if name == "atsp":
            return "lkh"
        if name == "acvrp":
            return "pyvrp"
        if name == "aspctsp":
            return None
        return "ortools"
    if name.startswith("md"):
        return "pyvrp"
    if name in {"cvrpl", "ocvrp"}:
        return "lkh"
    if name == "cvrptw":
        return "pyvrp"
    if name == "cvrp":
        return "hgs"
    if "pd" in name and "cvrp" in name:
        return "ortools"
    if name in {"pctsp", "spctsp"}:
        return "ils"
    if name in {"tsp", "pdtsp"}:
        return "lkh"
    if name == "op":
        return "compass"
    if "cvrp" in name:
        return "ortools"
    return None


class DatasetFinder:
    """Locate benchmark data and optional reference files deterministically."""

    def __init__(self, data_dir: Path | str = DEFAULT_DATASET_DIR):
        self.data_dir = Path(data_dir)

    @staticmethod
    def _matches_scale(filename: str, name: str, size: int) -> bool:
        return re.search(
            rf"{re.escape(name)}[_-]*{int(size)}(?!\d)", filename.lower()
        ) is not None

    @staticmethod
    def _solution_rank(path: Path) -> tuple[int, int, str]:
        limits = [int(value) for value in re.findall(r"(\d+)s", path.name.lower())]
        return (0 if limits else 1, -max(limits) if limits else 0, path.name.lower())

    def get(self, name: str, size: int) -> dict | None:
        name = name.lower()
        directory = self.data_dir / name
        if not directory.is_dir():
            raise FileNotFoundError(f"problem directory does not exist: {directory}")
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".pkl", ".pt", ".txt"}
        )
        matching = [path for path in files if self._matches_scale(path.name, name, size)]
        data_files = [
            path
            for path in matching
            if not any(keyword in path.name.lower() for keyword in _ORACLE_KEYWORDS)
        ]
        if not data_files:
            return None
        data_path = data_files[0]
        oracle = _oracle_for(name)
        solutions = [
            path
            for path in matching
            if path != data_path and oracle is not None and oracle in path.name.lower()
        ]
        solution_path = min(solutions, key=self._solution_rank) if solutions else None
        return {
            "problem_name": name,
            "scale": int(size),
            "oracle": oracle,
            "data_file": data_path.name,
            "data_path": data_path,
            "solution_file": None if solution_path is None else solution_path.name,
            "solution_path": solution_path,
        }


def _reference(
    solution_path: Path | None, name: str, start: int, count: int
) -> float | None:
    if solution_path is None:
        return None
    if solution_path.suffix == ".pt":
        saved = torch.load(solution_path, map_location="cpu", weights_only=False)
        values = torch.as_tensor(saved["cost"]).reshape(-1)
        # Preserve the existing evaluation contract: only these asymmetric
        # prize variants used an instance slice; the other PT references were
        # reported as the full saved-set mean.
        if name in {"aop", "apctsp", "aspctsp"}:
            values = values[start : start + count]
        return float(values.float().mean())
    if solution_path.suffix == ".pkl":
        with solution_path.open("rb") as source:
            saved = pickle.load(source)
        # The legacy CVRP-family loader used total_episodes as the stop index,
        # while TSP/PDTSP used a count relative to start. Keep that behavior so
        # this source decoupling does not also change validation semantics.
        stop = start + count if name in {"tsp", "pdtsp"} else count
        values = saved[start:stop]
        costs = [value[0] if isinstance(value, (tuple, list)) else value for value in values]
        return float(torch.tensor(costs, dtype=torch.float32).mean())
    raise ValueError(f"unsupported reference file: {solution_path}")


def _default_reference(name: str) -> float | None:
    return {
        "op": 33.19,
        "pctsp": 5.98,
        "pdtsp": 9.428,
        "spctsp": 6.16,
    }.get(name)


def _load_txt_tsp(path: Path, start: int, count: int) -> tuple[dict, float]:
    coordinates = []
    tours = []
    for line in path.read_text().splitlines()[start : start + count]:
        fields = line.split()
        marker = fields.index("output")
        coordinates.append(
            [[float(fields[i]), float(fields[i + 1])] for i in range(0, marker, 2)]
        )
        tours.append([int(node) - 1 for node in fields[marker + 1 : -1]])
    xy = torch.tensor(coordinates, dtype=torch.float32)
    tour = torch.tensor(tours, dtype=torch.long)
    ordered = xy.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    cost = torch.linalg.vector_norm(ordered - ordered.roll(-1, 1), dim=-1).sum(1)
    return {"xy": xy}, float(cost.mean())


def _load_pickle_data(path: Path, name: str, start: int, count: int) -> dict:
    with path.open("rb") as source:
        rows = pickle.load(source)[start : start + count]
    if name == "pdtsp":
        depot = torch.tensor([row[0] for row in rows], dtype=torch.float32)
        if depot.ndim == 2:
            depot = depot[:, None, :]
        nodes = torch.tensor([row[1] for row in rows], dtype=torch.float32)
        return {"xy": torch.cat((depot, nodes), dim=1)}
    if name in {"op", "pctsp", "spctsp"}:
        depot = torch.tensor([row[0] for row in rows], dtype=torch.float32)
        if depot.ndim == 2:
            depot = depot[:, None, :]
        nodes = torch.tensor([row[1] for row in rows], dtype=torch.float32)
        zeros = torch.zeros(count, 1)
        result = {"xy": torch.cat((depot, nodes), dim=1)}
        if name == "op":
            result["prize"] = torch.cat(
                (zeros, torch.tensor([row[2] for row in rows], dtype=torch.float32)), 1
            )
        else:
            result["penalty"] = torch.cat(
                (zeros, torch.tensor([row[2] for row in rows], dtype=torch.float32)), 1
            )
            result["prize"] = torch.cat(
                (zeros, torch.tensor([row[3] for row in rows], dtype=torch.float32)), 1
            )
        return result

    depot = torch.tensor([row[0] for row in rows], dtype=torch.float32)
    if depot.ndim == 2:
        depot = depot[:, None, :]
    nodes = torch.tensor([row[1] for row in rows], dtype=torch.float32)
    capacity = float(rows[0][3])
    customer_demand = torch.tensor([row[2] for row in rows], dtype=torch.float32) / capacity
    result = {
        "xy": torch.cat((depot, nodes), dim=1),
        "demand": torch.cat((torch.zeros(count, depot.shape[1]), customer_demand), 1),
    }
    if "l" in name:
        result["route_limit"] = torch.tensor([row[4] for row in rows], dtype=torch.float32)
    if "tw" in name:
        result["service_time"] = torch.cat(
            (
                torch.zeros(count, 1),
                torch.tensor([row[-3] for row in rows], dtype=torch.float32),
            ),
            1,
        )
        result["tw_start"] = torch.cat(
            (
                torch.zeros(count, 1),
                torch.tensor([row[-2] for row in rows], dtype=torch.float32),
            ),
            1,
        )
        result["tw_end"] = torch.cat(
            (
                torch.full((count, 1), 3.0),
                torch.tensor([row[-1] for row in rows], dtype=torch.float32),
            ),
            1,
        )
    return result


def _load_tensor_data(path: Path, name: str, start: int, count: int) -> tuple[dict, float | None]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    embedded = None
    if torch.is_tensor(saved):
        batch = saved[start : start + count].float()
        if name == "op":
            return {"xy": batch[:, :, :2], "prize": batch[:, :, 2]}, None
        if "pctsp" in name:
            return {
                "xy": batch[:, :, :2],
                "prize": batch[:, :, 2],
                "penalty": batch[:, :, -1],
            }, None
        return {"xy": batch}, None

    def sliced(key: str) -> torch.Tensor:
        return torch.as_tensor(saved[key])[start : start + count].float()

    result = {}
    if "xy" in saved:
        result["xy"] = sliced("xy")
    elif "dist" in saved:
        result["dist"] = sliced("dist")
    elif "dist_matrix" in saved:
        result["dist"] = sliced("dist_matrix")
    for source, target in (
        ("demand", "demand"),
        ("prize", "prize"),
        ("penalty", "penalty"),
        ("real_prize", "prize"),
        ("route_limit", "route_limit"),
    ):
        if source in saved:
            result[target] = sliced(source)
    if "node_demand" in saved:
        result["demand"] = torch.cat((torch.zeros(count, 1), sliced("node_demand")), 1)
    node_count = result.get("xy", result.get("dist")).shape[1]
    for field in ("service_time", "tw_start", "tw_end"):
        if field not in saved:
            continue
        values = sliced(field)
        if values.shape[1] != node_count:
            depot_value = 0.0
            if field == "tw_end":
                if "tw" not in name:
                    depot_value = float("inf")
                else:
                    depot_value = 1.0 if name.startswith("a") else 3.0
            values = torch.cat((torch.full((count, 1), depot_value), values), 1)
        result[field] = values
    for key in ("optimal", "result"):
        if key in saved:
            value = saved[key]
            if torch.is_tensor(value) and value.ndim:
                value = value[start : start + count].float().mean()
            embedded = float(value)
            break
    return result, embedded


def load_saved_data(
    path: Path | str,
    name: str,
    count: int,
    *,
    start: int = 0,
    solution_path: Path | str | None = None,
) -> tuple[dict, float | None]:
    """Read benchmark files into PRISM's neutral batched tensor schema."""
    path = Path(path)
    if path.suffix == ".txt":
        data, embedded = _load_txt_tsp(path, start, count)
    elif path.suffix == ".pkl":
        data = _load_pickle_data(path, name, start, count)
        embedded = None
    elif path.suffix == ".pt":
        data, embedded = _load_tensor_data(path, name, start, count)
    else:
        raise ValueError(f"unsupported dataset file: {path}")
    reference = _reference(
        None if solution_path is None else Path(solution_path),
        name,
        start,
        count,
    )
    if reference is None and embedded is None:
        reference = _default_reference(name)
    return data, reference if reference is not None else embedded


class SavedProblems:
    def __init__(
        self, size: int, data_dir: Path | str | None = DEFAULT_DATASET_DIR
    ):
        self.size = size
        self.finder = DatasetFinder(DEFAULT_DATASET_DIR if data_dir is None else data_dir)

    def load(self, name: str, index: int = 0) -> tuple[dict, float | None]:
        paths = self.finder.get(name, self.size)
        if paths is None:
            raise FileNotFoundError(
                f"no saved data for variant={name} scale={self.size}"
            )
        data, reference = load_saved_data(
            paths["data_path"],
            name,
            1,
            start=index,
            solution_path=paths["solution_path"],
        )
        return decoder_problem(name, data), reference
