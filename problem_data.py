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

# VRPTW activates only the time-window resource while allowing depot-separated
# routes. CVRPTW and OCVRPTW then teach capacity/TW and open-route/TW
# interactions in later curriculum phases. The md* entries put depot_count > 1
# in-distribution (previously only 0/1 was ever generated), including the
# open × multi-depot combination that was pure zero-shot before.
#
# Every FieldChannel must fire in training: capacity (cvrp*), time_windows
# (*tw), tour_limit (op), pickup_delivery (pdtsp), prize_quota (pctsp) were
# already covered, but route_limit and backhaul_order had no training variant at
# all -- their learned field channel + multiplier received zero gradient and
# emitted noise on any *l / *bp instance at inference. cvrpl and cvrpbp are the
# base problems that activate those two channels (both need capacity, so their
# minimal form is two-resource). Unlike URS, which leaves route_limit/backhaul
# to hand-coded stepwise feasibility masks, PRISM learns the channel geometry;
# that only works if the channel is exercised, so it must be trained rather than
# heuristically special-cased. Each of the two channels is given a closed and an
# open training context (cvrpl/ocvrpl, cvrpbp/ocvrpbp) so every field channel
# appears in >=2 training problems -- open routes change the route_limit return
# term and the backhaul-ordering regime, so a single closed base would tie the
# channel to one context. For the same reason pickup_delivery is trained not
# only as pdtsp (single-route, capacity-free) but composed with capacity and
# multi-route in a closed and an open base (cvrp/opdcvrp), so pd never has to
# interact with the capacity channel for the first time at inference on any
# pdcvrp* benchmark instance.
TRAIN_VARIANTS = _sort_variants(
    (
        "atsp",
        "acvrp",
        "tsp",
        "vrptw",
        "op",
        "pctsp",
        "cvrp",
        "cvrpb",
        "cvrpl",
        "cvrpbp",
        "ocvrpl",
        "ocvrpbp",
        "cvrptw",
        "ocvrp",
        "ocvrptw",
        "pdtsp",
        "pdcvrp",
        "opdcvrp",
        "mdocvrp",
        "amdocvrp",
        "mdcvrptw",
        "mdocvrptw",
    )
)
ALL_VARIANTS = _sort_variants(BENCHMARK_VARIANTS + ["vrptw"])

# Fixed validation coverage across objective type, pickup-delivery, symmetry,
# depot count, and one/two/three-resource VRP compositions.  The order is
# deliberately interleaved so smaller --val-heldout slices remain diverse.
# mdcvrpbp keeps a held-out probe on the now-trained backhaul_order channel
# (base cvrpbp is in TRAIN_VARIANTS) so its zero-shot composition is measured;
# the route_limit channel stays probed by acvrpl / mdcvrpl / *ltw.
VALIDATION_HELDOUT_VARIANTS = (
    "aop",
    "aopdcvrp",
    "ocvrpb",
    "acvrpb",
    "mdcvrp",
    "amdcvrp",
    "spctsp",
    "apdcvrp",
    "mdcvrpbp",
    "acvrpl",
    "mdcvrpl",
    "amdcvrpl",
    "cvrpltw",
    "acvrpltw",
    "mdcvrpltw",
    "amdcvrpltw",
)
assert len(set(VALIDATION_HELDOUT_VARIANTS)) == len(
    VALIDATION_HELDOUT_VARIANTS
)
assert set(VALIDATION_HELDOUT_VARIANTS) <= (
    set(BENCHMARK_VARIANTS) - set(TRAIN_VARIANTS)
)


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
    # EVRP = capacitated VRP whose battery resource enters through the resource
    # algebra (see decoder_problem), never as a CONSTRAINT_VOCAB entry -- this is
    # the zero-shot unseen-resource probe, so the schema stays a plain CVRP.
    is_evrp = name == "evrp"
    has_capacity = "cvrp" in name or is_evrp
    is_vrp = has_capacity or name == "vrptw"
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
    if name == "evrp":
        # Battery declared purely through the resource algebra: an edge-consumed,
        # depot/charger-replenished accumulator bounded at zero. No neural
        # parameter is battery-specific; the frozen model must interpret it from
        # these operational primitives alone (edge increment + node-event reset).
        charger = _first(data["charger"]).astype(np.float32)
        battery_range = float(_first(data["battery_range"]))
        problem["node_attributes"] = {"charger": charger}
        problem["resources"] = [
            {
                "name": "battery",
                "operator": "affine_accumulator",
                "scope": "route",
                "direction": "forward",
                "initial": battery_range,
                "scale": battery_range,
                "increment": {"edge_attribute": "distance", "coefficient": -1.0},
                "reset": {
                    "value": battery_range,
                    "at_depot": True,
                    "node_attribute": "charger",
                },
                "bounds": [{"lower": 0.0, "check": "transition"}],
            }
        ]
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


def _generated_vrptw(size: int) -> dict:
    """Generate capacity-free VRPTW with individually serviceable customers."""
    if size < 1:
        raise ValueError("vrptw requires at least one customer")
    node_count = size + 1
    xy = torch.rand(1, node_count, 2)
    travel = torch.linalg.vector_norm(xy[:, 1:] - xy[:, :1], dim=-1)
    service = torch.full((1, size), 0.2)
    # 3.2 exceeds the worst unit-square out-and-back distance plus service.
    # Every customer can therefore be served alone, while sampled windows still
    # determine which customers can profitably share a route.
    horizon = 3.2
    earliest = travel
    latest = horizon - travel - service
    center = earliest + (latest - earliest) * torch.rand(1, size)
    half_width = 0.1 + (horizon / 3 - 0.1) * torch.rand(1, size)
    start = torch.clamp(center - half_width, min=0.0)
    end = torch.minimum(center + half_width, latest)
    return decoder_problem(
        "vrptw",
        {
            "xy": xy,
            "service_time": torch.cat((torch.zeros(1, 1), service), dim=1),
            "tw_start": torch.cat((torch.zeros(1, 1), start), dim=1),
            "tw_end": torch.cat(
                (torch.full((1, 1), horizon), end), dim=1
            ),
        },
    )


def generated_problem(name: str, size: int, capacity: int = 50) -> dict:
    """Generate one training problem using PRISM-owned distributions."""
    name = name.lower()
    supported = set(TRAIN_VARIANTS) | {
        variant for variant in BENCHMARK_VARIANTS if "cvrp" in variant
    } | {"spctsp", "apctsp", "aspctsp", "aop", "apdtsp"}
    if name not in supported:
        raise NotImplementedError(f"no random generator for {name}")
    if name == "vrptw":
        return _generated_vrptw(size)

    if "pd" in name and size % 2:
        size += 1

    depot_count = 0 if name in {"tsp", "atsp"} else 3 if "md" in name else 1
    node_count = size + depot_count
    data: dict[str, torch.Tensor] = {}
    if name.startswith("a"):
        data["dist"] = _metric_distance(node_count).unsqueeze(0)
    else:
        data["xy"] = torch.rand(1, node_count, 2)

    if "cvrp" in name:
        demand = torch.randint(1, 10, (1, node_count)).float() / float(capacity)
        demand[:, :depot_count] = 0
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
            backhaul = torch.randperm(size)[:count] + depot_count
            demand[:, backhaul] *= -1
        data["demand"] = demand

        if "tw" in name:
            # Windows are sized against the nearest depot so multi-depot
            # instances stay individually serviceable from whichever depot a
            # route is anchored to (identical to depot 0 when depot_count == 1).
            if "dist" in data:
                matrix = data["dist"][0]
                travel_out = matrix[:depot_count, depot_count:].amin(
                    dim=0, keepdim=True
                )
                travel_back = matrix[depot_count:, :depot_count].amin(
                    dim=1
                ).unsqueeze(0)
            else:
                coordinates = data["xy"][0]
                radial = torch.linalg.vector_norm(
                    coordinates[depot_count:, None, :]
                    - coordinates[None, :depot_count, :],
                    dim=-1,
                )
                travel_out = travel_back = radial.amin(dim=1).unsqueeze(0)
            service = torch.full((1, size), 0.2)
            horizon = 1.0 if name.startswith("a") else 3.0
            earliest = travel_out
            latest = horizon - travel_back - service
            center = latest + (earliest - latest) * torch.rand(1, size)
            half_width = horizon / 3 + (service / 2 - horizon / 3) * torch.rand(1, size)
            start = torch.clamp(center - half_width, 0.0, horizon)
            end = torch.clamp(center + half_width, 0.0, horizon)
            depot_zeros = torch.zeros(1, depot_count)
            data["service_time"] = torch.cat((depot_zeros, service), dim=1)
            data["tw_start"] = torch.cat((depot_zeros, start), dim=1)
            data["tw_end"] = torch.cat(
                (torch.full((1, depot_count), horizon), end), dim=1
            )
        if "l" in name:
            if name.startswith("a"):
                # The tmat metric-closure distances shrink as node count grows,
                # so a fixed asymmetric budget is simultaneously infeasible at
                # small sizes and non-binding at large ones. Scale the limit to
                # the instance's own worst nearest-depot round trip so every
                # customer stays individually serviceable (feasible) while the
                # budget still forces multi-route structure (binding) -- the
                # same design as the symmetric 3.0 sitting just above the worst
                # unit-square round trip of 2*sqrt(2).
                matrix = data["dist"][0]
                out = matrix[:depot_count, depot_count:]
                back = matrix[depot_count:, :depot_count]
                worst_round_trip = (out.t() + back).amin(dim=1).amax()
                limit = float(worst_round_trip) * 1.1
            else:
                limit = 3.0
            data["route_limit"] = torch.full((1,), limit)

    if name in {"op", "aop"}:
        # Prize scales with remoteness from the depot: euclidean radius when
        # coordinates exist, otherwise the depot row of the asymmetric metric.
        if "dist" in data:
            radius = data["dist"][:, 0, :]
        else:
            xy = data["xy"]
            radius = torch.linalg.vector_norm(xy[:, :1] - xy, dim=-1)
        prize = (1 + (radius / radius.max(dim=-1, keepdim=True).values * 99).int()).float() / 100
        prize[:, 0] = 0
        data["prize"] = prize
    elif "pctsp" in name:
        prize = torch.cat((torch.zeros(1, 1), torch.rand(1, size) * 4 / size), dim=1)
        scale = {20: 2, 50: 3, 100: 4, 500: 9, 1000: 12}.get(size, max(2, round(size ** 0.4)))
        penalty = torch.cat(
            (torch.zeros(1, 1), torch.rand(1, size) * 3 * scale / size), dim=1
        )
        data.update(prize=prize, penalty=penalty)
    return decoder_problem(name, data)


def generate_vrptw_validation_data(
    size: int, count: int, seed: int = 0x54570000
) -> dict[str, torch.Tensor | int | str]:
    """Materialize a fixed batch from the current training distribution."""
    if count < 1:
        raise ValueError("VRPTW validation count must be positive")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        problems = [generated_problem("vrptw", size) for _ in range(count)]
    finally:
        torch.random.set_rng_state(state)
    return {
        "xy": torch.from_numpy(
            np.stack([problem["coordinates"] for problem in problems])
        ).float(),
        "service_time": torch.from_numpy(
            np.stack([problem["service_time"] for problem in problems])
        ).float(),
        "tw_start": torch.from_numpy(
            np.stack([problem["tw_start"] for problem in problems])
        ).float(),
        "tw_end": torch.from_numpy(
            np.stack([problem["tw_end"] for problem in problems])
        ).float(),
        "variant": "vrptw",
        "size": int(size),
        "count": int(count),
        "seed": int(seed),
        "distribution": "generated_problem:vrptw",
    }


def generate_evrp_data(
    size: int,
    count: int,
    seed: int = 0x45565250,
    capacity: int = 50,
    charger_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Generate battered EVRP instances in the neutral batched tensor schema.

    EVRP is never in TRAIN_VARIANTS: this is the zero-shot unseen-resource probe.
    The battery is declared through the resource algebra in ``decoder_problem``.
    ``battery_range`` is set to ``2.1 * max_i dist(depot, i)`` so the trivial
    depot->i->depot route is always feasible (guaranteeing a feasible complete
    solution) while multi-stop routes must respect the battery or charge en
    route -- chargers reset the battery when served, the depot resets it at every
    route start.
    """
    if size < 1:
        raise ValueError("evrp requires at least one customer")
    if count < 1:
        raise ValueError("evrp count must be positive")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        node_count = size + 1
        xy = torch.rand(count, node_count, 2)
        demand = torch.randint(1, 10, (count, node_count)).float() / float(capacity)
        demand[:, 0] = 0.0
        charger = torch.zeros(count, node_count)
        k = max(1, int(size * charger_fraction))
        for row in range(count):
            picked = torch.randperm(size)[:k] + 1
            charger[row, picked] = 1.0
        radius = torch.linalg.vector_norm(xy - xy[:, :1, :], dim=-1)
        battery_range = 2.1 * radius.max(dim=1).values
    finally:
        torch.random.set_rng_state(state)
    return {
        "xy": xy,
        "demand": demand,
        "charger": charger,
        "battery_range": battery_range,
    }


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


# Program-composition protocol: learned interaction sees at most pairs of
# resource rows, while evaluation contains strictly higher-order stacks. Keep
# this boundary explicit so future additions to TRAIN_VARIANTS cannot silently
# leak a 3+-resource program into training.
PROGRAM_TRAIN_MAX_RESOURCE_ORDER = 2
assert all(
    resource_count(name) <= PROGRAM_TRAIN_MAX_RESOURCE_ORDER
    for name in TRAIN_VARIANTS
)
HIGHER_ORDER_BENCHMARK_VARIANTS = tuple(
    name
    for name in BENCHMARK_VARIANTS
    if resource_count(name) > PROGRAM_TRAIN_MAX_RESOURCE_ORDER
)


@dataclass
class VariantCurriculum:
    variants: list[str]
    rng: random.Random
    seed: int = 0

    @classmethod
    def default(cls, seed: int) -> "VariantCurriculum":
        return cls(list(TRAIN_VARIANTS), random.Random(seed), seed)

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

    def schedule(
        self,
        epoch: int,
        epochs: int,
        steps: int,
        group_size: int,
        weights: dict[str, float] | None = None,
    ) -> list[str]:
        """Build a deterministic, balanced epoch schedule with distinct groups.

        With ``weights`` the balance is per-weight instead of uniform: a variant
        with twice the weight is scheduled roughly twice as often. The sort key
        becomes the weighted deficit ``count / weight`` so higher-weight
        variants tolerate more uses before being deprioritised, while the
        distinct-within-group and reproducibility guarantees are unchanged.
        ``weights=None`` reproduces the uniform schedule exactly.
        """
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if group_size < 1:
            raise ValueError("group_size must be positive")
        eligible = self.eligible(epoch, epochs) or self.variants
        if not eligible and steps:
            raise ValueError("cannot schedule variants from an empty curriculum")
        if group_size > len(eligible):
            raise ValueError(
                "group_size cannot exceed the number of eligible variants"
            )
        weight = {variant: 1.0 for variant in eligible}
        if weights is not None:
            for variant in eligible:
                value = weights.get(variant, 1.0)
                if value <= 0:
                    raise ValueError("sampling weights must be positive")
                weight[variant] = value

        # Epoch-local randomness makes resumes reproduce the same schedule
        # without depending on how many RNG calls an earlier epoch consumed.
        scheduler = random.Random(
            (int(self.seed) << 32) ^ (int(epoch) << 16) ^ int(epochs)
        )
        counts = {variant: 0 for variant in eligible}
        result = []
        while len(result) < steps:
            current_size = min(group_size, steps - len(result))
            candidates = list(eligible)
            scheduler.shuffle(candidates)
            candidates.sort(key=lambda variant: counts[variant] / weight[variant])
            selected = candidates[:current_size]
            result.extend(selected)
            for variant in selected:
                counts[variant] += 1
        return result


def channel_balanced_weights(variants: Iterable[str]) -> dict[str, float]:
    """Sampling weights that stop rare constraint channels being starved.

    Uniform sampling over variant *names* gives each field channel gradient in
    proportion to how many variants happen to carry it, so a singleton channel
    (tour_limit via op, prize_quota via pctsp) is starved next to capacity's
    many carriers. Each variant is weighted by the inverse coverage of its
    rarest channel -- the channel shared with the fewest other training
    variants -- which lifts the singletons without downweighting capacity (it
    rides along on every multi-resource variant). Weights are renormalised to
    mean 1 so the per-epoch step budget is unchanged; only the mix shifts.
    """
    names = list(variants)
    constraints = {name: problem_schema(name)["constraints"] for name in names}
    coverage: dict[str, int] = {}
    for channels in constraints.values():
        for channel in channels:
            coverage[channel] = coverage.get(channel, 0) + 1
    raw = {
        name: 1.0 / min((coverage[c] for c in channels), default=len(names))
        for name, channels in constraints.items()
    }
    total = sum(raw.values()) or 1.0
    scale = len(names) / total
    return {name: weight * scale for name, weight in raw.items()}


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
    solution_path: Path | None, start: int, count: int
) -> float | None:
    if solution_path is None:
        return None
    if solution_path.suffix == ".pt":
        saved = torch.load(solution_path, map_location="cpu", weights_only=False)
        values = torch.as_tensor(saved["cost"]).reshape(-1)
        values = values[start : start + count]
        if values.numel() != count:
            raise ValueError(
                f"requested {count} references from index {start}, "
                f"but {solution_path} contains only {values.numel()}"
            )
        reference = float(values.float().mean())
        if not np.isfinite(reference):
            raise ValueError(f"non-finite reference in {solution_path}")
        return reference
    if solution_path.suffix == ".pkl":
        with solution_path.open("rb") as source:
            saved = pickle.load(source)
        values = saved[start : start + count]
        if len(values) != count:
            raise ValueError(
                f"requested {count} references from index {start}, "
                f"but {solution_path} contains only {len(values)}"
            )
        costs = [
            value[0] if isinstance(value, (tuple, list)) else value
            for value in values
        ]
        reference = float(torch.tensor(costs, dtype=torch.float32).mean())
        if not np.isfinite(reference):
            raise ValueError(f"non-finite reference in {solution_path}")
        return reference
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


def _load_tensor_data(
    path: Path, name: str, start: int, count: int
) -> tuple[dict, float | None, bool]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    embedded = None
    embedded_is_aggregate = False
    if torch.is_tensor(saved):
        batch = saved[start : start + count].float()
        if name == "op":
            return (
                {"xy": batch[:, :, :2], "prize": batch[:, :, 2]},
                None,
                False,
            )
        if "pctsp" in name:
            return {
                "xy": batch[:, :, :2],
                "prize": batch[:, :, 2],
                "penalty": batch[:, :, -1],
            }, None, False
        return {"xy": batch}, None, False

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
                values = value[start : start + count].float()
                if values.shape[0] != count:
                    raise ValueError(
                        f"requested {count} embedded references from index "
                        f"{start}, but {path} contains only {values.shape[0]}"
                    )
                value = values.mean()
            else:
                embedded_is_aggregate = True
            embedded = float(value)
            break
    return result, embedded, embedded_is_aggregate


def load_saved_data(
    path: Path | str,
    name: str,
    count: int,
    *,
    start: int = 0,
    solution_path: Path | str | None = None,
    allow_aggregate_reference: bool = True,
) -> tuple[dict, float | None]:
    """Read benchmark files into PRISM's neutral batched tensor schema."""
    path = Path(path)
    if path.suffix == ".txt":
        data, embedded = _load_txt_tsp(path, start, count)
        embedded_is_aggregate = False
    elif path.suffix == ".pkl":
        data = _load_pickle_data(path, name, start, count)
        embedded = None
        embedded_is_aggregate = False
    elif path.suffix == ".pt":
        data, embedded, embedded_is_aggregate = _load_tensor_data(
            path, name, start, count
        )
    else:
        raise ValueError(f"unsupported dataset file: {path}")
    batch_sizes = {
        int(value.shape[0])
        for value in data.values()
        if torch.is_tensor(value) and value.ndim > 0
    }
    if batch_sizes != {count}:
        raise ValueError(
            f"requested {count} instances from index {start}, but {path} "
            f"loaded batch sizes {sorted(batch_sizes)}"
        )
    reference = _reference(
        None if solution_path is None else Path(solution_path),
        start,
        count,
    )
    selected_reference = reference
    if (
        selected_reference is None
        and embedded is not None
        and (allow_aggregate_reference or not embedded_is_aggregate)
    ):
        selected_reference = embedded
    if selected_reference is None and allow_aggregate_reference:
        selected_reference = _default_reference(name)
    if selected_reference is not None and not np.isfinite(selected_reference):
        raise ValueError(f"non-finite embedded reference in {path}")
    return data, selected_reference


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
            allow_aggregate_reference=False,
        )
        return decoder_problem(name, data), reference
