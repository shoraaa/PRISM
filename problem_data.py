"""PRISM-owned routing problem definitions, generators, and dataset readers.

The native decoder consumes a small, explicit dictionary schema.  This module
owns the conversion to that schema; benchmark repositories are data sources,
not Python dependencies.
"""

from __future__ import annotations

import os
import json
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
# PRISM's benchmark instances, one directory per variant holding every scale it
# has been generated or converted for (cvrp100_uniform.pkl, cvrp1000.pt, ...).
# The n=100 families came from URS and keep URS's per-instance oracle costs
# alongside them; larger scales are generated here and get their reference from
# whatever oracle the run solves with.
DEFAULT_DATASET_DIR = Path(
    os.environ.get("PRISM_DATASET_DIR", ROOT / "datasets" / "benchmarks")
)
VRPDB_BREAK_DURATION = 0.75
# Fraction of customers designated as rest areas (nodes where the mandatory
# break may be taken), mirroring EVRP's charger subset. A sparse rest-area set
# is what makes the driving-time resource genuinely bind the *distance*
# objective: a multi-stop route that would exceed the continuous-drive limit
# between rest-eligible nodes must detour to one, costing distance. With every
# node rest-eligible the reset is free and the problem collapses to CVRP.
VRPDB_REST_FRACTION = 0.15
# Continuous-driving limit as a multiple of the instance radius, mirroring
# EVRP's ``battery_range = 2.1 * max_radius``. Any value > 2 keeps every
# depot->i->depot singleton feasible without a break (round trip 2*radius <
# multiplier*max_radius), so a feasible all-singletons solution always exists,
# while multi-stop routes accumulate past the limit between sparse rest areas
# and must detour. A fixed wall-clock hours cap (e.g. 4.5) cannot both bind and
# stay singleton-feasible on a random-depot map, which is why the limit scales
# with the instance the same way the EVRP battery range does.
VRPDB_RANGE_MULTIPLIER = 2.1


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

# The 48 variants released by CCL-MTLVRP (ICLR 2026).  Its MTVRP grid is exactly
# PRISM's symmetric single- and multi-depot VRP families -- open x backhaul x
# route-limit x time-window -- and excludes the asymmetric, pickup-delivery,
# TSP, OP, and PCTSP benchmarks entirely.  Convert its .npz releases into a
# PRISM dataset directory with scripts/ccl_to_prism.py.
CCL_VARIANTS = _sort_variants(VRP_VARIANTS + MULTI_DEPOT_VRP_VARIANTS)
assert len(CCL_VARIANTS) == 48
assert set(CCL_VARIANTS) <= set(BENCHMARK_VARIANTS)

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
# Non-routing packing/selection classes. They reuse the resource algebra for
# hard feasibility but carry no geometry: every pairwise distance is equal, so
# they run complete candidate graphs and are deliberately excluded from
# BENCHMARK_VARIANTS (which the 110-variant routing results are keyed to) and
# from any locality or scale-transfer claim. Instance distributions follow
# DeepACO so its published numbers are a directly comparable reference.
PACKING_VARIANTS = _sort_variants(("bpp", "mkp"))

# Non-routing sequencing classes. Unlike the packing pair these carry a real
# (asymmetric, non-metric) distance matrix, so they keep the ordinary K-nearest
# neighbourhood and the geometric locality argument applies to them unchanged.
SEQUENCING_VARIANTS = _sort_variants(("sop",))
NON_ROUTING_VARIANTS = _sort_variants(PACKING_VARIANTS + SEQUENCING_VARIANTS)

# DeepACO's SOP generator constant: the probability that any admissible ordered
# pair is declared a precedence relation before transitive closure.
SOP_PRECEDENCE_DENSITY = 0.2

# DeepACO's generator constants, reproduced so instances are drawn from the same
# distribution rather than an approximation of it.
BPP_CAPACITY = 150
BPP_DEMAND_LOW = 20
BPP_DEMAND_HIGH = 100
MKP_DIMENSIONS = 5

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
        "ccl": CCL_VARIANTS,
        "packing": PACKING_VARIANTS,
        "sequencing": SEQUENCING_VARIANTS,
        "non_routing": NON_ROUTING_VARIANTS,
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


def _packing_schema(name: str) -> dict:
    """Schemas for the non-routing packing/selection classes.

    These are resolved before the routing name parsing because their names
    collide with it: "bpp" contains "bp", which would otherwise declare a
    backhaul ordering, and neither class is a "cvrp" so neither would receive
    the capacity it is entirely about.
    """
    if name == "bpp":
        # Bin packing is capacitated routing with the bin as the route: items
        # are the customers, a bin is a depot-to-depot trip, and the bin
        # capacity is the vehicle capacity. The objective is the number of bins,
        # charged as a fixed cost on each depot-out arc (see decoder_problem) --
        # the standard fixed-charge formulation, which keeps bin count exactly
        # equal to the accumulated distance and therefore incremental under
        # every existing search operator.
        return {
            "name": name,
            "constraints": ["visit_all", "capacity"],
            "objective": "distance",
            "depot_count": 1,
            "multi_route": True,
            "open_route": False,
            "capacity": 1.0,
            "prize_quota": 1.0,
        }
    if name == "mkp":
        # Multidimensional knapsack is single-route prize collection under m
        # simultaneous capacity rows. Nothing here is routing: distances are
        # uniformly zero and the visit order is immaterial, so the only thing
        # deciding the solution is which items the resource rows still admit.
        return {
            "name": name,
            "constraints": [],
            "objective": {
                "name": "prize",
                "distance_coeff": 0.0,
                "visit_coeff": 1.0,
                "miss_coeff": 0.0,
                "sense": -1.0,
            },
            "depot_count": 1,
            "multi_route": False,
            "open_route": False,
            "capacity": 1.0,
            "prize_quota": 0.0,
        }
    raise ValueError(f"{name!r} is not a packing variant")


def _sequencing_schema(name: str) -> dict:
    """Schemas for the non-routing sequencing classes."""
    if name == "sop":
        # The sequential ordering problem is an open asymmetric Hamiltonian path
        # from node 0 under a precedence DAG. Both halves are trained channels:
        # asymmetric distances (atsp, acvrp) and precedence (pdtsp, pdcvrp), so
        # a routing checkpoint reaches it without retraining. The relation
        # itself is declared in decoder_problem.
        return {
            "name": name,
            "constraints": ["visit_all"],
            "objective": "distance",
            "depot_count": 1,
            "multi_route": False,
            "open_route": True,
            "capacity": 1.0,
            "prize_quota": 1.0,
        }
    raise ValueError(f"{name!r} is not a sequencing variant")


def problem_schema(name: str) -> dict:
    """Describe routing semantics explicitly instead of relying on name parsing."""
    name = name.lower()
    if name in PACKING_VARIANTS:
        return _packing_schema(name)
    if name in SEQUENCING_VARIANTS:
        return _sequencing_schema(name)
    is_pctsp = "pctsp" in name
    is_op = name in {"op", "aop"}
    # EVRP = capacitated VRP whose battery resource enters through the resource
    # algebra (see decoder_problem), never as a named constraint -- this is
    # the zero-shot unseen-resource probe, so the schema stays a plain CVRP.
    # "evrp" may appear with a prefix (aevrp) or suffix (evrpl, evrptw), so match
    # the battery family anywhere in the name rather than only as a prefix.
    is_evrp = "evrp" in name
    is_vrpdb = name in {"vrpdb", "vrpdbtw"}
    has_capacity = "cvrp" in name or is_evrp or is_vrpdb
    is_vrp = has_capacity or name == "vrptw"
    constraints = []
    if not is_pctsp and not is_op:
        constraints.append("visit_all")
    if has_capacity:
        constraints.append("capacity")
    if "bp" in name:
        constraints.append("backhaul_order")
    if "pd" in name and not is_vrpdb:
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


def _packing_decoder_problem(name: str, data: dict) -> dict:
    """Materialize a packing/selection instance for the native decoder."""
    problem = problem_schema(name)
    if name == "bpp":
        demand = _first(data["demand"]).astype(np.float32)
        node_count = demand.shape[0]
        problem["demand"] = demand
        # Fixed charge on every depot-out arc, zero everywhere else: the
        # accumulated distance of a solution is then exactly its bin count, so
        # the existing distance objective *is* the bin-count objective and every
        # incremental search operator scores it correctly with no special case.
        distance = np.zeros((node_count, node_count), dtype=np.float32)
        distance[0, 1:] = 1.0
        problem["distance"] = distance
        return problem
    if name == "mkp":
        prize = _first(data["prize"]).astype(np.float32)
        weight = _first(data["weight"]).astype(np.float32)
        node_count = prize.shape[0]
        if weight.shape[0] != node_count:
            raise ValueError("mkp weight rows must match the node count")
        budget = float(_first(data["budget"]))
        problem["prize"] = prize
        # Order is immaterial: the solution is a set, so no arc carries cost.
        problem["distance"] = np.zeros((node_count, node_count), dtype=np.float32)
        dimensions = weight.shape[1]
        problem["node_attributes"] = {
            f"weight_{axis}": weight[:, axis] for axis in range(dimensions)
        }
        # One declared row per knapsack dimension. They are structurally
        # identical and differ only in the per-node attribute they consume,
        # which is the point: m simultaneously binding capacities is the same
        # composition the routing benchmark exercises, with the geometry
        # removed.
        problem["resources"] = [
            {
                "name": f"knapsack_{axis}",
                "operator": "affine_accumulator",
                "scope": "route",
                "direction": "forward",
                "initial": 0.0,
                "scale": budget,
                "increment": {
                    "node_attribute": f"weight_{axis}",
                    "coefficient": 1.0,
                },
                "bounds": [{"upper": budget, "check": "transition"}],
            }
            for axis in range(dimensions)
        ]
        return problem
    raise ValueError(f"{name!r} is not a packing variant")


def candidate_limit(problem: dict, default: int) -> int:
    """Candidate-graph width for one problem.

    Geometric variants keep the fixed K-nearest neighbourhood, which is what
    lets a single model transfer across instance sizes. The packing classes have
    no geometry -- every pairwise distance is equal -- so a truncated
    neighbourhood would retain an arbitrary index-ordered subset and make the
    remaining items unreachable. They run the complete graph instead, and are
    correspondingly outside any locality or scale-transfer claim.
    """
    if problem.get("name") not in PACKING_VARIANTS:
        return default
    node_count = len(problem.get("demand", problem.get("prize", ())))
    return max(1, node_count - 1)


def _sequencing_decoder_problem(name: str, data: dict) -> dict:
    """Materialize a sequencing instance for the native decoder."""
    problem = problem_schema(name)
    if name != "sop":
        raise ValueError(f"{name!r} is not a sequencing variant")
    distance = _first(data["distance"]).astype(np.float32)
    node_count = distance.shape[0]
    problem["distance"] = distance
    # One row per node listing what it must wait for. The relations reaching
    # node 0 are dropped: it is the depot, the path starts there, so "everything
    # follows the start" is structural rather than a constraint to enforce.
    predecessors = [
        [int(value) for value in row if int(value) > 0]
        for row in _first(data["predecessors"])
    ]
    problem["resources"] = [
        {
            "name": "precedence",
            "operator": "precedence",
            "relation": "dag",
            "scope": "solution",
            "direction": "forward",
            "predecessors": predecessors,
        }
    ]
    return problem


def decoder_problem(name: str, data: dict) -> dict:
    """Convert one batched tensor dictionary to the native decoder schema."""
    if name.lower() in PACKING_VARIANTS:
        return _packing_decoder_problem(name.lower(), data)
    if name.lower() in SEQUENCING_VARIANTS:
        return _sequencing_decoder_problem(name.lower(), data)
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
    if "prize_quota" in data:
        problem["prize_quota"] = float(_first(data["prize_quota"]))
    if name == "op":
        problem["tour_limit"] = 4.0
    elif name == "aop":
        problem["tour_limit"] = 1.0
    if "evrp" in name:
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
                "bounds": [
                    {
                        "lower": 0.0,
                        "check": "transition",
                        # Battery can be replenished at a charger, so the depot
                        # projection is a construction stranding guard rather
                        # than a necessary condition for a closed route.
                        "horizon": "return_construction",
                    }
                ],
            }
        ]
    elif name in {"vrpdb", "vrpdbtw"}:
        # Continuous driving time is accumulated on each travel edge. A break
        # is an optional pre-transition reset at the current node; native
        # execution inserts the latest feasible break for a route, which is the
        # minimum-break schedule in this simplified no-time-window setting.
        break_allowed = _first(data["break_allowed"]).astype(np.float32)
        max_continuous_drive = float(_first(data["max_continuous_drive"]))
        break_duration = float(_first(data["break_duration"]))
        problem["node_attributes"] = {"break_allowed": break_allowed}
        problem["resources"] = [
            {
                "name": "continuous_driving_time",
                "operator": "affine_accumulator",
                "scope": "route",
                "direction": "forward",
                "initial": 0.0,
                "scale": max_continuous_drive,
                "increment": {
                    "edge_attribute": "distance",
                    "coefficient": 1.0,
                },
                "reset": {
                    "value": 0.0,
                    "at_depot": True,
                    "node_attribute": "break_allowed",
                    "optional_before_transition": True,
                    "duration": break_duration,
                },
                "bounds": [
                    {
                        "upper": max_continuous_drive,
                        "check": "transition",
                        # A break resets driving time at an interior node, so
                        # the depot projection guards construction only.
                        "horizon": "return_construction",
                    }
                ],
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


# Benchmark constants from the URS appendix ("Capacity", "Duration Limit",
# "Asymmetric"). The saved evaluation instances carry exactly these values, so
# the generator has to use them too or training and test see different problems.
BENCHMARK_CAPACITY = 50
PICKUP_DELIVERY_CAPACITY = 20
SYMMETRIC_ROUTE_LIMIT = 3.0
ASYMMETRIC_ROUTE_LIMIT = 0.6


def benchmark_capacity(name: str) -> int:
    """Vehicle capacity the benchmark defines for a variant.

    Pickup-delivery instances use 20 rather than 50: at 50 the capacity stops
    binding and the solution degenerates toward a PDTSP tour.
    """
    return (
        PICKUP_DELIVERY_CAPACITY if "pd" in name.lower() else BENCHMARK_CAPACITY
    )


def generated_problem(
    name: str,
    size: int,
    capacity: int | None = None,
    randomize_resource_program: bool = False,
) -> dict:
    """Generate one training problem using PRISM-owned distributions."""
    name = name.lower()
    if capacity is None:
        capacity = benchmark_capacity(name)
    supported = set(TRAIN_VARIANTS) | {
        variant for variant in BENCHMARK_VARIANTS if "cvrp" in variant
    } | {"spctsp", "apctsp", "aspctsp", "aop", "apdtsp"}
    if name not in supported:
        raise NotImplementedError(f"no random generator for {name}")
    if name == "vrptw":
        problem = _generated_vrptw(size)
        if randomize_resource_program:
            append_random_resource_program(problem)
        return problem

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
                # The benchmark fixes the asymmetric budget at 0.6, and the
                # saved evaluation instances carry exactly that, so training has
                # to use it or the model never sees the regime it is tested in.
                #
                # The scaled round trip stays as a floor. tmat metric-closure
                # distances shrink as node count grows, so at small sizes a
                # fixed budget can leave a customer individually unserviceable;
                # the floor keeps every singleton feasible there. At the
                # benchmark size it lands near 0.17 and never binds, so the
                # generated limit is 0.6. The cost is that the budget is looser
                # -- and less structure-forcing -- than a self-scaled one would
                # be, which is the benchmark's choice rather than ours.
                matrix = data["dist"][0]
                out = matrix[:depot_count, depot_count:]
                back = matrix[depot_count:, :depot_count]
                worst_round_trip = float((out.t() + back).amin(dim=1).amax())
                limit = max(ASYMMETRIC_ROUTE_LIMIT, worst_round_trip * 1.1)
            else:
                limit = SYMMETRIC_ROUTE_LIMIT
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
    problem = decoder_problem(name, data)
    if randomize_resource_program:
        append_random_resource_program(problem)
    return problem


def append_random_resource_program(problem: dict) -> None:
    """Append a feasible anonymous row sampled in final term coordinates.

    This is training support, not a benchmark feature.  The row is deliberately
    nameless in semantics and varies source locality, term count, coefficient,
    join orientation, and post-bound work while retaining a singleton-feasible
    bound.  It exercises program coordinates instead of merely recombining the
    fixed benchmark rows.
    """
    node_count = len(problem.get("demand", problem.get("coordinates", [])))
    if node_count == 0:
        node_count = int(np.asarray(problem["distance"]).shape[0])
    depots = int(problem.get("depot_count", 0))
    workload = torch.rand(node_count, dtype=torch.float32) * 0.05
    if depots:
        workload[:depots] = 0.0
    ready = torch.rand(node_count, dtype=torch.float32) * 0.15
    if depots:
        ready[:depots] = 0.0
    attributes = dict(problem.get("node_attributes", {}))
    attributes["program_workload"] = workload.numpy()
    attributes["program_ready"] = ready.numpy()
    problem["node_attributes"] = attributes

    distance_coeff = float(0.35 + 1.3 * torch.rand(()))
    workload_coeff = float(0.5 + 1.5 * torch.rand(()))
    tropical = bool(torch.rand(()) < 0.35)
    semiring = "max_plus" if tropical else "arithmetic"
    terms: list[dict] = [
        {
            "edge_attribute": "distance",
            "coefficient": distance_coeff,
            "op": "add",
            "phase": "before_bound",
        },
        {
            "node_attribute": "program_workload",
            "coefficient": workload_coeff,
            "op": "add",
            "phase": "before_bound",
        },
    ]
    if bool(torch.rand(()) < 0.5):
        terms.append(
            {
                "node_attribute": "program_workload",
                "coefficient": float(0.1 + 0.4 * torch.rand(())),
                "op": "add",
                "phase": "after_bound",
            }
        )
    if tropical:
        terms.append(
            {
                "node_attribute": "program_ready",
                "op": "join",
                "phase": "before_bound",
            }
        )
    if depots:
        terms.append(
            {
                "value": 0.0,
                "op": "assign",
                "phase": "after_bound",
                "when": "reset_arrival",
                "at_depot": True,
            }
        )

    distance = np.asarray(problem["distance"], dtype=np.float32)
    if depots:
        out = distance[:depots, depots:].min(axis=0)
        back = distance[depots:, :depots].min(axis=1)
        singleton = distance_coeff * (out + back)
        singleton += workload_coeff * np.asarray(workload[depots:])
        # Pairwise precedence may force two customers into one route. Four
        # singleton legs keeps those routes feasible while still making the
        # randomized row observable on longer routes.
        route_factor = 8.0 if "pd" in problem["name"] else 4.0
        upper = max(0.5, float(singleton.max()) * route_factor)
        scope = "route"
        horizon = "return_construction"
    else:
        upper = max(1.0, distance_coeff * float(distance.max()) * node_count)
        scope = "solution"
        horizon = "transition"
    problem.setdefault("resources", []).append(
        {
            "name": "anonymous_program",
            "operator": "affine_accumulator",
            "semiring": semiring,
            "scope": scope,
            "initial": 0.0,
            "scale": upper,
            "terms": terms,
            "bounds": [
                {
                    "upper": upper,
                    "check": "transition",
                    "horizon": horizon,
                }
            ],
        }
    )


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


def generate_sop_data(
    size: int,
    count: int,
    seed: int = 0x534F5021,
    density: float = SOP_PRECEDENCE_DENSITY,
) -> dict[str, torch.Tensor]:
    """Generate sequential-ordering instances in the batched tensor schema.

    Reproduces DeepACO's generator. The cost matrix is asymmetric and
    non-metric: entries are uniform on [0, 1) and every row but the first has
    the destination's processing cost added, so ``d(i, j)`` depends on both
    ends. The precedence set is built backwards over node indices and closed
    under transitivity, which means the identity order is always feasible --
    a property of this generator worth knowing when reading its numbers.

    ``size`` counts every node including node 0, which is the fixed start.
    """
    if size < 4:
        raise ValueError("sop requires at least four nodes")
    if count < 1:
        raise ValueError("sop count must be positive")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        distances = torch.rand(count, size, size)
        # Row 0 keeps the raw draw; every other row is offset by the processing
        # cost of the node it enters.
        processing = distances[:, 0, :].unsqueeze(1)
        distances[:, 1:, :] = distances[:, 1:, :] + processing
        index = torch.arange(size)
        distances[:, index, index] = 0.0
        width = 0
        relations: list[list[list[int]]] = []
        for instance in range(count):
            precede: list[set[int]] = [set() for _ in range(size - 1)]
            for i in range(size - 3, -1, -1):
                for j in range(i + 1, size - 1):
                    if float(torch.rand(())) > density:
                        continue
                    precede[i].add(j)
                    precede[i].update(precede[j])
            # precede[i] holds the successors of node i + 1; invert it so each
            # node carries what it must wait for, which is what admission reads.
            required: list[list[int]] = [[] for _ in range(size)]
            for i, successors in enumerate(precede):
                for j in successors:
                    required[j + 1].append(i + 1)
            for node in range(1, size):
                required[node].append(0)
            width = max(width, max(len(row) for row in required))
            relations.append(required)
    finally:
        torch.random.set_rng_state(state)
    # Ragged predecessor lists are padded with -1 so the batch is one tensor;
    # decoder_problem drops the padding along with the structural node-0 edges.
    padded = torch.full((count, size, width), -1, dtype=torch.long)
    for instance, required in enumerate(relations):
        for node, values in enumerate(required):
            if values:
                padded[instance, node, : len(values)] = torch.tensor(values)
    return {
        "distance": distances,
        "predecessors": padded,
    }


def generate_bpp_data(
    size: int,
    count: int,
    seed: int = 0x42505021,
    capacity: int = BPP_CAPACITY,
) -> dict[str, torch.Tensor]:
    """Generate bin-packing instances in the neutral batched tensor schema.

    Item sizes follow DeepACO's Falkenauer-style generator: integers drawn
    uniformly from [BPP_DEMAND_LOW, BPP_DEMAND_HIGH] against a bin of
    ``capacity``. Demands are stored already divided by the capacity, matching
    the convention every other capacitated variant here uses, so the declared
    bound is a plain 1.0.

    Node 0 is the depot and stands for "close this bin and open the next one";
    it carries zero demand, exactly as DeepACO's dummy node does.
    """
    if size < 1:
        raise ValueError("bpp requires at least one item")
    if count < 1:
        raise ValueError("bpp count must be positive")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        raw = torch.randint(
            BPP_DEMAND_LOW, BPP_DEMAND_HIGH + 1, (count, size)
        ).float()
        demand = torch.cat((torch.zeros(count, 1), raw), dim=1) / float(capacity)
    finally:
        torch.random.set_rng_state(state)
    return {
        "demand": demand,
        "item_size": torch.cat((torch.zeros(count, 1), raw), dim=1),
        "capacity": torch.full((count,), float(capacity)),
    }


def generate_mkp_data(
    size: int,
    count: int,
    seed: int = 0x4D4B5021,
    dimensions: int = MKP_DIMENSIONS,
) -> dict[str, torch.Tensor]:
    """Generate multidimensional knapsack instances in the batched schema.

    This reproduces DeepACO's "well-stated" generator: prizes and weights are
    uniform on [0, 1), and each dimension's budget is drawn uniformly between
    the largest single weight (so no item is individually infeasible) and the
    total weight (so the constraint is not vacuous). DeepACO then rescales the
    weights so every budget equals ``size // 2``; we keep that rescaling because
    it fixes the budget, and a fixed budget is what lets one declared bound
    serve every instance.

    Node 0 is the depot and carries zero prize and zero weight in every
    dimension, matching DeepACO's dummy node.
    """
    if size < 1:
        raise ValueError("mkp requires at least one item")
    if count < 1:
        raise ValueError("mkp count must be positive")
    if dimensions < 1:
        raise ValueError("mkp requires at least one dimension")
    budget = float(size // 2)
    if budget <= 0.0:
        raise ValueError("mkp requires size >= 2 for a positive budget")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        prize = torch.rand(count, size)
        weight = torch.rand(count, size, dimensions)
        # Budget per dimension, uniform on (max weight, total weight).
        highest = weight.amax(dim=1)
        total = weight.sum(dim=1)
        budgets = highest + (total - highest) * torch.rand(count, dimensions)
        weight = weight * budget / budgets.unsqueeze(1)
        zeros = torch.zeros(count, 1)
        prize = torch.cat((zeros, prize), dim=1)
        weight = torch.cat((torch.zeros(count, 1, dimensions), weight), dim=1)
    finally:
        torch.random.set_rng_state(state)
    return {
        "prize": prize,
        "weight": weight,
        "budget": torch.full((count,), budget),
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


def generate_evrptw_data(
    size: int,
    count: int,
    seed: int = 0x45565257,
    capacity: int = 50,
    charger_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Generate Electric VRPTW instances in the neutral batched tensor schema.

    EVRPTW composes the EVRP battery (the zero-shot resource declared through the
    algebra in ``decoder_problem``) with the two trained channels capacity and
    time_windows -- an unseen *composition* probe rather than an unseen resource.
    Windows follow the vrptw generator (``horizon = 3.2``, ``service = 0.2``) so
    every customer stays individually serviceable as a depot singleton on time,
    while ``battery_range = 2.1 * max_i dist(depot, i)`` keeps that same singleton
    battery-feasible -- the two guarantees compose, so a feasible complete
    solution always exists.
    """
    if size < 1:
        raise ValueError("evrptw requires at least one customer")
    if count < 1:
        raise ValueError("evrptw count must be positive")
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
        travel = radius[:, 1:]
        service = torch.full((count, size), 0.2)
        horizon = 3.2
        earliest = travel
        latest = horizon - travel - service
        center = earliest + (latest - earliest) * torch.rand(count, size)
        half_width = 0.1 + (horizon / 3 - 0.1) * torch.rand(count, size)
        start = torch.clamp(center - half_width, min=0.0)
        end = torch.minimum(center + half_width, latest)
        depot_col = torch.zeros(count, 1)
        tw_start = torch.cat((depot_col, start), dim=1)
        tw_end = torch.cat((torch.full((count, 1), horizon), end), dim=1)
        service_time = torch.cat((depot_col, service), dim=1)
    finally:
        torch.random.set_rng_state(state)
    return {
        "xy": xy,
        "demand": demand,
        "charger": charger,
        "battery_range": battery_range,
        "tw_start": tw_start,
        "tw_end": tw_end,
        "service_time": service_time,
    }


def generate_vrpdb_data(
    size: int,
    count: int,
    seed: int = 0x56525044,
    capacity: int = 50,
    range_multiplier: float = VRPDB_RANGE_MULTIPLIER,
    break_duration: float = VRPDB_BREAK_DURATION,
    rest_fraction: float = VRPDB_REST_FRACTION,
) -> dict[str, torch.Tensor]:
    """Generate simplified driver-break VRP instances.

    Continuous driving time accumulates with travel and must not exceed a
    per-instance limit ``range_multiplier * max_i dist(depot, i)`` before a
    mandatory break resets it to zero. Breaks may be taken only at a sparse
    subset of ``rest_fraction`` customers -- the "rest areas" -- and at the
    depot, exactly mirroring EVRP's charger subset and range. Because the limit
    exceeds twice the radius, every ``depot->i->depot`` singleton is feasible
    without a break, so a feasible all-singletons solution always exists; but a
    multi-stop route that accumulates past the limit between two rest-eligible
    nodes must detour to a rest area, so the resource genuinely binds the
    distance objective rather than collapsing to CVRP (which the all-nodes,
    fixed-cap version did). The resource is not part of ``TRAIN_VARIANTS`` or
    the 110 registry.
    """
    if size < 1:
        raise ValueError("vrpdb requires at least one customer")
    if count < 1:
        raise ValueError("vrpdb count must be positive")
    if range_multiplier <= 2.0:
        raise ValueError(
            "range_multiplier must exceed 2 so every singleton stays feasible"
        )
    if break_duration < 0.0:
        raise ValueError("break duration must be non-negative")
    if not 0.0 < rest_fraction <= 1.0:
        raise ValueError("rest_fraction must be in (0, 1]")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        node_count = size + 1
        xy = 3.0 * torch.rand(count, node_count, 2)
        demand = torch.randint(1, 10, (count, node_count)).float() / float(capacity)
        demand[:, 0] = 0.0
        # Rest areas: a sparse subset of customers (depot resets are handled by
        # the resource's at_depot flag, so index 0 stays 0 here, exactly as
        # EVRP leaves the depot out of its sampled charger set).
        break_allowed = torch.zeros(count, node_count)
        k = max(1, int(size * rest_fraction))
        for row in range(count):
            picked = torch.randperm(size)[:k] + 1
            break_allowed[row, picked] = 1.0
        # Per-instance continuous-driving limit, scaled to the radius so
        # singletons stay feasible while multi-stop routes bind (see EVRP).
        radius = torch.linalg.vector_norm(xy - xy[:, :1, :], dim=-1)
        maximum = range_multiplier * radius.max(dim=1).values
        duration = torch.full((count,), float(break_duration))
    finally:
        torch.random.set_rng_state(state)
    return {
        "xy": xy,
        "demand": demand,
        "break_allowed": break_allowed,
        "max_continuous_drive": maximum,
        "break_duration": duration,
    }


def generate_vrpdbtw_data(
    size: int,
    count: int,
    seed: int = 0x56524454,
    capacity: int = 50,
    range_multiplier: float = VRPDB_RANGE_MULTIPLIER,
    break_duration: float = VRPDB_BREAK_DURATION,
) -> dict[str, torch.Tensor]:
    """Generate driver-break VRP with deadline time windows.

    Customer windows start at zero, so waiting cannot hide a break or make an
    earlier-than-required break advantageous. Their sampled upper bounds still
    constrain route order, and the depot horizon includes the break duration
    whenever a singleton round trip would exceed the continuous-driving limit.
    Because the per-instance limit exceeds twice the radius (see
    ``generate_vrpdb_data``), no singleton actually needs a mid-route break, so
    the depot horizon reduces to the plain round trip while multi-stop routes
    still exercise the unseen driver resource composed with the trained
    time-window channel.
    """
    data = generate_vrpdb_data(
        size,
        count,
        seed=seed,
        capacity=capacity,
        range_multiplier=range_multiplier,
        break_duration=break_duration,
    )
    state = torch.random.get_rng_state()
    torch.manual_seed(seed ^ 0x54574E44)
    try:
        xy = data["xy"]
        radius = torch.linalg.vector_norm(xy[:, 1:] - xy[:, :1], dim=-1)
        service = torch.full((count, size), 0.2)
        limit = data["max_continuous_drive"][:, None]
        return_break = (2.0 * radius > limit).float()
        return_break *= float(break_duration)
        singleton_finish = 2.0 * radius + service + return_break
        horizon = singleton_finish.max(dim=1).values + 0.5
        latest = horizon[:, None] - service - radius - return_break
        width = torch.clamp(latest - radius, min=0.0)
        deadline = radius + width * (0.35 + 0.65 * torch.rand(count, size))
        depot_zeros = torch.zeros(count, 1)
        data["service_time"] = torch.cat((depot_zeros, service), dim=1)
        data["tw_start"] = torch.zeros(count, size + 1)
        data["tw_end"] = torch.cat((horizon[:, None], deadline), dim=1)
    finally:
        torch.random.set_rng_state(state)
    return data


def generate_aevrp_data(
    size: int,
    count: int,
    seed: int = 0x41455650,
    capacity: int = 50,
    charger_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Generate Asymmetric EVRP instances in the neutral batched tensor schema.

    Battery consumption follows the *directed* metric-closure distance, so the
    return leg dist(i, depot) differs from the outbound dist(depot, i) -- this
    exercises the directional return-reachability guard. ``battery_range`` is set
    to ``1.1 * max_i (dist(depot, i) + dist(i, depot))`` so every customer stays
    feasible as a depot singleton under the asymmetric round trip. Instances
    carry an explicit ``dist`` matrix (no coordinates), matching how the
    asymmetric benchmarks enter the decoder.
    """
    if size < 1:
        raise ValueError("aevrp requires at least one customer")
    if count < 1:
        raise ValueError("aevrp count must be positive")
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        node_count = size + 1
        dist = torch.stack([_metric_distance(node_count) for _ in range(count)])
        demand = torch.randint(1, 10, (count, node_count)).float() / float(capacity)
        demand[:, 0] = 0.0
        charger = torch.zeros(count, node_count)
        k = max(1, int(size * charger_fraction))
        for row in range(count):
            picked = torch.randperm(size)[:k] + 1
            charger[row, picked] = 1.0
        round_trip = dist[:, 0, :] + dist[:, :, 0]
        battery_range = 1.1 * round_trip.max(dim=1).values
    finally:
        torch.random.set_rng_state(state)
    return {
        "dist": dist,
        "demand": demand,
        "charger": charger,
        "battery_range": battery_range,
    }


def generate_evrpl_data(
    size: int,
    count: int,
    seed: int = 0x4556504C,
    capacity: int = 50,
    charger_fraction: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Generate duration-limited EVRP (EVRP-L) in the neutral batched schema.

    Battery + capacity + a per-route distance cap. Two distance-like resources
    coexist: the battery (resets at chargers/depot) and the route_limit (total
    route length, resets only at the depot). ``route_limit = 1.5 * battery_range``
    keeps the battery the *binding return* constraint -- so the return-reachability
    guard still prevents stranding -- while the route cap still binds any route
    that would otherwise recharge repeatedly into a long shift.
    """
    if size < 1:
        raise ValueError("evrpl requires at least one customer")
    if count < 1:
        raise ValueError("evrpl count must be positive")
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
        route_limit = 1.5 * battery_range
    finally:
        torch.random.set_rng_state(state)
    return {
        "xy": xy,
        "demand": demand,
        "charger": charger,
        "battery_range": battery_range,
        "route_limit": route_limit,
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

    @staticmethod
    def _data_rank(path: Path) -> tuple[int, str]:
        """Prefer the materialized PRISM schema over retained source formats."""
        return (0 if path.name.lower().endswith("_prism.pt") else 1, path.name.lower())

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
        data_path = min(data_files, key=self._data_rank)
        oracle = _oracle_for(name)
        solutions = [
            path
            for path in matching
            if path != data_path and oracle is not None and oracle in path.name.lower()
        ]
        solution_path = min(solutions, key=self._solution_rank) if solutions else None
        metadata_path = data_path.with_suffix(".json")
        if metadata_path.is_file():
            with metadata_path.open() as source:
                metadata = json.load(source)
            if "solution_file" in metadata:
                declared = metadata["solution_file"]
                solution_path = None if declared is None else directory / declared
                if solution_path is not None and not solution_path.is_file():
                    raise FileNotFoundError(
                        f"declared solution file does not exist: {solution_path}"
                    )
        return {
            "problem_name": name,
            "scale": int(size),
            "oracle": oracle,
            "data_file": data_path.name,
            "data_path": data_path,
            "solution_file": None if solution_path is None else solution_path.name,
            "solution_path": solution_path,
            "metadata_path": metadata_path if metadata_path.is_file() else None,
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
