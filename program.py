"""Compile problem declarations into name-free executable resource programs.

This module is the semantic frontend of the Python reference decoder.  It is
the only place that knows benchmark constraint names or their input fields.
The decoder receives only :class:`ResourceProgram` objects and executes every
one through the same state-transition machinery.

These program objects define exact behavior; they are not neural descriptors
and are never supplied to the v14 model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


EPS = 1.0e-6


def _array(value: Any, size: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.ndim > 1 and result.shape[0] == 1:
        result = result[0]
    if result.shape != (size,):
        raise ValueError(f"expected a vector of shape ({size},), got {result.shape}")
    return result


@dataclass(frozen=True)
class Objective:
    name: str = "distance"
    distance_coeff: float = 1.0
    visit_coeff: float = 0.0
    miss_coeff: float = 0.0
    distance_regularizer: float = 0.0
    sense: float = 1.0

    @classmethod
    def compile(cls, value: str | Mapping[str, Any]) -> "Objective":
        if isinstance(value, Mapping):
            return cls(
                name=str(value.get("name", "custom")),
                distance_coeff=float(value.get("distance_coeff", 1.0)),
                visit_coeff=float(value.get("visit_coeff", 0.0)),
                miss_coeff=float(value.get("miss_coeff", 0.0)),
                distance_regularizer=float(value.get("distance_regularizer", 0.0)),
                sense=float(value.get("sense", 1.0)),
            )
        table = {
            "distance": cls(),
            "prize": cls(
                name="prize",
                distance_coeff=0.0,
                visit_coeff=1.0,
                distance_regularizer=1.0e-3,
                sense=-1.0,
            ),
            "distance_plus_penalty": cls(
                name="distance_plus_penalty", miss_coeff=1.0
            ),
        }
        try:
            return table[str(value)]
        except KeyError as error:
            raise ValueError(f"unknown objective: {value}") from error

    @property
    def direction(self) -> str:
        return "maximize" if self.sense < 0.0 else "minimize"

    def report(
        self, distance: float, visit_value: float, omission_value: float
    ) -> float:
        return (
            self.distance_coeff * distance
            + self.visit_coeff * visit_value
            + self.miss_coeff * omission_value
        )


@dataclass(frozen=True)
class Term:
    source: str
    operation: str = "add"
    phase: str = "before_bound"
    trigger: str = "always"
    point: str = "to"
    coefficient: float = 1.0
    constant: float = 0.0
    values: np.ndarray | None = None
    at_depot: bool = False
    trigger_nodes: np.ndarray | None = None
    gate: str = "always"
    gate_values: np.ndarray | None = None
    gate_sign: float = 0.0


@dataclass(frozen=True)
class Bound:
    lower: float = -math.inf
    upper: float = math.inf
    lower_values: np.ndarray | None = None
    upper_values: np.ndarray | None = None
    check: str = "transition"
    horizon: str = "transition"

    def lower_at(self, node: int) -> float:
        return self.lower if self.lower_values is None else float(self.lower_values[node])

    def upper_at(self, node: int) -> float:
        return self.upper if self.upper_values is None else float(self.upper_values[node])


@dataclass(frozen=True)
class ResourceProgram:
    name: str
    operator: str
    scope: str = "route"
    semiring: str = "arithmetic"
    initial: float = 0.0
    scale: float = 1.0
    terms: tuple[Term, ...] = ()
    bound: Bound = Bound()
    depot_release: bool = False

    relation: str | None = None
    predecessor: np.ndarray | None = None
    successors: tuple[tuple[int, ...], ...] | None = None
    node_class: np.ndarray | None = None
    predecessors: tuple[tuple[int, ...], ...] | None = None
    relation_count: int = 0

    # An optional checkpoint/restore may emit elapsed wall-clock time for
    # another generic accumulator to consume as an execution event.
    restore_duration: float = 0.0


@dataclass(frozen=True)
class CompiledProblem:
    programs: tuple[ResourceProgram, ...]
    requires_all_visits: bool
    objective: Objective
    visit_values: np.ndarray
    omission_values: np.ndarray


def _pairwise_program(
    name: str, predecessor: np.ndarray, scope: str, node_count: int
) -> ResourceProgram:
    successors: list[list[int]] = [[] for _ in range(node_count)]
    for node, required in enumerate(predecessor):
        if required >= 0:
            successors[int(required)].append(node)
    count = int(np.count_nonzero(predecessor >= 0))
    return ResourceProgram(
        name=name,
        operator="precedence",
        scope=scope,
        relation="pairwise",
        predecessor=predecessor,
        successors=tuple(tuple(row) for row in successors),
        relation_count=count,
        scale=max(float(count), 1.0),
    )


def compile_problem(
    problem: Mapping[str, Any], node_count: int, depot_count: int
) -> CompiledProblem:
    """Lower named schema conveniences at the decoder boundary.

    Nothing returned by this function requires the executor to inspect a
    constraint name or a benchmark-specific field.
    """

    constraints = frozenset(str(value) for value in problem["constraints"])
    customer_count = node_count - depot_count
    demand = _array(problem.get("demand"), node_count, 0.0)
    prize = _array(problem.get("prize"), node_count, 0.0)
    result: list[ResourceProgram] = []
    capacity = float(problem.get("capacity", 1.0))

    if "capacity" in constraints:
        terms = [
            Term(
                source="node_attribute",
                values=demand,
                coefficient=-1.0,
            )
        ]
        if "backhaul_order" in constraints:
            terms.append(
                Term(
                    source="node_attribute",
                    operation="assign",
                    phase="before_bound",
                    trigger="reset_departure",
                    values=np.where(demand < -EPS, 0.0, capacity),
                    at_depot=True,
                )
            )
        terms.extend(
            (
                Term(
                    source="constant",
                    operation="assign",
                    phase="after_bound",
                    trigger="reset_arrival",
                    constant=capacity,
                    at_depot=True,
                    gate="remainder_default",
                    gate_values=demand,
                    gate_sign=1.0,
                ),
                Term(
                    source="constant",
                    operation="assign",
                    phase="after_bound",
                    trigger="reset_arrival",
                    constant=0.0,
                    at_depot=True,
                    gate="remainder_alternative",
                    gate_values=demand,
                    gate_sign=1.0,
                ),
            )
        )
        result.append(
            ResourceProgram(
                name="capacity",
                operator="accumulator",
                initial=capacity,
                scale=capacity,
                terms=tuple(terms),
                bound=Bound(lower=0.0, upper=capacity),
                depot_release=True,
            )
        )

    if "time_windows" in constraints:
        earliest = _array(problem.get("tw_start"), node_count, 0.0)
        latest = _array(problem.get("tw_end"), node_count, math.inf)
        service = _array(problem.get("service_time"), node_count, 0.0)
        finite_latest = latest[np.isfinite(latest)]
        scale = max(
            float(np.max(finite_latest)) if finite_latest.size else 1.0,
            EPS,
        )
        result.append(
            ResourceProgram(
                name="time_window",
                operator="accumulator",
                semiring="max_plus",
                initial=0.0,
                scale=scale,
                terms=(
                    Term(source="distance"),
                    Term(source="event_duration"),
                    Term(
                        source="node_attribute",
                        operation="join",
                        values=earliest,
                    ),
                    Term(
                        source="node_attribute",
                        phase="after_bound",
                        values=service,
                    ),
                    Term(
                        source="constant",
                        operation="assign",
                        phase="after_bound",
                        trigger="reset_arrival",
                        constant=0.0,
                        at_depot=True,
                    ),
                ),
                bound=Bound(upper_values=latest, horizon="return"),
                depot_release=True,
            )
        )

    if "route_limit" in constraints:
        limit = float(problem.get("route_limit", math.inf))
        result.append(
            ResourceProgram(
                name="route_limit",
                operator="accumulator",
                initial=0.0,
                scale=limit,
                terms=(
                    Term(source="distance"),
                    Term(
                        source="constant",
                        operation="assign",
                        phase="after_bound",
                        trigger="reset_arrival",
                        constant=0.0,
                        at_depot=True,
                    ),
                ),
                bound=Bound(upper=limit, horizon="return"),
                depot_release=True,
            )
        )

    if "tour_limit" in constraints:
        limit = float(problem["tour_limit"])
        result.append(
            ResourceProgram(
                name="tour_limit",
                operator="accumulator",
                scope="solution",
                initial=0.0,
                scale=limit,
                terms=(
                    Term(source="distance"),
                    Term(
                        source="constant",
                        operation="assign",
                        phase="after_bound",
                        trigger="reset_arrival",
                        constant=0.0,
                        at_depot=True,
                    ),
                ),
                bound=Bound(upper=limit, horizon="return"),
                depot_release=True,
            )
        )

    if "backhaul_order" in constraints:
        node_class = np.where(demand > EPS, 0, 1).astype(np.int64)
        result.append(
            ResourceProgram(
                name="backhaul_order",
                operator="precedence",
                relation="class_order",
                node_class=node_class,
                relation_count=int(np.count_nonzero(node_class)),
            )
        )

    if "pickup_delivery" in constraints:
        predecessor = np.full(node_count, -1, dtype=np.int64)
        if "pickup_delivery_pairs" in problem:
            pairs = np.asarray(problem["pickup_delivery_pairs"], dtype=np.int64)
        else:
            if customer_count % 2:
                raise ValueError("pickup-delivery requires an even customer count")
            count = customer_count // 2
            pickup = np.arange(depot_count, depot_count + count)
            pairs = np.stack((pickup, pickup + count), axis=1)
        for pickup, delivery in pairs:
            predecessor[int(delivery)] = int(pickup)
        result.append(
            _pairwise_program(
                "pickup_delivery", predecessor, "route", node_count
            )
        )

    if "prize_quota" in constraints:
        quota = float(problem.get("prize_quota", 1.0))
        result.append(
            ResourceProgram(
                name="prize_quota",
                operator="accumulator",
                scope="solution",
                initial=0.0,
                scale=max(quota, EPS),
                terms=(Term(source="node_attribute", values=prize),),
                bound=Bound(lower=quota, check="route_end"),
            )
        )

    return CompiledProblem(
        programs=tuple(result),
        requires_all_visits="visit_all" in constraints,
        objective=Objective.compile(problem["objective"]),
        visit_values=prize,
        omission_values=_array(problem.get("penalty"), node_count, 0.0),
    )


__all__ = [
    "Bound",
    "CompiledProblem",
    "Objective",
    "ResourceProgram",
    "Term",
    "compile_problem",
]
