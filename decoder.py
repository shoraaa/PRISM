"""Readable executable-semantics decoder for PRISM.

This module is deliberately a semantic reference, not a replacement for the
native search implementation yet.  It has no compiled constraint kernels or
``FastPath`` dispatch.  Named benchmark constraints are lowered once into the
same small program representation accepted through ``problem["resources"]``;
after that, one interpreter supplies legality, next state, signed margin, and
events for every constraint.

The stable public surface is intentionally small:

* :meth:`Decoder.mask` replays a prefix and returns exact next-node legality;
* :meth:`Decoder.evaluate` replays and scores a complete route;
* :meth:`Decoder.probe` exposes the per-resource execution outcomes; and
* :meth:`Decoder.execution_features` exports the v14 learning pair
  ``(normalized next state, normalized signed margin)``.

Production construction, sampling, and SRR remain native.  The opt-in search
layer in :mod:`search` consumes this interpreter through the same exact behavior
contract, allowing a readable Python refinement path without changing native
execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from program import (
    Bound,
    Objective,
    ResourceProgram,
    Term,
    compile_problem,
)


EPS = 1.0e-6


class Event(IntFlag):
    NONE = 0
    RESET = 1
    CHECKPOINT = 2
    RESTORE = 4


@dataclass
class ResourceState:
    value: float
    shadow: float


@dataclass
class State:
    route: list[int]
    visited: np.ndarray
    route_visited: np.ndarray
    current: int
    route_depot: int
    start_node: int
    visited_customers: int
    at_depot: bool
    distance: float
    collected_visit_value: float
    served_omission_value: float
    resources: list[ResourceState]

    def copy(self) -> "State":
        return State(
            route=list(self.route),
            visited=self.visited.copy(),
            route_visited=self.route_visited.copy(),
            current=self.current,
            route_depot=self.route_depot,
            start_node=self.start_node,
            visited_customers=self.visited_customers,
            at_depot=self.at_depot,
            distance=self.distance,
            collected_visit_value=self.collected_visit_value,
            served_omission_value=self.served_omission_value,
            resources=[ResourceState(row.value, row.shadow) for row in self.resources],
        )


@dataclass(frozen=True)
class ResourceOutcome:
    admissible: bool
    next_value: float
    next_shadow: float
    bounded_value: float
    signed_margin: float
    normalized_next_state: float
    normalized_signed_margin: float
    events: Event = Event.NONE
    emitted_duration: float = 0.0


@dataclass(frozen=True)
class ActionOutcome:
    admissible: bool
    structural_admissible: bool
    resources: tuple[ResourceOutcome, ...]
    event_duration: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class ExecutedTransition:
    """One transition as observed through the executable semantics.

    ``live_state`` and ``resources`` are deliberately registry-indexed.  A
    consumer can value what the transition did without seeing a constraint
    name, a program term, or the search operator that proposed it.
    """

    origin: int
    destination: int
    live_state: tuple[float, ...]
    resources: tuple[ResourceOutcome, ...]
    implicit: bool = False


@dataclass(frozen=True)
class ExecutedRoute:
    """Exact route result plus the anonymous behavior that produced it."""

    route: tuple[int, ...]
    transitions: tuple[ExecutedTransition, ...]
    solution: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionCache:
    """Prefix states used to execute only a candidate's modified suffix.

    This object is an evaluator optimization, not part of the learned input.
    ``execution`` is the complete anonymous behavior exposed to consumers.
    """

    execution: ExecutedRoute
    prefix_states: tuple[State, ...]
    executed_transitions: int
    reused_transitions: int = 0


def _array(value: Any, size: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.ndim > 1 and result.shape[0] == 1:
        result = result[0]
    if result.shape != (size,):
        raise ValueError(f"expected a vector of shape ({size},), got {result.shape}")
    return result


def _matrix(value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim > 2 and result.shape[0] == 1:
        result = result[0]
    if result.shape != (size, size):
        raise ValueError(
            f"expected a matrix of shape ({size}, {size}), got {result.shape}"
        )
    return result


class Decoder:
    """Pure-Python executable constraint decoder.

    The class intentionally accepts only semantic arguments.  Native search
    tuning options are rejected rather than accepted and silently ignored.
    """

    def __init__(self, problem: Mapping[str, Any]):
        self.problem = dict(problem)
        for key in (
            "constraints",
            "objective",
            "depot_count",
            "multi_route",
            "open_route",
        ):
            if key not in self.problem:
                raise ValueError(f"explicit schema is missing {key!r}")

        coordinates = self.problem.get("coordinates")
        distance = self.problem.get("distance")
        if distance is None and coordinates is None:
            raise ValueError("one of 'distance' or 'coordinates' must be provided")
        if distance is not None:
            raw = np.asarray(distance)
            if raw.ndim > 2 and raw.shape[0] == 1:
                raw = raw[0]
            self.node_count = int(raw.shape[0])
            self.distance = _matrix(raw, self.node_count)
        else:
            xy = np.asarray(coordinates, dtype=np.float64)
            if xy.ndim > 2 and xy.shape[0] == 1:
                xy = xy[0]
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError("coordinates must have shape (node_count, 2)")
            self.node_count = int(xy.shape[0])
            self.distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)

        self.name = str(self.problem.get("name", "schema")).lower()
        self.depot_count = int(self.problem["depot_count"])
        if not 0 <= self.depot_count < self.node_count:
            raise ValueError("depot_count must be in [0, node_count)")
        self.customer_count = self.node_count - self.depot_count
        self.multi_route = bool(self.problem["multi_route"])
        self.open_route = bool(self.problem["open_route"])
        self.node_attributes = {
            str(name): _array(values, self.node_count, 0.0)
            for name, values in dict(self.problem.get("node_attributes", {})).items()
        }
        self.edge_attributes = {
            str(name): _matrix(values, self.node_count)
            for name, values in dict(self.problem.get("edge_attributes", {})).items()
        }

        compiled = compile_problem(
            self.problem, self.node_count, self.depot_count
        )
        self.requires_all_visits = compiled.requires_all_visits
        self.objective = compiled.objective
        self.visit_values = compiled.visit_values
        self.omission_values = compiled.omission_values
        self.programs = compiled.programs + tuple(
            self._compile_resource(row)
            for row in self.problem.get("resources", ())
        )

    def _pairwise_program(
        self, name: str, predecessor: np.ndarray, scope: str
    ) -> ResourceProgram:
        successors: list[list[int]] = [[] for _ in range(self.node_count)]
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

    def _scalar(self, expression: Any, field_name: str) -> float:
        if isinstance(expression, Mapping):
            try:
                return float(self.problem[str(expression["scalar"])])
            except KeyError as error:
                raise ValueError(f"invalid scalar reference for {field_name}") from error
        return float(expression)

    def _node_reference(self, expression: Mapping[str, Any]) -> np.ndarray:
        try:
            return self.node_attributes[str(expression["node_attribute"])]
        except KeyError as error:
            raise ValueError(f"unknown node attribute: {expression}") from error

    def _term(self, entry: Mapping[str, Any]) -> Term:
        if "node_attribute" in entry:
            source = "node_attribute"
            values = self.node_attributes[str(entry["node_attribute"])]
            constant = 0.0
        elif "edge_attribute" in entry:
            name = str(entry["edge_attribute"])
            source = "distance" if name == "distance" else "edge_attribute"
            values = None if source == "distance" else self.edge_attributes[name]
            constant = 0.0
        elif "value" in entry:
            source = "constant"
            values = None
            constant = self._scalar(entry["value"], "term.value")
        elif str(entry.get("op")) == "restore":
            source, values, constant = "constant", None, 0.0
        else:
            raise ValueError("a term needs a node, edge, or constant source")

        trigger_nodes = None
        if "at_nodes" in entry:
            trigger_nodes = self._node_reference(entry["at_nodes"]) > 0.5
        gate = entry.get("gate")
        gate_name = "always"
        gate_values = None
        gate_sign = 0.0
        if gate is not None:
            gate_values = self._node_reference(gate)
            gate_sign = 1.0 if gate.get("sign", "positive") == "positive" else -1.0
            gate_name = f"remainder_{gate.get('branch', 'default')}"
        return Term(
            source=source,
            operation=str(entry.get("op", "add")),
            phase=str(entry.get("phase", "before_bound")),
            trigger=str(entry.get("when", "always")),
            point=str(entry.get("at", "to")),
            coefficient=float(entry.get("coefficient", 1.0)),
            constant=constant,
            values=values,
            at_depot=bool(entry.get("at_depot", False)),
            trigger_nodes=trigger_nodes,
            gate=gate_name,
            gate_values=gate_values,
            gate_sign=gate_sign,
        )

    def _compile_resource(self, row: Mapping[str, Any]) -> ResourceProgram:
        operator = str(row["operator"])
        name = str(row["name"])
        scope = str(row.get("scope", "route"))
        if row.get("direction", "forward") != "forward":
            raise ValueError("the Python reference currently executes forward rows")

        if operator == "precedence":
            relation = str(row.get("relation", "pairwise"))
            if relation == "pairwise":
                predecessor = np.rint(
                    self._node_reference(row["predecessor"])
                ).astype(np.int64)
                return self._pairwise_program(name, predecessor, scope)
            if relation == "class_order":
                node_class = np.rint(self._node_reference(row["class"])).astype(np.int64)
                return ResourceProgram(
                    name=name,
                    operator="precedence",
                    scope=scope,
                    relation=relation,
                    node_class=node_class,
                    relation_count=int(np.count_nonzero(node_class > 0)),
                )
            if relation == "dag":
                predecessors = tuple(
                    tuple(int(value) for value in required)
                    for required in row["predecessors"]
                )
                successors: list[list[int]] = [[] for _ in range(self.node_count)]
                count = 0
                for node, required in enumerate(predecessors):
                    count += len(required)
                    for predecessor in required:
                        successors[predecessor].append(node)
                return ResourceProgram(
                    name=name,
                    operator="precedence",
                    scope=scope,
                    relation=relation,
                    predecessors=predecessors,
                    successors=tuple(tuple(row) for row in successors),
                    initial=float(count),
                    scale=max(float(count), 1.0),
                    relation_count=count,
                )
            raise ValueError(f"unknown precedence relation: {relation}")

        if operator not in {"affine_accumulator", "affine_max"}:
            raise ValueError(f"unknown resource operator: {operator}")
        semiring = str(
            row.get("semiring", "max_plus" if operator == "affine_max" else "arithmetic")
        )
        initial = self._scalar(row.get("initial", 0.0), "initial")
        scale = self._scalar(row.get("scale", 1.0), "scale")
        terms: list[Term] = []
        restore_duration = 0.0

        if "terms" in row:
            terms.extend(self._term(entry) for entry in row["terms"])
        else:
            increment = row.get("increment")
            if increment:
                if "edge_attribute" in increment:
                    terms.append(
                        self._term(
                            {
                                "edge_attribute": increment["edge_attribute"],
                                "coefficient": increment.get("coefficient", 1.0),
                            }
                        )
                    )
                if "node_attribute" in increment:
                    terms.append(
                        self._term(
                            {
                                "node_attribute": increment["node_attribute"],
                                "coefficient": increment.get("coefficient", 1.0),
                            }
                        )
                    )
            operand = row.get("join", row.get("clamp"))
            if operand:
                entry = dict(operand)
                if "value" not in entry and "node_attribute" not in entry:
                    raise ValueError("join needs value or node_attribute")
                entry["op"] = "join"
                terms.append(self._term(entry))
            departure = row.get("departure")
            if departure:
                terms.append(
                    self._term(
                        {
                            "node_attribute": departure["node_attribute"],
                            "coefficient": departure.get("coefficient", 1.0),
                            "phase": "after_bound",
                        }
                    )
                )
            reset = row.get("reset")
            if reset:
                reset_value = self._scalar(reset.get("value", initial), "reset.value")
                at_depot = bool(reset.get("at_depot", False))
                optional = bool(reset.get("optional_before_transition", False))
                nodes = None
                if "node_attribute" in reset:
                    nodes = self.node_attributes[str(reset["node_attribute"])] > 0.5
                if optional:
                    if nodes is None:
                        raise ValueError("optional reset requires node_attribute")
                    terms.extend(
                        (
                            Term(
                                source="constant",
                                operation="checkpoint",
                                phase="after_bound",
                                trigger="checkpoint_arrival",
                                constant=reset_value,
                                trigger_nodes=nodes,
                            ),
                            Term(
                                source="constant",
                                operation="restore",
                                trigger="bound_failure",
                            ),
                        )
                    )
                    restore_duration = float(reset.get("duration", 0.0))
                elif nodes is not None or at_depot:
                    gate = reset.get("guard")
                    base = {
                        "op": "assign",
                        "phase": "after_bound",
                        "when": "reset_arrival",
                        "value": reset_value,
                        "at_depot": at_depot,
                    }
                    if nodes is not None:
                        key = f"__reset_{name}"
                        self.node_attributes[key] = nodes.astype(np.float64)
                        base["at_nodes"] = {"node_attribute": key}
                    if gate is None:
                        terms.append(self._term(base))
                    else:
                        default = dict(base)
                        default["gate"] = {**gate, "branch": "default"}
                        alternative = dict(base)
                        alternative["value"] = reset.get("otherwise", initial)
                        alternative["gate"] = {**gate, "branch": "alternative"}
                        terms.extend((self._term(default), self._term(alternative)))
                if optional and at_depot:
                    terms.append(
                        Term(
                            source="constant",
                            operation="assign",
                            phase="after_bound",
                            trigger="reset_arrival",
                            constant=reset_value,
                            at_depot=True,
                        )
                    )

        has_depot_reset = any(
            term.operation == "assign"
            and term.trigger == "reset_arrival"
            and term.at_depot
            for term in terms
        )
        if scope == "route" and not has_depot_reset:
            terms.append(
                Term(
                    source="constant",
                    operation="assign",
                    phase="after_bound",
                    trigger="reset_arrival",
                    constant=initial,
                    at_depot=True,
                )
            )

        bounds = row.get("bounds", ({},))
        if len(bounds) != 1:
            raise ValueError("the Python reference requires one bound declaration")
        raw_bound = bounds[0]

        def side(name: str, default: float) -> tuple[float, np.ndarray | None]:
            if name not in raw_bound:
                return default, None
            value = raw_bound[name]
            if isinstance(value, Mapping) and "node_attribute" in value:
                return default, self._node_reference(value)
            return self._scalar(value, f"bound.{name}"), None

        lower, lower_values = side("lower", -math.inf)
        upper, upper_values = side("upper", math.inf)
        return ResourceProgram(
            name=name,
            operator="accumulator",
            scope=scope,
            semiring=semiring,
            initial=initial,
            scale=scale,
            terms=tuple(terms),
            bound=Bound(
                lower=lower,
                upper=upper,
                lower_values=lower_values,
                upper_values=upper_values,
                check=str(raw_bound.get("check", "transition")),
                horizon=str(raw_bound.get("horizon", "transition")),
            ),
            restore_duration=restore_duration,
        )

    # ------------------------------------------------------------------
    # Generic execution.  No named constraint appears below this line.
    # ------------------------------------------------------------------

    def initial_state(self, start: int) -> State:
        if not 0 <= start < self.node_count:
            raise ValueError("invalid start node")
        if self.depot_count and start >= self.depot_count:
            raise ValueError("a depot problem must start at a depot")
        visited = np.zeros(self.node_count, dtype=bool)
        route_visited = np.zeros(self.node_count, dtype=bool)
        visited_customers = 0
        visit_value = 0.0
        served_omission_value = 0.0
        at_depot = self.depot_count > 0
        route_depot = start if at_depot else -1
        if not at_depot:
            visited[start] = True
            route_visited[start] = True
            visited_customers = 1
            visit_value = float(self.visit_values[start])
            served_omission_value = float(self.omission_values[start])
        return State(
            route=[start],
            visited=visited,
            route_visited=route_visited,
            current=start,
            route_depot=route_depot,
            start_node=start,
            visited_customers=visited_customers,
            at_depot=at_depot,
            distance=0.0,
            collected_visit_value=visit_value,
            served_omission_value=served_omission_value,
            resources=[
                ResourceState(program.initial, program.initial)
                for program in self.programs
            ],
        )

    def _term_value(
        self,
        term: Term,
        origin: int,
        destination: int,
        event_duration: float,
    ) -> float:
        if term.source == "constant":
            raw = term.constant
        elif term.source == "distance":
            raw = float(self.distance[origin, destination])
        elif term.source == "event_duration":
            raw = event_duration
        elif term.source == "node_attribute":
            assert term.values is not None
            raw = float(term.values[destination if term.point == "to" else origin])
        elif term.source == "edge_attribute":
            assert term.values is not None
            raw = float(term.values[origin, destination])
        else:
            raise ValueError(f"unknown term source: {term.source}")
        return term.coefficient * raw

    def _event_matches(
        self,
        term: Term,
        origin: int,
        destination: int,
        bound_failed: bool,
    ) -> bool:
        depot = destination < self.depot_count
        if term.trigger == "always":
            return True
        if term.trigger == "bound_failure":
            return bound_failed
        if term.trigger == "reset_departure":
            return (
                origin < self.depot_count and term.at_depot
            ) or (
                term.trigger_nodes is not None and bool(term.trigger_nodes[origin])
            )
        if term.trigger in {"reset_arrival", "checkpoint_arrival"}:
            return (depot and term.at_depot) or (
                term.trigger_nodes is not None and bool(term.trigger_nodes[destination])
            )
        raise ValueError(f"unknown term trigger: {term.trigger}")

    def _gate_matches(self, term: Term, state: State) -> bool:
        if term.gate == "always":
            return True
        assert term.gate_values is not None
        customer = slice(self.depot_count, self.node_count)
        remaining = ~state.visited[customer]
        values = term.gate_values[customer][remaining]
        if term.gate_sign >= 0:
            matching = bool(np.any(values > EPS))
            opposing = bool(np.any(values < -EPS))
        else:
            matching = bool(np.any(values < -EPS))
            opposing = bool(np.any(values > EPS))
        alternative = not matching and opposing
        return alternative if term.gate == "remainder_alternative" else not alternative

    @staticmethod
    def _join(
        program: ResourceProgram, value: float, operand: float
    ) -> float:
        if program.semiring == "max_plus":
            return max(value, operand)
        if program.semiring == "min_plus":
            return min(value, operand)
        raise ValueError("an arithmetic program cannot execute a join term")

    def _normalize_state(
        self, program: ResourceProgram, value: float, node: int
    ) -> float:
        if program.operator == "precedence":
            return float(np.clip(value / max(program.relation_count, 1), 0.0, 1.0))
        lower = program.bound.lower_at(node)
        upper = program.bound.upper_at(node)
        if math.isfinite(lower) and math.isfinite(upper):
            result = (value - lower) / max(upper - lower, EPS)
        elif math.isfinite(lower):
            result = 1.0 - (value - lower) / max(program.scale, EPS)
        elif math.isfinite(upper):
            result = value / max(program.scale, EPS)
        else:
            result = abs(value) / max(program.scale, EPS)
        return float(np.clip(result, 0.0, 1.0))

    def live_state_features(self, state: State) -> tuple[float, ...]:
        """Return the anonymous normalized state of every executable row."""

        return tuple(
            self._normalize_state(program, row.value, state.current)
            for program, row in zip(self.programs, state.resources)
        )

    def _return_margin(
        self,
        program: ResourceProgram,
        state: State,
        origin: int,
        depot: int,
        value: float,
        event_duration: float = 0.0,
    ) -> float:
        projected = value
        for term in program.terms:
            if (
                term.operation in {"assign", "checkpoint"}
                and self._event_matches(term, origin, origin, False)
                and self._gate_matches(term, state)
            ):
                projected = self._term_value(term, origin, origin, event_duration)
        for term in program.terms:
            if (
                term.operation == "add"
                and term.phase == "before_bound"
                and term.trigger == "always"
            ):
                projected += self._term_value(term, origin, depot, event_duration)
        for term in program.terms:
            if term.operation == "join" and term.phase == "before_bound":
                projected = self._join(
                    program,
                    projected,
                    self._term_value(term, origin, depot, event_duration),
                )
        lower = program.bound.lower_at(depot)
        upper = program.bound.upper_at(depot)
        return min(projected - lower, upper - projected)

    def _probe_accumulator(
        self,
        program: ResourceProgram,
        row: ResourceState,
        state: State,
        destination: int,
        event_duration: float,
        force_route_end: bool,
        enforce_construction_horizon: bool,
    ) -> ResourceOutcome:
        origin = state.current
        depot = destination < self.depot_count
        value, shadow = row.value, row.shadow
        events = Event.NONE
        releases_at_depot = depot and program.depot_release and not force_route_end

        # A route-opening assignment models resources whose initial state
        # depends on the chosen first node (for example, an empty vehicle when
        # a class-ordered route opens on a pickup).  It is ordinary program
        # data, not a named decoder branch.
        for term in program.terms:
            if (
                term.operation == "assign"
                and term.phase == "before_bound"
                and self._event_matches(term, origin, destination, False)
                and self._gate_matches(term, state)
            ):
                value = self._term_value(term, origin, destination, event_duration)
                shadow = value
                events |= Event.RESET

        for term in program.terms:
            if (
                term.operation == "add"
                and term.phase == "before_bound"
                and term.trigger != "always"
                and self._event_matches(term, origin, destination, False)
                and self._gate_matches(term, state)
            ):
                delta = self._term_value(term, origin, destination, event_duration)
                value += delta
                shadow += delta
        previous_shadow = shadow
        for term in program.terms:
            if (
                term.operation == "add"
                and term.phase == "before_bound"
                and term.trigger == "always"
            ):
                value += self._term_value(term, origin, destination, event_duration)
        if any(term.operation == "checkpoint" for term in program.terms):
            shadow = previous_shadow
            for term in program.terms:
                if (
                    term.operation == "add"
                    and term.phase == "before_bound"
                    and term.trigger == "always"
                ):
                    shadow += self._term_value(term, origin, destination, event_duration)
        else:
            shadow = value
        for term in program.terms:
            if (
                term.operation == "join"
                and term.phase == "before_bound"
                and self._event_matches(term, origin, destination, False)
                and self._gate_matches(term, state)
            ):
                operand = self._term_value(term, origin, destination, event_duration)
                value = self._join(program, value, operand)
                shadow = self._join(program, shadow, operand)

        check = program.bound.check == "transition" or (
            (depot or force_route_end) and program.bound.check == "route_end"
        )
        lower = program.bound.lower_at(destination)
        upper = program.bound.upper_at(destination)
        admissible = not check or lower - EPS <= value <= upper + EPS
        restored = False
        if not admissible and math.isfinite(upper) and value > upper + EPS:
            if any(
                term.operation == "restore"
                and self._event_matches(term, origin, destination, True)
                for term in program.terms
            ) and shadow <= upper + EPS:
                value = shadow
                restored = True
                events |= Event.RESTORE
                admissible = not check or lower - EPS <= value <= upper + EPS

        bounded = value
        margin = min(value - lower, upper - value) if check else program.scale

        for term in program.terms:
            if (
                term.operation == "checkpoint"
                and self._event_matches(term, origin, destination, False)
                and self._gate_matches(term, state)
            ):
                shadow = self._term_value(term, origin, destination, event_duration)
                events |= Event.CHECKPOINT
        for term in program.terms:
            if (
                term.operation == "add"
                and term.phase == "after_bound"
                and term.trigger == "always"
            ):
                delta = self._term_value(term, origin, destination, event_duration)
                value += delta
                shadow += delta

        horizon_active = program.bound.horizon == "return" or (
            enforce_construction_horizon
            and program.bound.horizon == "return_construction"
        )
        if (
            admissible
            and horizon_active
            and not depot
            and not self.open_route
            and self.depot_count
            and program.bound.check != "solution_end"
        ):
            return_margin = self._return_margin(
                program, state, destination, state.route_depot, value
            )
            margin = min(margin, return_margin)
            admissible = return_margin >= -EPS

        for term in program.terms:
            if (
                term.operation == "assign"
                and term.phase == "after_bound"
                and self._event_matches(term, origin, destination, False)
                and self._gate_matches(term, state)
            ):
                value = self._term_value(term, origin, destination, event_duration)
                shadow = value
                events |= Event.RESET

        # A compiled depot release makes route closure admissible while keeping
        # the projected bound margin available to learning.  Forced closure
        # still uses the ordinary bound result so an implicit return is checked.
        if releases_at_depot:
            admissible = True
        normalized_margin = float(np.clip(margin / max(program.scale, EPS), -1.0, 1.0))
        if admissible and normalized_margin < 0.0:
            normalized_margin = 0.0
        if not admissible and normalized_margin >= 0.0:
            normalized_margin = -max(EPS, normalized_margin)
        return ResourceOutcome(
            admissible=admissible,
            next_value=value,
            next_shadow=shadow,
            bounded_value=bounded,
            signed_margin=margin,
            normalized_next_state=self._normalize_state(program, value, destination),
            normalized_signed_margin=normalized_margin,
            events=events,
            emitted_duration=program.restore_duration if restored else 0.0,
        )

    def _probe_precedence(
        self,
        program: ResourceProgram,
        row: ResourceState,
        state: State,
        destination: int,
    ) -> ResourceOutcome:
        depot = destination < self.depot_count
        route_scoped = program.scope == "route"
        value = row.value
        admissible = True

        if program.relation == "pairwise":
            assert program.predecessor is not None
            assert program.successors is not None
            if depot:
                admissible = not route_scoped or value <= EPS
                next_value = 0.0 if route_scoped else value
            else:
                required = int(program.predecessor[destination])
                seen = state.route_visited if route_scoped else state.visited
                admissible = required < 0 or bool(seen[required])
                next_value = value + len(program.successors[destination])
                if required >= 0:
                    next_value -= 1.0
        elif program.relation == "class_order":
            assert program.node_class is not None
            if depot:
                next_value = 0.0 if route_scoped else value
            else:
                node_class = float(program.node_class[destination])
                admissible = node_class >= value - EPS
                next_value = max(value, node_class)
        elif program.relation == "dag":
            assert program.predecessors is not None
            if depot:
                next_value = value
            else:
                seen = state.route_visited if route_scoped else state.visited
                admissible = all(
                    seen[required] for required in program.predecessors[destination]
                )
                next_value = value - len(program.predecessors[destination])
        else:
            raise ValueError(f"unknown relation: {program.relation}")

        margin = (
            1.0 - np.clip(next_value / max(program.relation_count, 1), 0.0, 1.0)
            if admissible
            else -1.0
        )
        return ResourceOutcome(
            admissible=admissible,
            next_value=float(next_value),
            next_shadow=float(next_value),
            bounded_value=float(next_value),
            signed_margin=float(margin),
            normalized_next_state=self._normalize_state(
                program, float(next_value), destination
            ),
            normalized_signed_margin=float(margin),
        )

    def _probe_resources(
        self,
        state: State,
        destination: int,
        *,
        force_route_end: bool = False,
        construction: bool = False,
    ) -> tuple[tuple[ResourceOutcome, ...], float]:
        first: list[ResourceOutcome] = []
        emitted_duration = 0.0
        for program, row in zip(self.programs, state.resources):
            outcome = (
                self._probe_precedence(program, row, state, destination)
                if program.operator == "precedence"
                else self._probe_accumulator(
                    program,
                    row,
                    state,
                    destination,
                    0.0,
                    force_route_end,
                    construction,
                )
            )
            first.append(outcome)
            emitted_duration += outcome.emitted_duration
        if emitted_duration <= 0.0:
            return tuple(first), 0.0

        final = list(first)
        for index, (program, row) in enumerate(zip(self.programs, state.resources)):
            if any(term.source == "event_duration" for term in program.terms):
                final[index] = self._probe_accumulator(
                    program,
                    row,
                    state,
                    destination,
                    emitted_duration,
                    force_route_end,
                    construction,
                )
        return tuple(final), emitted_duration

    def _structural_admissible(self, state: State, destination: int) -> bool:
        if not 0 <= destination < self.node_count:
            return False
        if destination >= self.depot_count:
            return not bool(state.visited[destination])
        if not self.depot_count or state.at_depot:
            return False
        return self.multi_route or not self.requires_all_visits

    def probe(
        self,
        state: State,
        destination: int,
        *,
        force_route_end: bool = False,
        construction: bool = False,
    ) -> ActionOutcome:
        structural = force_route_end or self._structural_admissible(state, destination)
        if not structural:
            return ActionOutcome(
                admissible=False,
                structural_admissible=False,
                resources=(),
                error=f"structural transition rejected: {state.current}->{destination}",
            )
        outcomes, duration = self._probe_resources(
            state,
            destination,
            force_route_end=force_route_end,
            construction=construction,
        )
        failed = next(
            (
                program.name
                for program, outcome in zip(self.programs, outcomes)
                if not outcome.admissible
            ),
            None,
        )
        return ActionOutcome(
            admissible=failed is None,
            structural_admissible=True,
            resources=outcomes,
            event_duration=duration,
            error="" if failed is None else f"resource transition failed: {failed}",
        )

    def commit(
        self, state: State, destination: int, outcome: ActionOutcome
    ) -> None:
        if not outcome.admissible:
            raise ValueError(outcome.error)
        for row, projected in zip(state.resources, outcome.resources):
            row.value = projected.next_value
            row.shadow = projected.next_shadow

        if destination < self.depot_count:
            if not self.open_route:
                state.distance += float(self.distance[state.current, state.route_depot])
            state.route.append(destination)
            state.current = destination
            state.route_depot = destination
            state.at_depot = True
            state.route_visited.fill(False)
            return

        state.distance += float(self.distance[state.current, destination])
        state.route.append(destination)
        state.current = destination
        state.at_depot = False
        state.visited[destination] = True
        state.route_visited[destination] = True
        state.visited_customers += 1
        state.collected_visit_value += float(self.visit_values[destination])
        state.served_omission_value += float(self.omission_values[destination])

    def complete(self, state: State) -> bool:
        if not self.depot_count:
            return state.visited_customers == self.customer_count
        if self.requires_all_visits:
            if state.visited_customers != self.customer_count:
                return False
            return state.at_depot if self.multi_route else True
        return len(state.route) > 1 and state.at_depot

    def _replay(self, route: Sequence[int], *, require_complete: bool) -> tuple[State | None, str]:
        values = [int(node) for node in route]
        if not values:
            return None, "route must not be empty"
        try:
            state = self.initial_state(values[0])
        except ValueError as error:
            return None, str(error)
        for destination in values[1:]:
            if self.complete(state):
                return None, "route continues after the problem is complete"
            outcome = self.probe(state, destination, construction=False)
            if not outcome.admissible:
                return None, outcome.error
            self.commit(state, destination, outcome)
        if require_complete and not self.complete(state):
            return None, "route ended before satisfying the completion condition"
        return state, ""

    def mask(self, prefix: Sequence[int]) -> np.ndarray:
        if not prefix:
            result = np.zeros(self.node_count, dtype=np.uint8)
            count = self.depot_count if self.depot_count else self.node_count
            result[:count] = 1
            return result
        state, _ = self._replay(prefix, require_complete=False)
        if state is None or self.complete(state):
            return np.zeros(self.node_count, dtype=np.uint8)
        result = np.zeros(self.node_count, dtype=np.uint8)
        for destination in range(self.node_count):
            result[destination] = self.probe(
                state, destination, construction=True
            ).admissible
        return result

    def execution_features(
        self, prefix: Sequence[int], candidates: Iterable[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return execution-derived v14 features for one live prefix.

        Rows follow ``candidates`` (all nodes by default), columns follow
        :attr:`programs`, and the last axis is ``(next_state, signed_margin)``.
        Structurally unavailable candidates retain a false mask; their resource
        outcomes are still evaluated when the resource transition is meaningful.
        """

        state, error = self._replay(prefix, require_complete=False)
        if state is None:
            raise ValueError(error)
        nodes = list(range(self.node_count) if candidates is None else candidates)
        features = np.zeros((len(nodes), len(self.programs), 2), dtype=np.float32)
        mask = np.zeros((len(nodes), len(self.programs)), dtype=bool)
        for row, destination in enumerate(nodes):
            outcomes, _ = self._probe_resources(state, int(destination))
            for resource, outcome in enumerate(outcomes):
                features[row, resource] = (
                    outcome.normalized_next_state,
                    outcome.normalized_signed_margin,
                )
                mask[row, resource] = True
        return features, mask

    def _finish(self, state: State) -> tuple[State | None, str]:
        if not self.open_route and not state.at_depot:
            end = state.start_node if not self.depot_count else state.route_depot
            outcome = self.probe(state, end, force_route_end=True)
            if not outcome.admissible:
                return None, outcome.error
            for row, projected in zip(state.resources, outcome.resources):
                row.value = projected.next_value
                row.shadow = projected.next_shadow
            state.distance += float(self.distance[state.current, end])

        for program, row in zip(self.programs, state.resources):
            if program.operator == "precedence":
                if program.relation != "class_order" and row.value > EPS:
                    return None, f"terminal resource bound failed: {program.name}"
            elif program.bound.check == "solution_end":
                lower = program.bound.lower_at(state.current)
                upper = program.bound.upper_at(state.current)
                if not lower - EPS <= row.value <= upper + EPS:
                    return None, f"terminal resource bound failed: {program.name}"
        return state, ""

    def _execute_cached(
        self,
        values: tuple[int, ...],
        base: ExecutionCache | None,
    ) -> ExecutionCache:
        """Execute ``values`` from its last state shared with ``base``."""

        array = np.asarray(values, dtype=np.int32)
        transitions: list[ExecutedTransition]
        prefix_states: list[State]
        reused = 0
        executed = 0
        if not values:
            solution = self._solution(array, False, error="route must not be empty")
            execution = ExecutedRoute(values, (), solution)
            return ExecutionCache(execution, (), 0)

        common = 0
        if base is not None:
            base_route = base.execution.route
            limit = min(len(values), len(base_route), len(base.prefix_states))
            while common < limit and values[common] == base_route[common]:
                common += 1
        if common:
            state = base.prefix_states[common - 1].copy()
            transitions = list(base.execution.transitions[: common - 1])
            prefix_states = list(base.prefix_states[:common])
            reused = max(common - 1, 0)
            suffix = values[common:]
        else:
            try:
                state = self.initial_state(values[0])
            except ValueError as error:
                solution = self._solution(array, False, error=str(error))
                execution = ExecutedRoute(values, (), solution)
                return ExecutionCache(execution, (), 0)
            transitions = []
            prefix_states = [state.copy()]
            suffix = values[1:]

        error = ""
        for destination in suffix:
            if self.complete(state):
                error = "route continues after the problem is complete"
                state = None
                break
            outcome = self.probe(state, destination, construction=False)
            executed += 1
            if not outcome.admissible:
                error = outcome.error
                state = None
                break
            transitions.append(
                ExecutedTransition(
                    origin=state.current,
                    destination=destination,
                    live_state=self.live_state_features(state),
                    resources=outcome.resources,
                )
            )
            self.commit(state, destination, outcome)
            prefix_states.append(state.copy())

        if state is not None and not self.complete(state):
            error = "route ended before satisfying the completion condition"
            state = None
        if state is not None and not self.open_route and not state.at_depot:
            end = state.start_node if not self.depot_count else state.route_depot
            outcome = self.probe(state, end, force_route_end=True)
            executed += 1
            if outcome.admissible:
                transitions.append(
                    ExecutedTransition(
                        origin=state.current,
                        destination=end,
                        live_state=self.live_state_features(state),
                        resources=outcome.resources,
                        implicit=True,
                    )
                )
            else:
                error = outcome.error
                state = None
        if state is not None:
            state, error = self._finish(state)
        if state is None:
            solution = self._solution(array, False, error=error)
            execution = ExecutedRoute(values, tuple(transitions), solution)
            return ExecutionCache(
                execution,
                tuple(prefix_states),
                executed,
                reused,
            )

        missed = float(
            np.sum(self.omission_values) - state.served_omission_value
        )
        objective = self.objective.report(
            state.distance, state.collected_visit_value, missed
        )
        solution = self._solution(
            array,
            math.isfinite(objective),
            objective=objective,
            distance=state.distance,
            collected_prize=state.collected_visit_value,
            missed_penalty=missed,
            error="" if math.isfinite(objective) else "route objective is not finite",
        )
        execution = ExecutedRoute(values, tuple(transitions), solution)
        return ExecutionCache(
            execution,
            tuple(prefix_states),
            executed,
            reused,
        )

    def execution_cache(self, route: Sequence[int]) -> ExecutionCache:
        """Execute a route and retain immutable prefix snapshots for edits."""

        return self._execute_cached(tuple(int(node) for node in route), None)

    def execute_candidate(
        self, base: ExecutionCache, route: Sequence[int]
    ) -> ExecutionCache:
        """Execute only the suffix after ``route`` diverges from ``base``."""

        return self._execute_cached(tuple(int(node) for node in route), base)

    def execute(self, route: Sequence[int]) -> ExecutedRoute:
        """Execute a complete route and publish its anonymous behavior trace."""

        return self.execution_cache(route).execution

    def evaluate(self, route: Sequence[int]) -> dict[str, Any]:
        return dict(self.execute(route).solution)

    def _solution(
        self,
        route: np.ndarray,
        feasible: bool,
        *,
        objective: float = math.inf,
        distance: float = 0.0,
        collected_prize: float = 0.0,
        missed_penalty: float = 0.0,
        error: str = "",
    ) -> dict[str, Any]:
        return {
            "route": route,
            "feasible": feasible,
            "objective": objective,
            "objective_name": self.objective.name,
            "direction": self.objective.direction,
            "distance": distance,
            "collected_prize": collected_prize,
            "missed_penalty": missed_penalty,
            "raw_objective": objective,
            "changed_edges": 0,
            "srr_moves": 0,
            "srr_scope_nodes": 0,
            "srr_revisits": 0,
            "srr_evaluations": 0,
            "srr_certified_evaluations": 0,
            "srr_incremental_rebuilds": 0,
            "srr_full_rebuilds": 0,
            "srr_rebuilt_nodes": 0,
            "off_graph_edges": 0,
            "error": error,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_count": self.node_count,
            "customer_count": self.customer_count,
            "depot_count": self.depot_count,
            "objective": self.objective.name,
            "direction": self.objective.direction,
            "multi_route": self.multi_route,
            "open_route": self.open_route,
            "resource_count": len(self.programs),
            "resource_names": [program.name for program in self.programs],
            "backend": "python_executable_semantics",
        }


__all__ = [
    "ActionOutcome",
    "Decoder",
    "Event",
    "ExecutionCache",
    "ExecutedRoute",
    "ExecutedTransition",
    "Objective",
    "ResourceOutcome",
    "ResourceProgram",
    "State",
]
