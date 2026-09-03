"""Search over exact, anonymously executed routing behavior.

The executable decoder owns feasibility and objective semantics.  Neighborhoods
own route-edit generation.  Guidance sees only the resulting execution trace;
it never receives a constraint name or a move/operator label.  This keeps the
same learned valuation usable by construction, insertion, and refinement
algorithms while allowing the native implementation to remain untouched.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np

from decoder import Decoder, ExecutedRoute, ExecutionCache


EPS = 1.0e-12


class CandidateGenerator(Protocol):
    """Produce complete candidate routes without exposing edit types downstream."""

    def generate(
        self,
        route: Sequence[int],
        decoder: Decoder,
        scope: frozenset[int] | None = None,
    ) -> Iterable[tuple[int, ...]]: ...


class CandidateScorer(Protocol):
    """Lower scores are preferred among exact objective improvements."""

    def score(self, current: ExecutedRoute, candidate: ExecutedRoute) -> float: ...


@dataclass(frozen=True)
class NeighborhoodConfig:
    nearest_neighbors: int = 16
    max_segment_length: int = 2
    relocate: bool = True
    exchange: bool = True
    two_opt: bool = True
    tail_exchange: bool = True
    optional_visits: bool = True

    def __post_init__(self) -> None:
        if self.nearest_neighbors <= 0:
            raise ValueError("nearest_neighbors must be positive")
        if self.max_segment_length <= 0:
            raise ValueError("max_segment_length must be positive")


class RoutingNeighborhood:
    """Deterministic routing edits whose semantics are checked by ``Decoder``.

    Move implementation stays here.  Neither the search loop nor learned
    guidance can observe which generator produced a candidate.
    """

    def __init__(self, decoder: Decoder, config: NeighborhoodConfig | None = None):
        self.config = config or NeighborhoodConfig()
        order = np.argsort(decoder.distance, axis=1, kind="stable")
        self.base_neighbors = tuple(
            tuple(
                int(node)
                for node in row
                if int(node) != source
            )[: self.config.nearest_neighbors]
            for source, row in enumerate(order)
        )
        self.neighbors = self.base_neighbors
        self.edge_priority: np.ndarray | None = None

    def prioritize(self, edge_energy: np.ndarray) -> None:
        """Order the fixed candidate graph by a learned edge valuation."""

        energy = np.asarray(edge_energy)
        if energy.shape != (len(self.base_neighbors), len(self.base_neighbors)):
            raise ValueError("edge_energy must have shape [node_count, node_count]")
        # A frozen learned context returns the same matrix on every descent
        # iteration.  Sorting every candidate row again was pure interpreter
        # overhead (and dominated small instances).
        if self.edge_priority is energy:
            return
        self.edge_priority = energy
        self.neighbors = tuple(
            tuple(
                sorted(
                    row,
                    key=lambda destination: (
                        float(energy[source, destination]),
                        destination,
                    ),
                )
            )
            for source, row in enumerate(self.base_neighbors)
        )

    @staticmethod
    def _customer_positions(route: Sequence[int], depot_count: int) -> list[int]:
        positions = [
            index for index, node in enumerate(route) if int(node) >= depot_count
        ]
        # Fix one representative of a depot-free cycle.
        if depot_count == 0 and positions:
            positions = positions[1:]
        return positions

    @staticmethod
    def _in_scope(nodes: Sequence[int], scope: frozenset[int] | None) -> bool:
        return scope is None or any(int(node) in scope for node in nodes)

    @staticmethod
    def _round_robin(
        iterables: Iterable[Iterable[tuple[int, ...]]],
    ) -> Iterator[tuple[int, ...]]:
        active = deque(iter(values) for values in iterables)
        while active:
            values = active.popleft()
            try:
                yield next(values)
            except StopIteration:
                continue
            active.append(values)

    @staticmethod
    def _without_empty_routes(
        route: Sequence[int], depot_count: int
    ) -> tuple[int, ...]:
        """Collapse consecutive depot markers created by an edit.

        One depot marker both closes the preceding route and declares the next
        route's depot.  Keeping the latter marker preserves that declaration
        when removing an otherwise empty route in a multi-depot solution.
        """

        result: list[int] = []
        for node in route:
            value = int(node)
            if (
                result
                and value < depot_count
                and result[-1] < depot_count
            ):
                result[-1] = value
            else:
                result.append(value)
        return tuple(result)

    def _relocations(
        self,
        route: tuple[int, ...],
        decoder: Decoder,
        scope: frozenset[int] | None,
    ) -> Iterator[tuple[int, ...]]:
        depot_count = decoder.depot_count
        def from_start(start: int) -> Iterator[tuple[int, ...]]:
            for length in range(1, self.config.max_segment_length + 1):
                stop = start + length
                if stop > len(route):
                    break
                segment = route[start:stop]
                if any(node < depot_count for node in segment):
                    break
                if not self._in_scope(segment, scope):
                    continue
                remaining = self._without_empty_routes(
                    route[:start] + route[stop:], depot_count
                )
                first = segment[0]
                neighbor_rank = {
                    node: rank for rank, node in enumerate(self.neighbors[first])
                }
                anchors: list[tuple[float, int]] = []
                for anchor, node in enumerate(remaining[:-1]):
                    value = int(node)
                    if value not in neighbor_rank and value >= depot_count:
                        continue
                    if self.edge_priority is None:
                        priority = float(neighbor_rank.get(value, -1))
                    else:
                        following = int(remaining[anchor + 1])
                        priority = float(
                            self.edge_priority[value, first]
                            + self.edge_priority[segment[-1], following]
                            - self.edge_priority[value, following]
                        )
                    anchors.append((priority, anchor))
                anchors.sort(key=lambda item: (item[0], item[1]))
                for _priority, anchor in anchors:
                    insertion = anchor + 1
                    candidate = (
                        remaining[:insertion] + segment + remaining[insertion:]
                    )
                    if candidate != route:
                        yield candidate

        yield from self._round_robin(
            from_start(start)
            for start in self._customer_positions(route, depot_count)
        )

    def _exchanges(
        self,
        route: tuple[int, ...],
        decoder: Decoder,
        scope: frozenset[int] | None,
    ) -> Iterator[tuple[int, ...]]:
        positions = self._customer_positions(route, decoder.depot_count)
        position_of = {int(route[index]): index for index in positions}
        def from_left(left: int) -> Iterator[tuple[int, ...]]:
            left_node = int(route[left])
            for right_node in self.neighbors[left_node]:
                right = position_of.get(right_node)
                if right is None or right <= left:
                    continue
                if not self._in_scope((left_node, right_node), scope):
                    continue
                candidate = list(route)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                yield tuple(candidate)

        yield from self._round_robin(from_left(left) for left in positions)

    def _two_opt(
        self,
        route: tuple[int, ...],
        decoder: Decoder,
        scope: frozenset[int] | None,
    ) -> Iterator[tuple[int, ...]]:
        positions = self._customer_positions(route, decoder.depot_count)
        position_of = {int(route[index]): index for index in positions}
        def from_left(left: int) -> Iterator[tuple[int, ...]]:
            left_node = int(route[left])
            for right_node in self.neighbors[left_node]:
                right = position_of.get(right_node)
                if right is None or right <= left:
                    continue
                segment = route[left : right + 1]
                if any(node < decoder.depot_count for node in segment):
                    continue
                if not self._in_scope(segment, scope):
                    continue
                yield route[:left] + tuple(reversed(segment)) + route[right + 1 :]

        yield from self._round_robin(from_left(left) for left in positions)

    def _tail_exchanges(
        self,
        route: tuple[int, ...],
        decoder: Decoder,
        scope: frozenset[int] | None,
    ) -> Iterator[tuple[int, ...]]:
        if decoder.depot_count == 0:
            return
        depots = [
            index for index, node in enumerate(route)
            if int(node) < decoder.depot_count
        ]
        if len(depots) < 3:
            return

        def exchange_pair(left_route: int, right_route: int):
            left_start = depots[left_route]
            left_end = depots[left_route + 1]
            right_start = depots[right_route]
            right_end = depots[right_route + 1]
            left_nodes = route[left_start + 1 : left_end]
            right_nodes = route[right_start + 1 : right_end]
            left_cuts = range(len(left_nodes), -1, -1)
            right_cuts = range(len(right_nodes), -1, -1)
            for left_cut in left_cuts:
                for right_cut in right_cuts:
                    left_tail = left_nodes[left_cut:]
                    right_tail = right_nodes[right_cut:]
                    if not left_tail and not right_tail:
                        continue
                    touched = left_tail + right_tail
                    if not self._in_scope(touched, scope):
                        continue
                    new_left = left_nodes[:left_cut] + right_tail
                    new_right = right_nodes[:right_cut] + left_tail
                    if left_route + 1 == right_route:
                        candidate = (
                            route[: left_start + 1]
                            + new_left
                            + (route[left_end],)
                            + new_right
                            + route[right_end:]
                        )
                    else:
                        candidate = (
                            route[: left_start + 1]
                            + new_left
                            + route[left_end : right_start + 1]
                            + new_right
                            + route[right_end:]
                        )
                    candidate = self._without_empty_routes(
                        candidate, decoder.depot_count
                    )
                    if candidate == route:
                        continue
                    yield candidate

        yield from self._round_robin(
            exchange_pair(left, right)
            for left in range(len(depots) - 2)
            for right in range(left + 1, len(depots) - 1)
        )

    def _optional_edits(
        self,
        route: tuple[int, ...],
        decoder: Decoder,
        scope: frozenset[int] | None,
    ) -> Iterator[tuple[int, ...]]:
        if decoder.requires_all_visits:
            return
        present = set(route)
        for node in range(decoder.depot_count, decoder.node_count):
            if node in present or not self._in_scope((node,), scope):
                continue
            nearby = set(self.neighbors[node])
            for anchor, anchor_node in enumerate(route[:-1]):
                if int(anchor_node) not in nearby and int(anchor_node) >= decoder.depot_count:
                    continue
                insertion = anchor + 1
                yield route[:insertion] + (node,) + route[insertion:]
        for position in self._customer_positions(route, decoder.depot_count):
            node = int(route[position])
            if self._in_scope((node,), scope):
                yield self._without_empty_routes(
                    route[:position] + route[position + 1 :],
                    decoder.depot_count,
                )

    def generate(
        self,
        route: Sequence[int],
        decoder: Decoder,
        scope: frozenset[int] | None = None,
    ) -> Iterable[tuple[int, ...]]:
        values = tuple(int(node) for node in route)
        families: list[Iterable[tuple[int, ...]]] = []
        if self.config.relocate:
            families.append(self._relocations(values, decoder, scope))
        if self.config.exchange:
            families.append(self._exchanges(values, decoder, scope))
        if self.config.two_opt:
            families.append(self._two_opt(values, decoder, scope))
        if self.config.tail_exchange:
            families.append(self._tail_exchanges(values, decoder, scope))
        if self.config.optional_visits:
            families.append(self._optional_edits(values, decoder, scope))
        yield from self._round_robin(families)


class ObjectiveScorer:
    """Best exact objective improvement, expressed as a lower-is-better score."""

    @staticmethod
    def score(current: ExecutedRoute, candidate: ExecutedRoute) -> float:
        value = float(candidate.solution["objective"])
        return -value if candidate.solution["direction"] == "maximize" else value


class ExecutedBehaviorScorer:
    """Apply current edge outputs to exact Python execution traces.

    The edge arrays stay aligned to the graph on which the model was evaluated.
    Off-graph transitions receive neutral learned residuals, matching the native
    contract.  Unlike the native SRR aggregate, the live-state terms are
    evaluated at every transition's replayed live state rather than once at a
    move anchor.
    """

    def __init__(
        self,
        decoder: Decoder,
        edge_index: np.ndarray,
        *,
        edge_field: np.ndarray,
        multipliers: np.ndarray,
        objective_energy_scale: float,
        objective_residual: np.ndarray | None = None,
        coupler_weights: np.ndarray | None = None,
        coupler_bias: np.ndarray | None = None,
        edge_state_field: np.ndarray | None = None,
    ) -> None:
        edges = np.asarray(edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge_count]")
        self.decoder = decoder
        self.edge_lookup = {
            (int(origin), int(destination)): index
            for index, (origin, destination) in enumerate(edges.T)
        }
        self.edge_field = np.asarray(edge_field, dtype=np.float64)
        self.multipliers = np.asarray(multipliers, dtype=np.float64)
        self.objective_energy_scale = max(float(objective_energy_scale), 1.0e-12)
        resources = len(decoder.programs)
        edge_count = edges.shape[1]
        self.objective_residual = (
            np.zeros(edge_count, dtype=np.float64)
            if objective_residual is None
            else np.asarray(objective_residual, dtype=np.float64)
        )
        if self.edge_field.shape != (edge_count, resources):
            raise ValueError(
                f"edge_field must have shape ({edge_count}, {resources})"
            )
        if self.objective_residual.shape != (edge_count,):
            raise ValueError(
                f"objective_residual must have shape ({edge_count},)"
            )
        if self.multipliers.shape != (resources + 1,):
            raise ValueError(f"multipliers must have shape ({resources + 1},)")
        if coupler_weights is None:
            self.coupler_weights = np.zeros((resources + 1, resources))
        else:
            self.coupler_weights = np.asarray(coupler_weights, dtype=np.float64)
        if coupler_bias is None:
            self.coupler_bias = np.zeros(resources + 1)
        else:
            self.coupler_bias = np.asarray(coupler_bias, dtype=np.float64)
        if edge_state_field is None:
            self.edge_state_field = np.zeros((edge_count, resources, resources))
        else:
            self.edge_state_field = np.asarray(
                edge_state_field, dtype=np.float64
            )
        if self.coupler_weights.shape != (resources + 1, resources):
            raise ValueError(
                "coupler_weights must have shape [resource_count + 1, resource_count]"
            )
        if self.coupler_bias.shape != (resources + 1,):
            raise ValueError("coupler_bias must have shape [resource_count + 1]")
        if self.edge_state_field.shape != (edge_count, resources, resources):
            raise ValueError(
                "edge_state_field must have shape "
                "[edge_count, resource_count, resource_count]"
            )
        if not all(
            np.isfinite(values).all()
            for values in (
                self.edge_field,
                self.multipliers,
                self.objective_residual,
                self.coupler_weights,
                self.coupler_bias,
                self.edge_state_field,
            )
        ):
            raise ValueError("guidance arrays must be finite")
        finite_distance = decoder.distance[np.isfinite(decoder.distance)]
        self.distance_scale = max(
            float(np.max(finite_distance)) if finite_distance.size else 0.0,
            1.0e-6,
        )
        self._cache: dict[tuple[int, ...], float] = {}
        self._proxy_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]], float
        ] = {}
        self._proxy_matrix_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._frozen_context: ExecutedRoute | None = None

    def freeze_context(self, execution: ExecutedRoute) -> None:
        """Keep one exactly executed context for a static model emission."""

        self._frozen_context = execution

    def _coupled_multiplier(self, slot: int, live_state: np.ndarray) -> float:
        logit = float(self.coupler_bias[slot])
        if live_state.size:
            logit += float(self.coupler_weights[slot] @ live_state)
        # Numerically stable 2 * sigmoid(logit), matching the native decoder.
        if logit >= 0.0:
            modulation = 2.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            modulation = 2.0 * exponential / (1.0 + exponential)
        return float(self.multipliers[slot]) * modulation

    def _objective_edge_cost(self, origin: int, destination: int) -> float:
        objective = self.decoder.objective
        travel = float(self.decoder.distance[origin, destination])
        if destination < self.decoder.depot_count:
            if self.decoder.open_route:
                return 0.0
            return (
                objective.sense * objective.distance_coeff * travel
                + objective.distance_regularizer * travel / self.distance_scale
            )
        return (
            objective.sense
            * (
                objective.distance_coeff * travel
                + objective.visit_coeff
                * float(self.decoder.visit_values[destination])
                - objective.miss_coeff
                * float(self.decoder.omission_values[destination])
            )
            + objective.distance_regularizer * travel / self.distance_scale
        )

    def route_energy(self, execution: ExecutedRoute) -> float:
        cached = self._cache.get(execution.route)
        if cached is not None:
            return cached
        resources = len(self.decoder.programs)
        energy = 0.0
        for transition in execution.transitions:
            live = np.asarray(transition.live_state, dtype=np.float64)
            edge = self.edge_lookup.get(
                (transition.origin, transition.destination)
            )
            objective_residual = (
                float(self.objective_residual[edge]) if edge is not None else 0.0
            )
            objective = (
                self._objective_edge_cost(
                    transition.origin, transition.destination
                )
                / self.objective_energy_scale
                + objective_residual
            )
            energy += self._coupled_multiplier(resources, live) * objective
            if edge is not None and resources:
                for row in range(resources):
                    energy += self._coupled_multiplier(row, live) * float(
                        self.edge_field[edge, row]
                    )
                    # The per-edge half: linear in the live state, scaled by the
                    # same base multiplier the native decoder uses.
                    energy += float(self.multipliers[row]) * float(
                        self.edge_state_field[edge, row] @ live
                    )
        self._cache[execution.route] = energy
        return energy

    def _proxy_live_state(
        self, current: ExecutedRoute
    ) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for transition in current.transitions:
            result[transition.origin] = np.asarray(
                transition.live_state, dtype=np.float64
            )
        return result

    def _proxy_energy_matrix(self, current: ExecutedRoute) -> np.ndarray:
        cached = self._proxy_matrix_cache.get(current.route)
        if cached is not None:
            return cached
        count = self.decoder.node_count
        resources = len(self.decoder.programs)
        objective = self.decoder.objective
        travel = self.decoder.distance
        node_term = objective.sense * (
            objective.visit_coeff * self.decoder.visit_values
            - objective.miss_coeff * self.decoder.omission_values
        )
        objective_cost = (
            objective.sense * objective.distance_coeff * travel
            + node_term[None, :]
            + objective.distance_regularizer * travel / self.distance_scale
        )
        if self.decoder.depot_count:
            if self.decoder.open_route:
                objective_cost[:, : self.decoder.depot_count] = 0.0
            else:
                objective_cost[:, : self.decoder.depot_count] = (
                    objective.sense
                    * objective.distance_coeff
                    * travel[:, : self.decoder.depot_count]
                    + objective.distance_regularizer
                    * travel[:, : self.decoder.depot_count]
                    / self.distance_scale
                )

        live = np.zeros((count, resources), dtype=np.float64)
        for origin, values in self._proxy_live_state(current).items():
            live[origin] = values

        residual = np.zeros((count, count), dtype=np.float64)
        field = np.zeros((count, count, resources), dtype=np.float64)
        # Off-graph transitions carry no learned row, so their state-conditioned
        # term is zero and the dense matrix below leaves it at zero.
        state_term = np.zeros((count, count), dtype=np.float64)
        for (origin, destination), edge in self.edge_lookup.items():
            residual[origin, destination] = self.objective_residual[edge]
            if resources:
                field[origin, destination] = self.edge_field[edge]

        logits = self.coupler_bias[None, :] + live @ self.coupler_weights.T
        modulation = 2.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        weights = modulation * self.multipliers[None, :]
        energy = weights[:, resources, None] * (
            objective_cost / self.objective_energy_scale + residual
        )
        if resources:
            energy += np.einsum("ir,ijr->ij", weights[:, :resources], field)
            for (origin, destination), edge in self.edge_lookup.items():
                state_term[origin, destination] = float(
                    self.multipliers[:resources]
                    @ (self.edge_state_field[edge] @ live[origin])
                )
            energy += state_term
        self._proxy_matrix_cache[current.route] = energy
        return energy

    def edge_energy_matrix(self, current: ExecutedRoute) -> np.ndarray:
        """Incumbent-conditioned learned energy for candidate ordering."""

        context = self._frozen_context or current
        return self._proxy_energy_matrix(context)

    def _route_edges(self, route: Sequence[int]) -> Iterator[tuple[int, int]]:
        if not route:
            return
        current = int(route[0])
        route_depot = current
        at_depot = current < self.decoder.depot_count
        for raw_destination in route[1:]:
            destination = int(raw_destination)
            if destination < self.decoder.depot_count:
                if not self.decoder.open_route and not at_depot:
                    yield current, route_depot
                current = destination
                route_depot = destination
                at_depot = True
            else:
                yield current, destination
                current = destination
                at_depot = False
        if not self.decoder.open_route and not at_depot:
            end = int(route[0]) if not self.decoder.depot_count else route_depot
            yield current, end

    def proxy_score(
        self, current: ExecutedRoute, route: Sequence[int]
    ) -> float:
        """Cheap incumbent-conditioned energy for shortlist construction.

        This uses the model outputs already computed from the incumbent's exact
        execution features.  Exact semantics are intentionally deferred until
        after ranking; the score cannot make a route feasible or acceptable.
        """

        values = tuple(int(node) for node in route)
        context = self._frozen_context or current
        cache_key = (context.route, values)
        cached = self._proxy_cache.get(cache_key)
        if cached is not None:
            return cached
        matrix = self._proxy_energy_matrix(context)
        energy = sum(matrix[origin, destination] for origin, destination in self._route_edges(values))
        self._proxy_cache[cache_key] = energy
        return energy

    def score(self, current: ExecutedRoute, candidate: ExecutedRoute) -> float:
        return self.route_energy(candidate) - self.route_energy(current)


@dataclass(frozen=True)
class SearchConfig:
    max_iterations: int = 32
    max_evaluations: int = 4096
    max_candidates_per_iteration: int = 512
    shortlist_size: int = 32
    selection: str = "objective"
    materialize_each_move: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        if self.max_candidates_per_iteration <= 0:
            raise ValueError("max_candidates_per_iteration must be positive")
        if self.shortlist_size <= 0:
            raise ValueError("shortlist_size must be positive")
        if self.selection not in {
            "first",
            "objective_first",
            "objective",
            "guidance",
        }:
            raise ValueError(
                "selection must be 'first', 'objective_first', 'objective', "
                "or 'guidance'"
            )


@dataclass(frozen=True)
class SearchDecision:
    """Operator-agnostic trace suitable for learned ranking replay."""

    current: ExecutedRoute
    candidates: tuple[ExecutedRoute, ...]
    selected_index: int
    scores: tuple[float, ...]


@dataclass(frozen=True)
class SearchResult:
    execution: ExecutedRoute
    moves: int
    iterations: int
    evaluations: int
    generated: int
    executed_transitions: int
    reused_transitions: int
    elapsed_seconds: float
    decisions: tuple[SearchDecision, ...] = ()

    @property
    def solution(self) -> dict:
        result = dict(self.execution.solution)
        result.update(
            {
                "srr_moves": self.moves,
                "srr_evaluations": self.evaluations,
                "python_search_iterations": self.iterations,
                "python_search_generated": self.generated,
                "python_executed_transitions": self.executed_transitions,
                "python_reused_transitions": self.reused_transitions,
                "python_search_seconds": self.elapsed_seconds,
            }
        )
        return result


def _better(candidate: ExecutedRoute, current: ExecutedRoute) -> bool:
    candidate_value = float(candidate.solution["objective"])
    current_value = float(current.solution["objective"])
    if candidate.solution["direction"] == "maximize":
        return candidate_value > current_value + EPS
    return candidate_value < current_value - EPS


def _changed_nodes(before: ExecutedRoute, after: ExecutedRoute) -> frozenset[int]:
    before_edges = {
        (transition.origin, transition.destination)
        for transition in before.transitions
    }
    after_edges = {
        (transition.origin, transition.destination)
        for transition in after.transitions
    }
    nodes: set[int] = set()
    for origin, destination in before_edges.symmetric_difference(after_edges):
        nodes.add(origin)
        nodes.add(destination)
    return frozenset(nodes)


class ExecutableRefiner:
    """Monotone local refinement over exact executable candidate behavior."""

    def __init__(
        self,
        decoder: Decoder,
        generator: CandidateGenerator,
        scorer: CandidateScorer | None = None,
        config: SearchConfig | None = None,
        solution_evaluator: Callable[[Sequence[int]], Mapping] | None = None,
    ) -> None:
        self.decoder = decoder
        self.generator = generator
        self.scorer = scorer or ObjectiveScorer()
        self.config = config or SearchConfig()
        self.solution_evaluator = solution_evaluator

    @staticmethod
    def _solution_execution(
        route: tuple[int, ...], solution: Mapping
    ) -> ExecutedRoute:
        return ExecutedRoute(route, (), dict(solution))

    def _objective_values(
        self, routes: Sequence[tuple[int, ...]]
    ) -> np.ndarray:
        """Vectorized declared objective, without making feasibility claims."""

        result = np.empty(len(routes), dtype=np.float64)
        groups: dict[int, list[int]] = {}
        for index, route in enumerate(routes):
            groups.setdefault(len(route), []).append(index)
        objective = self.decoder.objective
        for length, indices in groups.items():
            matrix = np.asarray([routes[index] for index in indices], dtype=np.int64)
            rows = matrix.shape[0]
            distance = np.zeros(rows, dtype=np.float64)
            if self.decoder.depot_count == 0:
                if length > 1:
                    distance += self.decoder.distance[
                        matrix[:, :-1], matrix[:, 1:]
                    ].sum(axis=1)
                if not self.decoder.open_route and length:
                    distance += self.decoder.distance[
                        matrix[:, -1], matrix[:, 0]
                    ]
                visit = self.decoder.visit_values[matrix].sum(axis=1)
                served_omission = self.decoder.omission_values[matrix].sum(axis=1)
            else:
                route_depot = matrix[:, 0].copy()
                at_depot = np.ones(rows, dtype=bool)
                for position in range(1, length):
                    origin = matrix[:, position - 1]
                    destination = matrix[:, position]
                    customer = destination >= self.decoder.depot_count
                    distance += np.where(
                        customer,
                        self.decoder.distance[origin, destination],
                        np.where(
                            (~at_depot) & (not self.decoder.open_route),
                            self.decoder.distance[origin, route_depot],
                            0.0,
                        ),
                    )
                    route_depot = np.where(customer, route_depot, destination)
                    at_depot = ~customer
                if not self.decoder.open_route and length:
                    distance += np.where(
                        at_depot,
                        0.0,
                        self.decoder.distance[matrix[:, -1], route_depot],
                    )
                customer = matrix >= self.decoder.depot_count
                visit = np.where(
                    customer, self.decoder.visit_values[matrix], 0.0
                ).sum(axis=1)
                served_omission = np.where(
                    customer, self.decoder.omission_values[matrix], 0.0
                ).sum(axis=1)
            missed = float(np.sum(self.decoder.omission_values)) - served_omission
            values = (
                objective.distance_coeff * distance
                + objective.visit_coeff * visit
                + objective.miss_coeff * missed
            )
            result[np.asarray(indices, dtype=np.int64)] = values
        return result

    def _guided_shortlist(
        self,
        current_cache: ExecutionCache,
        routes: Sequence[tuple[int, ...]],
        remaining_evaluations: int,
    ) -> tuple[ExecutionCache | None, int, int, int]:
        """Return one exact improving move after cheap learned prescoring."""

        proxy_score = getattr(self.scorer, "proxy_score", None)
        if self.solution_evaluator is None or proxy_score is None:
            return None, -1, 0, 0
        current = current_cache.execution
        if self.config.selection == "objective_first":
            objectives = self._objective_values(routes)
            direction = current.solution["direction"]
            canonical = -objectives if direction == "maximize" else objectives
            order = np.argsort(canonical, kind="stable")
            limit = min(
                len(order),
                self.config.shortlist_size,
                remaining_evaluations,
            )
            checked = 0
            current_value = float(current.solution["objective"])
            for index in order[:limit]:
                predicted = float(objectives[index])
                if direction == "maximize":
                    if predicted <= current_value + EPS:
                        break
                elif predicted >= current_value - EPS:
                    break
                checked += 1
                route = routes[int(index)]
                solution = self.solution_evaluator(route)
                execution = self._solution_execution(route, solution)
                if not bool(solution["feasible"]) or not _better(execution, current):
                    continue
                if not self.config.materialize_each_move:
                    return ExecutionCache(execution, (), 0, 0), checked, 0, 0
                selected_cache = self.decoder.execute_candidate(
                    current_cache, route
                )
                exact = selected_cache.execution
                if bool(exact.solution["feasible"]) and _better(exact, current):
                    return (
                        selected_cache,
                        checked,
                        selected_cache.executed_transitions,
                        selected_cache.reused_transitions,
                    )
            return None, checked, 0, 0
        if self.config.selection == "first":
            limit = min(
                len(routes),
                self.config.shortlist_size,
                remaining_evaluations,
            )
            executed = 0
            reused = 0
            checked = 0
            for route in routes[:limit]:
                checked += 1
                solution = self.solution_evaluator(route)
                execution = self._solution_execution(route, solution)
                if not bool(solution["feasible"]) or not _better(execution, current):
                    continue
                if not self.config.materialize_each_move:
                    return ExecutionCache(execution, (), 0, 0), checked, 0, 0
                selected_cache = self.decoder.execute_candidate(
                    current_cache, route
                )
                executed += selected_cache.executed_transitions
                reused += selected_cache.reused_transitions
                exact = selected_cache.execution
                if bool(exact.solution["feasible"]) and _better(exact, current):
                    return selected_cache, checked, executed, reused
            return None, checked, executed, reused
        ranked = sorted(
            (
                (float(proxy_score(current, route)), route)
                for route in routes
            ),
            key=lambda item: (item[0], item[1]),
        )
        limit = min(
            len(ranked),
            self.config.shortlist_size,
            remaining_evaluations,
        )
        improving: list[tuple[float, ExecutedRoute]] = []
        for score, route in ranked[:limit]:
            solution = self.solution_evaluator(route)
            execution = self._solution_execution(route, solution)
            if bool(solution["feasible"]) and _better(execution, current):
                improving.append((score, execution))
        if not improving:
            return None, limit, 0, 0
        if self.config.selection == "guidance":
            improving.sort(
                key=lambda item: (
                    item[0],
                    ObjectiveScorer.score(current, item[1]),
                    item[1].route,
                )
            )
        else:
            improving.sort(
                key=lambda item: (
                    ObjectiveScorer.score(current, item[1]),
                    item[0],
                    item[1].route,
                )
            )
        executed = 0
        reused = 0
        for _score, selected in improving:
            if not self.config.materialize_each_move:
                return ExecutionCache(selected, (), 0, 0), limit, 0, 0
            selected_cache = self.decoder.execute_candidate(
                current_cache, selected.route
            )
            executed += selected_cache.executed_transitions
            reused += selected_cache.reused_transitions
            exact = selected_cache.execution
            # Native evaluation accumulates float32 while the reference uses
            # float64. Near a local optimum that can create a sub-micro-unit
            # apparent improvement on a mathematically tied route. Keep the
            # executable behavior authoritative and try the next shortlisted
            # candidate rather than committing a false improvement.
            if bool(exact.solution["feasible"]) and _better(exact, current):
                return selected_cache, limit, executed, reused
        return None, limit, executed, reused

    def refine(
        self,
        route: Sequence[int],
        *,
        scope: Iterable[int] | None = None,
        record_decisions: bool = False,
    ) -> SearchResult:
        started = time.perf_counter()
        current_cache = self.decoder.execution_cache(route)
        current = current_cache.execution
        if not bool(current.solution["feasible"]):
            raise ValueError(
                "refinement requires a feasible incumbent: "
                + str(current.solution.get("error", "unknown execution error"))
            )
        active_scope = None if scope is None else frozenset(int(node) for node in scope)
        evaluations = 0
        generated = 0
        moves = 0
        iterations = 0
        executed_transitions = 0
        reused_transitions = 0
        decisions: list[SearchDecision] = []
        freeze_context = getattr(self.scorer, "freeze_context", None)
        if freeze_context is not None:
            freeze_context(current)

        while (
            iterations < self.config.max_iterations
            and evaluations < self.config.max_evaluations
        ):
            iterations += 1
            edge_energy_matrix = getattr(
                self.scorer, "edge_energy_matrix", None
            )
            prioritize = getattr(self.generator, "prioritize", None)
            if edge_energy_matrix is not None and prioritize is not None:
                prioritize(edge_energy_matrix(current))
            seen: set[tuple[int, ...]] = {current.route}
            routes: list[tuple[int, ...]] = []
            for candidate_route in self.generator.generate(
                current.route, self.decoder, active_scope
            ):
                generated += 1
                values = tuple(int(node) for node in candidate_route)
                if values in seen:
                    continue
                seen.add(values)
                routes.append(values)
                if len(routes) >= self.config.max_candidates_per_iteration:
                    break

            selected_cache, checked, executed, reused = self._guided_shortlist(
                current_cache,
                routes,
                self.config.max_evaluations - evaluations,
            )
            if checked >= 0:
                evaluations += checked
                executed_transitions += executed
                reused_transitions += reused
                if selected_cache is None:
                    break
                previous = current
                current_cache = selected_cache
                current = selected_cache.execution
                moves += 1
                changed = _changed_nodes(previous, current)
                active_scope = changed if active_scope is not None else None
                continue

            candidates: list[ExecutionCache] = []
            for values in routes:
                candidate_cache = self.decoder.execute_candidate(
                    current_cache, values
                )
                candidate = candidate_cache.execution
                executed_transitions += candidate_cache.executed_transitions
                reused_transitions += candidate_cache.reused_transitions
                evaluations += 1
                if bool(candidate.solution["feasible"]) and _better(candidate, current):
                    candidates.append(candidate_cache)
                if (
                    evaluations >= self.config.max_evaluations
                ):
                    break
            if not candidates:
                break

            candidate_executions = tuple(item.execution for item in candidates)
            scores = tuple(
                float(self.scorer.score(current, item))
                for item in candidate_executions
            )
            finite = [index for index, value in enumerate(scores) if math.isfinite(value)]
            if not finite:
                break
            selected = min(
                finite,
                key=lambda index: (
                    scores[index],
                    ObjectiveScorer.score(current, candidate_executions[index]),
                    candidate_executions[index].route,
                ),
            )
            if record_decisions:
                decisions.append(
                    SearchDecision(current, candidate_executions, selected, scores)
                )
            previous = current
            current_cache = candidates[selected]
            current = current_cache.execution
            moves += 1
            changed = _changed_nodes(previous, current)
            active_scope = changed if active_scope is not None else None

        if not self.config.materialize_each_move and self.solution_evaluator is not None:
            final_cache = self.decoder.execution_cache(current.route)
            final = final_cache.execution
            if not bool(final.solution["feasible"]):
                raise RuntimeError(
                    "solution evaluator accepted a route rejected by executable behavior"
                )
            current = final
            executed_transitions += final_cache.executed_transitions
            reused_transitions += final_cache.reused_transitions

        return SearchResult(
            execution=current,
            moves=moves,
            iterations=iterations,
            evaluations=evaluations,
            generated=generated,
            executed_transitions=executed_transitions,
            reused_transitions=reused_transitions,
            elapsed_seconds=time.perf_counter() - started,
            decisions=tuple(decisions),
        )


__all__ = [
    "CandidateGenerator",
    "CandidateScorer",
    "ExecutableRefiner",
    "ExecutedBehaviorScorer",
    "NeighborhoodConfig",
    "ObjectiveScorer",
    "RoutingNeighborhood",
    "SearchConfig",
    "SearchDecision",
    "SearchResult",
]
