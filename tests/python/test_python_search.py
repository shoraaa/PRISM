"""Generic Python refinement over exact executed behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import prism_decoder  # noqa: E402
from decoder import Decoder  # noqa: E402
from net import ConstraintFieldNet, decode_python_refinement  # noqa: E402
from problem_data import generated_problem, problem_schema  # noqa: E402
from search import (  # noqa: E402
    ExecutableRefiner,
    ExecutedBehaviorScorer,
    NeighborhoodConfig,
    RoutingNeighborhood,
    SearchConfig,
)


def _problem(name: str, size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    problem = problem_schema(name)
    problem.update(generated_problem(name, size))
    return problem


def test_executed_trace_is_anonymous_and_includes_implicit_closure() -> None:
    decoder = Decoder(_problem("cvrptw", 8, 11))
    route = [0]
    for node in range(1, decoder.node_count):
        route.extend((node, 0))

    execution = decoder.execute(route)

    assert execution.solution["feasible"]
    assert execution.transitions
    assert all(
        len(transition.live_state) == len(decoder.programs)
        for transition in execution.transitions
    )
    assert all(
        len(transition.resources) == len(decoder.programs)
        for transition in execution.transitions
    )
    tsp = Decoder(_problem("tsp", 8, 12)).execute(range(8))
    assert tsp.transitions[-1].implicit
    assert (tsp.transitions[-1].origin, tsp.transitions[-1].destination) == (7, 0)


@pytest.mark.parametrize("name", ["tsp", "cvrp", "cvrptw"])
def test_suffix_execution_matches_full_execution(name: str) -> None:
    decoder = Decoder(_problem(name, 12, 70 + len(name)))
    if decoder.depot_count == 0:
        route = list(range(decoder.node_count))
        candidate = route[:-2] + [route[-1], route[-2]]
    else:
        route = [0]
        for node in range(decoder.depot_count, decoder.node_count):
            route.extend((node, 0))
        candidate = route[:-4] + route[-2:] + route[-4:-2]

    base = decoder.execution_cache(route)
    cached = decoder.execute_candidate(base, candidate)
    full = decoder.execution_cache(candidate)

    assert cached.execution.solution["feasible"] is full.execution.solution["feasible"]
    assert cached.execution.solution["objective"] == pytest.approx(
        full.execution.solution["objective"], abs=1e-12
    )
    assert cached.execution.transitions == full.execution.transitions
    assert cached.reused_transitions > 0
    assert cached.executed_transitions < full.executed_transitions


@pytest.mark.parametrize("name", ["tsp", "cvrp", "cvrptw", "op"])
def test_python_refinement_preserves_feasibility_and_is_monotone(name: str) -> None:
    decoder = Decoder(_problem(name, 14, 20 + len(name)))
    if decoder.depot_count == 0:
        route = list(range(decoder.node_count))
    elif decoder.requires_all_visits:
        route = [0]
        for node in range(decoder.depot_count, decoder.node_count):
            route.extend((node, 0))
    else:
        route = [0, decoder.depot_count, 0]
    before = decoder.evaluate(route)
    assert before["feasible"]

    result = ExecutableRefiner(
        decoder,
        RoutingNeighborhood(
            decoder,
            NeighborhoodConfig(nearest_neighbors=8, max_segment_length=2),
        ),
        config=SearchConfig(
            max_iterations=2,
            max_evaluations=500,
            max_candidates_per_iteration=250,
        ),
    ).refine(route)

    assert result.solution["feasible"]
    if before["direction"] == "maximize":
        assert result.solution["objective"] >= before["objective"]
    else:
        assert result.solution["objective"] <= before["objective"]
    assert decoder.evaluate(result.solution["route"])["feasible"]


class _FixedCandidates:
    def __init__(self, candidates: list[list[int]]):
        self.candidates = candidates

    def generate(self, route, decoder, scope=None):
        del route, decoder, scope
        yield from self.candidates


def test_learned_execution_energy_ranks_moves_without_move_labels() -> None:
    distance = np.full((4, 4), 50.0, dtype=np.float32)
    np.fill_diagonal(distance, 0.0)
    for origin, destination, value in (
        (0, 1, 10), (1, 2, 10), (2, 3, 10), (3, 0, 10),
        (0, 2, 1), (2, 1, 10), (1, 3, 1),
        (3, 2, 10), (2, 0, 1),
    ):
        distance[origin, destination] = value
    problem = {
        "name": "schema",
        "constraints": ["visit_all"],
        "objective": "distance",
        "depot_count": 0,
        "multi_route": False,
        "open_route": False,
        "distance": distance,
    }
    decoder = Decoder(problem)
    edge_index = np.asarray(
        [
            (origin, destination)
            for origin in range(4)
            for destination in range(4)
            if origin != destination
        ],
        dtype=np.int64,
    ).T
    edge_lookup = {
        tuple(edge): index for index, edge in enumerate(edge_index.T.tolist())
    }
    residual = np.zeros(edge_index.shape[1], dtype=np.float32)
    # Both alternatives have exact objective 22.  Anonymous learned edge
    # behavior prefers the second route [0, 1, 3, 2].
    for edge in ((0, 1), (1, 3), (3, 2), (2, 0)):
        residual[edge_lookup[edge]] = -2.0
    scorer = ExecutedBehaviorScorer(
        decoder,
        edge_index,
        edge_field=np.zeros((edge_index.shape[1], 0), dtype=np.float32),
        multipliers=np.ones(1, dtype=np.float32),
        objective_residual=residual,
        objective_energy_scale=1.0,
    )
    result = ExecutableRefiner(
        decoder,
        _FixedCandidates([[0, 2, 1, 3], [0, 1, 3, 2]]),
        scorer=scorer,
        config=SearchConfig(
            max_iterations=1,
            max_evaluations=2,
            max_candidates_per_iteration=2,
        ),
    ).refine([0, 1, 2, 3], record_decisions=True)

    assert result.execution.route == (0, 1, 3, 2)
    assert len(result.decisions) == 1
    assert result.decisions[0].selected_index == 1


def test_model_can_refine_through_python_without_native_srr() -> None:
    problem = _problem("tsp", 10, 31)
    native = prism_decoder.Decoder(problem)
    incumbent = np.arange(10, dtype=np.int32)
    assert native.evaluate(incumbent)["feasible"]
    native.set_incumbent(incumbent)
    before = float(native.best_solution["objective"])

    solution, output, result = decode_python_refinement(
        problem,
        native,
        ConstraintFieldNet(depth=2, units=16).eval(),
        search_config=SearchConfig(
            max_iterations=1,
            max_evaluations=80,
            max_candidates_per_iteration=80,
        ),
        neighborhood_config=NeighborhoodConfig(
            nearest_neighbors=5,
            max_segment_length=1,
        ),
        install=False,
        record_decisions=True,
    )

    assert output["residual"].shape[0] == native.metadata["edge_count"]
    assert solution["feasible"]
    assert solution["objective"] <= before
    assert result.evaluations > 0
