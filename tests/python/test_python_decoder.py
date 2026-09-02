"""Differential contract for the readable Python semantic decoder.

The native backend remains the search implementation.  These tests define the
smaller compatibility surface that must agree before search can safely switch
backends: prefix legality, complete-route scoring, and execution-derived model
features.
"""

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
from decoder import Decoder as PythonDecoder  # noqa: E402
from problem_data import (  # noqa: E402
    BENCHMARK_VARIANTS,
    generated_problem,
    problem_schema,
)


def _problem(variant: str, seed: int, size: int = 8) -> dict:
    torch.manual_seed(seed)
    explicit = problem_schema(variant)
    explicit.update(generated_problem(variant, size))
    return explicit


def _deterministic_route(problem: dict) -> list[int]:
    variant = str(problem["name"])
    node_count = len(problem.get("coordinates", problem.get("distance")))
    depot_count = int(problem["depot_count"])
    customers = list(range(depot_count, node_count))
    if depot_count == 0:
        return list(range(node_count))
    if variant == "pdcvrp":
        pair_count = len(customers) // 2
        route = [0]
        for pickup in customers[:pair_count]:
            route.extend((pickup, pickup + pair_count, 0))
        return route
    if variant == "op":
        return [0, customers[0], 0]
    if variant == "pctsp":
        return [0, *customers, 0]
    route = [0]
    for customer in customers:
        route.extend((customer, 0))
    return route


@pytest.mark.parametrize(
    "variant",
    [
        "tsp",
        "cvrp",
        "cvrptw",
        "cvrpl",
        "cvrpb",
        "cvrpbp",
        "pdcvrp",
        "op",
        "pctsp",
    ],
)
def test_named_problem_masks_and_evaluation_match_native(variant: str) -> None:
    problem = _problem(variant, seed=101 + len(variant))
    native = prism_decoder.Decoder(problem)
    python = PythonDecoder(problem)
    route = _deterministic_route(problem)

    for length in range(len(route) + 1):
        np.testing.assert_array_equal(native.mask(route[:length]), python.mask(route[:length]))

    expected = native.evaluate(route)
    actual = python.evaluate(route)
    assert actual["feasible"] is expected["feasible"]
    if expected["feasible"]:
        assert actual["objective"] == pytest.approx(expected["objective"], abs=2e-6)
        assert actual["distance"] == pytest.approx(expected["distance"], abs=2e-6)
        assert actual["collected_prize"] == pytest.approx(
            expected["collected_prize"], abs=2e-6
        )
        assert actual["missed_penalty"] == pytest.approx(
            expected["missed_penalty"], abs=2e-6
        )
    else:
        assert actual["error"]


def test_v14_execution_state_and_margin_match_native() -> None:
    problem = _problem("cvrptw", seed=17)
    native = prism_decoder.Decoder(problem)
    python = PythonDecoder(problem)
    route = _deterministic_route(problem)
    assert native.evaluate(route)["feasible"]
    native.set_incumbent(np.asarray(route, dtype=np.int32))

    edge_index = np.asarray(native.edge_index)
    native_features = np.asarray(native.incumbent_transition_features)
    outgoing = np.flatnonzero(edge_index[0] == 1)
    destinations = edge_index[1, outgoing].tolist()
    python_features, python_mask = python.execution_features([0, 1], destinations)

    assert native.metadata["resource_names"] == python.metadata["resource_names"]
    assert python_mask.all()
    np.testing.assert_allclose(
        python_features, native_features[outgoing], rtol=1e-6, atol=1e-6
    )


def _pairwise_problem(scope: str) -> dict:
    size = 8
    predecessor = np.full(size, -1.0, dtype=np.float32)
    predecessor[5] = 1
    return {
        "name": "pairwise_reference",
        "constraints": ["visit_all"],
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "coordinates": np.random.default_rng(9).random((size, 2), dtype=np.float32),
        "capacity": 1.0,
        "demand": np.zeros(size, dtype=np.float32),
        "node_attributes": {"predecessor": predecessor},
        "resources": [
            {
                "name": "assembly_order",
                "operator": "precedence",
                "relation": "pairwise",
                "scope": scope,
                "predecessor": {"node_attribute": "predecessor"},
            }
        ],
    }


@pytest.mark.parametrize("scope", ["route", "solution"])
def test_declared_resource_program_matches_native(scope: str) -> None:
    problem = _pairwise_problem(scope)
    native = prism_decoder.Decoder(problem)
    python = PythonDecoder(problem)
    split = [0, 1, 2, 0, 5, 3, 4, 6, 7, 0]
    wrong_order = [0, 5, 1, 2, 3, 4, 6, 7, 0]

    np.testing.assert_array_equal(native.mask([0, 1, 2]), python.mask([0, 1, 2]))
    assert python.evaluate(split)["feasible"] is native.evaluate(split)["feasible"]
    assert python.evaluate(wrong_order)["feasible"] is False
    assert native.evaluate(wrong_order)["feasible"] is False


def test_new_accumulator_uses_the_same_legality_and_learning_interface() -> None:
    size = 5
    consumption = np.array([0.0, 0.4, 0.7, 0.3, 0.8], dtype=np.float32)
    problem = {
        "name": "energy_reference",
        "constraints": ["visit_all"],
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "coordinates": np.random.default_rng(4).random((size, 2), dtype=np.float32),
        "capacity": 1.0,
        "demand": np.zeros(size, dtype=np.float32),
        "node_attributes": {"energy": consumption},
        "resources": [
            {
                "name": "energy_budget",
                "operator": "affine_accumulator",
                "state_dim": 1,
                "direction": "forward",
                "scope": "route",
                "initial": 0.0,
                "scale": 1.0,
                "increment": {
                    "node_attribute": "energy",
                    "coefficient": 1.0,
                },
                "reset": {"at_depot": True, "value": 0.0},
                "bounds": [{"upper": 1.0}],
            }
        ],
    }
    route = [0, 1, 0, 2, 0, 3, 0, 4, 0]
    native = prism_decoder.Decoder(problem)
    python = PythonDecoder(problem)

    np.testing.assert_array_equal(native.mask([0, 1]), python.mask([0, 1]))
    assert native.mask([0, 1])[2] == 0
    assert native.evaluate(route)["feasible"]
    assert python.evaluate(route)["feasible"]

    native.set_incumbent(np.asarray(route, dtype=np.int32))
    edge_index = np.asarray(native.edge_index)
    outgoing = np.flatnonzero(edge_index[0] == 1)
    destinations = edge_index[1, outgoing].tolist()
    python_features, _ = python.execution_features([0, 1], destinations)
    np.testing.assert_allclose(
        python_features,
        np.asarray(native.incumbent_transition_features)[outgoing],
        rtol=1e-6,
        atol=1e-6,
    )


def test_prefix_legality_matches_across_the_110_variant_grid() -> None:
    checked_states = 0
    for index, variant in enumerate(BENCHMARK_VARIANTS):
        problem = _problem(variant, seed=1000 + index)
        native = prism_decoder.Decoder(problem)
        python = PythonDecoder(problem)
        prefix: list[int] = []

        # Follow a deterministic native-legal walk. Prefer customers so the
        # sweep exercises live resource state rather than depot-only prefixes.
        for _ in range(24):
            expected = np.asarray(native.mask(prefix), dtype=np.uint8)
            np.testing.assert_array_equal(
                expected,
                python.mask(prefix),
                err_msg=f"{variant}: prefix={prefix}",
            )
            checked_states += 1
            legal = np.flatnonzero(expected)
            if not len(legal):
                break
            customers = [
                int(node)
                for node in legal
                if node >= int(problem["depot_count"])
            ]
            prefix.append(customers[0] if customers else int(legal[0]))

    assert checked_states >= 1000


def test_decoder_core_contains_no_named_constraint_frontend() -> None:
    source = (ROOT / "decoder.py").read_text()
    forbidden = (
        '"capacity"',
        '"time_windows"',
        '"time_window"',
        '"route_limit"',
        '"tour_limit"',
        '"backhaul_order"',
        '"pickup_delivery"',
        '"prize_quota"',
        '"visit_all"',
        "self.demand",
        "self.tw_start",
        "self.tw_end",
        "self.service_time",
    )
    assert not [token for token in forbidden if token in source]

    frontend = (ROOT / "program.py").read_text()
    assert '"time_windows"' in frontend
    assert '"pickup_delivery"' in frontend
