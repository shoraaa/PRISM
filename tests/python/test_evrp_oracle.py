from __future__ import annotations

import numpy as np
import pytest

import prism_decoder
from evrp_oracle import solve_evrp


def _single_customer_driver_problem(depot_deadline: float) -> dict:
    coordinates = np.array([[0.0, 0.0], [2.5, 0.0]], dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    return {
        "name": "vrpdbtw-oracle",
        "coordinates": coordinates,
        "distance": distance,
        "constraints": ["visit_all", "time_windows"],
        "objective": "distance",
        "depot_count": 1,
        "multi_route": True,
        "open_route": False,
        "tw_start": np.zeros(2, dtype=np.float32),
        "tw_end": np.array([depot_deadline, 100.0], dtype=np.float32),
        "service_time": np.zeros(2, dtype=np.float32),
        "node_attributes": {"break_allowed": np.ones(2, dtype=np.float32)},
        "resources": [
            {
                "name": "continuous_driving_time",
                "operator": "affine_accumulator",
                "scope": "route",
                "direction": "forward",
                "initial": 0.0,
                "scale": 4.5,
                "increment": {
                    "edge_attribute": "distance",
                    "coefficient": 1.0,
                },
                "reset": {
                    "value": 0.0,
                    "at_depot": True,
                    "node_attribute": "break_allowed",
                    "optional_before_transition": True,
                    "duration": 0.75,
                },
                "bounds": [{"upper": 4.5, "check": "transition"}],
            }
        ],
    }


def test_ortools_driver_break_route_matches_native_decoder() -> None:
    problem = _single_customer_driver_problem(depot_deadline=6.0)

    oracle = solve_evrp(problem, time_limit_s=0.2)
    evaluated = prism_decoder.Decoder(problem).evaluate(
        np.asarray(oracle.route, dtype=np.int32)
    )

    assert oracle.route == [0, 1, 0]
    assert oracle.break_nodes == [1]
    assert evaluated["feasible"]


def test_ortools_driver_break_duration_respects_depot_deadline() -> None:
    problem = _single_customer_driver_problem(depot_deadline=5.5)

    with pytest.raises(RuntimeError, match="no feasible"):
        solve_evrp(problem, time_limit_s=0.1)
