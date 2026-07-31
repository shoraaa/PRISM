import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import prism_decoder  # noqa: E402
from problem_data import generated_problem  # noqa: E402


def euclidean_problem(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coordinates = rng.random((size, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    return coordinates, distance


def test_tsp_perturbation_and_srr_improve_incumbent() -> None:
    coordinates, distance = euclidean_problem(60, 7)
    solver = prism_decoder.Decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_ants=8,
    )
    solver.seed(20260727)

    bootstrap = solver.solve(1)
    refined = solver.solve(1)

    assert bootstrap["feasible"]
    assert refined["feasible"]
    assert solver.evaluate(refined["route"])["feasible"]
    assert refined["objective"] < bootstrap["objective"]
    assert refined["changed_edges"] > 0
    assert refined["srr_moves"] > 0
    assert 0 < refined["srr_scope_nodes"] <= 60
    assert refined["srr_revisits"] > 0
    assert refined["objective"] <= refined["raw_objective"]


def test_cvrp_uses_same_perturbation_backend() -> None:
    coordinates, distance = euclidean_problem(61, 8)
    rng = np.random.default_rng(9)
    demand = np.r_[0.0, rng.uniform(0.02, 0.09, 60)].astype(np.float32)
    solver = prism_decoder.Decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.6,
        },
        n_ants=8,
    )
    solver.seed(20260728)

    bootstrap = solver.solve(1)
    refined = solver.solve(1)

    assert bootstrap["feasible"]
    assert refined["feasible"]
    assert solver.evaluate(refined["route"])["feasible"]
    assert refined["objective"] < bootstrap["objective"]
    assert refined["changed_edges"] > 0
    assert refined["srr_moves"] > 0


def test_capacity_free_vrptw_uses_closed_multi_route_semantics() -> None:
    problem = generated_problem("vrptw", 20)
    solver = prism_decoder.Decoder(problem, n_ants=2)
    solver.seed(20260731)

    solution = solver.solve(2)

    assert solver.metadata["constraints"] == ["visit_all", "time_windows"]
    assert solver.metadata["multi_route"] is True
    assert solver.metadata["open_route"] is False
    assert solution["feasible"]
    assert solver.evaluate(solution["route"])["feasible"]


def test_search_configuration_is_exposed() -> None:
    coordinates, distance = euclidean_problem(20, 10)
    solver = prism_decoder.Decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        candidate_config={"max_candidates": 12},
        search_config={
            "min_changed_edges": 5,
            "max_perturb_attempts": 20,
            "or_opt_max_segment": 2,
        },
        n_ants=2,
    )

    assert solver.metadata["max_candidates"] == 12
    assert solver.metadata["search"] == {
        "min_changed_edges": 5,
        "max_perturb_attempts": 20,
        "or_opt_max_segment": 2,
        "feasibility_lookahead_depth": 2,
        "use_srr": True,
        "classical_behavior": True,
        "use_pheromone": True,
        "verify_screening_resources": False,
        "verify_incremental_srr": False,
    }


@pytest.mark.parametrize(
    "variant", ["cvrp", "cvrpb", "cvrpl", "cvrptw", "pctsp"]
)
def test_o1_screening_resources_match_full_evaluation_and_search(
    variant: str,
) -> None:
    problem = generated_problem(variant, 50 if variant == "pctsp" else 30, 20)

    def make_solver(verify: bool) -> prism_decoder.Decoder:
        solver = prism_decoder.Decoder(
            problem,
            search_config={
                "use_pheromone": False,
                "verify_screening_resources": verify,
                "verify_incremental_srr": verify,
            },
            n_ants=4,
        )
        solver.seed(8601)
        solver.solve(1)
        return solver

    ordinary = make_solver(False).sample()
    verified_solver = make_solver(True)
    traced = verified_solver.sample_traced()

    if variant in {"cvrp", "cvrpl", "pctsp"}:
        assert traced["trace"]["screening_fast_evaluations"] > 0
    assert traced["trace"]["screening_verification_failures"] == 0
    if variant == "cvrp":
        assert sum(
            solution["srr_incremental_rebuilds"]
            for solution in traced["solutions"]
        ) > 0
    assert len(ordinary) == len(traced["solutions"])
    for expected, actual in zip(ordinary, traced["solutions"]):
        assert np.array_equal(expected["route"], actual["route"])
        assert expected["objective"] == actual["objective"]
        assert expected["srr_moves"] == actual["srr_moves"]
        assert actual["srr_incremental_rebuilds"] == actual["srr_moves"]
        assert actual["srr_full_rebuilds"] == 0


def test_classical_behavior_flag_matches_default() -> None:
    coordinates, distance = euclidean_problem(35, 101)

    def run(search_config: dict | None = None) -> dict:
        solver = prism_decoder.Decoder(
            {"name": "tsp", "coordinates": coordinates, "distance": distance},
            search_config=search_config or {},
            n_ants=8,
        )
        solver.seed(818)
        return solver.solve(1)

    implicit = run()
    explicit = run({"classical_behavior": True})

    assert np.array_equal(implicit["route"], explicit["route"])
    assert implicit["objective"] == explicit["objective"]


def test_typed_field_mode_is_exposed_and_feasible() -> None:
    coordinates, distance = euclidean_problem(36, 102)
    solver = prism_decoder.Decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        search_config={"classical_behavior": False},
        n_ants=8,
    )
    channels = list(prism_decoder.FIELD_CHANNEL_NAMES)
    assert channels == [
        "capacity",
        "time_window",
        "route_limit",
        "tour_limit",
        "backhaul_order",
        "pickup_delivery",
        "prize_quota",
    ]
    assert solver.metadata["guidance_mode"] == "field"
    assert np.array_equal(
        solver.metadata["field_channel_mask"],
        np.zeros(len(channels), dtype=np.uint8),
    )

    with np.testing.assert_raises_regex(ValueError, "edge_field is required"):
        solver.solve(1)

    field = np.ones(
        (solver.metadata["edge_count"], len(channels)), dtype=np.float32
    )
    multipliers = np.zeros(len(channels), dtype=np.float32)
    version = solver.graph_version
    result = solver.solve(2, edge_field=field, multipliers=multipliers)

    assert result["feasible"]
    assert solver.evaluate(result["route"])["feasible"]
    assert solver.graph_version > version


def test_all_decoder_gnn_inputs_are_normalized() -> None:
    coordinates, distance = euclidean_problem(32, 120)
    rng = np.random.default_rng(121)
    demand = np.r_[0.0, rng.uniform(0.01, 0.08, 31)].astype(np.float32)
    tw_start = np.r_[0.0, rng.uniform(0.0, 2.0, 31)].astype(np.float32)
    tw_end = tw_start + 5.0
    solver = prism_decoder.Decoder(
        {
            "name": "cvrptw",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.8,
            "tw_start": tw_start,
            "tw_end": tw_end,
        },
        n_ants=4,
    )
    for values, width in (
        (solver.node_features, prism_decoder.NODE_FEATURE_COUNT),
        (solver.edge_features, prism_decoder.EDGE_FEATURE_COUNT),
        (solver.resource_features, prism_decoder.FIELD_CHANNEL_COUNT),
    ):
        assert values.ndim == 2
        assert values.shape[1] == width
        assert np.isfinite(values).all()
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    solver.seed(2121)
    solver.solve(1)
    assert np.any(solver.node_features[:, 12] == 1.0)
    assert np.isfinite(solver.node_features).all()
    assert np.all((solver.node_features >= 0.0) & (solver.node_features <= 1.0))


def test_resource_features_use_exported_cpp_scales() -> None:
    coordinates, distance = euclidean_problem(3, 123)
    solver = prism_decoder.Decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.array([0.0, 1.2, 1.2], dtype=np.float32),
            "capacity": 2.0,
        },
        n_ants=1,
    )

    scales = solver.resource_scales
    expected = np.clip(solver.resource_pressure / scales[None, :], 0.0, 1.0)
    assert scales.shape == (prism_decoder.FIELD_CHANNEL_COUNT,)
    assert np.all(scales > 0.0)
    assert np.allclose(solver.resource_features, expected)
    assert np.allclose(
        solver.edge_features[:, 1 : 1 + prism_decoder.FIELD_CHANNEL_COUNT],
        expected,
    )

    resources = solver.evaluate_resources(np.array([0, 1, 2, 0]))
    assert resources["structurally_valid"]
    assert np.isclose(resources["violation"][0], 0.2)


def test_resource_evaluator_returns_aligned_labels() -> None:
    coordinates, distance = euclidean_problem(25, 122)
    demand = np.r_[0.0, np.full(24, 0.04, dtype=np.float32)]
    solver = prism_decoder.Decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.5,
        },
        n_ants=4,
    )
    solver.seed(2222)
    solution = solver.solve(1)
    labels = solver.evaluate_resources(solution["route"])

    assert labels["structurally_valid"]
    assert labels["error"] == ""
    assert labels["violation"].shape == (prism_decoder.FIELD_CHANNEL_COUNT,)
    assert labels["binding"].shape == (prism_decoder.FIELD_CHANNEL_COUNT,)
    assert np.all(labels["violation"] >= 0.0)
    assert np.all((labels["binding"] >= 0.0) & (labels["binding"] <= 1.0))
    assert np.allclose(labels["violation"], 0.0, atol=1e-5)


def test_guidance_validation_and_inactive_channel_masking() -> None:
    coordinates, distance = euclidean_problem(28, 103)

    def make_solver() -> prism_decoder.Decoder:
        solver = prism_decoder.Decoder(
            {"name": "tsp", "coordinates": coordinates, "distance": distance},
            search_config={"classical_behavior": False},
            n_ants=4,
        )
        solver.seed(919)
        return solver

    solver = make_solver()
    shape = (solver.metadata["edge_count"], len(prism_decoder.FIELD_CHANNEL_NAMES))
    ones = np.ones(shape, dtype=np.float32)
    inactive_noise = np.random.default_rng(104).uniform(0.0, 20.0, shape).astype(
        np.float32
    )

    clean = solver.solve(1, edge_field=ones)
    noisy = make_solver().solve(1, edge_field=inactive_noise)
    assert np.array_equal(clean["route"], noisy["route"])
    assert clean["objective"] == noisy["objective"]

    with np.testing.assert_raises_regex(ValueError, "must have shape"):
        make_solver().solve(1, edge_field=ones[:, :-1])
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        make_solver().solve(1, edge_field=-ones)
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        make_solver().solve(
            1,
            edge_field=ones,
            multipliers=-np.ones(shape[1], dtype=np.float32),
        )

    classical = prism_decoder.Decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance}
    )
    with np.testing.assert_raises_regex(ValueError, "classical_behavior"):
        classical.solve(
            1,
            edge_field=np.zeros(
                (classical.metadata["edge_count"], shape[1]), dtype=np.float32
            ),
        )


def test_typed_field_changes_greedy_construction() -> None:
    coordinates, distance = euclidean_problem(24, 105)
    demand = np.r_[0.0, np.full(23, 0.02, dtype=np.float32)]

    def make_solver() -> prism_decoder.Decoder:
        solver = prism_decoder.Decoder(
            {
                "name": "cvrp",
                "coordinates": coordinates,
                "distance": distance,
                "demand": demand,
                "capacity": 1.0,
            },
            search_config={"classical_behavior": False},
            n_ants=1,
        )
        solver.seed(1001)
        return solver

    baseline_solver = make_solver()
    shape = (
        baseline_solver.metadata["edge_count"],
        len(prism_decoder.FIELD_CHANNEL_NAMES),
    )
    baseline = baseline_solver.solve(
        1,
        edge_field=np.ones(shape, np.float32),
        multipliers=np.full(shape[1], 100.0, dtype=np.float32),
    )
    start, original_next = baseline["route"][:2]

    guided_solver = make_solver()
    edge_index = guided_solver.edge_index
    alternatives = np.flatnonzero(
        (edge_index[0] == start) & (edge_index[1] != original_next)
    )
    assert alternatives.size > 0
    chosen_edge = int(alternatives[0])
    field = np.full(shape, 20.0, dtype=np.float32)
    field[chosen_edge, 0] = 0.0
    guided = guided_solver.solve(
        1,
        edge_field=field,
        multipliers=np.full(shape[1], 100.0, dtype=np.float32),
    )

    assert guided["route"][0] == start
    assert guided["route"][1] == edge_index[1, chosen_edge]
    assert guided["route"][1] != original_next


def test_additive_field_guides_zero_pressure_edge() -> None:
    coordinates, distance = euclidean_problem(20, 107)
    demand = np.zeros(20, dtype=np.float32)

    def make_solver() -> prism_decoder.Decoder:
        return prism_decoder.Decoder(
            {
                "name": "cvrp",
                "coordinates": coordinates,
                "distance": distance,
                "demand": demand,
                "capacity": 1.0,
            },
            search_config={"classical_behavior": False},
            n_ants=1,
        )

    baseline_solver = make_solver()
    shape = (
        baseline_solver.metadata["edge_count"],
        prism_decoder.FIELD_CHANNEL_COUNT,
    )
    field = np.ones(shape, dtype=np.float32)
    additive = np.zeros(shape, dtype=np.float32)
    multipliers = np.zeros(shape[1], dtype=np.float32)
    multipliers[0] = 100.0
    baseline = baseline_solver.sample_greedy(
        edge_field=field,
        edge_additive=additive,
        multipliers=multipliers,
    )
    start, original_next = baseline["route"][:2]

    guided_solver = make_solver()
    alternatives = np.flatnonzero(
        (guided_solver.edge_index[0] == start)
        & (guided_solver.edge_index[1] != original_next)
    )
    assert alternatives.size > 0
    chosen_edge = int(alternatives[0])
    assert guided_solver.resource_pressure[chosen_edge, 0] == 0.0
    additive[:, 0] = 20.0
    additive[chosen_edge, 0] = 0.0
    guided = guided_solver.sample_greedy(
        edge_field=field,
        edge_additive=additive,
        multipliers=multipliers,
    )

    assert guided["route"][0] == start
    assert guided["route"][1] == guided_solver.edge_index[1, chosen_edge]


def test_lookahead_risk_labels_and_avoids_time_window_dead_end() -> None:
    distance = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 10.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    problem = {
        "name": "tsptw",
        "distance": distance,
        "tw_start": np.zeros(3, dtype=np.float32),
        "tw_end": np.array([100.0, 5.0, 5.0], dtype=np.float32),
    }

    def make_solver() -> prism_decoder.Decoder:
        solver = prism_decoder.Decoder(
            problem,
            search_config={
                "classical_behavior": False,
                "use_pheromone": False,
                "feasibility_lookahead_depth": 1,
            },
            n_ants=1,
        )
        solver.seed(123)
        return solver

    solver = make_solver()
    shape = (solver.metadata["edge_count"], prism_decoder.FIELD_CHANNEL_COUNT)
    guidance = {
        "edge_field": np.ones(shape, dtype=np.float32),
        "multipliers": np.zeros(shape[1], dtype=np.float32),
    }
    traced = solver.sample_traced(**guidance)["trace"]
    first_edges = traced["feasibility_edges"][:2]
    first_labels = traced["feasibility_risk_labels"][:2]
    destinations = solver.edge_index[1, first_edges]
    labels_by_node = dict(zip(destinations.tolist(), first_labels.tolist()))

    assert labels_by_node[1] == 1.0
    assert labels_by_node[2] == 0.0

    baseline = make_solver().sample_greedy(**guidance)
    guided_solver = make_solver()
    risk = np.zeros(guided_solver.metadata["edge_count"], dtype=np.float32)
    risky_edge = np.flatnonzero(
        (guided_solver.edge_index[0] == 0)
        & (guided_solver.edge_index[1] == 1)
    )[0]
    risk[risky_edge] = 1.0
    guided = guided_solver.sample_greedy(
        **guidance, edge_risk=risk, risk_penalty=10.0
    )

    assert not baseline["feasible"]
    assert guided["feasible"]
    assert np.array_equal(guided["route"], np.array([0, 2, 1]))


def test_pheromone_ablation_skips_all_updates() -> None:
    coordinates, distance = euclidean_problem(40, 106)

    def run(use_pheromone: bool) -> tuple[dict, np.ndarray]:
        solver = prism_decoder.Decoder(
            {"name": "tsp", "coordinates": coordinates, "distance": distance},
            search_config={"use_pheromone": use_pheromone},
            n_ants=8,
        )
        solver.seed(1101)
        result = solver.solve(2)
        return result, solver.pheromone

    enabled, enabled_pheromone = run(True)
    disabled, disabled_pheromone = run(False)

    assert enabled["feasible"]
    assert disabled["feasible"]
    assert not np.all(enabled_pheromone == 1.0)
    assert np.all(disabled_pheromone == 1.0)


def test_directed_srr_improves_atsp_without_reversal() -> None:
    rng = np.random.default_rng(33)
    size = 30
    distance = rng.uniform(0.05, 1.0, (size, size)).astype(np.float32)
    np.fill_diagonal(distance, 0.0)
    for intermediate in range(size):
        distance = np.minimum(
            distance,
            distance[:, intermediate, None] + distance[intermediate, None, :],
        )
    solver = prism_decoder.Decoder(
        {"name": "atsp", "distance": distance},
        n_ants=8,
    )
    solver.seed(77)

    bootstrap = solver.solve(1)
    refined = solver.solve(1)

    assert refined["objective"] < bootstrap["objective"]
    assert refined["srr_moves"] > 0
    assert solver.evaluate(refined["route"])["feasible"]


def test_optional_srr_can_insert_unserved_nodes() -> None:
    coordinates, distance = euclidean_problem(31, 44)
    rng = np.random.default_rng(45)
    prize = np.r_[0.0, rng.uniform(0.1, 1.0, 30)].astype(np.float32)
    solver = prism_decoder.Decoder(
        {
            "name": "op",
            "coordinates": coordinates,
            "distance": distance,
            "prize": prize,
            "tour_limit": 3.0,
        },
        n_ants=8,
    )
    solver.seed(88)

    bootstrap = solver.solve(1)
    refined = solver.solve(1)

    assert refined["objective"] > bootstrap["objective"]
    assert len(refined["route"]) > len(bootstrap["route"])
    assert refined["srr_moves"] > 0
    assert solver.evaluate(refined["route"])["feasible"]


def test_static_ant_parallelism_is_deterministic() -> None:
    coordinates, distance = euclidean_problem(40, 91)
    configured = prism_decoder.get_max_threads()
    available = prism_decoder.get_available_threads()

    def solve_with(threads: int) -> dict:
        prism_decoder.set_num_threads(threads)
        solver = prism_decoder.Decoder(
            {"name": "tsp", "coordinates": coordinates, "distance": distance},
            n_ants=8,
        )
        solver.seed(909)
        solver.solve(1)
        return solver.solve(1)

    try:
        serial = solve_with(1)
        parallel = solve_with(min(4, available))
    finally:
        prism_decoder.set_num_threads(configured)

    assert np.array_equal(serial["route"], parallel["route"])
    assert serial["objective"] == parallel["objective"]
    assert serial["srr_moves"] == parallel["srr_moves"]
    assert serial["srr_evaluations"] == parallel["srr_evaluations"]


def test_coordinate_backed_distance_matches_euclidean_evaluation() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    solver = prism_decoder.Decoder(
        {"name": "tsp", "coordinates": coordinates}, n_ants=1
    )

    solution = solver.evaluate(np.array([0, 1, 2], dtype=np.int32))

    assert solution["feasible"]
    assert solution["distance"] == pytest.approx(2.0 + np.sqrt(2.0))


def test_large_coordinate_problem_keeps_sparse_candidate_storage() -> None:
    size = 600
    rng = np.random.default_rng(907)
    coordinates = rng.random((size, 2), dtype=np.float32)
    demand = np.r_[0.0, np.full(size - 1, 0.01, dtype=np.float32)]
    solver = prism_decoder.Decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "demand": demand,
            "capacity": 1.0,
        },
        candidate_config={"max_candidates": 64},
        n_ants=1,
    )

    # Customers have at most K edges; the depot keeps its mandatory overlay.
    assert solver.metadata["edge_count"] <= (size - 1) * 65
