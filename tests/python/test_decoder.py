import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import prism_decoder  # noqa: E402
from problem_data import (  # noqa: E402
    BENCHMARK_VARIANTS,
    generated_problem,
    problem_schema,
)


def euclidean_problem(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coordinates = rng.random((size, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    return coordinates, distance


def make_decoder(problem: dict, *args, **kwargs):
    """Materialize explicit fixture semantics before calling the native API."""
    explicit = problem_schema(str(problem.get("name", "schema")))
    explicit.update(problem)
    return prism_decoder.Decoder(explicit, *args, **kwargs)


def test_tsp_perturbation_and_srr_improve_incumbent() -> None:
    coordinates, distance = euclidean_problem(60, 7)
    solver = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=8,
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
    solver = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.6,
        },
        n_rollouts=8,
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


def test_srr_uses_bounded_row_best_improvement_policy() -> None:
    problem = generated_problem("cvrp", 60, 77)
    solver = make_decoder(problem, n_rollouts=4)
    solver.seed(2026)
    solver.solve(1)
    refined = solver.solve(1)

    assert refined["feasible"]
    assert refined["srr_moves"] > 0
    assert refined["srr_evaluations"] > 0
    assert refined["srr_full_rebuilds"] == 0


def test_capacity_free_vrptw_uses_closed_multi_route_semantics() -> None:
    problem = generated_problem("vrptw", 20)
    solver = make_decoder(problem, n_rollouts=2)
    solver.seed(20260731)

    solution = solver.solve(2)

    assert solver.metadata["constraints"] == ["visit_all", "time_windows"]
    assert solver.metadata["multi_route"] is True
    assert solver.metadata["open_route"] is False
    assert solution["feasible"]
    assert solver.evaluate(solution["route"])["feasible"]


def test_search_configuration_is_exposed() -> None:
    coordinates, distance = euclidean_problem(20, 10)
    solver = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        candidate_config={"max_candidates": 12},
        search_config={
            "min_changed_edges": 5,
            "max_perturb_attempts": 20,
            "or_opt_max_segment": 2,
        },
        n_rollouts=2,
    )

    assert solver.metadata["max_candidates"] == 12
    assert solver.metadata["candidate_strategy"] == "distance"
    assert solver.metadata["search"] == {
        "min_changed_edges": 5,
        "max_perturb_attempts": 20,
        "or_opt_max_segment": 2,
        "feasibility_lookahead_depth": 2,
        "use_srr": True,
        "classical_behavior": True,
        "verify_screening_resources": False,
        "verify_incremental_srr": False,
    }


def test_all_110_benchmark_schemas_are_explicit_and_normalizable() -> None:
    for variant in BENCHMARK_VARIANTS:
        explicit = problem_schema(variant)
        if "tour_limit" in explicit["constraints"]:
            explicit["tour_limit"] = 1.0 if variant == "aop" else 4.0
        normalized = prism_decoder.normalize_problem_schema(explicit)
        for key in (
            "name",
            "constraints",
            "objective",
            "depot_count",
            "multi_route",
            "open_route",
            "capacity",
            "prize_quota",
        ):
            assert normalized[key] == explicit[key], (variant, key)


def test_name_only_input_is_rejected_as_an_incomplete_schema() -> None:
    coordinates, _ = euclidean_problem(8, 22)
    with pytest.raises(ValueError, match="explicit schema is missing 'constraints'"):
        prism_decoder.Decoder({"name": "cvrp", "coordinates": coordinates})


def test_explicit_schema_execution_is_independent_of_variant_name() -> None:
    named = generated_problem("mdcvrptw", 16, 50)
    renamed = dict(named, name="custom_stateful_schema")
    nameless = dict(named)
    nameless.pop("name")
    named_solver = make_decoder(named, n_rollouts=2)
    renamed_solver = make_decoder(renamed, n_rollouts=2)
    nameless_solver = make_decoder(nameless, n_rollouts=2)

    assert nameless_solver.metadata["name"] == "schema"
    assert named_solver.metadata["constraint_kernels"] == renamed_solver.metadata[
        "constraint_kernels"
    ]
    assert np.array_equal(named_solver.edge_features, renamed_solver.edge_features)
    assert np.array_equal(named_solver.edge_features, nameless_solver.edge_features)

    named_solver.seed(20260803)
    renamed_solver.seed(20260803)
    nameless_solver.seed(20260803)
    for _ in range(2):
        named_solution = named_solver.solve(1)
        renamed_solution = renamed_solver.solve(1)
        nameless_solution = nameless_solver.solve(1)
        assert np.array_equal(named_solution["route"], renamed_solution["route"])
        assert np.array_equal(named_solution["route"], nameless_solution["route"])
        assert named_solution["objective"] == renamed_solution["objective"]
        assert named_solution["objective"] == nameless_solution["objective"]
        assert named_solution["srr_evaluations"] == renamed_solution[
            "srr_evaluations"
        ]
        assert named_solution["srr_evaluations"] == nameless_solution[
            "srr_evaluations"
        ]


def test_candidate_graph_uses_only_kd_tree_distance_and_is_incumbent_stable(
) -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [100.0, 0.0]],
        dtype=np.float32,
    )
    solver = make_decoder(
        {
            "name": "tsp",
            "coordinates": coordinates,
            "constraints": ["visit_all", "pickup_delivery"],
            "pickup_delivery_pairs": np.array([[1, 3]], dtype=np.int32),
        },
        candidate_config={"max_candidates": 1},
        n_rollouts=1,
    )

    initial_edges = solver.edge_index.copy()
    assert np.array_equal(initial_edges[:, initial_edges[0] == 0], [[0], [1]])
    assert np.array_equal(initial_edges[:, initial_edges[0] == 1], [[1], [0]])

    # Neither the far pickup-delivery pair nor an incumbent edge is allowed to
    # override the purely spatial neighbourhood.
    solver.set_incumbent(np.array([0, 2, 1, 3], dtype=np.int32))
    assert np.array_equal(solver.edge_index, initial_edges)


def test_candidate_graph_keeps_required_depot_overlay() -> None:
    coordinates = np.array(
        [[100.0, 100.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    solver = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "demand": np.array([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
            "capacity": 0.8,
        },
        candidate_config={"max_candidates": 2},
        n_rollouts=1,
    )

    for customer in range(1, 4):
        destinations = solver.edge_index[1, solver.edge_index[0] == customer]
        assert 0 in destinations
    assert np.array_equal(
        solver.edge_index[1, solver.edge_index[0] == 0], np.array([1, 2, 3])
    )


@pytest.mark.parametrize(
    "variant",
    [
        "cvrp",
        "cvrpb",
        "cvrpl",
        "cvrptw",
        "cvrpbltw",
        "cvrpbp",
        "mdcvrptw",
        "pdtsp",
        "pdcvrp",
        "pctsp",
    ],
)
def test_incremental_screening_resources_match_full_evaluation_and_search(
    variant: str,
) -> None:
    problem = generated_problem(variant, 50 if variant == "pctsp" else 30, 20)

    def make_solver(verify: bool) -> prism_decoder.Decoder:
        solver = make_decoder(
            problem,
            search_config={
                "verify_screening_resources": verify,
                "verify_incremental_srr": verify,
            },
            n_rollouts=4,
        )
        solver.seed(8601)
        solver.solve(1)
        return solver

    ordinary = make_solver(False).sample()
    verified_solver = make_solver(True)
    traced = verified_solver.sample_traced()

    if variant in {"cvrp", "cvrpl", "cvrptw", "pctsp"}:
        assert traced["trace"]["screening_fast_evaluations"] > 0
    assert traced["trace"]["screening_verification_failures"] == 0
    if variant == "cvrp":
        assert sum(
            solution["srr_incremental_rebuilds"]
            for solution in traced["solutions"]
        ) > 0
    if variant in {
        "cvrpl",
        "cvrptw",
        "cvrpbltw",
        "cvrpbp",
        "mdcvrptw",
        "pdtsp",
        "pdcvrp",
    }:
        assert sum(
            solution["srr_certified_evaluations"]
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
        solver = make_decoder(
            {"name": "tsp", "coordinates": coordinates, "distance": distance},
            search_config=search_config or {},
            n_rollouts=8,
        )
        solver.seed(818)
        return solver.solve(1)

    implicit = run()
    explicit = run({"classical_behavior": True})

    assert np.array_equal(implicit["route"], explicit["route"])
    assert implicit["objective"] == explicit["objective"]


def test_typed_field_mode_is_exposed_and_feasible() -> None:
    coordinates, distance = euclidean_problem(36, 102)
    solver = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        search_config={"classical_behavior": False},
        n_rollouts=8,
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
    multipliers = np.zeros(prism_decoder.MULTIPLIER_COUNT, dtype=np.float32)
    multipliers[prism_decoder.FIELD_CHANNEL_COUNT] = 1.0
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
    solver = make_decoder(
        {
            "name": "cvrptw",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.8,
            "tw_start": tw_start,
            "tw_end": tw_end,
        },
        n_rollouts=4,
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


def test_incumbent_live_state_matches_transition_scales_and_timing() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    service_time = np.array([0.0, 2.0, 2.0, 2.0], dtype=np.float32)
    solver = make_decoder(
        {
            "name": "cvrpltw",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.zeros(4, dtype=np.float32),
            "route_limit": 20.0,
            "tw_start": np.zeros(4, dtype=np.float32),
            "tw_end": np.full(4, 30.0, dtype=np.float32),
            "service_time": service_time,
        },
        n_rollouts=1,
    )

    solver.set_incumbent(np.array([0, 1, 2, 3, 0], dtype=np.int32))

    channels = list(prism_decoder.FIELD_CHANNEL_NAMES)
    live = solver.incumbent_live_state[1]
    assert live[channels.index("time_window")] == pytest.approx(4.0 / 30.0)
    assert live[channels.index("route_limit")] == pytest.approx(2.0 / 20.0)


def test_incumbent_tour_state_uses_tour_limit() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    solver = make_decoder(
        {
            "name": "tour-test",
            "coordinates": coordinates,
            "distance": distance,
            "constraints": ["tour_limit"],
            "depot_count": 1,
            "tour_limit": 20.0,
            "service_time": np.array(
                [0.0, 2.0, 2.0, 2.0], dtype=np.float32
            ),
        },
        n_rollouts=1,
    )

    solver.set_incumbent(np.array([0, 1, 2, 3, 0], dtype=np.int32))

    channels = list(prism_decoder.FIELD_CHANNEL_NAMES)
    live = solver.incumbent_live_state[1]
    assert live[channels.index("tour_limit")] == pytest.approx(2.0 / 20.0)


def test_resource_features_use_exported_cpp_scales() -> None:
    coordinates, distance = euclidean_problem(3, 123)
    solver = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.array([0.0, 1.2, 1.2], dtype=np.float32),
            "capacity": 2.0,
        },
        n_rollouts=1,
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
    solver = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.5,
        },
        n_rollouts=4,
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
        solver = make_decoder(
            {"name": "tsp", "coordinates": coordinates, "distance": distance},
            search_config={"classical_behavior": False},
            n_rollouts=4,
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
    # Signed learned fields are valid; only non-finite values are rejected.
    signed = make_solver().solve(1, edge_field=-ones)
    assert signed["feasible"]
    invalid = ones.copy()
    invalid[0, 0] = np.nan
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        make_solver().solve(1, edge_field=invalid)
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        make_solver().solve(
            1,
            edge_field=ones,
            multipliers=-np.ones(
                prism_decoder.MULTIPLIER_COUNT, dtype=np.float32
            ),
        )

    classical = make_decoder(
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
        solver = make_decoder(
            {
                "name": "cvrp",
                "coordinates": coordinates,
                "distance": distance,
                "demand": demand,
                "capacity": 1.0,
            },
            search_config={"classical_behavior": False},
            n_rollouts=1,
        )
        solver.seed(1001)
        return solver

    baseline_solver = make_solver()
    shape = (
        baseline_solver.metadata["edge_count"],
        len(prism_decoder.FIELD_CHANNEL_NAMES),
    )
    # High resource-field intensities with a unit objective weight (final slot),
    # so the field dominates the plain objective as this test intends.
    multipliers = np.full(prism_decoder.MULTIPLIER_COUNT, 100.0, dtype=np.float32)
    multipliers[prism_decoder.FIELD_CHANNEL_COUNT] = 1.0
    baseline = baseline_solver.solve(
        1,
        edge_field=np.zeros(shape, np.float32),
        multipliers=multipliers,
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
        multipliers=multipliers,
    )

    assert guided["route"][0] == start
    assert guided["route"][1] == edge_index[1, chosen_edge]
    assert guided["route"][1] != original_next


def test_additive_field_guides_zero_pressure_edge() -> None:
    coordinates, distance = euclidean_problem(20, 107)
    demand = np.zeros(20, dtype=np.float32)

    def make_solver() -> prism_decoder.Decoder:
        return make_decoder(
            {
                "name": "cvrp",
                "coordinates": coordinates,
                "distance": distance,
                "demand": demand,
                "capacity": 1.0,
            },
            search_config={"classical_behavior": False},
            n_rollouts=1,
        )

    baseline_solver = make_solver()
    shape = (
        baseline_solver.metadata["edge_count"],
        prism_decoder.FIELD_CHANNEL_COUNT,
    )
    field = np.ones(shape, dtype=np.float32)
    additive = np.zeros(shape, dtype=np.float32)
    multipliers = np.zeros(prism_decoder.MULTIPLIER_COUNT, dtype=np.float32)
    multipliers[0] = 100.0
    multipliers[prism_decoder.FIELD_CHANNEL_COUNT] = 1.0
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


def test_signed_objective_residual_guides_multi_constraint_objective() -> None:
    coordinates, distance = euclidean_problem(20, 108)
    rng = np.random.default_rng(109)
    demand = np.r_[0.0, rng.uniform(0.01, 0.03, 19)].astype(np.float32)
    tw_start = np.zeros(20, dtype=np.float32)
    tw_end = np.full(20, 10.0, dtype=np.float32)
    problem = {
        "name": "cvrptw",
        "coordinates": coordinates,
        "distance": distance,
        "demand": demand,
        "capacity": 0.5,
        "tw_start": tw_start,
        "tw_end": tw_end,
    }

    def make_solver() -> prism_decoder.Decoder:
        solver = make_decoder(
            problem,
            search_config={"classical_behavior": False},
            n_rollouts=1,
            beta=2.0,
        )
        solver.seed(10109)
        return solver

    baseline_solver = make_solver()
    field = np.zeros(
        (
            baseline_solver.metadata["edge_count"],
            prism_decoder.FIELD_CHANNEL_COUNT,
        ),
        dtype=np.float32,
    )
    multipliers = np.zeros(prism_decoder.MULTIPLIER_COUNT, dtype=np.float32)
    multipliers[prism_decoder.FIELD_CHANNEL_COUNT] = 1.0
    baseline = baseline_solver.sample_greedy(
        edge_field=field,
        multipliers=multipliers,
    )
    start, original_next = baseline["route"][:2]

    guided_solver = make_solver()
    alternatives = np.flatnonzero(
        (guided_solver.edge_index[0] == start)
        & (guided_solver.edge_index[1] != original_next)
        & (guided_solver.edge_index[1] >= guided_solver.metadata["depot_count"])
    )
    assert alternatives.size > 0
    chosen_edge = int(alternatives[0])
    objective_residual = np.zeros(guided_solver.metadata["edge_count"], np.float32)
    # Residuals are energy corrections: lower energy is preferred.
    objective_residual[chosen_edge] = -100.0
    guided = guided_solver.sample_greedy(
        edge_field=field,
        multipliers=multipliers,
        objective_residual=objective_residual,
    )

    assert guided["feasible"]
    assert guided["route"][0] == start
    assert guided["route"][1] == guided_solver.edge_index[1, chosen_edge]
    active = guided_solver.metadata["field_channel_mask"]
    assert active[0] and active[1]

    with pytest.raises(ValueError, match="objective_residual must have shape"):
        guided_solver.sample_greedy(
            edge_field=field,
            multipliers=multipliers,
            objective_residual=np.zeros((objective_residual.size, 1), np.float32),
        )


def test_srr_aggregate_comparison_uses_the_same_edge_energy() -> None:
    rng = np.random.default_rng(1)
    size = 18
    coordinates = rng.random((size, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.03, 0.12, size - 1)].astype(np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": distance,
        "demand": demand,
        "capacity": 0.35,
    }
    bootstrap = make_decoder(problem, n_rollouts=4)
    bootstrap.seed(9001)
    incumbent = bootstrap.solve(1)["route"]

    def make_solver() -> prism_decoder.Decoder:
        solver = make_decoder(
            problem,
            search_config={"classical_behavior": False},
            n_rollouts=1,
        )
        solver.seed(7001)
        solver.set_incumbent(incumbent)
        return solver

    ordinary_solver = make_solver()
    shape = (
        ordinary_solver.metadata["edge_count"],
        prism_decoder.FIELD_CHANNEL_COUNT,
    )
    field = rng.uniform(0.0, 2.0, shape).astype(np.float32)
    additive = rng.uniform(0.0, 0.2, shape).astype(np.float32)
    risk = rng.uniform(0.0, 0.3, shape[0]).astype(np.float32)
    multipliers = np.zeros(prism_decoder.MULTIPLIER_COUNT, dtype=np.float32)
    multipliers[0] = 1.3
    multipliers[prism_decoder.FIELD_CHANNEL_COUNT] = 0.8
    risk_penalty = 0.7
    ordinary = ordinary_solver.sample_greedy(
        edge_field=field,
        edge_additive=additive,
        multipliers=multipliers,
        edge_risk=risk,
        risk_penalty=risk_penalty,
    )

    # Scaling every term in E(e|s) by the same positive constant preserves all
    # greedy edge rankings and SRR aggregate comparisons. This catches either
    # reintroducing analytic_pressure * edge_field in SRR or omitting the
    # objective multiplier there: both break the common scale and change moves.
    scale = 25.0
    scaled = make_solver().sample_greedy(
        edge_field=field,
        edge_additive=additive,
        multipliers=multipliers * scale,
        edge_risk=risk,
        risk_penalty=risk_penalty * scale,
    )

    assert ordinary["srr_moves"] > 0
    assert scaled["srr_moves"] == ordinary["srr_moves"]
    assert np.array_equal(scaled["route"], ordinary["route"])
    assert scaled["objective"] == ordinary["objective"]


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
        solver = make_decoder(
            problem,
            search_config={
                "classical_behavior": False,
                "feasibility_lookahead_depth": 1,
            },
            n_rollouts=1,
        )
        solver.seed(123)
        return solver

    solver = make_solver()
    shape = (solver.metadata["edge_count"], prism_decoder.FIELD_CHANNEL_COUNT)
    multipliers = np.zeros(prism_decoder.MULTIPLIER_COUNT, dtype=np.float32)
    multipliers[prism_decoder.FIELD_CHANNEL_COUNT] = 1.0
    guidance = {
        "edge_field": np.ones(shape, dtype=np.float32),
        "multipliers": multipliers,
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
    solver = make_decoder(
        {"name": "atsp", "distance": distance},
        n_rollouts=8,
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
    solver = make_decoder(
        {
            "name": "op",
            "coordinates": coordinates,
            "distance": distance,
            "prize": prize,
            "tour_limit": 3.0,
        },
        n_rollouts=8,
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
        solver = make_decoder(
            {"name": "tsp", "coordinates": coordinates, "distance": distance},
            n_rollouts=8,
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
    solver = make_decoder(
        {"name": "tsp", "coordinates": coordinates}, n_rollouts=1
    )

    solution = solver.evaluate(np.array([0, 1, 2], dtype=np.int32))

    assert solution["feasible"]
    assert solution["distance"] == pytest.approx(2.0 + np.sqrt(2.0))


def test_large_coordinate_problem_keeps_sparse_candidate_storage() -> None:
    size = 600
    rng = np.random.default_rng(907)
    coordinates = rng.random((size, 2), dtype=np.float32)
    demand = np.r_[0.0, np.full(size - 1, 0.01, dtype=np.float32)]
    solver = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "demand": demand,
            "capacity": 1.0,
        },
        candidate_config={"max_candidates": 64},
        n_rollouts=1,
    )

    # Customers have at most K edges; the depot keeps its mandatory overlay.
    assert solver.metadata["edge_count"] <= (size - 1) * 65


def test_runtime_battery_resource_enforces_reset_and_exports_dynamic_rows() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [0.8, 0.0], [0.4, 0.0]], dtype=np.float32
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": distance,
        "constraints": [],
        "multi_route": False,
        "node_attributes": {
            "charging_station": np.array([0.0, 0.0, 1.0], dtype=np.float32)
        },
        "resources": [
            {
                "name": "battery",
                "operator": "affine_accumulator",
                "initial": 1.2,
                "scale": 1.2,
                "increment": {
                    "edge_attribute": "distance",
                    "coefficient": -1.0,
                },
                "reset": {
                    "node_attribute": "charging_station",
                    "value": 1.2,
                },
                "bounds": [{"lower": 0.0, "check": "transition"}],
            }
        ],
    }
    solver = make_decoder(problem, n_rollouts=1)

    assert solver.metadata["resource_count"] == prism_decoder.FIELD_CHANNEL_COUNT + 1
    assert solver.metadata["multiplier_count"] == solver.metadata["resource_count"] + 1
    assert solver.resource_features.shape[1] == solver.metadata["resource_count"]
    assert solver.resource_descriptors.shape == (
        solver.metadata["resource_count"],
        prism_decoder.RESOURCE_DESCRIPTOR_DIM,
    )
    assert [row["operator"] for row in solver.metadata["resources"][:7]] == [
        "capacity",
        "time_window",
        "route_limit",
        "tour_limit",
        "backhaul_order",
        "pickup_delivery",
        "prize_quota",
    ]
    assert solver.metadata["resources"][-1]["name"] == "battery"
    assert solver.metadata["resources"][-1]["operator"] == "affine_accumulator"
    assert solver.metadata["field_channel_mask"][-1] == 1
    coordinate_only = dict(problem)
    coordinate_only.pop("distance")
    coordinate_solver = make_decoder(coordinate_only, n_rollouts=1)
    assert coordinate_solver.resource_features.shape[1] == solver.metadata[
        "resource_count"
    ]

    depleted = solver.evaluate(np.array([0, 1, 0], dtype=np.int32))
    recharged = solver.evaluate(np.array([0, 1, 2, 0], dtype=np.int32))
    assert not depleted["feasible"]
    assert "battery" in depleted["error"]
    assert recharged["feasible"]
    labels = solver.evaluate_resources(np.array([0, 1, 0], dtype=np.int32))
    assert labels["violation"].shape == (solver.metadata["resource_count"],)
    assert labels["violation"][-1] > 0.0


def test_dynaco_policy_refines_through_runtime_resource_schema() -> None:
    rng = np.random.default_rng(71)
    coordinates = rng.random((25, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    problem = {
        "name": "schema_route_cardinality",
        "coordinates": coordinates,
        "distance": distance,
        "constraints": ["visit_all"],
        "multi_route": True,
        "node_attributes": {
            "unit": np.r_[0.0, np.ones(24)].astype(np.float32),
            "depot_reset": np.r_[1.0, np.zeros(24)].astype(np.float32),
        },
        "resources": [
            {
                "name": "route_cardinality",
                "operator": "affine_accumulator",
                "initial": 0.0,
                "scale": 4.0,
                "increment": {
                    "node_attribute": "unit",
                    "coefficient": 1.0,
                },
                "reset": {
                    "node_attribute": "depot_reset",
                    "value": 0.0,
                },
                "bounds": [{"upper": 4.0, "check": "transition"}],
            }
        ],
    }
    solver = make_decoder(
        problem,
        search_config={"verify_incremental_srr": True},
        n_rollouts=4,
    )
    solver.seed(71)
    solver.solve(1)
    refined = solver.solve(1)

    exact = solver.evaluate(refined["route"])
    resources = solver.evaluate_resources(refined["route"])
    assert refined["feasible"] and exact["feasible"]
    assert refined["srr_moves"] > 0
    assert refined["srr_certified_evaluations"] > 0
    assert refined["srr_incremental_rebuilds"] == refined["srr_moves"]
    assert refined["srr_full_rebuilds"] == 0
    assert resources["violation"][-1] == 0.0


def test_typed_candidate_quota_admits_reset_edge_behind_distance_gate() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    solver = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "constraints": [],
            "multi_route": False,
            "node_attributes": {
                "charging_station": np.array(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                )
            },
            "resources": [
                {
                    "name": "battery",
                    "operator": "affine_accumulator",
                    "initial": 10.0,
                    "scale": 10.0,
                    "increment": {
                        "edge_attribute": "distance",
                        "coefficient": -1.0,
                    },
                    "reset": {
                        "node_attribute": "charging_station",
                        "value": 10.0,
                    },
                    "bounds": [{"lower": 0.0}],
                }
            ],
        },
        candidate_config={"max_candidates": 2},
        n_rollouts=1,
    )
    incumbent = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    solver.set_incumbent(incumbent)

    def neighbors(node: int) -> set[int]:
        start, end = solver.edge_offsets[node : node + 2]
        return set(solver.edge_index[1, start:end].tolist())

    assert neighbors(1) == {0, 2}
    quotas = np.zeros(solver.metadata["resource_count"], dtype=np.float32)
    quotas[-1] = 1.0
    solver.set_candidate_resource_quotas(quotas)
    solver.set_incumbent(incumbent)
    assert neighbors(1) == {0, 3}
    assert solver.metadata["candidate_strategy"] == "typed_resource_quota"


def test_schema_mode_admits_resource_candidate_without_learned_quota() -> None:
    """The schema-derived neighborhood covers a resource-relevant node with no
    learned quota and no per-variant tuning; the geometric ablation drops it."""
    coordinates = np.array(
        [[5.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [10.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": distance,
        "constraints": [],
        "demand": np.zeros(6, dtype=np.float32),
        "capacity": 1.0,
        "node_attributes": {
            "charging": np.array([0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        },
        "resources": [
            {
                "name": "battery",
                "operator": "affine_accumulator",
                "initial": 100.0,
                "scale": 100.0,
                "increment": {"edge_attribute": "distance", "coefficient": -1.0},
                "reset": {"node_attribute": "charging", "value": 100.0},
                "bounds": [{"lower": 0.0}],
            }
        ],
    }

    def neighbors(solver, node: int) -> set[int]:
        start, end = solver.edge_offsets[node : node + 2]
        return set(solver.edge_index[1, start:end].tolist())

    schema = make_decoder(
        problem, candidate_config={"max_candidates": 3}, n_rollouts=1
    )
    geometric = make_decoder(
        problem,
        candidate_config={"max_candidates": 3, "candidate_mode": "geometric"},
        n_rollouts=1,
    )
    # Node 5 is the farthest node (highest battery pressure) but not among node
    # 1's nearest neighbours. Schema mode reserves it via the uniform equal-share
    # prior; geometric mode fills purely by distance and excludes it.
    assert 5 in neighbors(schema, 1)
    assert 5 not in neighbors(geometric, 1)
    assert schema.metadata["candidate_strategy"] == "uniform_schema"
    assert geometric.metadata["candidate_strategy"] == "distance"
