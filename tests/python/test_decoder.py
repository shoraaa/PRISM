import sys
from pathlib import Path

import numpy as np
import torch
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
        "verify_screening_resources": False,
        "verify_incremental_srr": False,
        "srr_exploration_budget": 0,
        "random_escape": False,
        "srr_exploration_margin": pytest.approx(1.0e-6),
    }


def _registry(solver) -> list[str]:
    """Registry order for THIS problem.

    A row's position depends on which constraints the instance declares, so it
    cannot be read off a global list: prism_decoder.FIELD_CHANNEL_NAMES
    enumerates the compiled fast paths, not any one problem's rows.
    """
    return list(solver.metadata["resource_names"])


def test_random_escape_lets_flat_constant_guidance_spend_budget() -> None:
    coordinates, distance = euclidean_problem(18, 1)
    problem = {
        "name": "tsp",
        "coordinates": coordinates,
        "distance": distance,
    }
    incumbent_solver = make_decoder(
        problem,
        candidate_config={"max_candidates": 17},
        search_config={"min_changed_edges": 4},
        n_rollouts=4,
        beta=2.0,
    )
    incumbent_solver.seed(9001)
    # Multiplier width is a per-problem fact: one slot per registry row plus the
    # objective slot. A TSP declares no constraint, so this is the objective
    # slot alone.
    constant_multipliers = np.zeros(
        incumbent_solver.metadata["multiplier_count"], dtype=np.float32
    )
    bootstrap = incumbent_solver.sample_greedy(
        multipliers=constant_multipliers
    )
    incumbent_solver.set_incumbent(bootstrap["route"])
    incumbent = incumbent_solver.solve(
        12, multipliers=constant_multipliers
    )

    def refine(random_escape: bool) -> dict:
        solver = make_decoder(
            problem,
            candidate_config={"max_candidates": 17},
            search_config={
                "min_changed_edges": 4,
                "srr_exploration_budget": 1,
                "random_escape": random_escape,
            },
            n_rollouts=1,
            beta=2.0,
        )
        solver.seed(23)
        solver.set_incumbent(incumbent["route"])
        return solver.solve(1, multipliers=constant_multipliers)

    disabled = refine(False)
    enabled = refine(True)

    assert disabled["srr_moves"] == 0
    assert enabled["srr_moves"] > 0
    assert enabled["objective"] <= incumbent["objective"] + 1.0e-6


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
    # generated_problem draws from the global torch RNG, so without a seed the
    # instance depends on whichever tests ran first -- and the coverage
    # assertions below (that the certified and incremental SRR paths are
    # exercised at all) are instance-dependent.
    torch.manual_seed(20260828)
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


def test_typed_field_mode_is_exposed_and_feasible() -> None:
    coordinates, distance = euclidean_problem(36, 102)
    solver = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=8,
    )
    # A TSP declares no constraint, so it carries no resource rows at all. The
    # registry used to open with one inactive row per compiled channel, which is
    # why this asserted seven names and an all-zero mask; a row's presence now
    # means the problem declared it, so the mask has no zeros left to check.
    channels = _registry(solver)
    assert channels == []
    assert solver.metadata["guidance_mode"] == "energy"
    assert solver.metadata["field_channel_mask"].shape == (0,)

    default_energy = solver.solve(1)
    assert default_energy["feasible"]

    field = np.ones(
        (solver.metadata["edge_count"], len(channels)), dtype=np.float32
    )
    multipliers = np.zeros(solver.metadata["multiplier_count"], dtype=np.float32)
    multipliers[-1] = 1.0
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
        (solver.resource_features, solver.metadata["resource_count"]),
    ):
        assert values.ndim == 2
        assert values.shape[1] == width
        assert np.isfinite(values).all()
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    solver.seed(2121)
    solver.solve(1)
    assert np.any(solver.node_features[:, 5] == 1.0)
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

    channels = _registry(solver)
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

    channels = _registry(solver)
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
    assert scales.shape == (solver.metadata["resource_count"],)
    assert np.all(scales > 0.0)
    assert np.allclose(solver.resource_features, expected)
    # The per-channel copy that used to sit in edge_features slots 1..7 is gone;
    # resource_features is the single per-resource carrier.
    assert "capacity" not in prism_decoder.CANDIDATE_FEATURE_NAMES

    resources = solver.evaluate_resources(np.array([0, 1, 2, 0]))
    assert resources["structurally_valid"]
    assert np.isclose(resources["violation"][0], 0.2)


def test_resource_semantics_are_invariant_to_physical_unit_rescaling() -> None:
    coordinates, distance = euclidean_problem(20, 124)
    rng = np.random.default_rng(125)
    demand = np.r_[0.0, rng.uniform(0.01, 0.04, 19)].astype(np.float32)

    def make_solver(factor: float) -> prism_decoder.Decoder:
        return make_decoder(
            {
                "name": "cvrp",
                "coordinates": coordinates,
                "distance": distance,
                "demand": demand * factor,
                "capacity": 0.4 * factor,
            },
            n_rollouts=1,
        )

    reference = make_solver(1.0)
    scaled = make_solver(100.0)
    capacity = _registry(reference).index("capacity")

    assert scaled.resource_scales[capacity] == pytest.approx(
        100.0 * reference.resource_scales[capacity]
    )
    assert np.allclose(reference.resource_features, scaled.resource_features)
    assert np.allclose(
        reference.resource_row_properties,
        scaled.resource_row_properties,
        atol=1e-7,
    )
    assert np.allclose(
        reference.resource_term_properties,
        scaled.resource_term_properties,
        atol=1e-7,
    )


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
    assert labels["violation"].shape == (solver.metadata["resource_count"],)
    assert labels["binding"].shape == (solver.metadata["resource_count"],)
    assert np.all(labels["violation"] >= 0.0)
    assert np.all((labels["binding"] >= 0.0) & (labels["binding"] <= 1.0))
    assert np.allclose(labels["violation"], 0.0, atol=1e-5)


def test_guidance_validation_and_every_registry_row_is_declared() -> None:
    coordinates, distance = euclidean_problem(28, 103)
    demand = np.r_[0.0, np.full(27, 0.03, dtype=np.float32)]

    def make_solver() -> prism_decoder.Decoder:
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
        solver.seed(919)
        return solver

    solver = make_solver()
    shape = (solver.metadata["edge_count"], solver.metadata["resource_count"])
    ones = np.ones(shape, dtype=np.float32)

    # This used to check that noise on an INACTIVE channel could not reach the
    # route. There is no inactive row to hide behind any more: the registry
    # holds exactly the constraints the problem declares, so guidance on a row
    # is guidance on something the instance actually enforces.
    assert _registry(solver) == ["capacity"]
    assert np.all(solver.metadata["field_channel_mask"] == 1)

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
                make_solver().metadata["multiplier_count"], dtype=np.float32
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
            n_rollouts=1,
        )
        solver.seed(1001)
        return solver

    baseline_solver = make_solver()
    shape = (
        baseline_solver.metadata["edge_count"],
        baseline_solver.metadata["resource_count"],
    )
    # High resource-field intensities with a unit objective weight (final slot),
    # so the field dominates the plain objective as this test intends.
    multipliers = np.full(baseline_solver.metadata["multiplier_count"], 100.0, dtype=np.float32)
    multipliers[-1] = 1.0
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
            n_rollouts=1,
        )

    baseline_solver = make_solver()
    shape = (
        baseline_solver.metadata["edge_count"],
        baseline_solver.metadata["resource_count"],
    )
    field = np.ones(shape, dtype=np.float32)
    additive = np.zeros(shape, dtype=np.float32)
    multipliers = np.zeros(baseline_solver.metadata["multiplier_count"], dtype=np.float32)
    multipliers[0] = 100.0
    multipliers[-1] = 1.0
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
            n_rollouts=1,
            beta=2.0,
        )
        solver.seed(10109)
        return solver

    baseline_solver = make_solver()
    field = np.zeros(
        (
            baseline_solver.metadata["edge_count"],
            baseline_solver.metadata["resource_count"],
        ),
        dtype=np.float32,
    )
    multipliers = np.zeros(baseline_solver.metadata["multiplier_count"], dtype=np.float32)
    multipliers[-1] = 1.0
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


def test_native_policy_is_invariant_to_positive_objective_rescaling() -> None:
    coordinates, distance = euclidean_problem(24, 110)

    def make_solver(factor: float) -> prism_decoder.Decoder:
        solver = make_decoder(
            {
                "name": "tsp",
                "coordinates": coordinates,
                "distance": distance * factor,
            },
            n_rollouts=5,
            beta=2.5,
        )
        solver.seed(10110)
        return solver

    reference_solver = make_solver(1.0)
    scaled_solver = make_solver(100.0)
    reference_scale = reference_solver.objective_energy_scale
    scaled_scale = scaled_solver.objective_energy_scale

    assert reference_scale > 0.0
    assert scaled_scale == pytest.approx(100.0 * reference_scale, rel=2e-6)
    assert np.array_equal(reference_solver.edge_index, scaled_solver.edge_index)
    assert np.allclose(
        reference_solver.objective_edge_costs / reference_scale,
        scaled_solver.objective_edge_costs / scaled_scale,
        atol=2e-6,
    )

    edge_count = reference_solver.metadata["edge_count"]
    objective_residual = np.linspace(-0.3, 0.3, edge_count, dtype=np.float32)
    reference = reference_solver.sample_traced(
        edge_field=np.zeros_like(reference_solver.resource_pressure),
        objective_residual=objective_residual,
    )
    scaled = scaled_solver.sample_traced(
        edge_field=np.zeros_like(scaled_solver.resource_pressure),
        objective_residual=objective_residual,
    )
    assert np.array_equal(
        reference["trace"]["chosen_indices"],
        scaled["trace"]["chosen_indices"],
    )
    assert np.allclose(
        reference["trace"]["log_probabilities"],
        scaled["trace"]["log_probabilities"],
        atol=2e-6,
    )
    for reference_solution, scaled_solution in zip(
        reference["solutions"], scaled["solutions"]
    ):
        assert np.array_equal(reference_solution["route"], scaled_solution["route"])
        assert scaled_solution["objective"] == pytest.approx(
            100.0 * reference_solution["objective"], rel=2e-6
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
            n_rollouts=1,
        )
        solver.seed(7001)
        solver.set_incumbent(incumbent)
        return solver

    ordinary_solver = make_solver()
    shape = (
        ordinary_solver.metadata["edge_count"],
        ordinary_solver.metadata["resource_count"],
    )
    field = rng.uniform(0.0, 2.0, shape).astype(np.float32)
    additive = rng.uniform(0.0, 0.2, shape).astype(np.float32)
    multipliers = np.zeros(ordinary_solver.metadata["multiplier_count"], dtype=np.float32)
    multipliers[0] = 1.3
    multipliers[-1] = 0.8
    ordinary = ordinary_solver.sample_greedy(
        edge_field=field,
        edge_additive=additive,
        multipliers=multipliers,
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
    )

    assert ordinary["srr_moves"] > 0
    assert scaled["srr_moves"] == ordinary["srr_moves"]
    assert np.array_equal(scaled["route"], ordinary["route"])
    assert scaled["objective"] == ordinary["objective"]


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

    # The problem declares no compiled constraint, so the battery row it appends
    # is the whole registry. It used to sit at position 7 behind seven rows the
    # instance never used.
    assert _registry(solver) == ["battery"]
    assert solver.metadata["multiplier_count"] == solver.metadata["resource_count"] + 1
    assert solver.resource_features.shape[1] == solver.metadata["resource_count"]
    assert solver.resource_row_properties.shape == (
        solver.metadata["resource_count"],
        prism_decoder.RESOURCE_ROW_PROPERTY_DIM,
    )
    assert solver.resource_term_properties.shape[1] == (
        prism_decoder.RESOURCE_TERM_PROPERTY_DIM
    )
    assert solver.resource_term_counts.shape == (
        solver.metadata["resource_count"],
    )
    # The seven compiled kernels remain the fast-path vocabulary, but a problem
    # that uses none of them publishes none of them: this list used to be
    # asserted as a fixed prefix ahead of the appended row.
    assert list(prism_decoder.FIELD_CHANNEL_NAMES) == [
        "capacity",
        "time_window",
        "route_limit",
        "tour_limit",
        "backhaul_order",
        "pickup_delivery",
        "prize_quota",
    ]
    assert [row["operator"] for row in solver.metadata["resources"]] == [
        "affine_accumulator"
    ]
    assert solver.metadata["resources"][-1]["name"] == "battery"
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


def test_optional_pre_transition_reset_enforces_driver_breaks() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    resource = {
        "name": "continuous_driving_time",
        "operator": "affine_accumulator",
        "initial": 0.0,
        "scale": 4.5,
        "increment": {"edge_attribute": "distance", "coefficient": 1.0},
        "reset": {
            "node_attribute": "break_allowed",
            "value": 0.0,
            "at_depot": True,
            "optional_before_transition": True,
            "duration": 0.75,
        },
        "bounds": [{"upper": 4.5, "check": "transition"}],
    }
    solver = make_decoder(
        {
            "name": "vrpdb",
            "coordinates": coordinates,
            "distance": distance,
            "constraints": [],
            "multi_route": False,
            "node_attributes": {
                "break_allowed": np.ones(4, dtype=np.float32)
            },
            "resources": [resource],
        },
        n_rollouts=1,
    )

    with_break = solver.evaluate(np.array([0, 1, 2, 0], dtype=np.int32))
    overlong_leg = solver.evaluate(np.array([0, 3, 0], dtype=np.int32))

    assert with_break["feasible"]
    assert not overlong_leg["feasible"]
    assert "continuous_driving_time" in overlong_leg["error"]
    labels = solver.evaluate_resources(np.array([0, 1, 2, 0], dtype=np.int32))
    assert labels["violation"][-1] == pytest.approx(0.0)


def test_driver_break_duration_advances_route_time() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    base = {
        "name": "vrpdb-time",
        "coordinates": coordinates,
        "distance": distance,
        "constraints": ["time_windows"],
        "multi_route": False,
        "tw_start": np.zeros(4, dtype=np.float32),
        "tw_end": np.array([100.0, 100.0, 100.0, 5.5], dtype=np.float32),
        "node_attributes": {"break_allowed": np.ones(4, dtype=np.float32)},
    }
    reset = {
        "node_attribute": "break_allowed",
        "value": 0.0,
        "at_depot": True,
        "optional_before_transition": True,
    }
    resource = {
        "name": "continuous_driving_time",
        "operator": "affine_accumulator",
        "initial": 0.0,
        "scale": 4.5,
        "increment": {"edge_attribute": "distance", "coefficient": 1.0},
        "reset": {**reset, "duration": 0.75},
        "bounds": [{"upper": 4.5, "check": "transition"}],
    }
    no_duration = {**resource, "reset": {**reset, "duration": 0.0}}
    route = np.array([0, 1, 2, 3, 0], dtype=np.int32)

    assert make_decoder(
        {**base, "resources": [no_duration]}, n_rollouts=1
    ).evaluate(route)["feasible"]
    assert not make_decoder(
        {**base, "resources": [resource]}, n_rollouts=1
    ).evaluate(route)["feasible"]


def test_driver_break_duration_is_charged_on_depot_return() -> None:
    coordinates = np.array([[0.0, 0.0], [2.5, 0.0]], dtype=np.float32)
    resource = {
        "name": "continuous_driving_time",
        "operator": "affine_accumulator",
        "initial": 0.0,
        "scale": 4.5,
        "increment": {"edge_attribute": "distance", "coefficient": 1.0},
        "reset": {
            "node_attribute": "break_allowed",
            "value": 0.0,
            "at_depot": True,
            "optional_before_transition": True,
            "duration": 0.75,
        },
        "bounds": [{"upper": 4.5, "check": "transition"}],
    }
    problem = {
        "name": "vrpdbtw-return",
        "coordinates": coordinates,
        "constraints": ["visit_all", "time_windows"],
        "multi_route": True,
        "tw_start": np.zeros(2, dtype=np.float32),
        "tw_end": np.array([5.5, 100.0], dtype=np.float32),
        "node_attributes": {"break_allowed": np.ones(2, dtype=np.float32)},
        "resources": [resource],
    }

    result = make_decoder(problem, n_rollouts=1).evaluate(
        np.array([0, 1, 0], dtype=np.int32)
    )

    assert not result["feasible"]


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


def test_objective_edge_term_slot_carries_the_declared_objective() -> None:
    """The edge slot reports the declared objective, not a route-structure flag.

    It replaced a hard-coded waived-open-return lever. On an open route the
    depot leg is free in the declared objective, so the slot reports that
    directly -- and it stays meaningful on objectives that declare no travel
    term at all, which the lever could not express.
    """
    names = list(prism_decoder.CANDIDATE_FEATURE_NAMES)
    slot = names.index("objective_edge_term")
    # No slot is named after a constraint or a route structure any more.
    assert not ({"capacity", "backhaul_order", "open_return"} & set(names))

    problem = generated_problem("ocvrpb", 16)
    solver = prism_decoder.Decoder(problem)
    edges = np.asarray(solver.edge_features)
    heads = np.asarray(solver.edge_index)[1]
    depot_arcs = heads < problem["depot_count"]
    # 0.5 is the squash's neutral point: a genuinely free leg. Every other arc
    # of a distance objective costs something, so it sits strictly above.
    assert np.allclose(edges[depot_arcs, slot], 0.5)
    assert np.all(edges[~depot_arcs, slot] > 0.5)

    # A closed route charges the return leg, so no arc is free.
    closed = prism_decoder.Decoder(generated_problem("cvrpb", 16))
    assert np.all(np.asarray(closed.edge_features)[:, slot] > 0.5)


def test_reverse_distance_slot_carries_the_opposite_leg() -> None:
    """The reverse leg is encoded even when its arc is not a candidate."""
    slot = list(prism_decoder.CANDIDATE_FEATURE_NAMES).index("reverse_distance")
    problem = generated_problem("acvrp", 40)
    # A tight candidate budget is what makes the point: rows are ranked per
    # source, so many reverse arcs are pruned away entirely.
    solver = prism_decoder.Decoder(problem, candidate_config={"max_candidates": 5})
    distance = np.asarray(problem["distance"], dtype=np.float32)
    # The distance scale is the instance's own magnitude, so a metric-closure
    # matrix normalizes by its maximum rather than by a floor.
    scale = float(distance.max())
    edges = np.asarray(solver.edge_features)
    tails, heads = np.asarray(solver.edge_index)
    assert np.allclose(edges[:, slot], distance[heads, tails] / scale, atol=1e-6)
    # Asymmetry is visible: the two legs of a candidate arc differ.
    assert np.abs(edges[:, 0] - edges[:, slot]).max() > 1e-3
    # A reverse leg is reported even where the reverse arc was pruned.
    candidates = {(int(a), int(b)) for a, b in zip(tails, heads)}
    assert any((b, a) not in candidates for a, b in candidates)

    symmetric = prism_decoder.Decoder(generated_problem("cvrp", 24))
    symmetric_edges = np.asarray(symmetric.edge_features)
    assert np.allclose(symmetric_edges[:, 0], symmetric_edges[:, slot])


def test_node_distance_profiles_replace_missing_coordinates() -> None:
    """Asymmetric instances have no x/y, so the profiles are their only geometry."""
    names = list(prism_decoder.NODE_FEATURE_NAMES)
    out_slot = names.index("mean_out_distance")
    in_slot = names.index("mean_in_distance")
    solver = prism_decoder.Decoder(generated_problem("acvrp", 24))
    nodes = np.asarray(solver.node_features)
    assert np.all(nodes[:, names.index("x")] == 0.0)
    assert np.all(nodes[:, names.index("y")] == 0.0)
    assert np.all(nodes[:, out_slot] > 0.0)
    assert np.all(nodes[:, in_slot] > 0.0)
    # Their difference is the node's own asymmetry, which x/y could never carry.
    assert np.abs(nodes[:, out_slot] - nodes[:, in_slot]).max() > 1e-3

    # Each profile is the mean candidate-arc distance on its own side.
    tails, heads = np.asarray(solver.edge_index)
    forward = np.asarray(solver.edge_features)[:, 0]
    for node in range(nodes.shape[0]):
        outgoing = forward[tails == node]
        incoming = forward[heads == node]
        assert nodes[node, out_slot] == pytest.approx(outgoing.mean(), abs=1e-6)
        assert nodes[node, in_slot] == pytest.approx(incoming.mean(), abs=1e-6)


def test_metric_symmetry_is_reported_separately_from_reversal_safety() -> None:
    """The metric's symmetry is model conditioning; reversal safety is not.

    A symmetric time-window instance is reversal-unsafe (the kernel is
    reversal-sensitive) while its metric is still perfectly symmetric, so the
    two predicates cannot share one flag.
    """
    euclidean = prism_decoder.Decoder(generated_problem("cvrptw", 16))
    assert euclidean.metadata["metric_symmetric"] is True
    assert euclidean.metadata["metric_skew"] == 0.0

    asymmetric = prism_decoder.Decoder(generated_problem("acvrp", 16))
    assert asymmetric.metadata["metric_symmetric"] is False
    assert 0.0 < asymmetric.metadata["metric_skew"] <= 1.0


def test_features_are_invariant_to_distance_and_time_unit_rescaling() -> None:
    """The same instance in different units must encode identically.

    Every scale used to be seeded at one and only grown, so it acted as
    max(1, true_max) and stopped normalizing anything measuring less than a unit
    across. That put the whole asymmetric family -- whose metric-closure
    distances shrink toward zero as node count grows -- into the encoder at a
    few percent of the range its Euclidean counterpart used, and made this test
    false for any sub-unit instance. Only the energy scale is unit-bearing: it
    converts energy, so it scales with the unit by construction.
    """
    coordinates, distance = euclidean_problem(24, 321)
    rng = np.random.default_rng(322)
    demand = np.r_[0.0, rng.uniform(0.01, 0.06, 23)].astype(np.float32)
    tw_start = np.r_[0.0, rng.uniform(0.0, 2.0, 23)].astype(np.float32)
    tw_end = (tw_start + 5.0).astype(np.float32)
    service = np.r_[0.0, np.full(23, 0.05, dtype=np.float32)].astype(np.float32)

    def make_solver(factor: float) -> prism_decoder.Decoder:
        return make_decoder(
            {
                "name": "cvrptw",
                "coordinates": (coordinates * factor).astype(np.float32),
                "distance": (distance * factor).astype(np.float32),
                "demand": demand,
                "capacity": 0.8,
                "tw_start": (tw_start * factor).astype(np.float32),
                "tw_end": (tw_end * factor).astype(np.float32),
                "service_time": (service * factor).astype(np.float32),
            },
            n_rollouts=1,
        )

    reference = make_solver(1.0)
    # 0.05 sits far below the unit floor that used to be applied; 100 far above.
    for factor in (0.05, 100.0):
        scaled = make_solver(factor)
        for name in (
            "node_features",
            "edge_features",
            "resource_features",
            "node_resource_features",
            "resource_row_properties",
            "resource_term_properties",
        ):
            assert np.allclose(
                getattr(reference, name), getattr(scaled, name), atol=1e-5
            ), f"{name} is not invariant to a {factor}x unit change"
        assert scaled.metadata["objective_scale"] == pytest.approx(
            reference.metadata["objective_scale"], rel=1e-4
        )
        assert scaled.metadata["metric_skew"] == pytest.approx(
            reference.metadata["metric_skew"], abs=1e-6
        )
        # The energy scale is the one quantity that must move with the unit.
        assert scaled.metadata["objective_energy_scale"] == pytest.approx(
            factor * reference.metadata["objective_energy_scale"], rel=1e-4
        )
