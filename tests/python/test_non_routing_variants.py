"""The non-routing classes: bin packing, MKP, and sequential ordering.

All three reuse the resource algebra for hard feasibility. The properties worth
pinning are the ones that would silently degrade rather than raise: that the
routing name parser does not claim them, that the fixed-charge distance really
equals the bin count, that the declared bounds actually bind, and that a
precedence DAG orders the path it is declared over.

bpp and mkp carry no geometry and run complete candidate graphs; sop carries a
real asymmetric distance matrix and keeps the ordinary K-nearest neighbourhood.
"""

from __future__ import annotations

import numpy as np
import pytest

import prism_decoder
import problem_data
from problem_data import (
    BENCHMARK_VARIANTS,
    BPP_CAPACITY,
    BPP_DEMAND_HIGH,
    BPP_DEMAND_LOW,
    MKP_DIMENSIONS,
    PACKING_VARIANTS,
    candidate_limit,
    decoder_problem,
    generate_bpp_data,
    generate_mkp_data,
    problem_schema,
)


def _instance(variant: str, data: dict, index: int = 0) -> dict:
    return decoder_problem(variant, {key: value[index : index + 1] for key, value in data.items()})


def _solve(problem: dict, iterations: int = 32, seed: int = 7) -> dict:
    decoder = prism_decoder.Decoder(
        problem,
        candidate_config={"max_candidates": candidate_limit(problem, 64)},
    )
    decoder.seed(seed)
    return decoder.solve(iterations)


def _bins(route: list[int], demand: np.ndarray) -> list[float]:
    loads: list[float] = []
    current = 0.0
    for node in route:
        if node == 0:
            if current > 0.0:
                loads.append(current)
            current = 0.0
        else:
            current += float(demand[node])
    if current > 0.0:
        loads.append(current)
    return loads


def test_packing_variants_stay_out_of_the_routing_benchmark() -> None:
    # The 110-variant routing results are keyed to BENCHMARK_VARIANTS; adding a
    # class there would silently rebase every published mean.
    assert PACKING_VARIANTS == ["bpp", "mkp"]
    assert not set(PACKING_VARIANTS) & set(BENCHMARK_VARIANTS)
    assert problem_data.problem_variants("packing") == PACKING_VARIANTS


def test_bpp_schema_is_capacitated_and_not_backhauled() -> None:
    # "bpp" contains "bp", which the routing name parser reads as a backhaul
    # ordering, and contains no "cvrp", so it would receive no capacity at all.
    schema = problem_schema("bpp")
    assert schema["constraints"] == ["visit_all", "capacity"]
    assert "backhaul_order" not in schema["constraints"]
    assert schema["multi_route"] is True


def test_mkp_schema_selects_a_subset_and_maximizes_prize() -> None:
    schema = problem_schema("mkp")
    assert schema["constraints"] == []
    assert "visit_all" not in schema["constraints"]
    assert schema["multi_route"] is False
    assert schema["objective"]["sense"] == -1.0
    assert schema["objective"]["distance_coeff"] == 0.0


def test_bpp_generator_follows_the_deepaco_distribution() -> None:
    data = generate_bpp_data(64, 3)
    demand = data["demand"]
    assert demand.shape == (3, 65)
    assert (demand[:, 0] == 0.0).all()
    raw = data["item_size"][:, 1:]
    assert raw.min() >= BPP_DEMAND_LOW
    assert raw.max() <= BPP_DEMAND_HIGH
    assert (raw == raw.round()).all()
    # Demands are stored pre-divided by the capacity, so the declared bound is 1.
    assert np.allclose(demand[:, 1:].numpy(), (raw / BPP_CAPACITY).numpy())


def test_mkp_generator_normalizes_every_budget_to_the_same_bound() -> None:
    size = 40
    data = generate_mkp_data(size, 3)
    assert data["weight"].shape == (3, size + 1, MKP_DIMENSIONS)
    assert (data["budget"] == size // 2).all()
    assert (data["prize"][:, 0] == 0.0).all()
    assert (data["weight"][:, 0] == 0.0).all()
    # Well-stated: no single item exceeds a budget, and no budget is vacuous.
    weight = data["weight"][:, 1:]
    assert (weight.amax(dim=1) <= float(size // 2)).all()
    assert (weight.sum(dim=1) > float(size // 2)).all()


def test_packing_classes_run_complete_candidate_graphs() -> None:
    problem = _instance("bpp", generate_bpp_data(30, 1))
    # Every pairwise distance is equal, so a truncated neighbourhood would keep
    # an arbitrary index-ordered subset and strand the remaining items.
    assert candidate_limit(problem, 64) == 30
    routing = decoder_problem("cvrp", {"xy": np.zeros((1, 10, 2), dtype=np.float32)})
    assert candidate_limit(routing, 64) == 64


def test_bpp_objective_equals_the_bin_count() -> None:
    data = generate_bpp_data(60, 2)
    for index in range(2):
        problem = _instance("bpp", data, index)
        solution = _solve(problem)
        route = [int(node) for node in solution["route"]]
        loads = _bins(route, problem["demand"])
        assert solution["feasible"]
        # The fixed charge sits on the depot-out arc, so accumulated distance is
        # the bin count exactly -- that identity is what lets the existing
        # incremental search operators score bin packing with no special case.
        assert solution["objective"] == pytest.approx(len(loads))


def test_bpp_packs_every_item_within_capacity() -> None:
    data = generate_bpp_data(60, 2)
    for index in range(2):
        problem = _instance("bpp", data, index)
        demand = problem["demand"]
        route = [int(node) for node in _solve(problem)["route"]]
        assert sorted(node for node in route if node != 0) == list(range(1, 61))
        loads = _bins(route, demand)
        assert max(loads) <= 1.0 + 1e-6
        # A trivial packing is one item per bin; search must beat that.
        assert len(loads) < 60


def test_mkp_respects_every_declared_dimension() -> None:
    size = 40
    data = generate_mkp_data(size, 2)
    budget = float(data["budget"][0])
    for index in range(2):
        problem = _instance("mkp", data, index)
        assert len(problem["resources"]) == MKP_DIMENSIONS
        solution = _solve(problem)
        assert solution["feasible"]
        selected = [int(node) for node in solution["route"] if int(node) != 0]
        assert len(selected) == len(set(selected))
        assert selected, "a feasible knapsack should hold at least one item"
        weight = np.stack(
            [problem["node_attributes"][f"weight_{axis}"] for axis in range(MKP_DIMENSIONS)],
            axis=1,
        )
        used = weight[selected].sum(axis=0)
        assert (used <= budget + 1e-4).all()
        assert solution["objective"] == pytest.approx(
            float(problem["prize"][selected].sum()), abs=1e-4
        )


def test_mkp_bounds_actually_bind() -> None:
    # Every item is individually feasible, so an unbound row would take all of
    # them; the row is only doing work if the selection is a strict subset.
    size = 40
    problem = _instance("mkp", generate_mkp_data(size, 1))
    selected = [int(node) for node in _solve(problem)["route"] if int(node) != 0]
    assert 0 < len(selected) < size


# ---------------------------------------------------------------------------
# Sequential ordering: the general precedence DAG
# ---------------------------------------------------------------------------


def _sop_predecessors(data: dict, index: int = 0) -> list[list[int]]:
    rows = data["predecessors"][index].tolist()
    return [[value for value in row if value > 0] for row in rows]


def test_sop_schema_is_an_open_asymmetric_path() -> None:
    schema = problem_data.problem_schema("sop")
    assert schema["constraints"] == ["visit_all"]
    assert schema["open_route"] is True, "SOP is a path, not a closed tour"
    assert schema["multi_route"] is False


def test_sop_keeps_the_geometric_neighbourhood() -> None:
    # Unlike bpp/mkp, SOP carries a real distance matrix, so the K-nearest
    # neighbourhood -- and the locality argument resting on it -- still applies.
    problem = decoder_problem("sop", problem_data.generate_sop_data(40, 1))
    assert candidate_limit(problem, 64) == 64
    assert "sop" not in PACKING_VARIANTS
    assert "sop" in problem_data.SEQUENCING_VARIANTS


def test_sop_declares_one_dag_row_holding_every_relation() -> None:
    data = problem_data.generate_sop_data(30, 1)
    problem = decoder_problem("sop", data)
    rows = problem["resources"]
    assert len(rows) == 1
    assert rows[0]["relation"] == "dag"
    # A DAG is not a set of pairwise rows: decoder.cpp rejects a node taking
    # part in more than one pairwise relation across the whole registry, so a
    # decomposition would not load at all.
    assert rows[0]["predecessors"] == _sop_predecessors(data)
    assert max(len(row) for row in rows[0]["predecessors"]) > 1


def test_sop_solutions_respect_every_precedence_relation() -> None:
    size = 40
    data = problem_data.generate_sop_data(size, 2)
    for index in range(2):
        problem = decoder_problem(
            "sop", {key: value[index : index + 1] for key, value in data.items()}
        )
        predecessors = _sop_predecessors(data, index)
        solution = _solve(problem)
        assert solution["feasible"]
        route = [int(node) for node in solution["route"]]
        assert sorted(route) == list(range(size))
        assert route[0] == 0
        position = {node: at for at, node in enumerate(route)}
        for node, required in enumerate(predecessors):
            for before in required:
                assert position[before] < position[node], (
                    f"{before} must precede {node}"
                )
        # An open path scores its traversed arcs and nothing else -- no return.
        distance = problem["distance"]
        expected = sum(
            distance[route[at]][route[at + 1]] for at in range(len(route) - 1)
        )
        assert solution["objective"] == pytest.approx(expected, rel=1e-5)


def test_sop_precedence_actually_constrains_the_order() -> None:
    # The relation is only doing work if some pair is ordered against what the
    # distances alone would choose; check the declared relations are non-trivial.
    data = problem_data.generate_sop_data(40, 1)
    predecessors = _sop_predecessors(data)
    assert sum(len(row) for row in predecessors) > 40
