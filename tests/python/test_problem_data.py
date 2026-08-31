from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import problem_data
from problem_data import (
    ALL_VARIANTS,
    BENCHMARK_VARIANTS,
    TRAIN_VARIANTS,
    DatasetFinder,
    SavedProblems,
    VariantCurriculum,
    generate_vrptw_validation_data,
    generated_problem,
    load_saved_data,
)


def test_registry_includes_capacity_free_vrptw_training_problem() -> None:
    assert len(BENCHMARK_VARIANTS) == 110
    assert len(ALL_VARIANTS) == 111
    assert "vrptw" in TRAIN_VARIANTS
    assert "vrptw" not in BENCHMARK_VARIANTS
    assert "tsptw" not in ALL_VARIANTS
    assert TRAIN_VARIANTS == sorted(
        [
            "atsp",
            "acvrp",
            "tsp",
            "vrptw",
            "op",
            "pctsp",
            "cvrp",
            "cvrpb",
            "cvrpl",
            "cvrpbp",
            "ocvrpl",
            "ocvrpbp",
            "cvrptw",
            "ocvrp",
            "ocvrptw",
            "pdtsp",
            "pdcvrp",
            "opdcvrp",
            "mdocvrp",
            "amdocvrp",
            "mdcvrptw",
            "mdocvrptw",
        ],
        key=len,
    )


def test_problem_data_has_no_baseline_python_dependency() -> None:
    source = Path(problem_data.__file__).read_text()

    assert "sys.path" not in source
    assert "from data." not in source
    assert "from problem." not in source


def test_dataset_finder_prefers_configured_oracle_and_longest_run(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "cvrptw"
    directory.mkdir()
    torch.save({"xy": torch.zeros(1, 101, 2)}, directory / "cvrptw100_data.pt")
    torch.save({"cost": torch.ones(1)}, directory / "cvrptw100_pyvrp20s.pt")
    torch.save({"cost": torch.ones(1)}, directory / "cvrptw100_pyvrp400s.pt")

    paths = DatasetFinder(tmp_path).get("cvrptw", 100)

    assert paths is not None
    assert paths["data_file"] == "cvrptw100_data.pt"
    assert paths["solution_file"] == "cvrptw100_pyvrp400s.pt"


def test_dataset_finder_prefers_canonical_prism_data(tmp_path: Path) -> None:
    directory = tmp_path / "op"
    directory.mkdir()
    torch.save(torch.rand(2, 4, 3), directory / "op3_legacy.pt")
    canonical = {"xy": torch.rand(2, 4, 2), "prize": torch.rand(2, 4)}
    torch.save(canonical, directory / "op3_prism.pt")

    paths = DatasetFinder(tmp_path).get("op", 3)

    assert paths is not None
    assert paths["data_path"] == directory / "op3_prism.pt"


def test_dataset_metadata_can_reject_a_stale_solution_sidecar(tmp_path: Path) -> None:
    directory = tmp_path / "cvrp"
    directory.mkdir()
    data_path = directory / "cvrp4_prism.pt"
    torch.save({"xy": torch.rand(2, 5, 2), "demand": torch.rand(2, 5)}, data_path)
    torch.save({"cost": torch.ones(2)}, directory / "cvrp4_hgs.pt")
    data_path.with_suffix(".json").write_text('{"solution_file": null}\n')

    paths = DatasetFinder(tmp_path).get("cvrp", 4)

    assert paths is not None
    assert paths["solution_path"] is None


def test_persistent_suite_generator_covers_all_variants() -> None:
    from scripts.generate_benchmark_suite import generate_suite, infeasibility_report

    suite = generate_suite(size=8, count=2, seed=1234, capacity=20)

    assert set(suite) == set(BENCHMARK_VARIANTS)
    assert suite["tsp"]["xy"].shape == (2, 8, 2)
    assert suite["atsp"]["dist"].shape == (2, 8, 8)
    assert suite["mdcvrptw"]["xy"].shape == (2, 11, 2)
    assert suite["mdcvrptw"]["tw_end"].shape == (2, 11)
    assert suite["aopdcvrp"]["demand"].shape == (2, 9)
    assert infeasibility_report("cvrptw", suite["cvrptw"])["instances"] >= 0
    assert infeasibility_report("amdcvrpl", suite["amdcvrpl"])["instances"] == 0


def test_normalizes_packed_tensor_benchmark_to_canonical_schema(
    tmp_path: Path,
) -> None:
    from scripts.normalize_benchmarks import normalize_variant

    directory = tmp_path / "op"
    directory.mkdir()
    packed = torch.rand(2, 4, 3)
    torch.save(packed, directory / "op3_legacy.pt")

    target, count, status = normalize_variant(tmp_path, "op", 3)
    saved = torch.load(target, map_location="cpu", weights_only=False)

    assert count == 2
    assert status == "converted:op3_legacy.pt"
    assert set(saved) == {"xy", "prize"}
    assert torch.equal(saved["xy"], packed[:, :, :2])
    assert torch.equal(saved["prize"], packed[:, :, 2])
    assert DatasetFinder(tmp_path).get("op", 3)["data_path"] == target

    rebuilt, rebuilt_count, rebuilt_status = normalize_variant(
        tmp_path, "op", 3, force=True
    )
    assert rebuilt == target
    assert rebuilt_count == 2
    assert rebuilt_status == "converted:op3_legacy.pt"


def test_saved_problems_loads_without_baseline_source(tmp_path: Path) -> None:
    directory = tmp_path / "tsp"
    directory.mkdir()
    xy = torch.rand(2, 20, 2)
    torch.save({"xy": xy, "optimal": torch.tensor([3.0, 5.0])}, directory / "tsp20.pt")

    problem, reference = SavedProblems(20, tmp_path).load("tsp", index=1)

    assert reference == 5.0
    assert np.array_equal(problem["coordinates"], xy[1].numpy())
    assert problem["name"] == "tsp"


def test_materialized_vrptw_is_reused_for_validation(tmp_path: Path) -> None:
    directory = tmp_path / "vrptw"
    directory.mkdir()
    path = directory / "vrptw20_n4_seed123.pt"
    torch.save(generate_vrptw_validation_data(20, 4, seed=123), path)
    saved = SavedProblems(20, tmp_path)

    first, first_reference = saved.load("vrptw", 3)
    torch.rand(100)
    second, second_reference = saved.load("vrptw", 3)

    assert first_reference is None
    assert second_reference is None
    assert np.array_equal(first["coordinates"], second["coordinates"])
    assert np.array_equal(first["tw_start"], second["tw_start"])
    assert np.array_equal(first["tw_end"], second["tw_end"])


def test_pickle_reference_uses_count_relative_to_nonzero_start(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "cvrp2.pkl"
    solution_path = tmp_path / "cvrp2_hgs.pkl"
    rows = [
        ([0.0, 0.0], [[0.1, 0.1], [0.2, 0.2]], [1, 2], 10),
        ([0.0, 0.0], [[0.3, 0.3], [0.4, 0.4]], [2, 1], 10),
        ([0.0, 0.0], [[0.5, 0.5], [0.6, 0.6]], [1, 1], 10),
    ]
    with data_path.open("wb") as target:
        pickle.dump(rows, target)
    with solution_path.open("wb") as target:
        pickle.dump([(11.0, []), (22.0, [])], target)

    _, reference = load_saved_data(
        data_path,
        "cvrp",
        1,
        start=1,
        solution_path=solution_path,
    )

    assert reference == 22.0
    with pytest.raises(ValueError, match="requested 1 references"):
        load_saved_data(
            data_path,
            "cvrp",
            1,
            start=2,
            solution_path=solution_path,
        )


def test_pt_reference_uses_count_relative_to_nonzero_start(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "cvrp2.pt"
    solution_path = tmp_path / "cvrp2_hgs.pt"
    torch.save({"xy": torch.rand(3, 3, 2)}, data_path)
    torch.save({"cost": torch.tensor([11.0, 22.0])}, solution_path)

    _, reference = load_saved_data(
        data_path,
        "cvrp",
        1,
        start=1,
        solution_path=solution_path,
    )

    assert reference == 22.0
    with pytest.raises(ValueError, match="requested 1 references"):
        load_saved_data(
            data_path,
            "cvrp",
            1,
            start=2,
            solution_path=solution_path,
        )


def test_saved_data_rejects_short_instance_slice(tmp_path: Path) -> None:
    data_path = tmp_path / "cvrp2.pt"
    torch.save({"xy": torch.rand(2, 3, 2)}, data_path)

    with pytest.raises(ValueError, match="requested 2 instances from index 1"):
        load_saved_data(data_path, "cvrp", 2, start=1)


def test_saved_problems_excludes_population_reference_constants(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "op"
    directory.mkdir()
    torch.save(torch.rand(2, 4, 3), directory / "op3.pt")

    _, reference = SavedProblems(3, tmp_path).load("op", index=1)

    assert reference is None


def test_saved_problems_excludes_scalar_embedded_reference(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "atsp"
    directory.mkdir()
    distance = torch.rand(2, 3, 3)
    distance.diagonal(dim1=1, dim2=2).zero_()
    torch.save(
        {"dist": distance, "optimal": torch.tensor(1.5)},
        directory / "atsp3.pt",
    )

    _, reference = SavedProblems(3, tmp_path).load("atsp", index=1)

    assert reference is None


def test_owned_generators_cover_the_existing_training_curriculum() -> None:
    for variant in TRAIN_VARIANTS:
        problem = generated_problem(variant, 20)
        assert problem["name"] == variant
        assert "coordinates" in problem or "distance" in problem


@pytest.mark.parametrize("variant", TRAIN_VARIANTS)
def test_randomized_training_rows_use_final_term_coordinates(variant: str) -> None:
    problem = generated_problem(
        variant, 20, randomize_resource_program=True
    )
    anonymous = [
        row for row in problem["resources"]
        if row["name"] == "anonymous_program"
    ]
    assert len(anonymous) == 1
    row = anonymous[0]
    assert row["terms"]
    assert all("op" in term and "phase" in term for term in row["terms"])
    assert all("stage" not in term for term in row["terms"])
    assert not ({"increment", "departure", "join", "reset"} & row.keys())
    assert {term["op"] for term in row["terms"]} <= {
        "add", "join", "assign"
    }


def test_generated_vrptw_is_capacity_free_and_multi_route() -> None:
    problem = generated_problem("vrptw", 20)

    assert problem["constraints"] == ["visit_all", "time_windows"]
    assert problem["depot_count"] == 1
    assert problem["multi_route"] is True
    assert problem["open_route"] is False
    assert "demand" not in problem


def test_curriculum_exposes_tw_only_task_in_the_first_phase() -> None:
    curriculum = VariantCurriculum.default(seed=1234)

    early = curriculum.eligible(epoch=0, epochs=100)
    middle = curriculum.eligible(epoch=33, epochs=100)

    assert [name for name in early if "tw" in name] == ["vrptw"]
    assert {name for name in middle if "tw" in name} == {
        "vrptw",
        "cvrptw",
        "ocvrptw",
        "mdcvrptw",
        "mdocvrptw",
    }


BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "datasets" / "benchmarks"


def _saved_benchmark(variant: str) -> dict:
    """Load the benchmark's own instance file for a variant, if present."""
    candidates = sorted(
        path
        for path in (BENCHMARK_DIR / variant).glob("*.pt")
        if "prism" not in path.name and "ortools" not in path.name
        and "pyvrp" not in path.name
    )
    if not candidates:
        pytest.skip(f"no saved benchmark instances for {variant}")
    return torch.load(candidates[0], weights_only=False)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [("cvrpl", problem_data.SYMMETRIC_ROUTE_LIMIT),
     ("acvrpl", problem_data.ASYMMETRIC_ROUTE_LIMIT)],
)
def test_generated_duration_limit_matches_the_saved_benchmark(
    variant: str, expected: float
) -> None:
    """Training and evaluation must impose the same duration budget.

    The asymmetric limit was self-scaled to each instance's worst depot round
    trip, which put training at roughly 0.17 against an evaluation budget of
    0.6 -- and no asymmetric duration-limit variant is in the training split,
    so every one of them is a held-out composition tested in a regime the
    model never saw.
    """
    saved = _saved_benchmark(variant)
    saved_limit = float(np.asarray(saved["route_limit"]).reshape(-1)[0])
    assert saved_limit == pytest.approx(expected, abs=1e-6)

    torch.manual_seed(0)
    generated = generated_problem(variant, 100)
    assert float(generated["route_limit"]) == pytest.approx(expected, abs=1e-6)


def test_pickup_delivery_capacity_matches_the_saved_benchmark() -> None:
    """Pickup-delivery uses capacity 20, not the 50 every other variant uses."""
    assert problem_data.benchmark_capacity("pdcvrp") == 20
    assert problem_data.benchmark_capacity("cvrp") == 50
    assert problem_data.benchmark_capacity("mdcvrpbp") == 50

    saved = _saved_benchmark("pdcvrp")
    saved_demand = np.abs(np.asarray(saved["demand"][0]))
    torch.manual_seed(0)
    generated = np.abs(np.asarray(generated_problem("pdcvrp", 100)["demand"]))
    assert generated[generated > 0].max() == pytest.approx(
        saved_demand[saved_demand > 0].max(), abs=1e-6
    )

    # An explicit capacity still overrides, and other variants are untouched.
    torch.manual_seed(0)
    override = generated_problem("pdcvrp", 100, capacity=50)
    assert np.abs(np.asarray(override["demand"])).max() < 0.2
