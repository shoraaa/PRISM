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
            "cvrptw",
            "ocvrp",
            "ocvrptw",
            "pdtsp",
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
