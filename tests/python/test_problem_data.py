from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import problem_data
from problem_data import (
    ALL_VARIANTS,
    BENCHMARK_VARIANTS,
    TRAIN_VARIANTS,
    DatasetFinder,
    SavedProblems,
    generate_vrptw_validation_data,
    generated_problem,
)


def test_registry_owns_benchmark_and_tw_only_collections() -> None:
    assert len(BENCHMARK_VARIANTS) == 110
    assert len(ALL_VARIANTS) == 111
    assert "vrptw" in TRAIN_VARIANTS
    assert "vrptw" not in BENCHMARK_VARIANTS
    assert "tsptw" not in ALL_VARIANTS


def test_problem_data_has_no_baseline_python_dependency() -> None:
    source = Path(problem_data.__file__).read_text()

    assert "sys.path" not in source
    assert "from data." not in source
    assert "from problem." not in source


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


def test_generated_vrptw_is_capacity_free_and_multi_route() -> None:
    problem = generated_problem("vrptw", 20)

    assert problem["constraints"] == ["visit_all", "time_windows"]
    assert problem["multi_route"] is True
    assert "demand" not in problem
