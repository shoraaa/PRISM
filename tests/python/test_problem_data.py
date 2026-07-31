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
    generated_problem,
)


def test_registry_preserves_the_existing_urs_problem_sets() -> None:
    assert len(BENCHMARK_VARIANTS) == 110
    assert ALL_VARIANTS == BENCHMARK_VARIANTS
    assert TRAIN_VARIANTS == sorted(
        [
            "atsp",
            "acvrp",
            "tsp",
            "op",
            "pctsp",
            "cvrp",
            "cvrpb",
            "cvrptw",
            "ocvrp",
            "ocvrptw",
            "pdtsp",
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


def test_owned_generators_cover_the_existing_training_curriculum() -> None:
    for variant in TRAIN_VARIANTS:
        problem = generated_problem(variant, 20)
        assert problem["name"] == variant
        assert "coordinates" in problem or "distance" in problem
