from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
URS_ROOT = ROOT / "baselines" / "URS"
if str(URS_ROOT) not in sys.path:
    sys.path.insert(0, str(URS_ROOT))

from data.DataFinder import DataFinder  # noqa: E402
from data.DataReader import get_saved_data  # noqa: E402
from problem.ProblemDef import get_random_problems  # noqa: E402
from problem.ProblemSet import ProblemSet  # noqa: E402


def _first(value):
    if torch.is_tensor(value):
        value = value[0].detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return value.astype(np.float32, copy=False)
    return value


def decoder_problem(name: str, data: dict) -> dict:
    """Convert one URS tensor dictionary to the C++ Decoder schema."""
    problem = {"name": name, "capacity": 1.0, "prize_quota": 1.0}
    if "xy" in data:
        coordinates = _first(data["xy"])
        problem["coordinates"] = coordinates
        # Preserve exact small-instance parity. Large Euclidean problems stay
        # O(n) in memory and let the native decoder compute distances on demand.
        if coordinates.shape[0] <= 512:
            problem["distance"] = np.linalg.norm(
                coordinates[:, None] - coordinates[None, :], axis=-1
            ).astype(np.float32)
        node_count = coordinates.shape[0]
    else:
        problem["distance"] = _first(data["dist"])
        node_count = problem["distance"].shape[0]
    for field in (
        "demand",
        "prize",
        "penalty",
        "tw_start",
        "tw_end",
        "service_time",
    ):
        if field in data:
            values = _first(data[field])
            if len(values) == node_count:
                problem[field] = values
    if "route_limit" in data:
        problem["route_limit"] = float(_first(data["route_limit"]))
    if name == "op":
        problem["tour_limit"] = 4.0
    elif name == "aop":
        problem["tour_limit"] = 1.0
    return problem


def generated_problem(name: str, size: int, capacity: int = 50) -> dict:
    kwargs = {
        "problem_gen_params": {
            "int_min": 0,
            "int_max": 1_000_000,
            "scaler": 1_000_000,
        }
    }
    data = get_random_problems(1, size, capacity, name, **kwargs)
    return decoder_problem(name, data)


def resource_count(name: str) -> int:
    resources = 0
    if "cvrp" in name:
        resources += 1
        suffix = name.split("cvrp", 1)[1]
        resources += int("l" in suffix)
        resources += int("tw" in suffix)
        resources += int("bp" in suffix)
    resources += int("pd" in name)
    resources += int("pctsp" in name)
    resources += int(name in {"op", "aop"})
    return resources


@dataclass
class VariantCurriculum:
    variants: list[str]
    rng: random.Random

    @classmethod
    def urs_seen(cls, seed: int) -> "VariantCurriculum":
        return cls(list(ProblemSet.get(name="train_problem_list")), random.Random(seed))

    @property
    def held_out(self) -> list[str]:
        selected = set(self.variants)
        return [name for name in ProblemSet.get() if name not in selected]

    def sample(self, epoch: int, epochs: int) -> str:
        progress = (epoch + 1) / max(epochs, 1)
        maximum = 1 if progress <= 1 / 3 else 2 if progress <= 2 / 3 else 7
        eligible = [name for name in self.variants if resource_count(name) <= maximum]
        return self.rng.choice(eligible or self.variants)


class SavedURS:
    def __init__(self, size: int):
        self.size = size
        self.finder = DataFinder(URS_ROOT / "dataset")

    def load(self, name: str, index: int = 0) -> tuple[dict, float | None]:
        # DataFinder emits a warning when a problem keeps its oracle embedded in
        # the data file (op/atsp/tsp/pctsp/...) instead of in a separate solution
        # file. That case is expected, so hush the warning to avoid mistaking it
        # for a fatal error.
        finder_logger = logging.getLogger("data.DataFinder")
        previous_level = finder_logger.level
        finder_logger.setLevel(logging.ERROR)
        try:
            paths = self.finder.get(name, self.size)
        finally:
            finder_logger.setLevel(previous_level)
        if paths is None:
            raise FileNotFoundError(
                f"no saved data for variant={name} scale={self.size}"
            )
        data, embedded = get_saved_data(
            paths["data_path"],
            name,
            index + 1,
            "cpu",
            start=index,
            solution_name=paths["solution_path"],
        )
        reference = float(embedded) if embedded is not None else None
        # A missing separate-solution oracle falls back to a placeholder of 1
        # (see DataReader). Treat that as "no reference" so it never yields a
        # bogus gap; embedded-oracle problems keep their real value.
        if paths["solution_path"] is None and reference == 1.0:
            reference = None
        return decoder_problem(name, data), reference
