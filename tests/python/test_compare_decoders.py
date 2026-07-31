from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "compare_decoders.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("compare_decoders", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compare_decoders = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_decoders)


def test_compare_defaults_to_eight_dynamic_instances() -> None:
    args = compare_decoders.parse_args(["--checkpoint", "model.pt"])

    assert args.val_size == 8
    assert args.static_field is False


def test_instance_data_preserves_batch_dimension() -> None:
    data = {
        "xy": torch.arange(24).reshape(3, 4, 2),
        "capacity": 1.0,
    }

    selected = compare_decoders._instance_data(data, 1)

    assert selected["xy"].shape == (1, 4, 2)
    assert torch.equal(selected["xy"], data["xy"][1:2])
    assert selected["capacity"] == 1.0


def test_mean_reference_uses_requested_prefix(tmp_path: Path) -> None:
    solution = tmp_path / "solutions.pt"
    torch.save({"cost": torch.tensor([2.0, 4.0, 100.0])}, solution)
    paths = {
        "solution_path": solution,
        "problem_name": "cvrp",
    }

    has_reference, reference = compare_decoders._mean_reference(paths, 0.0, 2)

    assert has_reference is True
    assert reference == pytest.approx(3.0)
