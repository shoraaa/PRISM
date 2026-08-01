from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test.py"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("decoder_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
decoder_evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decoder_evaluation)


def test_compare_defaults_to_eight_dynamic_instances() -> None:
    args = decoder_evaluation.parse_args(["--checkpoint", "model.pt"])

    assert args.val_size == 8
    assert args.static_field is False
    assert args.variants == "all"


def test_all_benchmarks_are_partitioned_into_training_splits() -> None:
    variants = decoder_evaluation.selected_variants("all")

    assert len(variants) == 110
    assert sum(
        decoder_evaluation.variant_split(name) == "seen" for name in variants
    ) == 15
    assert sum(
        decoder_evaluation.variant_split(name) == "heldout" for name in variants
    ) == 95


def test_instance_data_preserves_batch_dimension() -> None:
    data = {
        "xy": torch.arange(24).reshape(3, 4, 2),
        "capacity": 1.0,
    }

    selected = decoder_evaluation._instance_data(data, 1)

    assert selected["xy"].shape == (1, 4, 2)
    assert torch.equal(selected["xy"], data["xy"][1:2])
    assert selected["capacity"] == 1.0


def test_split_summary_keeps_results_and_failures_separate() -> None:
    variants = ["cvrp", "aop", "acvrpb"]
    rows = [
        {
            "variant": "cvrp",
            "reference": 1.0,
            "winner": "neural",
            "neural_improvement_pct": 2.0,
            "baseline_gap_pct": 3.0,
            "neural_gap_pct": 1.0,
        },
        {
            "variant": "aop",
            "reference": 1.0,
            "winner": "baseline",
            "neural_improvement_pct": -4.0,
            "baseline_gap_pct": 2.0,
            "neural_gap_pct": 6.0,
        },
    ]
    failures = [
        {"variant": "acvrpb", "split": "heldout", "error": "failed"}
    ]

    seen = decoder_evaluation._split_summary("seen", variants, rows, failures)
    heldout = decoder_evaluation._split_summary(
        "heldout", variants, rows, failures
    )

    assert seen == {
        "split": "seen",
        "variants": 1,
        "passed": 1,
        "failed": 0,
        "neural_wins": 1,
        "ties": 0,
        "baseline_wins": 0,
        "reference_variants": 1,
        "neural_improvement_mean": 2.0,
        "neural_improvement_median": 2.0,
        "baseline_gap_mean": 3.0,
        "neural_gap_mean": 1.0,
    }
    assert heldout["variants"] == 2
    assert heldout["passed"] == 1
    assert heldout["failed"] == 1
    assert heldout["baseline_wins"] == 1
