from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_construction_guidance.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_construction_guidance", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_defaults_to_requested_checkpoint_and_construction_only() -> None:
    args = comparison.parse_args([])

    assert args.checkpoint == ROOT / "pretrained" / "newer" / "best.pt"
    assert args.variants == "all"
    assert args.val_size == 8
    assert args.rollouts is None


def test_model_kwargs_recovers_training_cli_pooling_alias() -> None:
    kwargs = comparison.model_constructor_kwargs(
        {
            "config": {
                "resource_pooling": False,
                "couple_resource_tokens": False,
            }
        }
    )

    assert kwargs["pool_node_resources"] is False
    assert kwargs["couple_resource_tokens"] is False


def test_resolved_settings_inherit_checkpoint_construction_contract() -> None:
    args = comparison.parse_args([])
    settings = comparison.resolve_settings(
        args,
        {
            "config": {
                "n_rollouts": 7,
                "candidates": 33,
                "beta": 1.5,
                "seed": 99,
                "feasibility_lookahead_depth": 3,
                "feasibility_risk_penalty": 2.0,
            }
        },
    )

    assert settings.n_rollouts == 7
    assert settings.candidates == 33
    assert settings.beta == 1.5
    assert settings.seed == 99
    assert settings.feasibility_lookahead_depth == 3
    assert settings.risk_penalty == 2.0


def _row(method: str, instance: int, rollout: int, feasible: bool, objective=10.0):
    return {
        "variant": "cvrp",
        "instance": instance,
        "method": method,
        "rollout": rollout,
        "feasible": int(feasible),
        "direction": "minimize",
        "objective": objective if feasible else "",
        "guidance_seconds": 0.0,
        "construction_seconds": 0.0,
    }


def test_summary_separates_rollout_and_instance_feasibility() -> None:
    records = [
        _row("learned", 0, 0, True, 8.0),
        _row("learned", 0, 1, False),
        _row("heuristic", 0, 0, True, 10.0),
        _row("heuristic", 0, 1, True, 12.0),
        _row("uniform", 0, 0, False),
        _row("uniform", 0, 1, False),
    ]

    summary = comparison.summarize(records)

    assert summary["methods"]["learned"]["rollout_feasibility_rate"] == 0.5
    assert summary["methods"]["learned"]["instance_feasibility_rate"] == 1.0
    uniform = summary["comparisons"]["uniform"]
    assert uniform["learned_only_feasible_instances"] == 1
    assert uniform["learned_minus_baseline_instance_feasibility_pp"] == 100.0
    heuristic = summary["comparisons"]["heuristic"]
    assert heuristic["learned_wins"] == 1
    assert heuristic[
        "macro_variant_best_objective_improvement_percent"
    ] == pytest.approx(20.0)
