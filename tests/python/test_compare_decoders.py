from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    assert args.tsptw_source == "dataset"
    assert args.tsptw_size == 100
    assert args.tsptw_hardness == "hard"
    assert args.tsptw_dataset_seed == 2025
    assert args.baselines == "constant"
    assert args.shared_greedy_bootstrap is False
    assert args.min_changed_edges == 8
    assert args.max_perturb_attempts == 64


def test_compare_perturbation_cli_overrides_defaults() -> None:
    args = decoder_evaluation.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--min-changed-edges",
            "6",
            "--max-perturb-attempts",
            "19",
        ]
    )

    assert args.min_changed_edges == 6
    assert args.max_perturb_attempts == 19


def test_compare_baseline_selection() -> None:
    for name in ("constant", "distance", "random"):
        args = decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--baselines", name]
        )
        assert decoder_evaluation.selected_baselines(args.baselines) == (name,)

    args = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--baselines", "all"]
    )
    assert decoder_evaluation.selected_baselines(args.baselines) == (
        "constant",
        "distance",
        "random",
    )

    all_with_urs_id = decoder_evaluation.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--baselines",
            "all",
            "--urs-baseline-id",
            "urs.pt",
        ]
    )
    assert decoder_evaluation.selected_baselines(all_with_urs_id.baselines) == (
        "constant",
        "distance",
        "random",
    )

    urs = decoder_evaluation.parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--baselines",
            "urs",
            "--urs-baseline-id",
            "urs.pt",
        ]
    )
    assert decoder_evaluation.selected_baselines(urs.baselines) == ("urs",)
    assert urs.urs_checkpoint == Path("urs.pt")

    none = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--baseline", "none"]
    )
    assert none.baselines == "none"
    assert decoder_evaluation.selected_baselines(none.baselines) == ("none",)


def test_shared_greedy_bootstrap_requires_one_live_native_baseline() -> None:
    assert decoder_evaluation._shared_greedy_baseline(
        True, ("constant",), None
    ) == "constant"
    assert decoder_evaluation._shared_greedy_baseline(
        False, ("constant", "distance"), None
    ) is None

    for baselines in (("constant", "distance"), ("urs",), ("none",)):
        with pytest.raises(ValueError):
            decoder_evaluation._shared_greedy_baseline(
                True, baselines, None
            )
    with pytest.raises(ValueError):
        decoder_evaluation._shared_greedy_baseline(
            True, ("constant",), Path("cached.csv")
        )


def test_shared_greedy_route_uses_selected_baseline_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDecoder:
        def sample_greedy(self, **guidance):
            assert guidance == {"mode": "distance"}
            return {"feasible": True, "route": [0, 2, 1]}

    decoder = FakeDecoder()
    monkeypatch.setattr(
        decoder_evaluation,
        "_new_decoder",
        lambda problem, args, deterministic: decoder,
    )
    monkeypatch.setattr(
        decoder_evaluation,
        "_distance_guidance",
        lambda actual_decoder, problem: {"mode": "distance"},
    )

    route = decoder_evaluation._greedy_baseline_route(
        {"name": "tsp"}, SimpleNamespace(seed=7), "distance"
    )

    assert route == [0, 2, 1]


def test_neural_only_split_summary_omits_comparison_metrics() -> None:
    rows = [
        {
            "variant": "cvrp",
            "baseline": "none",
            "reference": 10.0,
            "baseline_gap_pct": "",
            "neural_gap_pct": -2.0,
            "winner": "",
            "neural_improvement_pct": "",
        }
    ]

    summary = decoder_evaluation._split_summary(
        "seen", ["cvrp"], rows, [], baseline="none"
    )

    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["neural_gap_mean"] == -2.0
    assert "neural_wins" not in summary
    assert "baseline_gap_mean" not in summary
    assert "neural_improvement_mean" not in summary


def test_cached_csv_supplies_available_baseline_rows(tmp_path: Path) -> None:
    cached_path = tmp_path / "cached.csv"
    cached_path.write_text(
        "variant,baseline,val_size,direction,baseline_objective\n"
        "tsp,distance,8,minimize,7.5\n"
        "op,distance,8,maximize,30.0\n"
        "cvrp,none,8,minimize,\n"
    )

    args = decoder_evaluation.parse_args(
        ["--checkpoint", "model.pt", "--cached", str(cached_path)]
    )
    rows, baselines = decoder_evaluation.load_cached_baseline_rows(cached_path)

    assert args.baselines == "cached"
    assert baselines == ("distance",)
    assert rows[("tsp", "distance", 8)] == {
        "baseline_objective": 7.5,
        "baseline_construction_objective": "",
        "direction": "minimize",
    }
    assert rows[("op", "distance", 8)]["direction"] == "maximize"
    assert all(key[0] != "cvrp" for key in rows)


def test_cached_csv_preserves_construction_objective(tmp_path: Path) -> None:
    cached_path = tmp_path / "cached.csv"
    cached_path.write_text(
        "variant,baseline,val_size,direction,baseline_construction_objective,"
        "baseline_objective\n"
        "tsp,distance,8,minimize,8.5,7.5\n"
    )

    rows, _ = decoder_evaluation.load_cached_baseline_rows(cached_path)

    assert rows[("tsp", "distance", 8)][
        "baseline_construction_objective"
    ] == 8.5


def test_formats_construction_before_final() -> None:
    assert decoder_evaluation._format_construction_final(
        12.5, 10.0, format_spec=".6g"
    ) == "12.5/10"
    assert decoder_evaluation._format_construction_final(
        25.0, 0.0, format_spec=".3f", suffix="%"
    ) == "25.000%/0.000%"
    assert decoder_evaluation._format_construction_final(
        "", 10.0, format_spec=".6g"
    ) == "n/a/10"


def test_urs_baseline_requires_identifier() -> None:
    with pytest.raises(SystemExit):
        decoder_evaluation.parse_args(
            ["--checkpoint", "model.pt", "--baselines", "urs"]
        )


def test_all_benchmarks_are_partitioned_into_training_splits() -> None:
    variants = decoder_evaluation.selected_variants("all")

    assert len(variants) == 110
    seen = len(decoder_evaluation.SEEN_VARIANTS)
    assert sum(
        decoder_evaluation.variant_split(name) == "seen" for name in variants
    ) == seen
    assert sum(
        decoder_evaluation.variant_split(name) == "heldout" for name in variants
    ) == 110 - seen
    assert "tsptw" not in variants


def test_tsptw_is_an_explicit_heldout_evaluator_variant() -> None:
    assert decoder_evaluation.selected_variants("tsptw") == ["tsptw"]
    assert decoder_evaluation.variant_split("tsptw") == "heldout"


def test_loads_car_tsptw_dataset_and_lkh_reference(tmp_path: Path) -> None:
    rows = [
        (
            [[0.0, 0.0], [3.0, 4.0]],
            [0.0, 0.0],
            [0.0, 4.0],
            [100.0, 10.0],
        ),
        (
            [[0.0, 0.0], [6.0, 8.0]],
            [0.0, 0.0],
            [0.0, 9.0],
            [100.0, 20.0],
        ),
    ]
    with (tmp_path / "tsptw2_easy.pkl").open("wb") as destination:
        pickle.dump(rows, destination)
    with (tmp_path / "lkh_tsptw2_easy.pkl").open("wb") as destination:
        pickle.dump([(10.0, [1]), (20.0, [1])], destination)

    data, reference = decoder_evaluation.load_car_tsptw_data(
        tmp_path, size=2, hardness="easy", count=2
    )
    problem = decoder_evaluation.solver_problem(
        "tsptw", decoder_evaluation._instance_data(data, 0)
    )

    assert data["xy"].shape == (2, 2, 2)
    assert reference == 15.0
    assert problem["constraints"] == ["visit_all", "time_windows"]
    assert problem["depot_count"] == 1
    assert problem["multi_route"] is False
    assert problem["open_route"] is False
    assert problem["distance"][0, 1] == 5.0


def test_recovers_car_hard_generator_feasible_routes() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        xy = torch.rand(2, 5, 2) * 100.0
        expected = []
        customers = torch.arange(1, 5)
        for _ in range(2):
            expected.append(
                torch.cat(
                    (
                        torch.zeros(1, dtype=torch.long),
                        customers[torch.randperm(4)],
                    )
                )
            )

    actual = decoder_evaluation._car_hard_feasible_routes(
        xy, seed=123, generated_count=2
    )

    assert torch.equal(actual, torch.stack(expected))


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
        "baseline": "constant",
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
