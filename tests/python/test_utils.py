from utils import Logger, validation_metrics_for_wandb


def test_validation_wandb_metrics_exclude_static_bookkeeping() -> None:
    metrics = {
        "instances": 224.0,
        "gap_instances": 192.0,
        "gap_coverage": 0.85,
        "is_classical_baseline": 0.0,
        "feasibility_rate": 1.0,
        "worst_variant_feasibility_rate": 1.0,
        "average_baseline_improvement_percent": 1.2,
        "macro_baseline_improvement_percent": 1.1,
        "macro_score": 1.2,
        "variants/cvrp/instances": 8.0,
        "variants/cvrp/feasible_instances": 8.0,
        "variants/cvrp/feasibility_rate": 1.0,
        "variants/cvrp/objective": 12.0,
        "variants/cvrp/gap": -1.0,
        "variants/cvrp/baseline_improvement_percent": 2.0,
    }

    selected = validation_metrics_for_wandb(metrics)

    assert selected == {
        "feasibility_rate": 1.0,
        "worst_variant_feasibility_rate": 1.0,
        "variants/cvrp/objective": 12.0,
        "variants/cvrp/gap": -1.0,
        "variants/cvrp/baseline_improvement_percent": 2.0,
    }


def test_log_validation_sends_only_curated_metrics() -> None:
    logger = Logger(use_wandb=False)
    captured = {}
    logger._wandb_log = lambda values, step: captured.update(values)

    logger.log_validation(
        1.0,
        2.0,
        -0.5,
        3,
        {
            "instances": 224.0,
            "average_baseline_improvement_percent": 1.5,
            "macro_baseline_improvement_percent": 1.4,
            "macro_score": 1.5,
        },
        step=96,
    )

    assert "val/instances" not in captured
    assert "val/gap" not in captured
    assert captured["val/epoch"] == 3
    assert captured["val_summary/macro_gap"] == -0.5
    assert captured["val_summary/macro_improvement"] == 1.4
    assert captured["val_summary/macro_score"] == 1.5


def test_log_baseline_uses_run_summary_without_history(monkeypatch) -> None:
    class Run:
        summary = {}

    monkeypatch.setattr("wandb.run", Run())
    logger = Logger(use_wandb=True)
    history_calls = []
    logger._wandb_log = lambda values, step: history_calls.append(values)

    logger.log_baseline(0.75)

    assert Run.summary == {"baseline/gap": 0.75}
    assert history_calls == []
