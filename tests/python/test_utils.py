from utils import Logger, validation_metrics_for_wandb


def test_validation_wandb_metrics_exclude_static_bookkeeping() -> None:
    metrics = {
        "instances": 224.0,
        "gap_instances": 224.0,
        "gap_coverage": 1.0,
        "feasibility_rate": 1.0,
        "worst_variant_feasibility_rate": 1.0,
        "macro_score": 1.2,
        "variants/cvrp/instances": 8.0,
        "variants/cvrp/feasibility_rate": 1.0,
        "variants/cvrp/objective": 12.0,
        "variants/cvrp/gap": -1.0,
        "variants/cvrp/baseline_improvement_percent": 2.0,
    }

    selected = validation_metrics_for_wandb(metrics)

    assert selected == {
        "feasibility_rate": 1.0,
        "worst_variant_feasibility_rate": 1.0,
        "variants/cvrp/feasibility_rate": 1.0,
        "variants/cvrp/objective": 12.0,
        "variants/cvrp/gap": -1.0,
        "variants/cvrp/baseline_improvement_percent": 2.0,
    }


def test_log_validation_uses_macro_summary_namespace() -> None:
    logger = Logger(use_wandb=False)
    captured = {}
    logger._wandb_log = lambda values, step: captured.update(values)

    logger.log_validation(
        1.0,
        2.0,
        -0.5,
        3,
        {
            "macro_baseline_improvement_percent": 1.4,
            "macro_score": 1.5,
        },
        step=96,
    )

    assert "val/gap" not in captured
    assert captured["val/epoch"] == 3
    assert captured["val_summary/macro_gap"] == -0.5
    assert captured["val_summary/macro_improvement"] == 1.4
    assert captured["val_summary/macro_score"] == 1.5


def test_train_step_identifies_variant_without_mixed_cost_metric() -> None:
    logger = Logger(use_wandb=False)
    captured = {}
    logger._wandb_log = lambda values, step: captured.update(values)

    logger.log_train_step("cvrp", 10.0, 9.0, 2, {"emissions": 1.0})

    assert captured["train/variant"] == "cvrp"
    assert captured["train/variants/cvrp/avg_cost"] == 10.0
    assert captured["train/variants/cvrp/best_cost"] == 9.0
    assert "train/avg_cost" not in captured
