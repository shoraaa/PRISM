"""Logging and metric utilities retained from the original trainer."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


_VALIDATION_WANDB_AGGREGATES = {
    "emissions",
    "time_neural",
    "time_decoder",
    "net_evals",
    "feasibility_rate",
    "worst_variant_feasibility_rate",
    "worst_variant_gap",
    "worst_variant_baseline_improvement_percent",
    "saved_reference_instances",
    "missing_reference_instances",
    "gap_instances",
    "gap_coverage",
    "seen_gap",
    "heldout_gap",
    "seen_baseline_improvement_percent",
    "heldout_baseline_improvement_percent",
}
_VALIDATION_WANDB_VARIANT_SUFFIXES = {
    "objective",
    "gap",
    "baseline_improvement_percent",
    "feasibility_rate",
}


def validation_metrics_for_wandb(
    metrics: dict[str, float],
) -> dict[str, float]:
    """Drop invariant validation-manifest bookkeeping from W&B histories."""
    selected = {}
    for name, value in metrics.items():
        if name in _VALIDATION_WANDB_AGGREGATES:
            selected[name] = value
            continue
        if name.startswith("variants/") and name.rsplit("/", 1)[-1] in (
            _VALIDATION_WANDB_VARIANT_SUFFIXES
        ):
            selected[name] = value
    return selected


class Logger:
    def __init__(
        self,
        name: str = "train",
        use_wandb: bool = True,
        log_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.use_wandb = use_wandb
        self._step = 0
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        if not self._logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            self._logger.addHandler(console)
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_dir / f"{name}.log")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)

    def set_step(self, step: int) -> None:
        self._step = step

    def info(self, message: str) -> None:
        self._logger.info(message)

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def _wandb_log(
        self, values: dict[str, float | str], step: Optional[int]
    ) -> None:
        if not self.use_wandb:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(values, step=self._step if step is None else step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        prefix: str = "",
        step: Optional[int] = None,
    ) -> None:
        self._wandb_log(
            {f"{prefix}{key}": float(value) for key, value in metrics.items()},
            step,
        )

    def log_train_step(
        self,
        variant: str,
        avg_cost: float,
        best_cost: float,
        epoch: int,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        values = {
            "train/variant": variant,
            f"train/variants/{variant}/avg_cost": avg_cost,
            f"train/variants/{variant}/best_cost": best_cost,
            "train/epoch": epoch,
        }
        values.update({f"train/{key}": float(value) for key, value in metrics.items()})
        self._wandb_log(values, step)

    def log_validation(
        self,
        avg_last: float,
        avg_best: float,
        gap: float,
        epoch: int,
        metrics: dict[str, float],
        *,
        is_best: bool = False,
        timing: Optional[dict[str, float]] = None,
        step: Optional[int] = None,
    ) -> None:
        values = {
            # Raw canonical costs cannot be compared across distance, penalty,
            # and maximize-prize variants. Keep the mixed value diagnostic.
            "val/diagnostic_mixed_canonical_best": avg_best,
            "val/epoch": epoch,
            # This is the exact lower-is-better value used to rank best.pt.
            "val_summary/cost": avg_best,
            "val_summary/is_best": float(is_best),
            "val_summary/macro_gap": gap,
            "val_summary/macro_improvement": float(
                metrics["macro_baseline_improvement_percent"]
            ),
            "val_summary/macro_score": float(metrics["macro_score"]),
        }
        values.update(
            {
                f"val_summary/{name}": float(value)
                for name, value in metrics.items()
                if name.startswith("group_cost/")
            }
        )
        values.update(
            {
                f"val/{key}": float(value)
                for key, value in validation_metrics_for_wandb(metrics).items()
            }
        )
        if timing:
            values.update({f"time/{key}": float(value) for key, value in timing.items()})
        self._wandb_log(values, step)

    def log_baseline(self, gap: float) -> None:
        """Store the fields-off baseline gap as run metadata, not epoch history."""
        if self.use_wandb:
            try:
                import wandb
            except ImportError:
                wandb = None
            if wandb is not None and wandb.run is not None:
                wandb.run.summary["baseline/gap"] = float(gap)
        self.info(f"Baseline: Gap={gap:.2f}%")

    def log_epoch_summary(
        self,
        epoch: int,
        train_cost: float,
        val_best: float,
        gap: Optional[float],
        feasibility_rate: Optional[float] = None,
        *,
        is_best: bool = False,
    ) -> None:
        message = f"Epoch {epoch}: MixedTrainCost={train_cost:.4f}"
        message += (
            " Validation=skipped"
            if gap is None
            else f" ValCost={val_best:.4f} MacroGap={gap:.2f}%"
        )
        if feasibility_rate is not None:
            message += f" Feasible={feasibility_rate:.2%}"
        if is_best:
            message += " BEST"
        self.info(message)


@dataclass
class MetricsCollector:
    _metrics: dict[str, list[float | torch.Tensor]] = field(
        default_factory=dict
    )

    def reset(self) -> None:
        self._metrics.clear()

    def add(self, name: str, value: float | torch.Tensor) -> None:
        self._metrics.setdefault(name, []).append(value)

    def add_dict(self, metrics: dict[str, float | torch.Tensor]) -> None:
        for name, value in metrics.items():
            self.add(name, value)

    def get_mean(self, name: str) -> float:
        values = self._metrics.get(name, [])
        if not values:
            return 0.0
        if any(torch.is_tensor(value) for value in values):
            tensors = [
                value.detach().float().reshape(())
                if torch.is_tensor(value)
                else torch.tensor(value, dtype=torch.float32)
                for value in values
            ]
            device = next(
                value.device for value in tensors if value.device.type != "cpu"
            ) if any(value.device.type != "cpu" for value in tensors) else None
            if device is not None:
                tensors = [value.to(device) for value in tensors]
            return float(torch.stack(tensors).mean().cpu())
        return float(np.mean(values))

    def get_all_means(self) -> dict[str, float]:
        return {name: self.get_mean(name) for name in self._metrics}


_logger: Optional[Logger] = None


def get_logger() -> Logger:
    global _logger
    if _logger is None:
        _logger = Logger(use_wandb=False)
    return _logger


def init_logger(
    use_wandb: bool = True,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Logger:
    global _logger
    _logger = Logger(
        name="train",
        use_wandb=use_wandb,
        log_dir=log_dir,
        verbose=verbose,
    )
    return _logger
