"""Logging and metric utilities retained from the original trainer."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


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

    def _wandb_log(self, values: dict[str, float], step: Optional[int]) -> None:
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
        avg_cost: float,
        best_cost: float,
        epoch: int,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        values = {
            "train/avg_cost": avg_cost,
            "train/best_cost": best_cost,
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
        timing: Optional[dict[str, float]] = None,
        step: Optional[int] = None,
    ) -> None:
        values = {
            "val/avg_last": avg_last,
            "val/avg_best": avg_best,
            "val/gap": gap,
            "val/epoch": epoch,
        }
        values.update({f"val/{key}": float(value) for key, value in metrics.items()})
        if timing:
            values.update({f"time/{key}": float(value) for key, value in timing.items()})
        self._wandb_log(values, step)

    def log_epoch_summary(
        self, epoch: int, train_cost: float, val_best: float, gap: float
    ) -> None:
        self.info(
            f"Epoch {epoch}: TrainCost={train_cost:.4f} "
            f"ValBest={val_best:.4f} Gap={gap:.2f}%"
        )


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
