"""URS (Unified Routing Solver) as a subprocess adapter.

URS is a foreign checkpoint with its own repository and its own idea of a
dataset, so it runs out-of-process against a file: the batch's ``data_path``
is handed to ``baselines/URS/run_instances.py`` and one JSON line comes back.
That is the same shape every other foreign NCO baseline will take, which is
why the mechanics live in ``SubprocessMethod`` and only the argv, the payload
and the cache key are here.

It is best-effort by design. URS crashes on two variant families at scale
(closed-route time windows produce NaN logits; multi-depot exhausts memory in
its reward path), so a failure is reported as a status and the run keeps
PRISM's measurements for that variant.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

from ..cache import ResultCache, default_cache_path
from .base import MethodRequest, MethodResult, SubprocessMethod


ROOT = Path(__file__).resolve().parent.parent.parent
URS_RUNNER = ROOT / "baselines" / "URS" / "run_instances.py"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--urs-baseline-id",
        "--urs-checkpoint",
        dest="urs_checkpoint",
        type=Path,
        default=None,
        help=(
            "URS checkpoint required by the 'urs' method. It runs on the same "
            "instances in an isolated subprocess. Point at e.g. "
            "baselines/URS/pretrained/unified_checkpoint_500.pt."
        ),
    )
    parser.add_argument(
        "--urs-cuda",
        dest="urs_cuda",
        type=int,
        default=-1,
        help="CUDA device for the URS subprocess (-1 = CPU).",
    )
    parser.add_argument(
        "--urs-batch-size", dest="urs_batch_size", type=int, default=50
    )
    parser.add_argument(
        "--urs-no-aug",
        dest="urs_no_aug",
        action="store_true",
        help="Disable URS instance augmentation (faster, weaker URS).",
    )
    parser.add_argument(
        "--urs-cache",
        dest="urs_cache",
        type=Path,
        default=default_cache_path("urs"),
        help=(
            "JSON cache of URS per-instance objectives, keyed by variant + "
            "checkpoint + augmentation + data file (default: "
            "results/urs_cache.json). A changed checkpoint invalidates its "
            "entries."
        ),
    )
    parser.add_argument(
        "--urs-refresh",
        dest="urs_refresh",
        action="store_true",
        help="Ignore any cached URS results and recompute (and overwrite) them.",
    )


def validate_arguments(args: argparse.Namespace, error) -> None:
    if args.urs_no_aug and args.aug not in (None, 1):
        error("--urs-no-aug conflicts with --aug values other than 1")


class UrsMethod(SubprocessMethod):
    """Run URS on a variant's saved dataset file."""

    name = "urs"
    result_prefix = "URS_RESULT_JSON "

    def __init__(self, args: argparse.Namespace, cache: ResultCache | None = None):
        self.args = args
        self.cache = cache if cache is not None else ResultCache(
            args.urs_cache, refresh=args.urs_refresh
        )
        self._temporary: tempfile.TemporaryDirectory | None = None

    def config(self) -> str:
        checkpoint = (
            Path(self.args.urs_checkpoint).name
            if self.args.urs_checkpoint
            else "cached"
        )
        requested_aug = getattr(self.args, "aug", None)
        aug = (
            str(requested_aug)
            if requested_aug is not None
            else "off" if self.args.urs_no_aug else "on"
        )
        return (
            f"checkpoint={checkpoint},"
            f"aug={aug},"
            f"batch={self.args.urs_batch_size}"
        )

    def covers(self, problem: dict) -> str | None:
        return None

    def cache_key(self, request: MethodRequest) -> str:
        """Identity of a URS result: variant, checkpoint (path+mtime), aug, data."""
        checkpoint = Path(self.args.urs_checkpoint).resolve()
        try:
            stamp = checkpoint.stat().st_mtime_ns
        except OSError:
            stamp = 0
        requested_aug = getattr(self.args, "aug", None)
        aug = (
            f"aug{requested_aug}"
            if requested_aug is not None
            else "noaug" if self.args.urs_no_aug else "aug"
        )
        return (
            f"{request.batch.variant}|{request.batch.signature}|"
            f"{len(request.batch)}|{checkpoint}|{stamp}|{aug}"
        )

    def _export(self, request: MethodRequest) -> Path:
        """Write the exact PRISM-loaded batch in URS's neutral tensor schema."""
        if request.batch.data is None:
            raise ValueError("URS requires a tensor batch")
        self._temporary = tempfile.TemporaryDirectory(prefix="prism-urs-")
        target = Path(self._temporary.name) / f"{request.batch.variant}.pt"
        torch.save(request.batch.data, target)
        return target

    def command(self, request: MethodRequest) -> list[str]:
        dataset = self._export(request)
        command = [
            sys.executable,
            str(URS_RUNNER),
            "--problem", request.batch.variant,
            "--data-path", str(dataset),
            "--episodes", str(len(request.batch)),
            "--scale", str(request.batch.n),
            "--model-load", str(self.args.urs_checkpoint),
            "--cuda", str(self.args.urs_cuda),
            "--batch-size", str(self.args.urs_batch_size),
        ]
        requested_aug = getattr(self.args, "aug", None)
        if requested_aug is not None:
            command.extend(("--aug-factor", str(requested_aug)))
        elif self.args.urs_no_aug:
            command.append("--disable-aug")
        return command

    def parse(self, payload: dict, request: MethodRequest) -> MethodResult:
        if payload.get("error"):
            return MethodResult.failure(str(payload["error"]), config=self.config())
        objectives = payload.get("objectives") or []
        if len(objectives) < len(request.batch):
            return MethodResult.failure(
                f"returned {len(objectives)} of {len(request.batch)} objectives",
                config=self.config(),
            )
        return MethodResult(
            objectives=[float(value) for value in objectives[: len(request.batch)]],
            direction=payload.get("direction", ""),
            config=self.config(),
        )

    def run(self, request: MethodRequest) -> MethodResult:
        # Generated --n-node batches hold decoder problems rather than the
        # neutral tensor dictionary that URS's environment consumes.
        if request.batch.data is None:
            return MethodResult.unsupported("no_tensor_batch", config=self.config())
        if self.args.urs_checkpoint is None:
            return MethodResult.unsupported("no_checkpoint", config=self.config())

        key = self.cache_key(request)
        entry = self.cache.get(key)
        if entry is not None and len(entry.get("objectives", [])) >= len(
            request.batch
        ):
            return MethodResult(
                objectives=[
                    float(value)
                    for value in entry["objectives"][: len(request.batch)]
                ],
                direction=entry.get("direction", ""),
                seconds=float(entry.get("seconds", 0.0)),
                source="cached",
                config=self.config(),
            )

        try:
            result = super().run(request)
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
        if result.status == "ok":
            self.cache.put(
                key,
                {
                    "objectives": result.objectives,
                    "direction": result.direction,
                    "seconds": result.seconds,
                },
            )
        return result
