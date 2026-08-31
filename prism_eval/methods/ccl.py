"""CCL-MTLVRP as a subprocess adapter, evaluated on PRISM's own instances.

CCL ships its own instance generator and its own .npz releases, but a
comparison is only worth anything if both models see the same instances, so
this adapter ignores all of that: it exports whatever the run is already
evaluating -- PRISM's own benchmark instances, at whatever scale the run uses
-- into CCL's feature layout and hands that to CCL's policy.

CCL lives in its own repository with its own virtual environment (rl4co,
lightning, hydra), so the model runs out-of-process through
``ccl_worker.py``; nothing CCL needs is importable here.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..cache import ResultCache, default_cache_path
from .base import MethodRequest, MethodResult, SubprocessMethod


ROOT = Path(__file__).resolve().parent.parent.parent
CCL_ROOT = ROOT / "baselines" / "CCL-MTLVRP"
CCL_RELD = CCL_ROOT / "CCL-ReLD"
DEFAULT_CCL_PYTHON = CCL_ROOT / ".venv" / "bin" / "python"
DEFAULT_CCL_CHECKPOINT = (
    CCL_RELD
    / "logs"
    / "0424-routefinder-LO-CaDA-ReLD-dotNoise-P20-newO-100-main"
    / "2025-04-27_11-39-29"
    / "checkpoints"
    / "epoch_299.ckpt"
)
WORKER = Path(__file__).resolve().parent / "ccl_worker.py"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ccl-checkpoint",
        dest="ccl_checkpoint",
        type=Path,
        default=DEFAULT_CCL_CHECKPOINT,
        help="CCL RouteFinder checkpoint (default: the bundled size-100 one).",
    )
    parser.add_argument(
        "--ccl-python",
        dest="ccl_python",
        type=Path,
        default=DEFAULT_CCL_PYTHON,
        help="Python executable of CCL's own environment.",
    )
    parser.add_argument(
        "--ccl-device",
        dest="ccl_device",
        default=None,
        help="Device for the CCL subprocess (default: PRISM's --device).",
    )
    parser.add_argument(
        "--ccl-augmentations", dest="ccl_augmentations", type=int, default=8
    )
    parser.add_argument(
        "--ccl-starts",
        dest="ccl_starts",
        type=int,
        default=0,
        help=(
            "CCL starts per instance; 0 (default) uses the environment "
            "default, i.e. full POMO multi-start with starts = n, the same "
            "budget the URS baseline runs at."
        ),
    )
    parser.add_argument(
        "--ccl-batch-size",
        dest="ccl_batch_size",
        type=int,
        default=0,
        help=(
            "instances per CCL forward pass; 0 (default) does the whole "
            "variant at once. At full POMO the effective batch is "
            "batch x augmentations x starts, so set this low on a "
            "memory-limited GPU at large n."
        ),
    )
    parser.add_argument(
        "--ccl-cache",
        dest="ccl_cache",
        type=Path,
        default=default_cache_path("ccl"),
        help="JSON cache of CCL per-instance objectives.",
    )
    parser.add_argument(
        "--ccl-refresh",
        dest="ccl_refresh",
        action="store_true",
        help="Ignore any cached CCL results and recompute (and overwrite) them.",
    )


def validate_arguments(args: argparse.Namespace, error) -> None:
    if args.ccl_augmentations < 1:
        error("--ccl-augmentations must be positive")
    if args.ccl_starts < 0 or args.ccl_starts == 1:
        # CCL's decoder assumes a multi-start axis, so one start is not a
        # smaller budget, it is a broken one.
        error("--ccl-starts must be 0 (environment default) or at least 2")


def augmentation_factor(args: argparse.Namespace) -> int:
    """Return the shared override, or CCL's baseline-specific default."""
    shared = getattr(args, "aug", None)
    return int(shared if shared is not None else args.ccl_augmentations)


class CclMethod(SubprocessMethod):
    """Run CCL's RouteFinder policy on the variant's own saved instances."""

    name = "ccl"
    result_prefix = "CCL_RESULT_JSON "

    def __init__(self, args: argparse.Namespace, cache: ResultCache | None = None):
        self.args = args
        self.cache = cache if cache is not None else ResultCache(
            args.ccl_cache, refresh=args.ccl_refresh
        )
        self._dataset: Path | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None

    def config(self) -> str:
        starts = self.args.ccl_starts or "pomo"
        return (
            f"checkpoint={Path(self.args.ccl_checkpoint).name},"
            f"starts={starts},aug={augmentation_factor(self.args)}"
        )

    def covers(self, problem: dict) -> str | None:
        """CCL's feature space is capacity x open x backhaul x limit x window.

        That is exactly PRISM's symmetric single- and multi-depot VRP grid, and
        it cannot express the asymmetric families (no coordinates), nor
        pickup-delivery, TSP, OP or PCTSP.
        """
        from problem_data import CCL_VARIANTS

        if problem.get("name") not in CCL_VARIANTS:
            return "not_a_ccl_variant"
        return None

    def cache_key(self, request: MethodRequest) -> str:
        checkpoint = Path(self.args.ccl_checkpoint).resolve()
        try:
            stamp = checkpoint.stat().st_mtime_ns
        except OSError:
            stamp = 0
        return (
            f"{request.batch.variant}|{request.batch.signature}|"
            f"{len(request.batch)}|{checkpoint}|{stamp}|"
            f"{self.args.ccl_starts}|{augmentation_factor(self.args)}"
        )

    def _export(self, request: MethodRequest) -> tuple[Path, bool]:
        """Write the batch's instances in CCL's layout, unchanged in content."""
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.prism_to_ccl import ccl_name, to_ccl_arrays

        variant = request.batch.variant
        name = ccl_name(variant)
        arrays = to_ccl_arrays(variant, request.batch.data, len(request.batch))
        self._temporary = tempfile.TemporaryDirectory(prefix="prism-ccl-")
        target = Path(self._temporary.name) / name / "test"
        target.mkdir(parents=True, exist_ok=True)
        dataset = target / f"{request.batch.n}.npz"
        np.savez(dataset, **arrays)
        return dataset, name.startswith("md")

    def command(self, request: MethodRequest) -> list[str]:
        dataset, multi_depot = self._export(request)
        self._dataset = dataset
        command = [
            str(self.args.ccl_python),
            str(WORKER),
            "--dataset", str(dataset),
            "--checkpoint", str(self.args.ccl_checkpoint),
            "--ccl-reld", str(CCL_RELD),
            "--device", self.args.ccl_device or self.args.device,
            "--augmentations", str(augmentation_factor(self.args)),
            "--starts", str(self.args.ccl_starts),
            "--batch-size", str(self.args.ccl_batch_size),
            "--seed", str(request.seed),
        ]
        if multi_depot:
            command.append("--multi-depot")
        return command

    def parse(self, payload: dict, request: MethodRequest) -> MethodResult:
        objectives = payload.get("objectives") or []
        if len(objectives) < len(request.batch):
            return MethodResult.failure(
                f"returned {len(objectives)} of {len(request.batch)} objectives",
                config=self.config(),
            )
        return MethodResult(
            objectives=[float(value) for value in objectives[: len(request.batch)]],
            direction=payload.get("direction", "minimize"),
            seconds=float(payload.get("seconds", 0.0)),
            config=self.config(),
        )

    def run(self, request: MethodRequest) -> MethodResult:
        if request.batch.data is None:
            # --n-node builds problems directly; CCL needs the tensor batch.
            return MethodResult.unsupported("no_tensor_batch", config=self.config())
        problem, _initial_route = request.batch.problem(0)
        reason = self.covers(problem)
        if reason is not None:
            return MethodResult.unsupported(reason, config=self.config())
        if not Path(self.args.ccl_python).exists():
            return MethodResult.unsupported(
                f"no_ccl_environment({self.args.ccl_python})", config=self.config()
            )

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
                direction=entry.get("direction", "minimize"),
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
