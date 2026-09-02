"""PRISM itself, and the native controls that share its search.

The controls exist to separate what the learned field contributes from what
the decoder's search would find anyway, so they must run through exactly the
same decoder with exactly the same budget -- only the energy differs. Keeping
them next to PRISM is what makes that hard to break.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from train import infer_instance

from .base import InProcessMethod, MethodRequest, MethodResult


NATIVE_CONTROL_NAMES = ("constant", "distance", "random")


@dataclass(frozen=True)
class LoadedCheckpoint:
    """One checkpoint under test, already rebuilt into a model.

    ``name`` is the method name every row of this checkpoint carries, so a run
    comparing several of them is a group-by on the existing ``method`` column
    rather than a second dimension in the schema.
    """

    name: str
    path: Path
    model: Any
    epoch: Any = "unknown"


def method_names(paths: Sequence[Path]) -> list[str]:
    """Name one PRISM method per checkpoint, distinctly and readably.

    A single checkpoint keeps the bare ``prism`` name, so existing CSVs,
    ``--cached`` files and comparisons are unaffected. Several checkpoints are
    named by the parts of their paths that actually differ, which is what makes
    an ablation directory readable: ``pretrained/v15/best.pt`` and
    ``pretrained/only-ppo/best.pt`` become ``prism:v15`` and
    ``prism:only-ppo`` rather than two indistinguishable ``best``s.
    """
    if len(paths) == 1:
        return ["prism"]
    parts = [list(path.resolve().parts[:-1]) + [path.stem] for path in paths]
    depth = max(len(part) for part in parts)
    for candidate in range(1, depth + 1):
        if len({tuple(part[-candidate:]) for part in parts}) == len(parts):
            depth = candidate
            break
    # Pad on the left so shallower paths still align, then drop the components
    # every checkpoint shares -- they say nothing about which one a row is from.
    labels = [
        [""] * (depth - len(part)) + part[-depth:] for part in parts
    ]
    kept = [
        index
        for index in range(depth)
        if len({label[index] for label in labels}) > 1
    ]
    names = [
        "prism:" + "-".join(label[index] for index in kept if label[index])
        for label in labels
    ]
    if len(set(names)) != len(names):
        # Only reachable when the same checkpoint is listed twice; an index
        # keeps the method column a key rather than silently merging rows.
        names = [f"{name}#{index}" for index, name in enumerate(names)]
    return names


def decoder_namespace(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    """The search configuration PRISM and its controls both decode under."""
    return SimpleNamespace(
        candidates=args.candidates,
        n_rollouts=args.rollouts,
        beta=2.0,
        seed=seed,
        search_iterations=args.iterations,
        feasibility_lookahead_depth=2,
        device=args.device,
        static_field=args.static_field,
        min_changed_edges=args.min_changed_edges,
        random_escape=args.random_escape,
        srr_exploration_budget=args.srr_exploration_budget,
    )


def _search_config(args: argparse.Namespace) -> str:
    return (
        f"iterations={args.iterations},rollouts={args.rollouts},"
        f"candidates={args.candidates},"
        f"min_changed_edges={args.min_changed_edges},"
        f"field={'static' if args.static_field else 'dynamic'}"
    )


class PrismMethod(InProcessMethod):
    """The checkpoint under test."""

    name = "prism"

    def __init__(self, checkpoint: LoadedCheckpoint, args: argparse.Namespace):
        self.name = checkpoint.name
        self.model = checkpoint.model
        self.checkpoint = Path(checkpoint.path)
        self.args = args
        self._net_evals: list[float] = []

    def config(self) -> str:
        checkpoint = self.checkpoint.resolve()
        try:
            stamp = checkpoint.stat().st_mtime_ns
        except OSError:
            stamp = 0
        return (
            f"checkpoint={checkpoint}@{stamp},"
            f"{_search_config(self.args)},"
            f"srr={self.args.srr_exploration_budget}"
        )

    def run(self, request: MethodRequest) -> MethodResult:
        self._net_evals = []
        result = super().run(request)
        if self._net_evals:
            result.meta["net_evals_mean"] = sum(self._net_evals) / len(
                self._net_evals
            )
        return result

    def solve_one(
        self,
        problem: dict,
        initial_route,
        index: int,
        request: MethodRequest,
    ) -> float:
        _, solution, metrics = infer_instance(
            self.model,
            problem,
            decoder_namespace(self.args, request.instance_seed(index)),
            initial_route=initial_route,
        )
        if not solution["feasible"]:
            raise RuntimeError(f"instance {index} returned an infeasible solution")
        self._last_direction = solution["direction"]
        self._net_evals.append(metrics["net_evals"])
        return float(solution["objective"])

    def on_progress(
        self, request: MethodRequest, index: int, objective: float, seconds: float
    ) -> None:
        label = "PRISM" if self.name == "prism" else self.name
        print(
            f"{label} variant={request.batch.variant} "
            f"instance {index + 1}/{len(request.batch)} "
            f"objective={objective:.6g} "
            f"net_evals={self._net_evals[-1]} "
            f"seconds={seconds:.3f}",
            flush=True,
        )


class NativeControl(InProcessMethod):
    """A decoder-only control: same search, hand-written energy."""

    def __init__(self, name: str, args: argparse.Namespace):
        if name not in NATIVE_CONTROL_NAMES:
            raise ValueError(f"unknown native control: {name}")
        self.name = name
        self.args = args

    def config(self) -> str:
        # Only the escape budget separates a control's search from PRISM's, and
        # the controls get one only when it is asked for explicitly.
        budget = (
            self.args.srr_exploration_budget if self.args.random_escape else 0
        )
        escape = "random" if self.args.random_escape else "disabled"
        return f"{_search_config(self.args)},escape={escape},srr={budget}"

    def solve_one(
        self,
        problem: dict,
        initial_route,
        index: int,
        request: MethodRequest,
    ) -> float:
        _, solution, _ = infer_instance(
            None,
            problem,
            decoder_namespace(self.args, request.instance_seed(index)),
            initial_route=initial_route,
            baseline=self.name,
        )
        if not solution["feasible"]:
            raise RuntimeError(
                f"instance {index} returned an infeasible {self.name} solution"
            )
        self._last_direction = solution["direction"]
        return float(solution["objective"])
