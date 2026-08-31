"""The registry: every method the evaluator can put on the same instances.

Adding a baseline is adding an entry here plus the file it names. A spec owns
its own CLI flags (``add_arguments``) and its own construction, so nothing in
``test.py`` grows a branch per method, and the CSV never grows a column.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from .base import Method, MethodRequest, MethodResult  # noqa: F401


@dataclass(frozen=True)
class MethodSpec:
    """How to expose one method on the command line and build it."""

    name: str
    build: Callable[[argparse.Namespace, object], object]
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None
    validate: Callable[[argparse.Namespace, Callable[[str], None]], None] | None = None
    #: Whether ``--methods all`` includes it. External solvers and foreign
    #: checkpoints are opt-in: they cost wall clock, a checkpoint or a venv.
    in_all: bool = False
    help: str = ""


def _build_native(name: str):
    def build(args: argparse.Namespace, _model) -> object:
        from .prism import NativeControl

        return NativeControl(name, args)

    return build


def _build_prism(args: argparse.Namespace, model) -> object:
    from .prism import PrismMethod

    return PrismMethod(model, args)


def _build_oracle(args: argparse.Namespace, _model) -> object:
    from .oracle import OracleMethod

    return OracleMethod(args)


def _oracle_arguments(parser: argparse.ArgumentParser) -> None:
    from .oracle import add_arguments

    add_arguments(parser)


def _oracle_validate(args: argparse.Namespace, error) -> None:
    from .oracle import validate_arguments

    validate_arguments(args, error)


def _build_ccl(args: argparse.Namespace, _model) -> object:
    from .ccl import CclMethod

    return CclMethod(args)


def _ccl_arguments(parser: argparse.ArgumentParser) -> None:
    from .ccl import add_arguments

    add_arguments(parser)


def _ccl_validate(args: argparse.Namespace, error) -> None:
    from .ccl import validate_arguments

    validate_arguments(args, error)


def _build_urs(args: argparse.Namespace, _model) -> object:
    from .urs import UrsMethod

    return UrsMethod(args)


def _urs_arguments(parser: argparse.ArgumentParser) -> None:
    from .urs import add_arguments

    add_arguments(parser)


def _urs_validate(args: argparse.Namespace, error) -> None:
    from .urs import validate_arguments

    validate_arguments(args, error)


REGISTRY: dict[str, MethodSpec] = {
    "prism": MethodSpec(
        name="prism",
        build=_build_prism,
        help="the checkpoint under test",
    ),
    "constant": MethodSpec(
        name="constant",
        build=_build_native("constant"),
        in_all=True,
        help="control: identical energy on every candidate",
    ),
    "distance": MethodSpec(
        name="distance",
        build=_build_native("distance"),
        in_all=True,
        help="control: normalized edge distance as energy",
    ),
    "random": MethodSpec(
        name="random",
        build=_build_native("random"),
        in_all=True,
        help="control: stable edge-keyed random energy",
    ),
    "oracle": MethodSpec(
        name="oracle",
        build=_build_oracle,
        add_arguments=_oracle_arguments,
        validate=_oracle_validate,
        help="classical solver: PyVRP where it models the problem, else OR-Tools",
    ),
    "urs": MethodSpec(
        name="urs",
        build=_build_urs,
        add_arguments=_urs_arguments,
        validate=_urs_validate,
        help="foreign NCO checkpoint, run out-of-process on the same dataset",
    ),
    "ccl": MethodSpec(
        name="ccl",
        build=_build_ccl,
        add_arguments=_ccl_arguments,
        validate=_ccl_validate,
        help="CCL-MTLVRP's RouteFinder, run in its own environment on the same dataset",
    ),
}

#: PRISM is always measured; the rest are what "baselines" means here.
BASELINE_NAMES: tuple[str, ...] = tuple(
    name for name in REGISTRY if name != "prism"
)
DEFAULT_METHODS: tuple[str, ...] = ("prism", "constant")


def add_method_arguments(parser: argparse.ArgumentParser) -> None:
    """Let every registered method contribute its own flags."""
    parser.add_argument(
        "--aug",
        type=int,
        default=None,
        help=(
            "Override test-time augmentation for constructive neural baselines "
            "such as URS and CCL. Use --aug 1 for memory-bounded large-scale "
            "evaluation; omit it to retain each baseline's native default."
        ),
    )
    for spec in REGISTRY.values():
        if spec.add_arguments is not None:
            spec.add_arguments(parser)


def validate_method_arguments(args: argparse.Namespace, error) -> None:
    """Let every registered method check its own flags."""
    if args.aug is not None and args.aug < 1:
        error("--aug must be positive")
    for spec in REGISTRY.values():
        if spec.validate is not None:
            spec.validate(args, error)


def resolve(selection: str) -> tuple[str, ...]:
    """Turn a --methods selection into method names, PRISM always first.

    Accepts a comma-separated list, ``all`` (PRISM plus every method marked
    ``in_all``) and ``none`` (PRISM alone).
    """
    chosen: list[str] = []
    for token in selection.split(","):
        token = token.strip()
        if not token or token == "none":
            continue
        if token == "all":
            # Expandable inside a list, so "all,oracle" means what it reads as.
            chosen.extend(name for name, spec in REGISTRY.items() if spec.in_all)
        else:
            chosen.append(token)
    unknown = [name for name in chosen if name not in REGISTRY]
    if unknown:
        raise ValueError(
            "unknown method(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(REGISTRY)
        )
    ordered = ["prism"] + [name for name in chosen if name != "prism"]
    seen: list[str] = []
    for name in ordered:
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def build(
    names: tuple[str, ...], args: argparse.Namespace, model
) -> list[object]:
    """Construct the selected methods in the order they will run."""
    return [REGISTRY[name].build(args, model) for name in names]
