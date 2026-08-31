#!/usr/bin/env python3
"""Materialize PRISM's benchmark store in its neutral tensor-dictionary schema.

Imported benchmark sources use several historical layouts: pickled tuples,
text TSP rows, packed OP/PCTSP tensors, and URS dictionaries with keys such as
``dist_matrix`` and ``node_demand``. They remain useful provenance, but they are
not runtime formats once the files live under ``datasets/benchmarks``.

This command reads each source through ``problem_data.load_saved_data`` and
writes ``<variant><scale>_prism.pt`` atomically. ``DatasetFinder`` preferentially
selects that canonical artifact. Exact per-instance references embedded in a
source are retained under ``optimal``; aggregate constants are intentionally
not promoted to per-instance references.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from problem_data import (  # noqa: E402
    BENCHMARK_VARIANTS,
    DEFAULT_DATASET_DIR,
    DatasetFinder,
    _ORACLE_KEYWORDS,
    load_saved_data,
)


def _instance_count(path: Path) -> int:
    if path.suffix.lower() == ".pt":
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if torch.is_tensor(saved):
            return len(saved)
        if not isinstance(saved, dict):
            raise ValueError(f"{path}: unsupported PT payload {type(saved).__name__}")
        for key in ("xy", "dist", "dist_matrix", "demand", "node_demand"):
            value = saved.get(key)
            if torch.is_tensor(value) and value.ndim:
                return len(value)
        raise ValueError(f"{path}: no batched tensor identifies its instance count")
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as source:
            return len(pickle.load(source))
    if path.suffix.lower() == ".txt":
        return len(path.read_text().splitlines())
    raise ValueError(f"{path}: unsupported benchmark source")


def _embedded_references(path: Path, count: int) -> torch.Tensor | None:
    """Return only exact per-instance references, never aggregate defaults."""
    if path.suffix.lower() == ".txt":
        # TSP text rows include an explicit tour after the ``output`` marker.
        coordinates = []
        tours = []
        for line in path.read_text().splitlines():
            fields = line.split()
            marker = fields.index("output")
            coordinates.append(
                [[float(fields[i]), float(fields[i + 1])] for i in range(0, marker, 2)]
            )
            tours.append([int(node) - 1 for node in fields[marker + 1 : -1]])
        xy = torch.tensor(coordinates, dtype=torch.float32)
        tour = torch.tensor(tours, dtype=torch.long)
        ordered = xy.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
        return torch.linalg.vector_norm(
            ordered - ordered.roll(-1, 1), dim=-1
        ).sum(1)
    if path.suffix.lower() != ".pt":
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict):
        return None
    for key in ("optimal", "result"):
        value = saved.get(key)
        if torch.is_tensor(value) and value.ndim and len(value) == count:
            return value.detach().cpu().float().reshape(count)
    return None


def _canonical_path(root: Path, name: str, scale: int) -> Path:
    return root / name / f"{name}{scale}_prism.pt"


def _validate_canonical(path: Path, name: str, count: int) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError(f"{path}: canonical payload is not a dictionary")
    anchor = saved.get("xy", saved.get("dist"))
    if not torch.is_tensor(anchor) or len(anchor) != count:
        raise ValueError(f"{path}: canonical batch does not contain {count} instances")
    # Exercise the same reader and first-prefix contract used by evaluation.
    load_saved_data(path, name, min(8, count), allow_aggregate_reference=False)


def normalize_variant(
    root: Path, name: str, scale: int, *, force: bool = False
) -> tuple[Path, int, str]:
    target = _canonical_path(root, name, scale)
    if target.is_file() and not force:
        count = _instance_count(target)
        _validate_canonical(target, name, count)
        return target, count, "verified"

    finder = DatasetFinder(root)
    paths = finder.get(name, scale)
    if paths is None:
        raise FileNotFoundError(f"no size-{scale} source for {name} under {root}")
    source = Path(paths["data_path"])
    if source == target:
        candidates = [
            path
            for path in target.parent.iterdir()
            if path.is_file()
            and path != target
            and path.suffix.lower() in {".pkl", ".pt", ".txt"}
            and finder._matches_scale(path.name, name, scale)
            and not any(keyword in path.name.lower() for keyword in _ORACLE_KEYWORDS)
        ]
        if not candidates:
            raise ValueError(
                f"{target}: --force needs a retained non-canonical source"
            )
        source = min(candidates, key=finder._data_rank)
    count = _instance_count(source)
    data, _ = load_saved_data(
        source,
        name,
        count,
        allow_aggregate_reference=False,
    )
    reference = _embedded_references(source, count)
    if reference is not None:
        data["optimal"] = reference

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(target)
    _validate_canonical(target, name, count)
    return target, count, f"converted:{source.name}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--scale", type=int, default=100)
    parser.add_argument(
        "--variants",
        default="all",
        help="comma-separated benchmark variants, or all (default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild existing canonical artifacts from retained source files",
    )
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error("--scale must be positive")
    if args.variants == "all":
        variants = list(BENCHMARK_VARIANTS)
    else:
        variants = [value.strip() for value in args.variants.split(",") if value.strip()]
        unknown = sorted(set(variants) - set(BENCHMARK_VARIANTS))
        if unknown:
            parser.error("unknown variants: " + ", ".join(unknown))

    converted = 0
    verified = 0
    for index, name in enumerate(variants, 1):
        target, count, status = normalize_variant(
            args.dataset_dir, name, args.scale, force=args.force
        )
        converted += status.startswith("converted:")
        verified += status == "verified"
        print(
            "NORMALIZE",
            f"{index}/{len(variants)}",
            f"variant={name}",
            f"instances={count}",
            f"status={status}",
            f"target={target}",
            flush=True,
        )
    print(
        "NORMALIZE_SUMMARY",
        f"variants={len(variants)}",
        f"converted={converted}",
        f"verified={verified}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
