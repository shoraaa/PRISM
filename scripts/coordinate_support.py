#!/usr/bin/env python3
"""Coordinate support and separation census for the resource property maps.

Answers three questions about the declaration representation, with **no
checkpoint** -- everything here reads ``Decoder`` properties, so it runs the
moment a decoder builds.

1. ``--support``    Which row/term coordinates does training actually vary, and
                    is any constant coordinate recoverable by swapping training
                    variants for other benchmark variants? A coordinate that is
                    constant across the whole 110-variant family cannot be
                    covered from inside the benchmark at any variant selection.

2. ``--demand``     Which coordinates do the unseen-resource declarations place
                    outside the training support? These are the directions the
                    zero-shot claim leans on. Continuous coordinates are
                    reported separately: a new value on an axis training already
                    varies is interpolation, not an untrained direction.

3. ``--separation`` What does ``--program-blind-resources`` actually remove? For
                    every pair of co-active resources in a variant, compare two
                    views: the DESCRIPTOR (row properties plus the pooled term
                    set -- exactly what the ablation blanks) and the RESIDUAL
                    (per-resource node attributes, scales, bounds and edge
                    pressure -- what still reaches a program-blind model). A
                    pair separated by the descriptor but NOT by the residual is
                    a pair the ablated model provably cannot tell apart. That
                    count is the information the ablation costs.

Usage:
    python scripts/coordinate_support.py --all
    python scripts/coordinate_support.py --separation --variants cvrpbltw,evrp
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prism_decoder  # noqa: E402
import torch  # noqa: E402
from prism_eval import instances as pe_instances  # noqa: E402
from problem_data import (  # noqa: E402
    BENCHMARK_VARIANTS,
    TRAIN_VARIANTS,
    DatasetFinder,
    load_saved_data,
)

ROW_DIM = prism_decoder.RESOURCE_ROW_PROPERTY_DIM
TERM_DIM = prism_decoder.RESOURCE_TERM_PROPERTY_DIM
TOL = 1e-4


class Rows:
    """Per-resource views of one variant, as the model receives them."""

    def __init__(self, decoder):
        self.names = [
            row.get("name", f"row{i}")
            for i, row in enumerate(decoder.resource_declarations)
        ]
        self.row = np.asarray(
            decoder.resource_row_properties, dtype=np.float64
        ).reshape(-1, ROW_DIM)
        terms = np.asarray(
            decoder.resource_term_properties, dtype=np.float64
        ).reshape(-1, TERM_DIM)
        counts = np.asarray(decoder.resource_term_counts, dtype=np.int64).ravel()
        bounds = np.cumsum(np.concatenate(([0], counts)))
        self.terms = [terms[bounds[i] : bounds[i + 1]] for i in range(len(counts))]
        self.counts = counts
        # What a program-blind model still sees per resource: static node
        # attributes, the normalization scale, and candidate-edge pressure.
        node = np.asarray(decoder.node_resource_features, dtype=np.float64)
        pressure = np.asarray(decoder.resource_pressure, dtype=np.float64)
        scales = np.asarray(decoder.resource_scales, dtype=np.float64).ravel()
        count = self.row.shape[0]
        if count == 0:
            # TSP and ATSP declare no resources; there is nothing to compare.
            self.pressure_raw = np.zeros((0, 0))
            self.node_raw = np.zeros((0, 0, 0))
            self.residual = []
            return
        node = node.reshape(-1, count, node.shape[-1] if node.ndim == 3 else 1)
        pressure = pressure.reshape(-1, count)
        self.pressure_raw = pressure
        self.node_raw = node
        self.residual = [
            np.concatenate(
                (
                    node[:, i, :].mean(axis=0),
                    node[:, i, :].std(axis=0),
                    [pressure[:, i].mean(), pressure[:, i].std(), scales[i]],
                )
            )
            for i in range(count)
        ]

    def signature(self, index: int) -> np.ndarray:
        """Behavioral signature: what the row does to this instance.

        Quantiles of the resource pressure it induces over candidate edges plus
        the moments of the per-node attributes its algebra resolves. Derived by
        execution rather than by enumerating syntactic properties, so a new
        language primitive changes the response, not the width.
        """
        quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        node = self.node_raw[:, index, :]
        # Pressure is a ratio to the declared bound and is unbounded above: a
        # tight row can consume several times its budget on one edge. Compress
        # it so a row that is merely *tighter* than a trained one stays near it
        # instead of being pushed out of the space by magnitude alone.
        pressure = np.quantile(self.pressure_raw[:, index], quantiles)
        pressure = pressure / (1.0 + np.abs(pressure))
        return np.concatenate(
            (
                pressure,
                node.mean(axis=0),
                node.std(axis=0),
            )
        )

    def descriptor(self, index: int) -> np.ndarray:
        """Row properties plus the order-invariant pooled term set."""
        terms = self.terms[index]
        pooled = terms.sum(axis=0) if len(terms) else np.zeros(TERM_DIM)
        return np.concatenate((self.row[index], pooled, [self.counts[index]]))


def load(variant: str, finder: DatasetFinder, size: int, seed: int) -> Rows:
    if variant in pe_instances.GENERATORS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            data = pe_instances.GENERATORS[variant](size, 1)
    else:
        paths = finder.get(variant, size)
        data, _reference = load_saved_data(
            paths["data_path"],
            variant,
            1,
            solution_path=paths["solution_path"],
            allow_aggregate_reference=False,
        )
    problem = pe_instances.solver_problem(
        variant, pe_instances._instance_data(data, 0)
    )
    return Rows(prism_decoder.Decoder(problem))


def collect(variants, finder, args) -> dict[str, Rows]:
    loaded: dict[str, Rows] = {}
    for variant in variants:
        try:
            loaded[variant] = load(variant, finder, args.size, args.seed)
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"  {variant}: skipped ({type(error).__name__}: {error})")
    return loaded


def _values(stack: np.ndarray, index: int) -> set[float]:
    return set(np.round(stack[:, index], 4))


def report_support(loaded: dict[str, Rows]) -> dict[str, list[int]]:
    """Training coverage versus the whole benchmark family, per coordinate."""
    train = set(TRAIN_VARIANTS)
    seen = [name for name in loaded if name in train]
    blocks = {
        "row": (
            np.vstack([loaded[n].row for n in seen]),
            np.vstack([r.row for r in loaded.values()]),
            ROW_DIM,
        ),
        "term": (
            np.vstack([t for n in seen for t in loaded[n].terms]),
            np.vstack([t for r in loaded.values() for t in r.terms]),
            TERM_DIM,
        ),
    }
    constant: dict[str, list[int]] = {}
    for label, (train_stack, all_stack, dim) in blocks.items():
        recoverable, dead = [], []
        print(
            f"\n=== {label} properties ({dim} coords, "
            f"{len(train_stack)} training rows, {len(all_stack)} total) ==="
        )
        print(f"{'i':>3}  {'training values':<24}{'all-benchmark values':<24}status")
        for index in range(dim):
            train_values = sorted(_values(train_stack, index))
            all_values = sorted(_values(all_stack, index))
            if len(train_values) > 1:
                status = "varies in training"
            elif len(all_values) > 1:
                status = "RECOVERABLE by variant swap"
                recoverable.append(index)
            else:
                status = "constant across every variant"
                dead.append(index)

            def show(values: list[float]) -> str:
                text = ",".join(f"{value:g}" for value in values[:4])
                return (text + ("..." if len(values) > 4 else ""))[:23]

            print(f"{index:>3}  {show(train_values):<24}{show(all_values):<24}{status}")
        print(f"  recoverable by variant swap : {recoverable}")
        print(f"  constant across every variant: {dead}")
        constant[label] = dead
    return constant


def report_demand(loaded: dict[str, Rows], unseen: list[str], finder, args) -> None:
    """Coordinates the unseen resources place outside the training support."""
    train = set(TRAIN_VARIANTS)
    seen = [name for name in loaded if name in train]
    train_row = np.vstack([loaded[n].row for n in seen])
    train_term = np.vstack([t for n in seen for t in loaded[n].terms])
    continuous = {
        "row": [i for i in range(ROW_DIM) if len(_values(train_row, i)) > 2],
        "term": [i for i in range(TERM_DIM) if len(_values(train_term, i)) > 2],
    }
    print("\n=== unseen-resource coordinate demand ===")
    print("(coordinates marked * are continuous in training: a new value there")
    print(" is interpolation on a trained axis, not an untrained direction)")
    for variant in unseen:
        try:
            rows = load(variant, finder, args.size, args.seed)
        except Exception as error:  # noqa: BLE001
            print(f"  {variant}: skipped ({type(error).__name__}: {error})")
            continue

        def outside(stack, train_stack, axes):
            marked = []
            for index in range(stack.shape[1]):
                if not _values(stack, index) <= _values(train_stack, index):
                    marked.append(f"{index}*" if index in axes else str(index))
            return marked

        row_out = outside(rows.row, train_row, continuous["row"])
        term_out = outside(
            np.vstack(rows.terms), train_term, continuous["term"]
        )
        print(
            f"  {variant:>9}: row [{', '.join(row_out) or '-'}]"
            f"  term [{', '.join(term_out) or '-'}]"
        )


def report_signature(loaded: dict[str, Rows]) -> None:
    """Soundness and completeness of the behavioral signature.

    Soundness   rows with identical descriptors must get identical signatures.
    Completeness rows with different descriptors must get different signatures;
                 a collision is a pair the signature cannot tell apart, and is
                 fixed by extending the probe battery rather than by changing
                 any tensor width.
    """
    print("\n=== behavioral signature: soundness and completeness ===")
    collisions, unsound, pairs = [], [], 0
    for variant, rows in sorted(loaded.items()):
        for left, right in combinations(range(len(rows.names)), 2):
            pairs += 1
            same_syntax = (
                np.abs(rows.descriptor(left) - rows.descriptor(right)).sum() <= TOL
            )
            signature_gap = float(
                np.abs(rows.signature(left) - rows.signature(right)).sum()
            )
            pair = (variant, rows.names[left], rows.names[right], signature_gap)
            if same_syntax and signature_gap > TOL:
                unsound.append(pair)
            elif not same_syntax and signature_gap <= TOL:
                collisions.append(pair)
    print(f"  co-active pairs checked          : {pairs}")
    print(f"  separated by syntax, not signature: {len(collisions)}")
    print(f"  identical syntax, distinct signature: {len(unsound)}")
    for variant, left, right, gap in collisions[:10]:
        print(f"    collision  {variant}: {left} | {right} (gap {gap:.4f})")
    for variant, left, right, gap in unsound[:10]:
        print(f"    unsound    {variant}: {left} | {right} (gap {gap:.4f})")
    if not collisions and not unsound:
        print(
            "  the signature separates exactly the pairs the syntax does, so it"
            "\n  is a viable drop-in for the property maps on this family."
        )


def report_separation(loaded: dict[str, Rows]) -> None:
    """What --program-blind-resources removes, pair by co-active pair."""
    print("\n=== descriptor vs residual separation of co-active resources ===")
    print("descriptor = row + pooled terms (blanked by --program-blind-resources)")
    print("residual   = node attributes, pressure, scale (still reaching the model)")
    print(
        f"\n{'variant':<14}{'pair':<34}{'desc':>6}{'resid':>8}  verdict"
    )
    tallies = {"both": 0, "descriptor only": 0, "residual only": 0, "neither": 0}
    lost: list[tuple[str, str, str]] = []
    for variant, rows in sorted(loaded.items()):
        for left, right in combinations(range(len(rows.names)), 2):
            descriptor_gap = float(
                np.abs(rows.descriptor(left) - rows.descriptor(right)).sum()
            )
            residual_gap = float(
                np.abs(rows.residual[left] - rows.residual[right]).sum()
            )
            by_descriptor = descriptor_gap > TOL
            by_residual = residual_gap > TOL
            if by_descriptor and by_residual:
                verdict = "both"
            elif by_descriptor:
                verdict = "descriptor only"
                lost.append((variant, rows.names[left], rows.names[right]))
            elif by_residual:
                verdict = "residual only"
            else:
                verdict = "neither"
            tallies[verdict] += 1
            pair = f"{rows.names[left]} | {rows.names[right]}"
            print(
                f"{variant:<14}{pair:<34}{descriptor_gap:>6.2f}"
                f"{residual_gap:>8.2f}  {verdict}"
            )
    total = sum(tallies.values()) or 1
    print("\n  co-active pairs by what separates them:")
    for verdict, count in tallies.items():
        print(f"    {verdict:<16} {count:>4}  ({100.0 * count / total:.1f}%)")
    print(
        "\n  'descriptor only' is the cost of --program-blind-resources: pairs the\n"
        "  ablated model cannot distinguish. 'neither' is a pre-existing blind\n"
        "  spot that the descriptor does not fix either."
    )
    if lost:
        print("\n  pairs collapsed by the ablation:")
        for variant, left, right in lost:
            print(f"    {variant}: {left} | {right}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", action="store_true")
    parser.add_argument("--demand", action="store_true")
    parser.add_argument("--separation", action="store_true")
    parser.add_argument(
        "--signature",
        action="store_true",
        help="check the behavioral signature against the syntactic descriptor",
    )
    parser.add_argument("--all", action="store_true", help="run every report")
    parser.add_argument("--dataset-dir", default="datasets/benchmarks")
    parser.add_argument(
        "--variants",
        default="all",
        help="comma-separated variants, or 'all' for the benchmark family",
    )
    parser.add_argument(
        "--unseen",
        default="evrp,evrptw,vrpdb,vrpdbtw",
        help="generated variants to score against the training support",
    )
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.all or not (
        args.support or args.demand or args.separation or args.signature
    ):
        args.support = args.demand = args.separation = args.signature = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    finder = DatasetFinder(args.dataset_dir)
    if args.variants == "all":
        variants = list(BENCHMARK_VARIANTS)
    else:
        variants = [name.strip() for name in args.variants.split(",") if name.strip()]

    print(f"loading {len(variants)} variants at n={args.size}")
    loaded = collect(variants, finder, args)
    if not loaded:
        raise SystemExit("no variants loaded")
    covered = sum(1 for name in loaded if name in set(TRAIN_VARIANTS))
    print(f"loaded {len(loaded)}/{len(variants)}; {covered} are training variants")

    if args.support:
        report_support(loaded)
    if args.demand:
        unseen = [name.strip() for name in args.unseen.split(",") if name.strip()]
        report_demand(loaded, unseen, finder, args)
    if args.separation:
        report_separation(loaded)
    if args.signature:
        report_signature(loaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
