#!/usr/bin/env python3
"""E1a/E1b: does the model use resource-program properties semantically?

Each resource is represented by fixed row properties and a variable-size set
of executable term properties. The model maps every term with shared weights
and pools the set, so adding a term never adds an input dimension. This probe
intervenes on those semantic properties before the learned pooling step.

This script intervenes on program properties at inference and measures the effect,
with no retraining:

    zero     blank one component's slots -- the model is told nothing about it
    shuffle  roll one component across compatible active resource rows or term
             sets (term sets must have equal cardinality)

``zero`` is E1a: it measures whether a component is *read* at all. ``shuffle``
is E1b: it moves properties between compatible resource programs. An effect is
therefore not explained only by the model reacting to an all-zero input.

Reported per (variant, component):

    d_energy      mean |change in edge energy|, in units of the unperturbed
                  energy's own std -- 0 means the component is inert
    tau           mean Kendall tau between the perturbed and unperturbed
                  candidate ranking at each decision state, which is what the
                  decoder actually consumes; 1.0 means no behavioural change
    d_auc         change in the resource energy's feasibility AUROC (the E4a
                  statistic), i.e. whether constraint awareness survives

This tool intentionally supports only the current pooled-term schema. Old flat
descriptor checkpoints are incompatible with the current model contract.

Usage:
    python scripts/probe_descriptor.py --checkpoint ../epoch235_015.pt \
        --code-root . --dataset-dir datasets/benchmarks \
        --variants cvrp,cvrptw,cvrpbltw,pdcvrp --instances 4 --states 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_field import (  # noqa: E402
    build_decoder,
    build_model,
    install_incumbent,
    load_code_root,
    state_records,
)


COMPONENTS: dict[str, tuple[str, tuple[int, ...]]] = {
    "row_kind": ("row", (0, 1, 2, 3)),
    "bound": ("row", (4, 5, 6)),
    "check_phase": ("row", (7, 8, 9)),
    "scope": ("row", (10, 11)),
    "bound_horizon": ("row", (12, 13)),
    "state": ("row", (14, 15, 16)),
    "relation_density": ("row", (17,)),
    "direction": ("row", (18, 19)),
    "source": ("term", (0, 1, 2, 3)),
    "read_point": ("term", (4,)),
    "operation": ("term", (5, 6, 7, 8, 9, 10, 11)),
    "phase": ("term", (12,)),
    "when": ("term", (13, 14, 15, 16, 17, 18)),
    "gate": ("term", (19, 20, 21)),
    "coefficient": ("term", (22, 23)),
}


def _term_slices(counts) -> list[slice]:
    slices = []
    offset = 0
    for count in counts.tolist():
        slices.append(slice(offset, offset + int(count)))
        offset += int(count)
    return slices


def perturbed_program(graph, location, slots, mode, active):
    """Return intervened ``(row, term)`` tensors for one decoder graph."""
    rows = graph.resource_row_properties.clone()
    terms = graph.resource_term_properties.clone()
    target = rows if location == "row" else terms
    index = list(slots)
    if mode == "zero":
        target[:, index] = 0.0
        return rows, terms
    if mode == "shuffle":
        live = np.flatnonzero(active > 0.5)
        if len(live) < 2:
            return None
        if location == "row":
            replacement = rows[live.tolist()].clone()
            replacement[:, index] = graph.resource_row_properties[
                np.roll(live, 1).tolist()
            ][:, index]
            rows[live.tolist()] = replacement
            return rows, terms
        counts = graph.resource_term_counts.reshape(-1)
        slices = _term_slices(counts)
        changed = False
        for count in sorted({int(counts[i]) for i in live}):
            group = [int(i) for i in live if int(counts[i]) == count]
            if count == 0 or len(group) < 2:
                continue
            rolled = np.roll(group, 1)
            for destination, source in zip(group, rolled):
                terms[slices[destination], index] = (
                    graph.resource_term_properties[slices[int(source)], index]
                )
            changed = True
        return (rows, terms) if changed else None
    raise ValueError(f"unknown intervention: {mode}")


def ranking_tau(base: pd.DataFrame, other: pd.DataFrame) -> float:
    """Mean Kendall tau over decision states, on each state's candidate set.

    The decoder ranks candidates leaving one node, so the ranking that matters
    is per (state, source) -- a global correlation over all edges would be
    dominated by between-node energy differences the decoder never compares.
    """
    taus = []
    for key, group in base.groupby(["instance", "state"], sort=False):
        partner = other[
            (other.instance == key[0]) & (other.state == key[1])
        ]
        if len(partner) != len(group) or len(group) < 3:
            continue
        statistic = kendalltau(group.energy.values, partner.energy.values)[0]
        if np.isfinite(statistic):
            taus.append(statistic)
    return float(np.mean(taus)) if taus else float("nan")


def resource_auc(frame: pd.DataFrame) -> float:
    """E4a statistic: feasibility AUROC of the resource energy alone."""
    sub = frame[(frame.visited == 0) & (frame.is_depot == 0)]
    labels = sub.mask_feasible.values
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, -sub.energy_resource.values)


def run_variant(code, model, variant, data, args, components) -> list[dict]:
    torch = code.torch
    rows: list[dict] = []
    baseline_frames = []
    perturbed_frames: dict[tuple[str, str], list[pd.DataFrame]] = {}
    changed: dict[tuple[str, str], list[bool]] = {}

    for instance in range(args.instances):
        problem = code.solver_problem(
            variant, code.instance_data(data, instance)
        )
        decoder = build_decoder(code, problem, args, instance)
        incumbent = install_incumbent(code, model, decoder, args)
        if incumbent is None:
            continue

        graph = code.net.build_decoder_data(decoder, args.device)
        with torch.no_grad():
            base_output = model(graph)
        active = base_output["active_channels"][0].detach().cpu().numpy()
        base = pd.DataFrame(
            state_records(
                code, model, decoder, problem, base_output, incumbent, args
            )
        )
        base["instance"] = instance
        baseline_frames.append(base)

        original_rows = graph.resource_row_properties.clone()
        original_terms = graph.resource_term_properties.clone()
        for name, (location, slots) in components.items():
            for mode in args.modes:
                graph.resource_row_properties = original_rows
                graph.resource_term_properties = original_terms
                replacement = perturbed_program(
                    graph, location, slots, mode, active
                )
                if replacement is None:
                    continue
                # A shuffle is a no-op when the active resources already agree
                # on this component, and a zeroing is a no-op when the slots are
                # already zero. Both look exactly like "the model ignores it",
                # so track it rather than let it silently read as an effect.
                replacement_rows, replacement_terms = replacement
                effective = not (
                    torch.equal(replacement_rows, original_rows)
                    and torch.equal(replacement_terms, original_terms)
                )
                changed.setdefault((name, mode), []).append(effective)
                graph.resource_row_properties = replacement_rows
                graph.resource_term_properties = replacement_terms
                with torch.no_grad():
                    output = model(graph)
                frame = pd.DataFrame(
                    state_records(
                        code, model, decoder, problem, output, incumbent, args
                    )
                )
                frame["instance"] = instance
                perturbed_frames.setdefault((name, mode), []).append(frame)
        graph.resource_row_properties = original_rows
        graph.resource_term_properties = original_terms

    if not baseline_frames:
        return rows
    baseline = pd.concat(baseline_frames, ignore_index=True)
    spread = baseline.energy.std()
    base_auc = resource_auc(baseline)

    for (name, mode), frames in perturbed_frames.items():
        frame = pd.concat(frames, ignore_index=True)
        delta = np.abs(frame.energy.values - baseline.energy.values)
        rows.append(
            {
                "variant": variant,
                "R": int(baseline.n_active_resources.iloc[0]),
                "component": name,
                "mode": mode,
                "input_changed": float(np.mean(changed.get((name, mode), [0]))),
                "d_energy": float(delta.mean() / spread) if spread else 0.0,
                "tau": ranking_tau(baseline, frame),
                "auc_base": base_auc,
                "auc_perturbed": resource_auc(frame),
            }
        )
    for row in rows:
        row["d_auc"] = row["auc_perturbed"] - row["auc_base"]
    return rows


def _sorted_term_component(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values
    order = np.lexsort(values.T[::-1])
    return values[order]


def census(code, args, components) -> pd.DataFrame:
    """How much does the program distinguish one active resource from another?

    No checkpoint involved -- this is a property of the decoder's program
    construction. If two co-active resources receive identical rows, no model
    reading only program properties can price them differently, and the resource
    representation has collapsed to whatever still separates them.
    """
    dataset_dir = args.dataset_dir or code.problem_data.DEFAULT_DATASET_DIR
    finder = code.problem_data.DatasetFinder(dataset_dir)
    rows = []
    for variant in [n.strip() for n in args.variants.split(",") if n.strip()]:
        try:
            paths = finder.get(variant, 100)
            data, _ = code.problem_data.load_saved_data(
                paths["data_path"], variant, 1,
                solution_path=paths["solution_path"],
                allow_aggregate_reference=False,
            )
        except Exception as error:  # noqa: BLE001
            print(f"{variant}: skipped ({error})", flush=True)
            continue
        problem = code.solver_problem(
            variant, code.instance_data(data, 0)
        )
        decoder = code.prism_decoder.Decoder(
            problem, candidate_config={"max_candidates": args.candidates},
            n_rollouts=1, beta=2.0,
        )
        row_properties = np.asarray(decoder.resource_row_properties)
        term_properties = np.asarray(decoder.resource_term_properties)
        term_counts = np.asarray(decoder.resource_term_counts, dtype=np.int64)
        term_slices = _term_slices(term_counts)
        active = np.flatnonzero(
            np.asarray(decoder.metadata["field_channel_mask"]) > 0
        )
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                left_index, right_index = active[i], active[j]
                left = row_properties[left_index]
                right = row_properties[right_index]
                differing = int(np.count_nonzero(np.abs(left - right) > 1e-9))
                separating = []
                for component, (location, slots) in components.items():
                    if location == "row":
                        distinct = not np.allclose(
                            left[list(slots)], right[list(slots)], atol=1e-9
                        )
                    else:
                        left_terms = _sorted_term_component(
                            term_properties[term_slices[left_index]][:, list(slots)]
                        )
                        right_terms = _sorted_term_component(
                            term_properties[term_slices[right_index]][:, list(slots)]
                        )
                        distinct = left_terms.shape != right_terms.shape or not np.allclose(
                            left_terms, right_terms, atol=1e-9
                        )
                    if distinct:
                        separating.append(component)
                if term_counts[left_index] != term_counts[right_index]:
                    separating.append("term_count")
                    differing += 1
                rows.append({
                    "variant": variant,
                    "pair": f"r{active[i]}/r{active[j]}",
                    "differing_properties": differing,
                    "identical": not separating,
                    "separating_components": ",".join(separating) or "-",
                })
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--variants", default="cvrp,cvrptw,cvrpbltw,pdcvrp")
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--min-changed-edges", type=int, default=8)
    parser.add_argument("--candidate-mode", default="geometric")
    parser.add_argument("--risk-penalty", type=float, default=1.0)
    parser.add_argument("--incumbent", choices=("field", "distance"),
                        default="field")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--modes",
        default="zero,shuffle",
        help="comma-separated interventions to apply",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--census",
        action="store_true",
        help=(
            "report only how far the program separates co-active resources; "
            "needs no checkpoint, so it runs against either schema"
        ),
    )
    args = parser.parse_args(argv)
    args.modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code = load_code_root(args.code_root)
    schema = code.net.MODEL_SCHEMA
    if schema != "typed_resource_v13_pooled_terms":
        raise SystemExit(
            f"probe requires typed_resource_v13_pooled_terms, got {schema!r}"
        )
    components = COMPONENTS

    if args.census:
        table = census(code, args, components)
        pd.set_option("display.width", 170)
        pd.set_option("display.max_rows", 400)
        print(
            f"\nschema={schema} row_width="
            f"{code.prism_decoder.RESOURCE_ROW_PROPERTY_DIM} term_width="
            f"{code.prism_decoder.RESOURCE_TERM_PROPERTY_DIM}"
        )
        print(table.to_string(index=False))
        if not table.empty:
            print(
                f"\nidentical co-active pairs: "
                f"{int(table.identical.sum())}/{len(table)}   "
                "mean differing row properties/count: "
                f"{table.differing_properties.mean():.1f}"
            )
        return 0

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --census is given")
    model, checkpoint, _config = build_model(code, args.checkpoint, args.device)
    print(
        f"checkpoint epoch={checkpoint.get('epoch')} schema={schema} "
        f"components={len(components)}",
        flush=True,
    )

    dataset_dir = args.dataset_dir or code.problem_data.DEFAULT_DATASET_DIR
    finder = code.problem_data.DatasetFinder(dataset_dir)
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]

    rows: list[dict] = []
    for variant in variants:
        try:
            paths = finder.get(variant, 100)
            data, _reference = code.problem_data.load_saved_data(
                paths["data_path"],
                variant,
                args.instances,
                solution_path=paths["solution_path"],
                allow_aggregate_reference=False,
            )
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"{variant}: skipped ({error})", flush=True)
            continue
        variant_rows = run_variant(code, model, variant, data, args, components)
        rows += variant_rows
        print(f"{variant}: {len(variant_rows)} measurements", flush=True)

    table = pd.DataFrame(rows)
    if table.empty:
        print("no measurements")
        return 1
    pd.set_option("display.width", 170)
    pd.set_option("display.max_rows", 400)

    for mode in args.modes:
        section = table[table["mode"] == mode]
        if section.empty:
            continue
        print("\n" + "=" * 92)
        print(
            f"{mode}: per-variant effect  "
            f"(tau=1 means the candidate ranking is unchanged)"
        )
        print("=" * 92)
        pivot = section.pivot_table(
            index="component", columns="variant", values="tau"
        )
        order = [name for name in components if name in pivot.index]
        print(pivot.loc[order].to_string(float_format="%.3f"))

        print(f"\n{mode}: pooled over variants")
        pooled = (
            section.groupby("component")[
                ["input_changed", "d_energy", "tau", "d_auc"]
            ]
            .mean()
            .loc[order]
        )
        print(pooled.to_string(float_format="%.4f"))
        inert = pooled.index[pooled.input_changed < 1e-9].tolist()
        if inert:
            print(
                "  note: the intervention was a no-op on every instance for "
                + ", ".join(inert)
                + " -- those properties are already constant across the active "
                "resources, so this measures the program input, not the model."
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
