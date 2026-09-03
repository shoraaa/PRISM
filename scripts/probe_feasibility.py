#!/usr/bin/env python3
"""A1: do semantic prediction and search energy play their distinct roles?

For every candidate edge at replayed decision states this records the learned
search field ``phi_r(e)``, the separate semantic-margin prediction, and the
signed admissibility margin ``y_{r,e}`` the interpreter computed to decide
legality. The margin's sign is pinned to the authoritative verdict, so
``y_{r,e} >= 0`` is exactly "admissible under r".

Two measurements come out of that pairing:

  * per resource, separate AUROCs for semantic-margin prediction and search
    energy. The former tests executable representation; the latter tests
    whether independently trained energy prices dangerous choices.
  * per state, whether the lowest-energy candidate is admissible, against the
    shortest-distance candidate and the uniform expectation.  This is the
    original probe's question, restated on the surviving channel.

The incumbent is constructed with the objective alone, so the probed states are
not states the model chose for itself.

Unlike ``probe_field.py`` -- which loads a checkpoint's own checkout through
``--code-root`` and is kept for pre-v14 schemas -- this runs against the working
tree.

    python scripts/probe_feasibility.py \
        --checkpoint pretrained/program-blind-resource-v15/best.pt \
        --variants cvrpbltw,acvrpbltw,mdcvrpbltw \
        --instances 16 --states 12 \
        --out results/probe/feasibility_v15.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import net  # noqa: E402
import prism_decoder  # noqa: E402
from problem_data import DEFAULT_DATASET_DIR, DatasetFinder, load_saved_data  # noqa: E402
from prism_eval.instances import (  # noqa: E402
    _instance_data,
    selected_variants,
    solver_problem,
)

SIGNED_MARGIN = 1  # ResourceTransitionFeature::SIGNED_MARGIN


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so tied scores cannot inflate the AUROC."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < ordered.shape[0]:
        stop = start
        while stop + 1 < ordered.shape[0] and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties at 0.5."""
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


# --------------------------------------------------------------------------- #
# Model and instances
# --------------------------------------------------------------------------- #
def build_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    schema = checkpoint.get("model_schema")
    supported = (net.MODEL_SCHEMA, net.LEGACY_NO_OBJECTIVE_RESIDUAL_SCHEMA)
    if schema not in supported:
        raise SystemExit(
            f"checkpoint schema {schema!r} is not one of {supported!r}; "
            "use probe_field.py --code-root for older schemas"
        )
    config = checkpoint.get("config") or {}
    # Rebuild under the flags the checkpoint was trained with: several reshape
    # heads, and the forward-time ones would silently evaluate an ablation as
    # the full model.
    parameters = inspect.signature(net.ConstraintFieldNet.__init__).parameters
    kwargs = {
        name: config[name]
        for name in parameters
        if name != "self" and name in config
    }
    model = net.ConstraintFieldNet(**kwargs).to(device)
    # The schema must be forwarded: the loader migrates a legacy checkpoint only
    # when told which schema it was written under, and otherwise attempts a
    # strict load against the working tree's.
    net.load_constraint_field_state_dict(
        model,
        checkpoint["model_state_dict"],
        model_schema=schema,
        config=config,
    )
    model.eval()
    return model, checkpoint


def load_variant(variant: str, finder: DatasetFinder, instances: int, size: int):
    paths = finder.get(variant, size)
    data, _ = load_saved_data(
        paths["data_path"],
        variant,
        instances,
        solution_path=paths["solution_path"],
        allow_aggregate_reference=False,
    )
    return data


def build_decoder(problem: dict, args, instance: int):
    decoder = prism_decoder.Decoder(
        problem,
        candidate_config={"max_candidates": args.candidates},
        search_config={
            "use_srr": True,
            "min_changed_edges": 8,
            "feasibility_lookahead_depth": 2,
            "srr_exploration_budget": 0,
        },
        n_rollouts=args.rollouts,
        beta=2.0,
    )
    decoder.seed(args.seed + instance)
    return decoder


def distance_incumbent(decoder):
    """Objective-only construction: no learned guidance reaches the incumbent.

    Every guidance pointer defaults to None, which the decoder reads as a zero
    field with unit objective weight, so this is the field-off control.
    """
    incumbent = decoder.sample_greedy()
    if not incumbent["feasible"]:
        feasible = [item for item in decoder.sample() if item["feasible"]]
        if not feasible:
            return None
        incumbent = min(feasible, key=lambda item: float(item["objective"]))
    decoder.set_incumbent(incumbent["route"])
    return incumbent


def sampled_positions(route: np.ndarray, states: int) -> np.ndarray:
    """Decision positions along the incumbent, excluding start and terminus."""
    positions = np.arange(1, max(len(route) - 1, 1))
    if states and len(positions) > states:
        picked = np.unique(
            np.linspace(0, len(positions) - 1, states).round().astype(int)
        )
        positions = positions[picked]
    return positions


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def probe_instance(model, decoder, variant, instance, args, rows, states):
    incumbent = distance_incumbent(decoder)
    if incumbent is None:
        return

    # The candidate graph is rebuilt around the installed incumbent, so the
    # network must be evaluated on the graph the probe will actually read.
    graph = net.build_decoder_data(decoder, args.device)
    with torch.no_grad():
        output = model(graph)

    metadata = decoder.metadata
    resource_count = int(metadata["resource_count"])
    energy_scale = float(metadata["objective_energy_scale"])
    depot_count = int(metadata["depot_count"])

    names = [
        row["name"]
        for row in decoder.resource_declarations
    ][:resource_count]

    offsets = np.asarray(decoder.edge_offsets, dtype=np.int64)
    edge_index = np.asarray(decoder.edge_index, dtype=np.int64)
    objective_cost = np.asarray(decoder.objective_edge_costs, dtype=np.float64)
    live_states = np.asarray(decoder.incumbent_live_state, dtype=np.float32)

    field = output["residual"].detach().cpu().numpy()
    semantic_margin = output["semantic_margin"].detach().cpu().numpy()
    active = output["active_channels"][0].detach().cpu().numpy()
    margin = graph.resource_transition_features[..., SIGNED_MARGIN]
    margin = margin.detach().cpu().numpy()
    valid = graph.resource_transition_mask.detach().cpu().numpy()

    route = np.asarray(incumbent["route"], dtype=np.int32)
    for state_id, position in enumerate(sampled_positions(route, args.states)):
        prefix = route[: position + 1]
        legal = np.asarray(decoder.mask(prefix), dtype=np.uint8)
        seen = {int(node) for node in prefix[depot_count:]}
        source = int(route[position])

        live = torch.as_tensor(
            live_states[source], dtype=torch.float32, device=args.device
        ).view(1, -1)
        with torch.no_grad():
            multipliers = model.couple(output, live)[0].cpu().numpy()

        lo, hi = int(offsets[source]), int(offsets[source + 1])
        candidates = []
        for edge in range(lo, hi):
            target = int(edge_index[1, edge])
            # Published protocol: unvisited non-depot candidates. The depot is
            # admissible from almost every state, so including it would inflate
            # every selector equally and make the comparison less informative.
            if target < depot_count or target in seen:
                continue
            objective_term = float(
                multipliers[resource_count]
                * (objective_cost[edge] / energy_scale)
            )
            resource_term = float(
                (multipliers[:resource_count] * active * field[edge]).sum()
            )
            candidates.append(
                {
                    "edge": edge,
                    "target": target,
                    "energy": objective_term + resource_term,
                    "energy_flip": objective_term - resource_term,
                    "cost": float(objective_cost[edge]),
                    "admissible": int(legal[target]),
                }
            )
            for channel in range(resource_count):
                if not active[channel] or not valid[edge, channel]:
                    continue
                rows.append(
                    {
                        "variant": variant,
                        "instance": instance,
                        "state": state_id,
                        "edge": edge,
                        "resource": names[channel]
                        if channel < len(names)
                        else f"channel_{channel}",
                        "field": float(field[edge, channel]),
                        "semantic_margin": float(
                            semantic_margin[edge, channel]
                        ),
                        "margin": float(margin[edge, channel]),
                        "row_admissible": int(margin[edge, channel] >= 0.0),
                        "lambda": float(multipliers[channel]),
                        "mask_admissible": int(legal[target]),
                        # Kept so the candidate-level AUROC can be scored
                        # against a distance baseline from the same states.
                        "objective_cost": float(objective_cost[edge]),
                    }
                )

        admissible = sum(item["admissible"] for item in candidates)
        # An informative state has something for the ranking to get wrong.
        if candidates and 0 < admissible < len(candidates):
            best_energy = min(candidates, key=lambda item: item["energy"])
            best_cost = min(candidates, key=lambda item: item["cost"])
            # Energy-sign diagnostic. Semantic margin prediction is deliberately
            # separate and never enters this calculation; flipping here tests
            # only whether the independently trained search field has learned a
            # useful cost polarity.
            best_flipped = min(candidates, key=lambda item: item["energy_flip"])
            states.append(
                {
                    "variant": variant,
                    "instance": instance,
                    "state": state_id,
                    "candidates": len(candidates),
                    "energy_safe": best_energy["admissible"],
                    "flipped_safe": best_flipped["admissible"],
                    "distance_safe": best_cost["admissible"],
                    "uniform_safe": admissible / len(candidates),
                }
            )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def candidate_level_auroc(rows, stream=sys.stdout):
    """Per-state AUROC over CANDIDATES, not (candidate, resource) pairs.

    Pooling every pair into one AUROC mixes states whose admissible fraction
    differs, which washes out a real within-state ranking. The decoder ranks
    candidates within one state, so that is the level to score: per state, take
    each candidate's tightest predicted margin and its summed resource energy,
    and separate the candidates the composed mask rejects from those it admits.
    """
    per_state = defaultdict(dict)
    for row in rows:
        key = (row["variant"], row["instance"], row["state"])
        entry = per_state[key].setdefault(
            row["edge"], {"label": 1 - row["mask_admissible"],
                          "cost": row["objective_cost"],
                          "semantic": [], "energy": []}
        )
        entry["semantic"].append(row["semantic_margin"])
        entry["energy"].append(row["field"] * row["lambda"])

    scores = defaultdict(list)
    by_variant = defaultdict(lambda: defaultdict(list))
    for (variant, _, _), edges in per_state.items():
        labels = np.array([e["label"] for e in edges.values()], dtype=np.int64)
        if labels.sum() in (0, labels.size):
            continue          # not informative: one class only
        semantic = np.array([-min(e["semantic"]) for e in edges.values()])
        energy = np.array([sum(e["energy"]) for e in edges.values()])
        distance = np.array([e["cost"] for e in edges.values()])
        for tag, value in (("semantic", semantic), ("energy", energy),
                           ("distance", distance)):
            score = auroc(value, labels)
            if np.isfinite(score):
                scores[tag].append(score)
                by_variant[variant][tag].append(score)

    if not scores["semantic"]:
        return
    print("\nPer-state candidate-level AUROC (informative states)", file=stream)
    print(f"  informative states : {len(scores['semantic'])}", file=stream)
    for tag in ("semantic", "energy", "distance"):
        values = np.array(scores[tag])
        print(f"  {tag:18s} : {values.mean():.4f}", file=stream)
    for variant in sorted(by_variant):
        row = by_variant[variant]
        print(
            f"    {variant:16s} states={len(row['semantic']):5d} "
            f"semantic={np.mean(row['semantic']):.4f} "
            f"energy={np.mean(row['energy']):.4f}",
            file=stream,
        )


def report(rows, states, stream=sys.stdout):
    by_resource = defaultdict(list)
    for row in rows:
        by_resource[row["resource"]].append(row)

    candidate_level_auroc(rows, stream=stream)

    print("\nPer-resource separation of admissible candidates", file=stream)
    print(
        f"{'resource':26s} {'pairs':>9s} {'admiss%':>8s} "
        f"{'AUROC(energy)':>13s} {'AUROC(semantic)':>15s} {'check':>7s}",
        file=stream,
    )
    print("-" * 91, file=stream)
    summary = []
    for name in sorted(by_resource):
        group = by_resource[name]
        # The search field is a cost, while the separate semantic head predicts
        # signed slack. Report them independently: high energy should identify
        # inadmissibility; high semantic margin should identify admissibility.
        fields = np.array([r["field"] for r in group], dtype=np.float64)
        predicted_margins = np.array(
            [r["semantic_margin"] for r in group], dtype=np.float64
        )
        margins = np.array([r["margin"] for r in group], dtype=np.float64)
        labels = np.array([r["row_admissible"] for r in group], dtype=np.int64)
        a_field = auroc(fields, 1 - labels)
        a_semantic = auroc(predicted_margins, labels)
        # 1.0 by construction: the label is the margin's own sign. This is a
        # wiring check that the margin, mask and field are aligned on the same
        # (edge, resource) index -- not a result.
        a_margin = auroc(margins, labels)
        summary.append(
            (name, len(group), labels.mean(), a_field, a_semantic, a_margin)
        )
        print(
            f"{name:26s} {len(group):9d} {100*labels.mean():7.1f}% "
            f"{a_field:13.3f} {a_semantic:15.3f} {a_margin:7.3f}",
            file=stream,
        )

    if states:
        energy = np.mean([s["energy_safe"] for s in states])
        flipped = np.mean([s["flipped_safe"] for s in states])
        distance = np.mean([s["distance_safe"] for s in states])
        uniform = np.mean([s["uniform_safe"] for s in states])
        print(
            f"\nInformative states: {len(states)}\n"
            f"  energy    top-1 admissible : {100*energy:5.1f}%\n"
            f"  distance  top-1 admissible : {100*distance:5.1f}%\n"
            f"  uniform   expectation      : {100*uniform:5.1f}%\n"
            f"  [diagnostic] resource term negated : {100*flipped:5.1f}%",
            file=stream,
        )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    # The published protocol is the four-resource composition: capacity, time
    # windows, route limit, and backhaul precedence. The `bltw` set omits the
    # `p` row, so it probes three resources and silently reports a different
    # protocol from the one the paper describes.
    parser.add_argument(
        "--variants",
        default="cvrpbpltw,acvrpbpltw,mdcvrpbpltw,amdocvrpbpltw",
    )
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model, checkpoint = build_model(args.checkpoint, args.device)
    print(
        f"checkpoint epoch={checkpoint.get('epoch')} "
        f"val_gap={checkpoint.get('val_gap')} schema={net.MODEL_SCHEMA}",
        flush=True,
    )

    finder = DatasetFinder(args.dataset_dir or DEFAULT_DATASET_DIR)
    variants = (
        selected_variants("all")
        if args.variants == "all"
        else [v.strip() for v in args.variants.split(",") if v.strip()]
    )

    rows, states = [], []
    for variant in variants:
        try:
            data = load_variant(variant, finder, args.instances, args.size)
        except Exception as error:  # noqa: BLE001
            print(f"{variant}: skipped ({type(error).__name__}: {error})", flush=True)
            continue
        before = len(rows)
        for instance in range(args.instances):
            problem = solver_problem(variant, _instance_data(data, instance))
            decoder = build_decoder(problem, args, instance)
            probe_instance(model, decoder, variant, instance, args, rows, states)
        print(f"{variant}: {len(rows)-before} rows", flush=True)

    if not rows:
        raise SystemExit("no rows produced")

    summary = report(rows, states)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}", flush=True)
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
