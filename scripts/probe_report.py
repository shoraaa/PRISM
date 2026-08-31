#!/usr/bin/env python3
"""Analyses over a ``probe_field.py`` dump.

Each section corresponds to an experiment in ``experiments.md``:

E4a  Is the learned energy feasibility-aware, or does the hard mask carry all of
     it? AUROC of ``-energy`` against one-step feasibility, scored only on
     candidates the mask could plausibly have accepted -- unvisited, non-depot
     targets -- so the trivial "already served" rejections do not inflate it.
     Distance is the null model: the field only knows something distance does
     not when it beats that column.

E4c  How much of that is information the network adds, rather than information
     it was handed? Logistic regression on (distance) vs (distance + the
     resource-pressure features the encoder already receives) vs (+ energy),
     reported as incremental AUROC.

E4f  Do the state-dependent intensities stay in a "value" regime, or do they
     behave like diverging Lagrange multipliers? Distribution of lambda_r(s)
     over states, per variant and per resource.

E7b  Intrinsic heatmap quality: does the energy rank the incumbent's successor
     top-1 among the feasible candidates leaving that node? Only meaningful on
     a dump built with ``--incumbent distance``. Against a field-constructed
     incumbent the comparison is circular -- the energy is being scored against
     a solution it produced -- and the report labels it as such.

Usage:
    python scripts/probe_report.py results/probe/v6_e4.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def decidable(frame: pd.DataFrame) -> pd.DataFrame:
    """Candidates whose feasibility is a constraint question, not bookkeeping."""
    return frame[(frame.visited == 0) & (frame.is_depot == 0)]


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def e4a(frame: pd.DataFrame) -> pd.DataFrame:
    """Feasibility AUROC of the energy against the distance null model."""
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        sub = decidable(group)
        if sub.empty:
            continue
        labels = sub.mask_feasible.values
        rows.append(
            {
                "variant": variant,
                "R": int(sub.n_active_resources.iloc[0]),
                "n": len(sub),
                "feas_rate": labels.mean(),
                "auc_energy": safe_auc(labels, -sub.energy.values),
                "auc_distance": safe_auc(labels, -sub.distance.values),
                "auc_resource": safe_auc(labels, -sub.energy_resource.values),
                "auc_risk_head": safe_auc(labels, -sub.feasibility_logit.values),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # The quantity of interest: what the energy adds over distance alone.
    out["lift_energy"] = out.auc_energy - out.auc_distance
    out["lift_resource"] = out.auc_resource - out.auc_distance
    return out.sort_values("R")


def e4c(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Incremental AUROC of the energy over the features the encoder is given."""
    pressure_columns = [
        column for column in frame.columns if column.startswith("pressure_")
    ]
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        sub = decidable(group)
        labels = sub.mask_feasible.values
        if len(sub) < 50 or len(np.unique(labels)) < 2:
            continue
        # Only channels this variant actually activates carry signal; the rest
        # are structurally zero and would just add noise columns.
        active = [
            column
            for column in pressure_columns
            if sub[column].std() > 1e-9
        ]
        blocks = {
            "distance": sub[["distance"]].values,
            "distance+pressure": sub[["distance"] + active].values,
            "distance+pressure+energy": sub[
                ["distance"] + active + ["energy"]
            ].values,
        }
        scores = {}
        for name, matrix in blocks.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=seed),
            )
            folds = min(5, int(min(np.bincount(labels.astype(int)))))
            if folds < 2:
                scores[name] = float("nan")
                continue
            predicted = cross_val_predict(
                model, matrix, labels, cv=folds, method="predict_proba"
            )[:, 1]
            scores[name] = safe_auc(labels, predicted)
        rows.append({"variant": variant, "n": len(sub), **scores})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["incremental"] = (
        out["distance+pressure+energy"] - out["distance+pressure"]
    )
    return out


def e4f(frame: pd.DataFrame) -> pd.DataFrame:
    """Magnitude and spread of the state-dependent intensities."""
    rows = []
    lambda_columns = [
        column for column in frame.columns if column.startswith("lambda_")
    ]
    for variant, group in frame.groupby("variant", sort=False):
        for column in lambda_columns:
            channel = column.split("_")[1]
            # Under a variable-width registry the padded columns are blank, so
            # an absent channel reads as NaN rather than 0. NaN < 0.5 is False,
            # which would let it through as if it were active.
            flag = group[f"active_{channel}"].max()
            if not (flag >= 0.5):
                continue
            # One value per decision state, not per candidate edge: lambda is a
            # function of state alone, so per-edge rows would weight states by
            # their candidate count.
            per_state = group.groupby(["instance", "state"])[column].first()
            rows.append(
                {
                    "variant": variant,
                    "channel": int(channel),
                    "states": len(per_state),
                    "mean": per_state.mean(),
                    "std": per_state.std(),
                    "min": per_state.min(),
                    "p95": per_state.quantile(0.95),
                    "max": per_state.max(),
                }
            )
    return pd.DataFrame(rows)


def e7b(frame: pd.DataFrame) -> pd.DataFrame:
    """Does the energy rank the incumbent's successor first?

    Circular unless the incumbent came from the zero-field control; the caller
    is responsible for saying which dump this is.
    """
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        hits = total = 0
        for _, state in group.groupby(["instance", "state"]):
            feasible = state[state.mask_feasible == 1]
            if feasible.in_incumbent.sum() != 1:
                continue
            total += 1
            hits += int(
                feasible.loc[feasible.energy.idxmin(), "in_incumbent"] == 1
            )
        if total:
            rows.append(
                {"variant": variant, "states": total, "top1": hits / total}
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--skip-e4c", action="store_true",
                        help="skip the logistic-regression section (slowest)")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.csv)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    print(f"rows={len(frame)}  variants={frame.variant.nunique()}\n")

    print("=" * 78)
    print("E4a  feasibility AUROC (unvisited non-depot candidates only)")
    print("     lift_* > 0 means the learned signal beats the distance null")
    print("=" * 78)
    print(e4a(frame).to_string(index=False, float_format="%.3f"))

    if not args.skip_e4c:
        print("\n" + "=" * 78)
        print("E4c  incremental AUROC of the energy over encoder-visible features")
        print("=" * 78)
        table = e4c(frame)
        if table.empty:
            print("(insufficient label variation)")
        else:
            print(table.to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 78)
    print("E4f  state-dependent intensities lambda_r(s), per decision state")
    print("=" * 78)
    print(e4f(frame).to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 78)
    print("E7b  top-1 agreement between energy and the incumbent's successor")
    print("     circular unless the dump used --incumbent distance")
    print("=" * 78)
    print(e7b(frame).to_string(index=False, float_format="%.3f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
