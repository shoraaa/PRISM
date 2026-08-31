"""Aggregate result rows into per-method summaries and pairwise comparisons.

Everything here is computed by grouping rows, so it works for any set of
methods without naming them: adding a baseline adds a group, not a branch.
``test.py`` prints its end-of-run report through this module, and
``scripts/summarize.py`` prints the same report from a saved CSV.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

from .results import Row, measured


def gap_percent(direction: str, objective: float, reference: float) -> float:
    """Signed shortfall against a reference, positive meaning worse."""
    denominator = max(abs(reference), 1e-9)
    if direction == "maximize":
        return (reference - objective) / denominator * 100.0
    return (objective - reference) / denominator * 100.0


def better(direction: str, lhs: float, rhs: float) -> bool:
    tolerance = 1e-6 * max(abs(lhs), abs(rhs), 1.0)
    if direction == "maximize":
        return lhs > rhs + tolerance
    return lhs < rhs - tolerance


@dataclass
class VariantSummary:
    """One method's result on one variant, averaged over its instances."""

    variant: str
    split: str
    method: str
    method_config: str
    direction: str
    instances: int
    objective: float | None
    seconds: float
    reference: float | None
    status: str
    source: str

    @property
    def gap_pct(self) -> float | None:
        if self.objective is None or self.reference is None:
            return None
        return gap_percent(self.direction, self.objective, self.reference)


def variant_summaries(
    rows: Iterable[Row],
) -> dict[tuple[str, str], VariantSummary]:
    """Collapse per-instance rows into one summary per (variant, method)."""
    grouped: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        grouped.setdefault((row.variant, row.method), []).append(row)

    summaries: dict[tuple[str, str], VariantSummary] = {}
    for key, group in grouped.items():
        values = [row.objective for row in measured(group)]
        head = group[0]
        reference = next(
            (
                float(row.reference)
                for row in group
                if isinstance(row.reference, (int, float))
            ),
            None,
        )
        # A variant-level status row (no objective) reports the reason; when
        # instances were measured the status is whatever they carry.
        status = next(
            (row.status for row in group if row.status != "ok"), "ok"
        ) if not values else next(
            (row.status for row in measured(group)), "ok"
        )
        summaries[key] = VariantSummary(
            variant=head.variant,
            split=head.split,
            method=head.method,
            method_config=head.method_config,
            direction=next(
                (row.direction for row in group if row.direction), ""
            ),
            instances=len(values),
            objective=statistics.fmean(values) if values else None,
            seconds=sum(
                float(row.seconds)
                for row in group
                if isinstance(row.seconds, (int, float))
            ),
            reference=reference,
            status=status,
            source=head.source,
        )
    return summaries


def method_gaps(
    rows: Iterable[Row], method: str = "prism"
) -> dict[str, float]:
    """{variant: gap against the reference} for one method.

    Variants without a reference are omitted, which is what every report over
    oracle gaps wants.
    """
    return {
        variant: entry.gap_pct
        for (variant, name), entry in variant_summaries(rows).items()
        if name == method and entry.gap_pct is not None
    }


def methods(rows: Iterable[Row], base: str = "prism") -> list[str]:
    """Method names in a stable order, with ``base`` first when present."""
    seen: list[str] = []
    for row in rows:
        if row.method not in seen:
            seen.append(row.method)
    if base in seen:
        seen.remove(base)
        return [base, *seen]
    return seen


@dataclass
class Comparison:
    """A paired comparison of two methods over the variants both covered."""

    base: str
    method: str
    variants: int
    base_wins: int
    ties: int
    method_wins: int
    instances: int
    instance_base_wins: int
    instance_ties: int
    instance_method_wins: int
    improvements: list[float]

    @property
    def improvement_mean(self) -> float | None:
        return statistics.fmean(self.improvements) if self.improvements else None

    @property
    def improvement_median(self) -> float | None:
        return statistics.median(self.improvements) if self.improvements else None


def compare(rows: Sequence[Row], base: str, method: str) -> Comparison:
    """Compare ``base`` with ``method`` on variants and on paired instances."""
    summaries = variant_summaries(rows)
    per_instance: dict[tuple[str, str, int], float] = {}
    for row in measured(rows):
        if row.method in (base, method) and isinstance(row.instance, int):
            per_instance[(row.variant, row.method, row.instance)] = float(
                row.objective
            )

    variants = sorted(
        {
            variant
            for variant, name in summaries
            if name == method
            and summaries[(variant, method)].objective is not None
            and (variant, base) in summaries
            and summaries[(variant, base)].objective is not None
        }
    )
    result = Comparison(base, method, 0, 0, 0, 0, 0, 0, 0, 0, [])
    for variant in variants:
        base_summary = summaries[(variant, base)]
        other = summaries[(variant, method)]
        direction = base_summary.direction or other.direction
        if direction != other.direction and other.direction:
            # Methods that disagree on the objective's direction are not
            # measuring the same thing; leave the pair out rather than
            # inventing a winner.
            continue
        result.variants += 1
        if better(direction, base_summary.objective, other.objective):
            result.base_wins += 1
        elif better(direction, other.objective, base_summary.objective):
            result.method_wins += 1
        else:
            result.ties += 1
        result.improvements.append(
            -gap_percent(direction, base_summary.objective, other.objective)
        )
        index = 0
        while (variant, base, index) in per_instance and (
            variant,
            method,
            index,
        ) in per_instance:
            base_value = per_instance[(variant, base, index)]
            other_value = per_instance[(variant, method, index)]
            result.instances += 1
            if better(direction, base_value, other_value):
                result.instance_base_wins += 1
            elif better(direction, other_value, base_value):
                result.instance_method_wins += 1
            else:
                result.instance_ties += 1
            index += 1
    return result


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}%"


def method_report(rows: Sequence[Row], method: str) -> dict:
    summaries = [
        summary
        for (_variant, name), summary in variant_summaries(rows).items()
        if name == method
    ]
    gaps = [
        summary.gap_pct for summary in summaries if summary.gap_pct is not None
    ]
    # A method can run under more than one configuration across a report --
    # the oracle picks a solver per variant, and concatenated runs may differ --
    # so name them all rather than implying the first one held throughout.
    configs = []
    for entry in summaries:
        if entry.method_config and entry.method_config not in configs:
            configs.append(entry.method_config)
    return {
        "method": method,
        "config": (
            " | ".join(configs[:3]) + (" | ..." if len(configs) > 3 else "")
        ),
        "variants": len(summaries),
        "evaluated": sum(summary.objective is not None for summary in summaries),
        "unsupported": sum(
            summary.status.startswith("unsupported") for summary in summaries
        ),
        "failed": sum(
            summary.status.startswith("failed") for summary in summaries
        ),
        "instances": sum(summary.instances for summary in summaries),
        "seconds": sum(summary.seconds for summary in summaries),
        "cached": sum(summary.source == "cached" for summary in summaries),
        "gaps": gaps,
    }


def split_report(rows: Sequence[Row], method: str, split: str, base: str) -> dict:
    split_rows = [row for row in rows if row.split == split]
    report = method_report(split_rows, method)
    report["split"] = split
    if method != base:
        report["comparison"] = compare(split_rows, base, method)
    return report


def print_report(
    rows: Sequence[Row],
    *,
    base: str = "prism",
    splits: Sequence[str] = ("seen", "heldout"),
    stream=None,
) -> None:
    """Print the end-of-run report: per method, per comparison, per split."""
    stream = sys.stdout if stream is None else stream

    def emit(*fields) -> None:
        print(*fields, file=stream, flush=True)

    for method in methods(rows, base):
        report = method_report(rows, method)
        emit(
            "METHOD",
            f"method={method}",
            f"config=[{report['config']}]",
            f"variants={report['variants']}",
            f"evaluated={report['evaluated']}",
            f"unsupported={report['unsupported']}",
            f"failed={report['failed']}",
            f"instances={report['instances']}",
            f"cached_variants={report['cached']}",
            f"seconds={report['seconds']:.3f}",
        )
        if report["gaps"]:
            emit(
                "REFERENCE_GAPS",
                f"method={method}",
                f"n={len(report['gaps'])}",
                f"mean={_percent(statistics.fmean(report['gaps']))}",
                f"median={_percent(statistics.median(report['gaps']))}",
            )

    for method in methods(rows, base):
        if method == base:
            continue
        comparison = compare(rows, base, method)
        if not comparison.variants:
            continue
        emit(
            "COMPARE",
            f"base={base}",
            f"method={method}",
            f"variants={comparison.variants}",
            f"base_wins={comparison.base_wins}",
            f"ties={comparison.ties}",
            f"method_wins={comparison.method_wins}",
            f"instances={comparison.instances}",
            f"instance_base_wins={comparison.instance_base_wins}",
            f"instance_ties={comparison.instance_ties}",
            f"instance_method_wins={comparison.instance_method_wins}",
            f"base_improvement_mean={_percent(comparison.improvement_mean)}",
            f"base_improvement_median={_percent(comparison.improvement_median)}",
        )

    for split in splits:
        for method in methods(rows, base):
            report = split_report(rows, method, split, base)
            if not report["variants"]:
                continue
            fields = [
                "SPLIT",
                f"split={split.upper()}",
                f"method={method}",
                f"variants={report['variants']}",
                f"evaluated={report['evaluated']}",
                f"failed={report['failed']}",
            ]
            comparison = report.get("comparison")
            if comparison is not None and comparison.variants:
                fields.extend(
                    (
                        f"base_wins={comparison.base_wins}",
                        f"ties={comparison.ties}",
                        f"method_wins={comparison.method_wins}",
                        "base_improvement_mean="
                        f"{_percent(comparison.improvement_mean)}",
                    )
                )
            if report["gaps"]:
                fields.append(
                    f"gap_mean={_percent(statistics.fmean(report['gaps']))}"
                )
            emit(*fields)

    for (variant, method), summary in sorted(variant_summaries(rows).items()):
        if summary.objective is None:
            emit("BLANK", f"variant={variant}", f"method={method}",
                 f"status={summary.status}")
