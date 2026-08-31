"""The evaluation loop: every method, same instances, one row schema.

The loop knows nothing about any particular solver. It resolves a variant's
instances, runs PRISM first (so the objective's direction is established), then
each remaining method on exactly those instances, and turns whatever comes back
into rows. A method that fails or does not model a family is recorded with its
reason and the run continues -- one baseline can never cost another its
measurements.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, Sequence

from .instances import GENERATED_VARIANTS, InstanceBatch, build_instances, variant_split
from .methods.base import Method, MethodRequest, MethodResult
from .results import Row, RowWriter


def _row_key(row: Row) -> tuple[str, str, int | str]:
    return row.variant, row.method, row.instance


def _compatible_config(method: Method, recorded: str) -> bool:
    expected = method.config()
    if method.name == "oracle":
        # The selected solver is discovered from the problem and replaces
        # ``auto`` after the first solve; the remaining oracle knobs must match.
        recorded = ",".join(
            part for part in recorded.split(",") if not part.startswith("solver=")
        )
        expected = ",".join(
            part for part in expected.split(",") if not part.startswith("solver=")
        )
    return recorded == expected


def _resumed_result(
    prior: list[Row], batch: InstanceBatch, method: Method, seed: int
) -> MethodResult | None:
    measured = [
        row
        for row in prior
        if row.variant == batch.variant
        and row.method == method.name
        and isinstance(row.instance, int)
        and isinstance(row.objective, (int, float))
        and row.status == "ok"
    ]
    if not measured:
        return None
    measured.sort(key=lambda row: int(row.instance))
    if [row.instance for row in measured] != list(range(len(measured))):
        raise ValueError(
            f"resume rows for {batch.variant}/{method.name} are not a contiguous prefix"
        )
    head = measured[0]
    if any(row.n != batch.n or row.seed != seed for row in measured):
        raise ValueError(
            f"resume rows for {batch.variant}/{method.name} do not match "
            f"n={batch.n}, seed={seed}"
        )
    if any(
        row.direction != head.direction or row.method_config != head.method_config
        for row in measured
    ):
        raise ValueError(
            f"resume rows for {batch.variant}/{method.name} mix directions or configs"
        )
    if not _compatible_config(method, head.method_config):
        raise ValueError(
            f"resume config mismatch for {batch.variant}/{method.name}: "
            f"CSV has {head.method_config!r}, current run has {method.config()!r}"
        )
    if len(measured) != len(batch):
        return None
    return MethodResult(
        objectives=[float(row.objective) for row in measured],
        direction=head.direction,
        seconds=sum(float(row.seconds) for row in measured),
        source="resumed",
        config=head.method_config,
    )


def _rows_for(
    batch: InstanceBatch | None,
    variant: str,
    method: str,
    result: MethodResult,
    seed: int,
) -> list[Row]:
    """Turn one method's report into rows: one per instance, or one status row."""
    split = variant_split(variant)
    reference = None if batch is None else batch.reference
    common = dict(
        variant=variant,
        split=split,
        n="" if batch is None else batch.n,
        seed=seed,
        method=method,
        method_config=result.config,
        status=result.status,
        source=result.source,
        direction=result.direction,
        reference="" if reference is None else float(reference),
    )
    if not result.objectives:
        return [
            Row(instance="", objective="", feasible="", seconds="", **common)
        ]
    # A batched method measures one wall clock for the whole variant; spreading
    # it evenly keeps a sum over rows equal to the real total.
    per_instance = result.seconds / len(result.objectives)
    return [
        Row(
            instance=index,
            objective=float(objective),
            feasible=1,
            seconds=per_instance,
            **common,
        )
        for index, objective in enumerate(result.objectives)
    ]


def run(
    methods: Sequence[Method],
    variants: Sequence[str],
    args: argparse.Namespace,
    finder,
    writer: RowWriter,
    *,
    reference_hook: Callable[[str, InstanceBatch], float | None] | None = None,
    cached: dict | None = None,
    resume_rows: list[Row] | None = None,
) -> list[Row]:
    """Evaluate every method on every variant, streaming rows as they land."""
    rows: list[Row] = list(resume_rows or [])
    emitted = {_row_key(row) for row in rows}
    cached = cached or {}
    cached_hits = 0

    def emit(new_rows: list[Row]) -> None:
        for row in new_rows:
            key = _row_key(row)
            if key in emitted:
                continue
            emitted.add(key)
            rows.append(row)
            writer.write(row)

    for index, name in enumerate(variants):
        started = time.perf_counter()
        batch: InstanceBatch | None = None
        try:
            batch = build_instances(name, index, args, finder)
            if (
                reference_hook is not None
                and name in GENERATED_VARIANTS
                and batch.reference is None
            ):
                batch.reference = reference_hook(name, batch)

            request = MethodRequest(
                batch=batch, seed=args.seed + index * len(batch)
            )
            reports: list[str] = []
            for method in methods:
                resumed = _resumed_result(rows, batch, method, args.seed)
                entry = cached.get((name, method.name))
                if resumed is not None:
                    result = resumed
                elif (
                    entry is not None
                    and len(entry["objectives"]) >= len(batch)
                    and _compatible_config(method, entry.get("config", ""))
                ):
                    result = MethodResult(
                        objectives=[
                            float(value)
                            for value in entry["objectives"][: len(batch)]
                        ],
                        direction=entry["direction"],
                        source="cached",
                        config=entry.get("config", ""),
                    )
                    cached_hits += 1
                else:
                    def persist_instance(
                        instance: int,
                        objective: float,
                        seconds: float,
                        direction: str,
                        *,
                        current_method=method,
                    ) -> None:
                        if not request.direction and direction:
                            request.direction = direction
                        emit(
                            [
                                Row(
                                    variant=name,
                                    split=variant_split(name),
                                    n=batch.n,
                                    seed=args.seed,
                                    method=current_method.name,
                                    method_config=current_method.config(),
                                    instance=instance,
                                    objective=objective,
                                    feasible=1,
                                    seconds=seconds,
                                    status="ok",
                                    source="evaluated",
                                    direction=direction or request.direction,
                                    reference=""
                                    if batch.reference is None
                                    else float(batch.reference),
                                )
                            ]
                        )

                    request.on_result = persist_instance
                    try:
                        result = method.run(request)
                    finally:
                        request.on_result = None

                # PRISM runs first and establishes the objective's direction;
                # a method that reports a different one is not measuring the
                # same problem, so its numbers are withheld rather than mixed in.
                if not request.direction and result.direction:
                    request.direction = result.direction
                elif (
                    result.objectives
                    and result.direction
                    and result.direction != request.direction
                ):
                    result = MethodResult.failure(
                        f"direction={result.direction!r} != "
                        f"{request.direction!r}",
                        config=result.config,
                    )
                if result.objectives and not result.direction:
                    result.direction = request.direction

                emit(_rows_for(batch, name, method.name, result, args.seed))
                if result.objectives and result.status.startswith("failed"):
                    emit(
                        _rows_for(
                            batch,
                            name,
                            method.name,
                            MethodResult(status=result.status, config=result.config),
                            args.seed,
                        )
                    )
                elif result.status == "ok" and len(result.objectives) >= len(batch):
                    # Keep old crash/failure markers in the append-only CSV as
                    # provenance, but a now-complete resumed batch supersedes
                    # them for this run's summary and exit status.
                    rows[:] = [
                        row
                        for row in rows
                        if not (
                            row.variant == name
                            and row.method == method.name
                            and row.instance == ""
                            and row.status.startswith("failed")
                        )
                    ]
                if result.objectives:
                    label = result.meta.get("solver")
                    reports.append(
                        f"{method.name}["
                        + (f"{label}/" if label else "")
                        + f"{result.source}](objective="
                        f"{statistics.fmean(result.objectives):.6g})"
                    )
                else:
                    reports.append(f"{method.name}[{result.status}](objective=n/a)")

            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                f"results=[{'; '.join(reports)}]",
                f"seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            # Only instance construction can land here; a method's own failure
            # is a status, not an exception.
            detail = f"{type(error).__name__}: {error}"
            measured = {
                row.method
                for row in rows
                if row.variant == name
                and isinstance(row.objective, (int, float))
            }
            for method in methods:
                if method.name in measured:
                    continue
                emit(
                    _rows_for(
                        batch,
                        name,
                        method.name,
                        MethodResult.failure(detail, config=method.config()),
                        args.seed,
                    )
                )
            print(
                "TEST",
                f"{index + 1}/{len(variants)}",
                f"variant={name}",
                "status=FAIL",
                f"split={variant_split(name).upper()}",
                f"error={detail}",
                f"seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )

    if cached_hits:
        print("CACHED_ROWS", f"reused={cached_hits}", flush=True)
    return rows
