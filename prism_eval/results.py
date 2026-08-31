"""The one row every method writes, and the CSV it streams into.

A measurement is (instance, method) -> objective. Encoding the method in the
*value* of a ``method`` column rather than in column names is what keeps the
schema fixed as methods are added: PRISM, the native controls, an external
solver like PyVRP/OR-Tools and a foreign NCO checkpoint all produce the same
row, comparing any two of them is a group-by, and a new method never widens
the file or touches a reader.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Row:
    """One measurement of one method on one instance.

    ``instance`` is the index within the variant's batch, or ``""`` for a
    variant-level row that records why a method produced no measurement at all
    (``status`` then carries the reason and ``objective`` is blank).

    ``seconds`` is per row. Methods that solve a whole variant in one batched
    call (a foreign NCO checkpoint, say) spread their measured wall clock
    evenly over the instances, so a sum over rows still reproduces the real
    total. ``reference`` is the variant's reference objective repeated on every
    row, because the saved datasets carry one mean reference per variant rather
    than one per instance.
    """

    variant: str
    split: str
    n: int | str
    seed: int
    method: str
    method_config: str
    instance: int | str
    objective: float | str
    feasible: int | str
    seconds: float | str
    status: str
    source: str
    direction: str
    reference: float | str


FIELDS: tuple[str, ...] = tuple(field.name for field in fields(Row))

_NUMERIC = {"objective", "seconds", "reference"}
_INTEGER = {"n", "seed", "instance", "feasible"}


class RowWriter:
    """Stream rows to CSV so a partial run survives an interrupt."""

    def __init__(self, path: Path | None, *, append: bool = False):
        self.path = path
        self._stream = None
        self._writer: csv.DictWriter | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if append and path.is_file():
                # A hard kill can interrupt one final writerow syscall. Every
                # schema field is single-line, so only an unterminated tail can
                # be partial; discard that tail before appending new rows.
                with path.open("rb+") as raw:
                    content = raw.read()
                    if content and not content.endswith(b"\n"):
                        boundary = content.rfind(b"\n")
                        raw.seek(0)
                        raw.truncate(boundary + 1 if boundary >= 0 else 0)
                with path.open(newline="") as existing:
                    header = next(csv.reader(existing), [])
                if header != list(FIELDS):
                    raise ValueError(
                        f"{path} cannot be resumed: CSV header does not match "
                        "the current result schema"
                    )
            mode = "a" if append and path.is_file() else "w"
            self._stream = path.open(mode, newline="")
            self._writer = csv.DictWriter(self._stream, fieldnames=list(FIELDS))
            if mode == "w":
                self._writer.writeheader()
                self._sync()

    def _sync(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def write(self, row: Row) -> None:
        if self._writer is None or self._stream is None:
            return
        self._writer.writerow(asdict(row))
        self._sync()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "RowWriter":
        return self

    def __exit__(self, *_exception) -> None:
        self.close()


def _coerce(name: str, value: str):
    text = value.strip()
    if not text:
        return ""
    if name in _NUMERIC:
        return float(text)
    if name in _INTEGER:
        return int(float(text))
    return text


def read_rows(path: Path | str) -> list[Row]:
    """Read a results CSV back into rows, restoring numeric types.

    Files written before the tidy schema are converted on the fly, so results
    already on disk stay readable. A converted file carries variant-level
    means only (its per-instance objectives were never written), so summaries
    and gaps work while instance-paired comparisons do not.
    """
    path = Path(path)
    with path.open("rb") as raw:
        raw.seek(0, 2)
        size = raw.tell()
        if size:
            raw.seek(-1, 2)
            unterminated_tail = raw.read(1) != b"\n"
        else:
            unterminated_tail = False
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        missing = set(FIELDS) - columns
        if not missing:
            records = list(reader)
            restored = []
            for index, row in enumerate(records):
                if unterminated_tail and index == len(records) - 1:
                    continue
                if any(row.get(name) is None for name in FIELDS):
                    if index == len(records) - 1:
                        # Resume can safely discard an interrupted final row;
                        # RowWriter truncates it before appending.
                        continue
                    raise ValueError(f"{path}: malformed CSV row {index + 2}")
                restored.append(
                    Row(**{name: _coerce(name, row[name]) for name in FIELDS})
                )
            return restored
        if "neural_objective" not in columns:
            raise ValueError(
                f"{path} is not a PRISM results CSV; missing columns: "
                + ", ".join(sorted(missing))
            )
        return _convert_legacy(list(reader))


def _convert_legacy(records: list[dict]) -> list[Row]:
    """Convert the pre-tidy wide CSV (one row per variant x baseline)."""

    def number(record: dict, key: str):
        text = (record.get(key) or "").strip()
        return float(text) if text else ""

    rows: list[Row] = []
    seen_prism: set[str] = set()
    for record in records:
        variant = (record.get("variant") or "").strip()
        direction = (record.get("direction") or "").strip()
        reference = number(record, "reference")
        common = dict(
            variant=variant,
            split=(record.get("split") or "").strip(),
            n="",
            seed=0,
            instance="",
            feasible="",
            direction=direction,
            reference=reference,
        )
        if variant not in seen_prism:
            seen_prism.add(variant)
            rows.append(
                Row(
                    method="prism",
                    method_config=(record.get("field_mode") or "").strip(),
                    objective=number(record, "neural_objective"),
                    seconds=number(record, "neural_seconds"),
                    status="ok",
                    source="evaluated",
                    **common,
                )
            )
        baseline = (record.get("baseline") or "").strip()
        if baseline and baseline != "none":
            objective = number(record, "baseline_objective")
            rows.append(
                Row(
                    method=baseline,
                    method_config="",
                    objective=objective,
                    seconds=number(record, "baseline_seconds"),
                    status="ok" if objective != "" else "failed(legacy_blank)",
                    source=(record.get("baseline_source") or "").strip()
                    or "evaluated",
                    **common,
                )
            )
        urs_objective = number(record, "urs_objective")
        if urs_objective != "" and baseline != "urs":
            rows.append(
                Row(
                    method="urs",
                    method_config="",
                    objective=urs_objective,
                    seconds="",
                    status="ok",
                    source="evaluated",
                    **common,
                )
            )
    return rows


def load_cached_rows(
    path: Path,
) -> tuple[dict[tuple[str, str], dict], tuple[str, ...]]:
    """Reuse a prior run's per-instance baseline objectives.

    Reads the tidy results CSV, so reuse needs no bespoke format: group the
    rows of a method on a variant, keep them in instance order, and hand them
    back. PRISM's own rows are never reused -- the point of a run is to measure
    the checkpoint in front of it.
    """
    if not path.is_file():
        raise FileNotFoundError(f"cached results CSV does not exist: {path}")
    cached: dict[tuple[str, str], dict] = {}
    order: list[str] = []
    for row in read_rows(path):
        if row.method == "prism" or not isinstance(row.objective, (int, float)):
            continue
        if not isinstance(row.instance, int):
            continue
        if row.direction not in {"minimize", "maximize"}:
            raise ValueError(
                f"{path}: {row.method} on {row.variant} has no objective "
                "direction"
            )
        entry = cached.setdefault(
            (row.variant, row.method),
            {
                "objectives": [],
                "instances": [],
                "direction": row.direction,
                "config": row.method_config,
            },
        )
        if entry["direction"] != row.direction:
            raise ValueError(
                f"{path}: {row.method} on {row.variant} mixes objective "
                "directions"
            )
        entry["instances"].append(row.instance)
        entry["objectives"].append(float(row.objective))
        if row.method not in order:
            order.append(row.method)
    for (variant, method), entry in cached.items():
        if sorted(entry["instances"]) != list(range(len(entry["instances"]))):
            raise ValueError(
                f"{path}: {method} on {variant} has gaps or duplicates in its "
                "instance indices"
            )
        paired = sorted(zip(entry["instances"], entry["objectives"]))
        entry["objectives"] = [objective for _index, objective in paired]
    return cached, tuple(order)


def measured(rows: Iterable[Row]) -> Iterator[Row]:
    """Rows that carry an actual objective, skipping status-only rows."""
    for row in rows:
        if isinstance(row.objective, (int, float)):
            yield row
