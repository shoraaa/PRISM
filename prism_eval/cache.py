"""Durable per-variant result storage, shared by every method that needs it.

A method that costs real wall clock -- an external solver under a time limit, a
foreign checkpoint behind a subprocess -- should never be asked to recompute a
result the run already paid for. Storage and streaming live here; *identity*
stays with the method, which knows which of its knobs change its numbers, so
each adapter supplies its own key and no method can invalidate another's
entries.
"""

from __future__ import annotations

import json
from pathlib import Path


def default_cache_path(method: str) -> Path:
    """Where a method's cache lives unless the caller says otherwise.

    One file per method, so refreshing one never discards another's entries.
    """
    return Path(__file__).resolve().parent.parent / "results" / f"{method}_cache.json"


class ResultCache:
    """A JSON file of ``key -> payload``, written after every update.

    Writing on every update is deliberate: a variant interrupted halfway keeps
    the instances it already solved, and the file can be tailed while a long
    run is in flight. Unknown keys are preserved, so several methods (or
    several runs) can share one file.
    """

    def __init__(self, path: Path | None, *, refresh: bool = False):
        self.path = path
        self.refresh = refresh
        self._entries: dict = {} if refresh else self._load(path)

    @staticmethod
    def _load(path: Path | None) -> dict:
        if path is None or not path.exists():
            return {}
        try:
            with path.open() as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            # A truncated cache is a performance loss, never a run failure.
            return {}

    def get(self, key: str) -> dict | None:
        if self.refresh:
            return None
        entry = self._entries.get(key)
        return entry if isinstance(entry, dict) else None

    def put(self, key: str, payload: dict) -> None:
        self._entries[key] = payload
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w") as handle:
            json.dump(self._entries, handle)
        temporary.replace(self.path)
