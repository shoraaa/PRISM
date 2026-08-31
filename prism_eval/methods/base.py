"""What every solver in an evaluation run has to look like.

A method is asked for one variant at a time, not one instance at a time. That
is forced by the batched ones: a foreign NCO checkpoint wants the whole
dataset in a single forward pass, and a subprocess adapter wants to hand its
worker one file. Per-instance solvers get that loop for free from
``InProcessMethod``.

Two rules make an adapter safe to add:

* ``covers`` states up front which problems the method models, so an
  unsupported family is recorded as a reason rather than a wrong number;
* a failure is returned as a status, never raised, so one broken baseline
  cannot discard the measurements of everything else on that variant.

An adapter that can hand back routes rather than only objectives is worth
preferring, because the decoder can then re-score them on PRISM's own scale
and catch a foreign repo whose objective drifts from ours.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from ..instances import InstanceBatch


@dataclass
class MethodRequest:
    """One variant's work order, identical for every method."""

    batch: InstanceBatch
    seed: int
    #: What PRISM measured on this variant, when it ran first; "" otherwise.
    direction: str = ""
    # In-process methods call this immediately after each instance completes so
    # the evaluator can durably append its CSV row before the next solve starts.
    on_result: Callable[[int, float, float, str], None] | None = None

    def instance_seed(self, index: int) -> int:
        return self.seed + index


@dataclass
class MethodResult:
    """What a method reports back for one variant.

    ``objectives`` may be shorter than the batch when a method fails partway;
    the runner records what arrived and reports the status for the rest.
    """

    objectives: list[float] = field(default_factory=list)
    direction: str = ""
    seconds: float = 0.0
    status: str = "ok"
    source: str = "evaluated"
    config: str = ""
    #: Free-form detail a method wants reported but the schema should not fix
    #: (PRISM's net evaluations, the oracle's chosen solver, ...).
    meta: dict = field(default_factory=dict)

    @classmethod
    def unsupported(cls, reason: str, *, config: str = "") -> "MethodResult":
        return cls(status=f"unsupported({reason})", config=config)

    @classmethod
    def failure(cls, error: BaseException | str, *, config: str = "") -> "MethodResult":
        detail = (
            error
            if isinstance(error, str)
            else f"{type(error).__name__}: {error}"
        )
        return cls(status=f"failed({detail})", config=config)


@runtime_checkable
class Method(Protocol):
    """A solver the harness can put on the same instances as PRISM."""

    name: str

    def config(self) -> str:
        """The knobs that decide what this method's numbers mean."""

    def covers(self, problem: dict) -> str | None:
        """None when the problem is modelled, else the reason it is not."""

    def run(self, request: MethodRequest) -> MethodResult:
        """Solve one variant."""


class InProcessMethod:
    """Base for methods that solve one instance at a time in this process.

    Implement ``solve_one``; the loop, the timing and the partial-failure
    handling come from here.
    """

    name = "in-process"

    def config(self) -> str:
        return ""

    def covers(self, problem: dict) -> str | None:
        return None

    #: Set by ``solve_one`` when the method measures the objective's direction
    #: itself; PRISM does, and everything after it is checked against PRISM's.
    _last_direction = ""

    def solve_one(
        self,
        problem: dict,
        initial_route,
        index: int,
        request: MethodRequest,
    ) -> float:
        raise NotImplementedError

    def on_progress(
        self, request: MethodRequest, index: int, objective: float, seconds: float
    ) -> None:
        """Hook for per-instance reporting; overridden where it is useful."""

    def run(self, request: MethodRequest) -> MethodResult:
        result = MethodResult(config=self.config(), direction=request.direction)
        for index in range(len(request.batch)):
            problem, initial_route = request.batch.problem(index)
            reason = self.covers(problem)
            if reason is not None:
                result.status = f"unsupported({reason})"
                return result
            started = time.perf_counter()
            try:
                objective = self.solve_one(problem, initial_route, index, request)
            except Exception as error:  # noqa: BLE001
                result.status = MethodResult.failure(error).status
                return result
            elapsed = time.perf_counter() - started
            result.seconds += elapsed
            result.objectives.append(float(objective))
            result.direction = self._last_direction or result.direction
            if request.on_result is not None:
                request.on_result(
                    index,
                    float(objective),
                    elapsed,
                    self._last_direction or result.direction,
                )
            self.on_progress(request, index, float(objective), elapsed)
        return result


class SubprocessMethod:
    """Base for methods that live in another repository and another venv.

    The pattern every foreign NCO checkpoint needs: point at an interpreter and
    a script, hand it a dataset file, run it, and read one JSON line back. A
    subclass supplies ``command`` and reads its payload; isolation, failure
    reporting and stderr surfacing are handled here.
    """

    name = "subprocess"
    #: stdout lines starting with this prefix carry the JSON payload.
    result_prefix = "RESULT_JSON "

    def config(self) -> str:
        return ""

    def covers(self, problem: dict) -> str | None:
        return None

    def command(self, request: MethodRequest) -> list[str]:
        raise NotImplementedError

    def parse(self, payload: dict, request: MethodRequest) -> MethodResult:
        raise NotImplementedError

    def cwd(self):
        return None

    def run(self, request: MethodRequest) -> MethodResult:
        command = self.command(request)
        started = time.perf_counter()
        completed = subprocess.run(
            command, capture_output=True, text=True, cwd=self.cwd()
        )
        seconds = time.perf_counter() - started
        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.splitlines()[-20:])
            print(
                f"{self.name} exited {completed.returncode} on "
                f"{request.batch.variant}; stderr:\n{tail}",
                file=sys.stderr,
                flush=True,
            )
            return MethodResult.failure(
                f"exit {completed.returncode}", config=self.config()
            )
        payload = None
        for line in completed.stdout.splitlines():
            if line.startswith(self.result_prefix):
                payload = json.loads(line[len(self.result_prefix):])
        if payload is None:
            return MethodResult.failure(
                f"no {self.result_prefix.strip()} line", config=self.config()
            )
        result = self.parse(payload, request)
        result.seconds = result.seconds or seconds
        return result
