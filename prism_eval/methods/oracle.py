"""The classical-solver oracle: PyVRP where it models the problem, OR-Tools elsewhere.

PyVRP covers the capacitated multi-route grid natively and is the stronger
solver there; OR-Tools covers everything else PRISM benchmarks -- single tours,
pickup-delivery precedence, prize objectives, and the resource-algebra probes.
Which one ran is reported per variant rather than assumed, because the two do
not measure the same way: PyVRP reports its own objective, while an OR-Tools
route is scored by PRISM's decoder.

Both are anytime heuristics, so ``--oracle-time-limit`` sets how strong an
oracle this is. Results stream into a cache after every instance, which makes
an interrupted variant resumable and keeps a wall-clock-bound run reproducible.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

import prism_decoder

from ..cache import ResultCache, default_cache_path
from .base import MethodRequest, MethodResult


def _pyvrp_unsupported(problem: dict) -> str | None:
    """Name the feature that puts ``problem`` outside PyVRP's model, if any.

    PyVRP covers the capacitated multi-route families -- capacity, backhaul
    (mixed and ordered), route limits, time windows, open routes, multiple
    depots, and asymmetric distances -- which is exactly PRISM's cvrp* grid.
    Everything else in the 110 benchmarks (prize objectives, tour limits,
    pickup-delivery pairing, single-tour TSPs) and every resource-algebra probe
    (EVRP battery, VRPDB driving time) has no PyVRP equivalent and is solved by
    the OR-Tools oracle instead; see ``oracle_solver_for``.
    """
    constraints = set(problem.get("constraints", ()))
    outside = constraints & {"pickup_delivery", "tour_limit", "prize_quota"}
    if outside:
        return "constraints=" + ",".join(sorted(outside))
    if problem.get("objective") != "distance":
        return f"objective={problem.get('objective')}"
    if not problem.get("multi_route"):
        return "single_route"
    if int(problem.get("depot_count", 0)) < 1:
        return "no_depot"
    if problem.get("resources"):
        return "resource_algebra"
    return None


def _pyvrp_problem(problem: dict):
    """Build a PyVRP ``ProblemData`` from one PRISM decoder problem.

    Reads PRISM's schema directly: a single signed demand vector (backhaul
    negative), depot-inclusive time windows, and capacity normalised to
    ``problem["capacity"]``.
    Distances are scaled to integers, so the returned scale converts PyVRP's
    integral cost back onto PRISM's objective scale.

    Backhaul caveat: PyVRP's VRPB model loads the vehicle with the route's own
    delivery total, whereas PRISM (following URS) departs at full capacity, so
    PyVRP additionally admits prefixes whose pickups exceed their deliveries.
    On the b/bp families its objective is therefore a bound on a slightly
    relaxed problem rather than one PRISM could always attain. This is the same
    model CCL-MTLVRP's own PyVRP reference uses, which is what the ``optimal``
    column of the converted CCL datasets holds, so the two stay comparable.
    """
    import numpy as np
    from pyvrp import Client, Depot, Location, ProblemData, VehicleType

    scale = 1_000_000
    max_value = 1 << 32
    constraints = set(problem.get("constraints", ()))
    coordinates = problem.get("coordinates")
    matrix_source = problem.get("distance")
    if matrix_source is None:
        if coordinates is None:
            raise ValueError("PyVRP requires coordinates or a distance matrix")
        coordinates = np.asarray(coordinates, dtype=float)
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        matrix_source = np.sqrt(np.sum(delta * delta, axis=-1))
    matrix_source = np.asarray(matrix_source, dtype=float)
    node_count = matrix_source.shape[0]
    depot_count = int(problem["depot_count"])
    client_count = node_count - depot_count

    def scaled(values) -> np.ndarray:
        array = np.asarray(values, dtype=float) * scale
        array = np.where(np.isposinf(array), np.iinfo(np.int64).max, array)
        return np.rint(array).astype(np.int64)

    matrix = scaled(matrix_source)
    np.fill_diagonal(matrix, 0)

    demand = (
        scaled(problem["demand"]).reshape(node_count)
        if "demand" in problem
        else np.zeros(node_count, dtype=np.int64)
    )
    # PRISM inherits URS's signed demand: positive linehaul, negative backhaul.
    delivery = np.maximum(demand, 0)
    pickup = np.maximum(-demand, 0)
    capacity = (
        int(scaled(problem["capacity"]).reshape(-1)[0])
        if "capacity" in constraints
        else int(np.iinfo(np.int64).max)
    )

    if "time_windows" in constraints:
        windows = np.column_stack(
            (
                scaled(problem["tw_start"]).reshape(node_count),
                scaled(problem["tw_end"]).reshape(node_count),
            )
        )
        service = scaled(problem.get("service_time", np.zeros(node_count)))
        service = service.reshape(node_count)
    else:
        windows = np.column_stack(
            (
                np.zeros(node_count, dtype=np.int64),
                np.full(node_count, np.iinfo(np.int64).max, dtype=np.int64),
            )
        )
        service = np.zeros(node_count, dtype=np.int64)

    max_distance = (
        int(scaled(problem["route_limit"]).reshape(-1)[0])
        if "route_limit" in constraints
        else int(np.iinfo(np.int64).max)
    )

    if problem.get("open_route"):
        # PyVRP requires an end depot. Zero-cost/time return arcs reproduce the
        # open-route objective and keep route-distance constraints open-ended.
        matrix[:, :depot_count] = 0
    if "backhaul_order" in constraints:
        # Classic VRPB: no backhaul may precede a linehaul on the same route.
        linehaul = np.flatnonzero(delivery > 0)
        backhaul = np.flatnonzero(pickup > 0)
        matrix[np.ix_(backhaul, linehaul)] = max_value

    if coordinates is None:
        # Asymmetric variants ship a distance matrix and no embedding; PyVRP
        # solves from the matrices alone and uses coordinates only for display.
        locations = [Location(x=0.0, y=0.0) for _ in range(node_count)]
    else:
        locations = [
            Location(x=float(x), y=float(y))
            for x, y in np.asarray(coordinates, dtype=float)
        ]
    depots = [
        Depot(
            location=index,
            tw_early=int(windows[index, 0]),
            tw_late=int(windows[index, 1]),
        )
        for index in range(depot_count)
    ]
    clients = [
        Client(
            location=index,
            delivery=[int(delivery[index])],
            pickup=[int(pickup[index])],
            service_duration=int(service[index]),
            tw_early=int(windows[index, 0]),
            tw_late=int(windows[index, 1]),
        )
        for index in range(depot_count, node_count)
    ]
    vehicle_types = [
        VehicleType(
            num_available=client_count,
            capacity=[capacity],
            start_depot=depot_index,
            end_depot=depot_index,
            tw_early=int(windows[depot_index, 0]),
            tw_late=int(windows[depot_index, 1]),
            max_distance=max_distance,
        )
        for depot_index in range(depot_count)
    ]
    return ProblemData(
        locations,
        clients,
        depots,
        vehicle_types,
        [matrix],
        [matrix],
    ), scale


def solve_pyvrp_instance(problem: dict, time_limit: float, seed: int) -> float:
    """Solve one PRISM problem with PyVRP and return its objective."""
    from pyvrp import solve
    from pyvrp.stop import MaxRuntime

    data, scale = _pyvrp_problem(problem)
    result = solve(data, MaxRuntime(time_limit), seed=seed, display=False)
    if not result.best.is_feasible():
        raise RuntimeError("PyVRP found no feasible solution")
    return float(result.cost()) / scale


def solve_ortools_instance(
    problem: dict, time_limit: float, candidates: int
) -> float:
    """Solve one PRISM problem with OR-Tools, measured by the decoder itself.

    The OR-Tools route is handed back to ``Decoder.evaluate`` -- the feasibility
    and cost authority -- so the number reported is on PRISM's exact objective
    scale and any mismatch between the two models surfaces as an infeasible
    oracle route rather than a silently wrong baseline.
    """
    from evrp_oracle import solve_problem

    result = solve_problem(problem, time_limit_s=time_limit)
    decoder = prism_decoder.Decoder(
        problem, candidate_config={"max_candidates": candidates}
    )
    evaluated = decoder.evaluate(
        torch.tensor(result.route, dtype=torch.int32).numpy()
    )
    if not evaluated["feasible"]:
        raise RuntimeError(
            "OR-Tools produced a route the decoder rejects: "
            f"{evaluated.get('error', 'unknown')}"
        )
    return float(evaluated["objective"])


def oracle_solver_for(problem: dict) -> tuple[str, str | None]:
    """Pick the oracle solver for one problem, or say why neither applies.

    PyVRP models the capacitated multi-route grid natively and is the stronger
    solver there, so it is preferred; OR-Tools covers everything else PRISM
    benchmarks -- single tours, pickup-delivery precedence, prize objectives,
    and the resource-algebra probes.
    """
    if _pyvrp_unsupported(problem) is None:
        return "pyvrp", None
    if int(problem.get("depot_count", 1)) > 1:
        # Multi-depot instances are all PyVRP-modelled, so this is unreachable
        # in practice; the OR-Tools model is single-depot.
        return "ortools", "multi_depot"
    return "ortools", None


def solve_oracle_instance(
    problem: dict, solver: str, time_limit: float, seed: int, candidates: int
) -> float:
    if solver == "pyvrp":
        return solve_pyvrp_instance(problem, time_limit, seed)
    return solve_ortools_instance(problem, time_limit, candidates)


def _oracle_version(solver: str) -> str:
    from importlib.metadata import version

    return f"{solver} {version('pyvrp' if solver == 'pyvrp' else 'ortools')}"


def oracle_reference(batch, *, time_limit: float, candidates: int) -> float:
    """Mean decoder-measured OR-Tools cost over a generated probe's instances.

    The unseen-resource probes ship no saved optimum, so a gap needs one solved
    on the spot. Every route is validated through the decoder, which puts the
    reference on PRISM's exact objective scale and turns a model mismatch into
    a loud failure rather than a quietly wrong number.
    """
    return statistics.fmean(
        solve_ortools_instance(batch.problem(index)[0], time_limit, candidates)
        for index in range(len(batch))
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The oracle's own CLI surface, registered with the method."""
    parser.add_argument(
        "--oracle-time-limit",
        dest="oracle_time_limit",
        type=float,
        default=10.0,
        help=(
            "Solver wall-clock budget in seconds per instance for the 'oracle' "
            "method (default: 10). Both PyVRP and OR-Tools are anytime "
            "heuristics, so this sets the strength of the oracle."
        ),
    )
    parser.add_argument(
        "--oracle-cache",
        dest="oracle_cache",
        type=Path,
        default=default_cache_path("oracle"),
        help=(
            "JSON cache of oracle per-instance objectives, keyed by variant + "
            "instance source + val_size + seed + time limit (default: "
            "results/oracle_cache.json). It is written after every instance, "
            "so an interrupted variant resumes from what it already solved."
        ),
    )
    parser.add_argument(
        "--oracle-refresh",
        dest="oracle_refresh",
        action="store_true",
        help="Ignore any cached oracle results and recompute (and overwrite) them.",
    )


def validate_arguments(args: argparse.Namespace, error) -> None:
    """Checks that belong with the flags rather than with the CLI."""
    if args.oracle_time_limit <= 0:
        error("--oracle-time-limit must be positive")


class OracleMethod:
    """Solve each instance with whichever classical solver models it."""

    name = "oracle"

    def __init__(self, args: argparse.Namespace, cache: ResultCache | None = None):
        self.args = args
        self.cache = cache if cache is not None else ResultCache(
            args.oracle_cache, refresh=args.oracle_refresh
        )
        self.solver = ""

    def config(self) -> str:
        return (
            f"solver={self.solver or 'auto'},"
            f"time_limit={self.args.oracle_time_limit}"
        )

    def covers(self, problem: dict) -> str | None:
        return oracle_solver_for(problem)[1]

    def cache_key(self, request: MethodRequest) -> str:
        """What identifies an oracle result: instances, budget and seed.

        The batch signature pins the instances, and the per-instance solve seed
        derives from the run seed, so a matching key reproduces the same
        problems solved under the same budget.
        """
        return (
            f"{request.batch.variant}|{request.batch.signature}|"
            f"{len(request.batch)}|{request.seed}|{self.args.oracle_time_limit}"
        )

    def run(self, request: MethodRequest) -> MethodResult:
        key = self.cache_key(request)
        entry = self.cache.get(key) or {}
        objectives = [float(value) for value in entry.get("objectives", [])][
            : len(request.batch)
        ]
        seconds = float(entry.get("seconds", 0.0))
        self.solver = str(entry.get("solver", ""))
        if len(objectives) >= len(request.batch):
            return MethodResult(
                objectives=objectives,
                direction=request.direction,
                seconds=seconds,
                source="cached",
                config=self.config(),
                meta={"solver": self.solver},
            )
        if objectives:
            print(
                f"ORACLE variant={request.batch.variant} resuming from instance "
                f"{len(objectives) + 1}/{len(request.batch)}",
                flush=True,
            )

        for index in range(len(objectives), len(request.batch)):
            problem, _initial_route = request.batch.problem(index)
            solver, reason = oracle_solver_for(problem)
            if reason is not None:
                print(
                    f"No oracle models {request.batch.variant} ({reason})",
                    file=sys.stderr,
                    flush=True,
                )
                return MethodResult.unsupported(reason, config=self.config())
            self.solver = solver
            started = time.perf_counter()
            try:
                objective = solve_oracle_instance(
                    problem,
                    solver,
                    self.args.oracle_time_limit,
                    request.instance_seed(index),
                    self.args.candidates,
                )
            except Exception as error:  # noqa: BLE001
                print(
                    f"Oracle {solver} failed on {request.batch.variant} instance "
                    f"{index}: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return MethodResult.failure(error, config=self.config())
            instance_seconds = time.perf_counter() - started
            seconds += instance_seconds
            objectives.append(objective)
            # Persist after every instance so an interrupted run keeps the
            # instances it already paid for.
            self.cache.put(
                key,
                {
                    "objectives": objectives,
                    "seconds": seconds,
                    "solver": solver,
                    "version": _oracle_version(solver),
                },
            )
            print(
                f"ORACLE variant={request.batch.variant} solver={solver} "
                f"instance {index + 1}/{len(request.batch)} "
                f"objective={objective:.6g} seconds={instance_seconds:.3f}",
                flush=True,
            )
            if request.on_result is not None:
                request.on_result(
                    index,
                    float(objective),
                    instance_seconds,
                    request.direction,
                )
        return MethodResult(
            objectives=objectives,
            direction=request.direction,
            seconds=seconds,
            config=self.config(),
            meta={"solver": self.solver},
        )
