"""OR-Tools oracle for the problem families PyVRP cannot model.

This module builds a *faithful* OR-Tools routing model of the exact problem the
decoder enforces -- capacity, optional time windows, route/tour limits, open
routes, pickup-delivery precedence, prize collection with optional nodes,
battery as a remaining-charge dimension, and continuous driver time as an
optionally reset remaining-drive dimension that is reset to full at every
charger and depot (matching the decoder's ``value = reset_value`` reset, which
is a full reset, not a chosen partial refuel).

``test.py``'s ``oracle`` baseline solves with PyVRP wherever PyVRP has a model
(the capacitated multi-route grid) and falls back here for everything else: the
single-tour families (tsp, atsp, tsptw), pickup-delivery (pd*), prize
objectives (op, pctsp and their variants), and the generated resource-algebra
probes (evrp*, vrpdb*), which have no PyVRP equivalent at all.

The result is self-validating: the OR-Tools route is handed back to the decoder's
own ``evaluate`` (the feasibility+cost authority), so a model that does not match
decoder semantics surfaces immediately as an infeasible oracle route rather than
a silently wrong number. The reference objective returned is decoder-measured, so
it is on the exact scale PRISM's objective uses.

OR-Tools is integer-only, so distances/battery are scaled by ``_DIST_SCALE`` and
demand/capacity by ``_UNIT_SCALE``. The solver is a heuristic (guided local
search); at large sizes it yields a strong feasible upper bound, not a proven
optimum, so the reported gap is an upper bound on PRISM's true optimality gap.

One modelling choice is deliberately stricter than the decoder: OR-Tools keeps
each pickup-delivery pair on one vehicle, whereas the decoder only requires the
pickup to be visited first. Every route it returns is therefore decoder-feasible
(the caller validates this), but on multi-route pd* instances the bound it
reports may be slightly loose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

_DIST_SCALE = 100_000
_UNIT_SCALE = 100_000
# Weight on a forgone prize, so maximizing collected prize dominates the tour
# length that breaks ties between equally rewarding tours.
_PRIZE_WEIGHT = 10_000
# Search configuration. PATH_CHEAPEST_ARC is the right default for the
# visit-everything families, and it is the only strategy that reliably finds a
# first solution under a prize quota. It is a poor fit for optional nodes
# though: it builds a cheap tour blind to the prizes, and guided local search
# then penalises arcs rather than the node selection, so it settles for a tour
# of low-value nodes. Prize problems therefore try a ladder and keep the best
# solution any of them reaches, splitting the caller's budget between them.
_FIRST_SOLUTION_STRATEGY = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
_OPTIONAL_NODE_STRATEGIES = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
)
_METAHEURISTIC = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH


@dataclass
class OracleResult:
    route: list[int]
    scaled_distance: int
    vehicles_used: int
    break_nodes: list[int]
    # The routing model's own objective (arc costs plus the penalties of any
    # dropped node), used to compare solutions found by different strategies.
    objective_value: int = 0


def _distance_matrix(problem: dict) -> np.ndarray:
    if "distance" in problem and problem["distance"] is not None:
        return np.asarray(problem["distance"], dtype=np.float64)
    coordinates = np.asarray(problem["coordinates"], dtype=np.float64)
    return np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )


def _battery_spec(problem: dict) -> dict | None:
    for row in problem.get("resources", []) or []:
        if row.get("name") == "battery":
            return row
    return None


def _driver_break_spec(problem: dict) -> dict | None:
    for row in problem.get("resources", []) or []:
        if row.get("name") == "continuous_driving_time":
            return row
    return None


def _charger_mask(problem: dict, node_count: int) -> np.ndarray:
    attributes = problem.get("node_attributes", {}) or {}
    charger = attributes.get("charger")
    if charger is None:
        return np.zeros(node_count, dtype=bool)
    return np.asarray(charger, dtype=np.float64) > 0.5


def _break_mask(problem: dict, node_count: int) -> np.ndarray:
    """Nodes where a driver break may reset the drive accumulator.

    Mirrors the decoder's ``break_allowed`` node attribute. When absent (every
    node rest-eligible), all customers qualify; the depot resets at route
    boundaries regardless and is handled separately.
    """
    attributes = problem.get("node_attributes", {}) or {}
    allowed = attributes.get("break_allowed")
    if allowed is None:
        return np.ones(node_count, dtype=bool)
    return np.asarray(allowed, dtype=np.float64) > 0.5


def _pickup_delivery_pairs(problem: dict, node_count: int) -> list[tuple[int, int]]:
    """Pair pickups with deliveries the way the native decoder does.

    Mirrors ``set_pickup_delivery_relations`` in src/binding.cpp: explicit
    ``pickup_delivery_pairs`` when given, otherwise URS's convention that the
    first half of the customers are pickups and customer ``i`` is delivered by
    customer ``i + pair_count``.
    """
    if "pickup_delivery" not in problem.get("constraints", []):
        return []
    pairs = problem.get("pickup_delivery_pairs")
    if pairs is not None:
        return [(int(pickup), int(delivery)) for pickup, delivery in pairs]
    depot_count = int(problem.get("depot_count", 1))
    customers = node_count - depot_count
    if customers % 2:
        raise ValueError("pickup-delivery variants require an even customer count")
    pair_count = customers // 2
    return [
        (depot_count + index, depot_count + index + pair_count)
        for index in range(pair_count)
    ]


def _vehicle_bound(
    demand: np.ndarray,
    capacity: float,
    distance: np.ndarray,
    battery_range: float | None,
    node_count: int,
) -> int:
    customers = node_count - 1
    # Pickup-delivery demand is signed and sums to zero per pair, so the load a
    # route has to carry is bounded by the positive half alone.
    capacity_routes = math.ceil(
        float(np.maximum(demand[1:], 0.0).sum()) / max(capacity, 1e-9)
    )
    bounds = [capacity_routes, 1]
    if battery_range:
        round_trip = 2.0 * distance[0, 1:]
        bounds.append(
            math.ceil(float(round_trip.sum()) / max(battery_range, 1e-9))
        )
    # A generous margin keeps the model feasible without inflating it to the
    # singleton worst case; the caller retries with more vehicles if needed.
    return int(min(customers, max(bounds) + 4))


def _solve_once(
    problem: dict,
    num_vehicles: int,
    time_limit_s: float,
    first_solution_strategy=None,
) -> OracleResult | None:
    distance = _distance_matrix(problem)
    node_count = distance.shape[0]
    demand = np.asarray(problem.get("demand", np.zeros(node_count)), dtype=np.float64)
    capacity = float(problem.get("capacity", 1.0))
    battery = _battery_spec(problem)
    battery_range = float(battery["initial"]) if battery is not None else None
    charger = _charger_mask(problem, node_count)
    driver_break = _driver_break_spec(problem)

    scaled_distance = np.rint(distance * _DIST_SCALE).astype(np.int64)
    constraints = problem.get("constraints", [])
    depot_count = int(problem.get("depot_count", 1))
    if depot_count > 1:
        raise NotImplementedError("the OR-Tools oracle is single-depot")
    open_route = bool(problem.get("open_route", False))

    manager = pywrapcp.RoutingIndexManager(node_count, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    solver = routing.solver()

    def travel(i: int, j: int) -> int:
        # An open route is not charged for its return leg, exactly as the
        # decoder skips the closing edge when ``open_route`` is set. A
        # depot-less TSP closes its tour, so node 0 keeps its real arc costs.
        if open_route and depot_count and j < depot_count:
            return 0
        return int(scaled_distance[i, j])

    def distance_cb(from_index: int, to_index: int) -> int:
        return travel(
            manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        )

    transit_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    # Visit order, used to force each delivery after its own pickup. A constant
    # unit transit makes the cumul the position of a node along its route.
    order_dimension = None
    pairs = _pickup_delivery_pairs(problem, node_count)
    if pairs:
        routing.AddConstantDimension(1, node_count, True, "Order")
        order_dimension = routing.GetDimensionOrDie("Order")
        for pickup, delivery in pairs:
            pickup_index = manager.NodeToIndex(pickup)
            delivery_index = manager.NodeToIndex(delivery)
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            solver.Add(
                routing.VehicleVar(pickup_index)
                == routing.VehicleVar(delivery_index)
            )
            solver.Add(
                order_dimension.CumulVar(pickup_index)
                <= order_dimension.CumulVar(delivery_index)
            )

    # Capacity dimension. The cumul is the load taken on so far, which the
    # decoder mirrors as ``capacity - load``; both must stay within [0, cap].
    if "capacity" in constraints:
        scaled_demand = np.rint(demand * _UNIT_SCALE).astype(np.int64)
        scaled_demand[0] = 0

        def demand_cb(from_index: int) -> int:
            return int(scaled_demand[manager.IndexToNode(from_index)])

        demand_index = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_index,
            0,
            [int(round(capacity * _UNIT_SCALE))] * num_vehicles,
            True,
            "Capacity",
        )

    # Route-limit (duration) dimension: total route distance capped per vehicle.
    # It reuses the distance transit and resets only at the depot (route start),
    # which OR-Tools enforces automatically via fix_start_cumul_to_zero.
    if "route_limit" in constraints:
        limit = int(round(float(problem["route_limit"]) * _DIST_SCALE))
        routing.AddDimension(transit_index, 0, limit, True, "RouteLimit")

    # Orienteering tour budget: one route, capped total length.
    if "tour_limit" in constraints:
        limit = int(round(float(problem["tour_limit"]) * _DIST_SCALE))
        routing.AddDimension(transit_index, 0, limit, True, "TourLimit")

    # Optional nodes. Without "visit_all" the model may drop a customer at a
    # price: the forgone prize for an orienteering objective, the node's own
    # penalty for a prize-collecting one. Prize objectives are lexicographic --
    # collect as much prize as possible, then travel as little as possible --
    # so the forgone prize is weighted well above any attainable tour length.
    prize = np.asarray(
        problem.get("prize", np.zeros(node_count)), dtype=np.float64
    )
    if "visit_all" not in constraints:
        maximize_prize = problem.get("objective") == "prize"
        forgone = prize if maximize_prize else np.asarray(
            problem.get("penalty", np.zeros(node_count)), dtype=np.float64
        )
        weight = _PRIZE_WEIGHT if maximize_prize else 1
        for node in range(depot_count, node_count):
            routing.AddDisjunction(
                [manager.NodeToIndex(node)],
                int(round(float(forgone[node]) * _UNIT_SCALE)) * weight,
            )

    # Prize quota: the visited nodes must collect at least the required prize.
    if "prize_quota" in constraints:
        scaled_prize = np.rint(prize * _UNIT_SCALE).astype(np.int64)

        def prize_cb(from_index: int) -> int:
            return int(scaled_prize[manager.IndexToNode(from_index)])

        prize_index = routing.RegisterUnaryTransitCallback(prize_cb)
        routing.AddDimension(
            prize_index, 0, int(scaled_prize.sum()), True, "Prize"
        )
        prize_dimension = routing.GetDimensionOrDie("Prize")
        quota = int(round(float(problem.get("prize_quota", 1.0)) * _UNIT_SCALE))
        for vehicle in range(num_vehicles):
            prize_dimension.CumulVar(routing.End(vehicle)).SetMin(quota)

    # Elapsed route time. Driver-break variants need this dimension even without
    # time windows so each selected reset contributes its fixed duration.
    time_dim = None
    if "time_windows" in constraints or driver_break is not None:
        has_time_windows = "time_windows" in constraints
        service = np.asarray(
            problem.get("service_time", np.zeros(node_count)), dtype=np.float64
        )
        if has_time_windows:
            tw_start = np.asarray(problem["tw_start"], dtype=np.float64)
            tw_end = np.asarray(problem["tw_end"], dtype=np.float64)
            horizon_value = float(tw_end.max())
        else:
            tw_start = np.zeros(node_count, dtype=np.float64)
            horizon_value = (node_count + 1) * (
                float(distance.max()) + float(service.max()) + 1.0
            )
            tw_end = np.full(node_count, horizon_value, dtype=np.float64)
        horizon = int(math.ceil(horizon_value * _DIST_SCALE))

        def time_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            elapsed = float(travel(i, j)) / _DIST_SCALE + float(service[i])
            return int(round(elapsed * _DIST_SCALE))

        time_index = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(time_index, horizon, horizon, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        if has_time_windows:
            for node in range(1, node_count):
                index = manager.NodeToIndex(node)
                time_dim.CumulVar(index).SetRange(
                    int(round(float(tw_start[node]) * _DIST_SCALE)),
                    int(round(float(tw_end[node]) * _DIST_SCALE)),
                )
        for vehicle in range(num_vehicles):
            start = routing.Start(vehicle)
            end = routing.End(vehicle)
            time_dim.CumulVar(start).SetValue(0)
            time_dim.CumulVar(end).SetRange(
                0, int(round(float(tw_end[0]) * _DIST_SCALE))
            )

    # Battery: remaining charge, decreasing by distance (negative transit) and
    # reset to full at every charger/depot via an equality on the node slack.
    if battery_range is not None:
        cap = int(round(battery_range * _DIST_SCALE))

        def battery_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return -int(scaled_distance[i, j])

        battery_index = routing.RegisterTransitCallback(battery_cb)
        # slack_max = cap lets a charger top the charge back up to full.
        routing.AddDimension(battery_index, cap, cap, False, "Battery")
        battery_dim = routing.GetDimensionOrDie("Battery")
        for node in range(node_count):
            index = manager.NodeToIndex(node)
            if node == 0 or charger[node]:
                # Full reset: departing charge is exactly the capacity, so the
                # slack tops arrival charge up to full (matches decoder reset).
                battery_dim.CumulVar(index).SetRange(0, cap)
                routing.solver().Add(
                    battery_dim.CumulVar(index)
                    + battery_dim.SlackVar(index)
                    == cap
                )
            else:
                battery_dim.CumulVar(index).SetRange(0, cap)
                battery_dim.SlackVar(index).SetValue(0)
        for vehicle in range(num_vehicles):
            start = routing.Start(vehicle)
            battery_dim.CumulVar(start).SetRange(0, cap)
            routing.solver().Add(
                battery_dim.CumulVar(start) + battery_dim.SlackVar(start) == cap
            )

    # Driver breaks: store remaining continuous-drive allowance. Travel reduces
    # it; a Boolean reset at the current customer may top it back up to H_max.
    # The same Boolean contributes T_break through the elapsed-time slack.
    break_vars: dict[int, object] = {}
    if driver_break is not None:
        bounds = driver_break.get("bounds", [])
        if len(bounds) != 1 or "upper" not in bounds[0]:
            raise ValueError("driver-break resource requires one upper bound")
        max_drive = int(round(float(bounds[0]["upper"]) * _DIST_SCALE))
        reset = driver_break.get("reset", {})
        break_duration = int(round(float(reset.get("duration", 0.0)) * _DIST_SCALE))
        break_allowed = _break_mask(problem, node_count)

        def remaining_drive_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return -int(scaled_distance[i, j])

        driver_index = routing.RegisterTransitCallback(remaining_drive_cb)
        routing.AddDimension(driver_index, max_drive, max_drive, False, "Driver")
        driver_dim = routing.GetDimensionOrDie("Driver")
        for vehicle in range(num_vehicles):
            start = routing.Start(vehicle)
            driver_dim.CumulVar(start).SetValue(max_drive)
            driver_dim.SlackVar(start).SetValue(0)
        for node in range(1, node_count):
            index = manager.NodeToIndex(node)
            take_break = solver.BoolVar(f"driver_break_{node}")
            routing.AddToAssignment(take_break)
            # A break may only be taken at a rest-eligible node (matches the
            # decoder's break_allowed gate). Elsewhere the reset is forbidden.
            if not break_allowed[node]:
                solver.Add(take_break == 0)
            remaining = driver_dim.CumulVar(index)
            refill = driver_dim.SlackVar(index)
            # No break => refill=0. Break => remaining+refill=H_max.
            solver.Add(refill <= max_drive * take_break)
            solver.Add(remaining + refill >= max_drive * take_break)
            solver.Add(remaining + refill <= max_drive)
            if time_dim is not None and break_duration > 0:
                solver.Add(
                    time_dim.SlackVar(index) >= break_duration * take_break
                )
            break_vars[node] = take_break

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        _FIRST_SOLUTION_STRATEGY
        if first_solution_strategy is None
        else first_solution_strategy
    )
    parameters.local_search_metaheuristic = _METAHEURISTIC
    parameters.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return None

    # How a route terminates is the decoder's completion signal. A depot-less
    # problem (tsp, atsp) is one closed tour over the nodes themselves, and a
    # visit-all single tour (pdtsp, tsptw) is complete at its last customer --
    # both are closed implicitly, and a trailing depot would read as continuing
    # past completion. Multi-route problems and the optional-visit families
    # (op, pctsp) instead declare completion by returning to the depot.
    close_at_depot = bool(depot_count) and (
        num_vehicles > 1
        or bool(problem.get("multi_route", True))
        or "visit_all" not in constraints
    )
    route: list[int] = [0]
    used = 0
    total = 0
    for vehicle in range(num_vehicles):
        index = routing.Start(vehicle)
        customers: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                customers.append(node)
            next_index = solution.Value(routing.NextVar(index))
            total += routing.GetArcCostForVehicle(index, next_index, vehicle)
            index = next_index
        if customers:
            route.extend(customers)
            if close_at_depot:
                route.append(0)
            used += 1
    break_nodes = [
        node for node, variable in break_vars.items() if solution.Value(variable)
    ]
    return OracleResult(
        route=route,
        scaled_distance=total,
        vehicles_used=used,
        break_nodes=break_nodes,
        objective_value=int(solution.ObjectiveValue()),
    )


def _solve_best(
    problem: dict, num_vehicles: int, time_limit_s: float
) -> OracleResult | None:
    """Solve once per first-solution strategy and keep the best solution."""
    constraints = problem.get("constraints", [])
    # The ladder pays off only where nodes are optional and unconstrained in
    # aggregate. Under a prize quota the other strategies cannot even build a
    # feasible first solution, and they burn their whole share discovering
    # that, so a quota keeps the single reliable strategy and the full budget.
    ladder = "visit_all" not in constraints and "prize_quota" not in constraints
    strategies = _OPTIONAL_NODE_STRATEGIES if ladder else (_FIRST_SOLUTION_STRATEGY,)
    deadline = time.perf_counter() + time_limit_s
    best: OracleResult | None = None
    for position, strategy in enumerate(strategies):
        # Split what is left of the budget over the strategies still to run, so
        # a strategy that fails to start hands its share back to the others.
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        share = remaining / (len(strategies) - position)
        result = _solve_once(problem, num_vehicles, share, strategy)
        if result is not None and (
            best is None or result.objective_value < best.objective_value
        ):
            best = result
    return best


def solve_problem(problem: dict, time_limit_s: float = 5.0) -> OracleResult:
    """Solve one decoder problem, retrying vehicle counts if needed."""
    distance = _distance_matrix(problem)
    node_count = distance.shape[0]
    demand = np.asarray(problem.get("demand", np.zeros(node_count)), dtype=np.float64)
    battery = _battery_spec(problem)
    battery_range = float(battery["initial"]) if battery is not None else None
    # Single-tour families (tsp, tsptw, op, pctsp, pdtsp) admit exactly one
    # route, so the retry ladder below does not apply to them.
    if not problem.get("multi_route", True):
        result = _solve_best(problem, 1, time_limit_s)
        if result is None:
            raise RuntimeError("OR-Tools found no feasible solution")
        return result
    vehicles = _vehicle_bound(
        demand, float(problem.get("capacity", 1.0)), distance, battery_range, node_count
    )
    for _ in range(4):
        result = _solve_best(problem, vehicles, time_limit_s)
        if result is not None:
            return result
        if vehicles >= node_count - 1:
            break
        vehicles = min(node_count - 1, vehicles * 2)
    raise RuntimeError("OR-Tools found no feasible solution")


# The resource probes reach the same solver; kept as the name test.py and the
# oracle tests have always used for them.
solve_evrp = solve_problem
