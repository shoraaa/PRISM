"""Batched, GPU-friendly route evaluator (torch).

Reproduces the C++ RoutingDecoder::evaluate cost+feasibility for a *complete*
route as a deterministic scan over the sequence, so refinement training never
calls the C++ oracle per move and can batch thousands of rollouts on GPU. The
oracle is kept only as an eval-time authority and as the parity check in
tests/python (see test_route_eval_parity).

Scope: single-depot, multi-route, non-open variants whose constraints are a
subset of {visit_all, capacity, time_windows} and objective min-distance --
i.e. cvrp / cvrptw. This is the affine slice; structural ops (backhaul,
pickup-delivery, prize) and open/multi-depot come later, guarded by the same
parity test.

FEASIBILITY_EPS matches decoder.cpp.
"""

from __future__ import annotations

import torch

FEASIBILITY_EPS = 1e-4


class RouteEvaluator:
    def __init__(self, problem, device="cpu"):
        c = set(problem.get("constraints", []))
        self.depot_count = int(problem.get("depot_count", 1))
        self.objective = problem.get("objective", "distance")
        if self.objective not in ("distance", "prize", "distance_plus_penalty"):
            raise NotImplementedError(f"unsupported objective: {self.objective}")
        unsupported = c - {"visit_all", "capacity", "time_windows", "route_limit",
                           "backhaul_order", "pickup_delivery", "tour_limit",
                           "prize_quota"}
        if unsupported:
            raise NotImplementedError(f"unsupported constraints: {unsupported}")
        self.device = device
        self.multi_route = bool(problem.get("multi_route", False))
        self.open_route = bool(problem.get("open_route", False))
        self.has_capacity = "capacity" in c
        self.has_tw = "time_windows" in c
        self.has_route_limit = "route_limit" in c
        self.has_backhaul = "backhaul_order" in c
        self.has_pd = "pickup_delivery" in c
        self.has_tour_limit = "tour_limit" in c
        self.has_prize_quota = "prize_quota" in c
        self.has_visit_all = "visit_all" in c

        if "distance" in problem:
            dist = torch.as_tensor(problem["distance"], dtype=torch.float64, device=device)
        else:
            # problem_data only materializes the N^2 matrix for N<=512; for larger
            # instances build it from coordinates (Euclidean). Fine to ~a few k;
            # for huge N compute leg distances lazily instead.
            xy = torch.as_tensor(problem["coordinates"], dtype=torch.float64, device=device)
            dist = torch.cdist(xy, xy)
        self.dist = dist
        self.n = dist.shape[0]
        self.depot = 0
        # tsp/atsp have no depot -> a single Hamiltonian cycle over all nodes
        self.no_depot = self.depot_count == 0
        # pickup-delivery pairing is by index: pickup c pairs with delivery c+pd_k
        # (first half of customers are pickups, second half deliveries)
        self.pd_k = (self.n - self.depot_count) // 2 if self.has_pd else 0
        self.capacity = float(problem.get("capacity", 0.0))
        # demand is absent for capacity-free schemas (e.g. vrptw)
        if self.has_capacity:
            self.demand = torch.as_tensor(problem["demand"], dtype=torch.float64, device=device)
        if self.has_route_limit:
            self.route_limit = float(problem["route_limit"])
        if self.has_tour_limit:
            self.tour_limit = float(problem["tour_limit"])
        if self.has_prize_quota:
            self.prize_quota = float(problem["prize_quota"])
        if self.objective in ("prize", "distance_plus_penalty") or self.has_prize_quota:
            self.prize = torch.as_tensor(problem.get("prize", [0.0] * self.n), dtype=torch.float64, device=device)
        if self.objective == "distance_plus_penalty":
            self.penalty = torch.as_tensor(problem["penalty"], dtype=torch.float64, device=device)
        if self.has_tw:
            self.tw_start = torch.as_tensor(problem["tw_start"], dtype=torch.float64, device=device)
            self.tw_end = torch.as_tensor(problem["tw_end"], dtype=torch.float64, device=device)
            self.service = torch.as_tensor(problem["service_time"], dtype=torch.float64, device=device)

    def evaluate_batch(self, routes):
        """routes: list[list[int]]. Returns (cost[B], feasible[B], violation[B]).

        cost is the min-distance objective (only meaningful where feasible).
        violation is the summed constraint overrun (>=0), for reward shaping.
        """
        B = len(routes)
        L = max(len(r) for r in routes)
        rt = torch.full((B, L), -1, dtype=torch.long, device=self.device)
        for i, r in enumerate(routes):
            rt[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=self.device)
        return self.evaluate_padded(rt, rt != -1)

    @staticmethod
    def _segmented_cummax(X, seg):
        """Prefix max of X within each contiguous segment (reset when seg id
        changes), via Hillis-Steele doubling -- O(log L) passes, no L-loop."""
        B, L = X.shape
        m = X.clone()
        neg = float("-inf")
        k = 1
        while k < L:
            sm = torch.cat([X.new_full((B, k), neg), m[:, :-k]], dim=1)
            ss = torch.cat([seg.new_full((B, k), -1), seg[:, :-k]], dim=1)
            m = torch.where(ss == seg, torch.maximum(m, sm), m)
            k *= 2
        return m

    def _evaluate_affine(self, rt, valid):
        """Vectorized evaluate for the affine single-depot family (no TW / backhaul
        / pd / multi-depot). Distance + capacity load-profile + route_limit via
        segmented cumsum/scatter -- no O(L) Python scan. Matches evaluate_padded
        (parity-tested). Returns (obj [B], feas [B], viol [B])."""
        B, L = rt.shape
        dev = self.device
        dc = self.depot_count
        rows = torch.arange(B, device=dev)
        is_depot = (rt >= 0) & (rt < dc)
        is_cust = rt >= dc
        node = rt.clamp(min=0)
        feas = torch.ones(B, dtype=torch.bool, device=dev)
        viol = torch.zeros(B, dtype=torch.float64, device=dev)

        # structural: start depot; multi-route ends at a depot; no two depots
        feas &= is_depot[:, 0]
        last = rt[rows, valid.sum(1) - 1]
        if self.multi_route:
            feas &= (last >= 0) & (last < dc)
        consec = valid[:, 1:] & is_depot[:, 1:] & is_depot[:, :-1]
        feas &= ~consec.any(1)
        if self.has_visit_all:
            counts = torch.zeros(B, self.n, dtype=torch.float64, device=dev)
            counts.scatter_add_(1, node, valid.double())
            feas &= (counts[:, dc:] == 1).all(dim=1)

        route_id = torch.cumsum(is_depot.long(), dim=1)  # [B,L], route of each pos
        R = int(route_id.max().item()) + 1
        # multi-depot: each route returns to its OWN start depot (route_depot),
        # not the next marker. Single-depot: route_depot is always 0.
        route_start_node = torch.zeros(B, R, dtype=torch.long, device=dev)
        route_start_node.scatter_(1, torch.where(is_depot, route_id, torch.zeros_like(route_id)),
                                  torch.where(is_depot, node, torch.zeros_like(node)))
        route_depot = torch.gather(route_start_node, 1, route_id)  # [B,L] start depot node

        # distance cost. A leg INTO a depot marker is physically the return to the
        # route's own depot; open routes pay nothing for it.
        step_v = valid[:, 1:]
        next_is_depot = is_depot[:, 1:]
        leg_dest = torch.where(next_is_depot, route_depot[:, :-1], node[:, 1:])
        d = self.dist[node[:, :-1], leg_dest]
        charged = step_v & ~(next_is_depot & self.open_route)
        cost = torch.where(charged, d, torch.zeros_like(d)).sum(1)

        if self.has_capacity:
            load_delta = torch.where(is_cust, -self.demand[node], torch.zeros(B, L, dtype=torch.float64, device=dev))
            C = torch.cumsum(load_delta, dim=1)
            big = torch.finfo(torch.float64).max
            segmin = torch.full((B, R), big, dtype=torch.float64, device=dev)
            segmin.scatter_reduce_(1, route_id, C, "amin", include_self=True)
            segmax = torch.full((B, R), -big, dtype=torch.float64, device=dev)
            segmax.scatter_reduce_(1, route_id, C, "amax", include_self=True)
            startC = torch.zeros(B, R, dtype=torch.float64, device=dev)
            startC.scatter_(1, torch.where(is_depot, route_id, torch.zeros_like(route_id)),
                            torch.where(is_depot, C, torch.zeros_like(C)))
            lh = (is_cust & (self.demand[node] > FEASIBILITY_EPS)).double()
            bh = (is_cust & (self.demand[node] < -FEASIBILITY_EPS)).double()
            has_lh = torch.zeros(B, R, dtype=torch.float64, device=dev)
            has_bh = torch.zeros(B, R, dtype=torch.float64, device=dev)
            has_lh.scatter_reduce_(1, route_id, lh, "amax", include_self=True)
            has_bh.scatter_reduce_(1, route_id, bh, "amax", include_self=True)
            has_lh = has_lh > 0.5
            has_bh = has_bh > 0.5
            minp = segmin - startC
            maxp = segmax - startC
            initial = torch.where(has_lh | ~has_bh, torch.full_like(minp, self.capacity),
                                  torch.zeros_like(minp))
            excess = torch.maximum(
                torch.maximum(-(initial + minp), initial + maxp - self.capacity),
                torch.zeros_like(minp))
            # ignore route 0 (no positions before the first depot)
            excess[:, 0] = 0.0
            feas &= (excess <= FEASIBILITY_EPS).all(dim=1)
            viol = viol + excess.sum(1)

        if self.has_route_limit:
            # each leg belongs to its SOURCE's route (the return-to-depot leg's
            # source is the last customer, so it counts toward that route -- the
            # destination depot already belongs to the NEXT route)
            leg_rid = route_id[:, :-1]
            rdist = torch.zeros(B, R, dtype=torch.float64, device=dev)
            rdist.scatter_add_(1, leg_rid, torch.where(charged, d, torch.zeros_like(d)))
            over = (rdist - self.route_limit).clamp(min=0)
            over[:, 0] = 0.0
            feas &= (over <= FEASIBILITY_EPS).all(dim=1)
            viol = viol + over.sum(1)

        if self.has_tw:
            # time-warp arrival unrolled: a[i] = S[i] + max_{j<=i}(tw_start[j]-S[j])
            # per route; step[i] = travel_into_i + service[node[i-1]]
            step = torch.zeros(B, L, dtype=torch.float64, device=dev)
            step[:, 1:] = d + self.service[node[:, :-1]]
            step = torch.where(valid, step, torch.zeros_like(step))
            Cstep = torch.cumsum(step, dim=1)
            startC = torch.zeros(B, R, dtype=torch.float64, device=dev)
            startC.scatter_(1, torch.where(is_depot, route_id, torch.zeros_like(route_id)),
                            torch.where(is_depot, Cstep, torch.zeros_like(Cstep)))
            S = Cstep - torch.gather(startC, 1, route_id)
            X = self.tw_start[node] - S
            a = S + self._segmented_cummax(X, route_id)  # arrival (after any wait)
            late = (a - self.tw_end[node]).clamp(min=0)
            # return to this route's OWN depot (multi-depot aware)
            ret = a + self.service[node] + self.dist[node, route_depot]
            late_depot = (ret - self.tw_end[route_depot]).clamp(min=0)
            if self.open_route:
                late_depot = torch.zeros_like(late_depot)
            tw_over = torch.where(is_cust & valid, late + late_depot, torch.zeros_like(late))
            feas &= (tw_over <= FEASIBILITY_EPS).all(dim=1)
            viol = viol + tw_over.sum(1)

        if self.has_backhaul:
            # no linehaul (demand>0) after any backhaul (demand<0) within a route.
            dem = self.demand[node]
            is_lh = is_cust & (dem > FEASIBILITY_EPS)
            is_bh = (is_cust & (dem < -FEASIBILITY_EPS)).double()
            incl = self._segmented_cummax(is_bh, route_id)      # backhaul seen so far
            excl = torch.cat([incl.new_zeros(B, 1), incl[:, :-1]], dim=1)
            route_start = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=dev),
                 route_id[:, 1:] != route_id[:, :-1]], dim=1)
            excl = torch.where(route_start, torch.zeros_like(excl), excl)
            bad = is_lh & (excl > 0.5)
            feas &= ~bad.any(1)
            viol = viol + bad.double().sum(1)

        if self.has_pd:
            # a delivery is feasible only if its paired pickup (node-pd_k) was
            # visited earlier in the SAME route.
            local = node - dc
            is_pick = is_cust & (local < self.pd_k)
            is_deliv = is_cust & (local >= self.pd_k)
            pos = torch.full((B, self.n), -1, dtype=torch.long, device=dev)
            ar = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
            pos[rows.unsqueeze(1).expand(B, L)[valid], node[valid]] = ar[valid]
            pk_node = (node - self.pd_k).clamp(min=0)
            pk_pos = pos[rows.unsqueeze(1).expand(B, L), pk_node]  # [B,L]
            pk_rid = torch.gather(route_id, 1, pk_pos.clamp(min=0))
            ok_pd = (pk_pos >= 0) & (pk_pos < ar) & (pk_rid == route_id)
            bad = is_deliv & ~ok_pd
            feas &= ~bad.any(1)
            viol = viol + bad.double().sum(1)

        if not self.multi_route and not self.open_route:
            # single closed route with implicit trailing depot (pdtsp): add the
            # return leg when the route ends at a customer
            last_node = rt[rows, valid.sum(1) - 1]
            rd_last = route_depot[rows, valid.sum(1) - 1]
            need_close = last_node >= dc
            cost = cost + torch.where(need_close, self.dist[last_node.clamp(min=0), rd_last],
                                      torch.zeros_like(cost))

        # _finalize turns tour distance into the schema objective (distance /
        # prize / distance+penalty) and applies tour_limit / prize_quota.
        return self._finalize(rt, valid, cost, feas, viol, is_cust)

    def evaluate_padded(self, rt, valid):
        """Tensor-in core of evaluate_batch. rt: long [B, L] (pad=-1); valid bool.

        Lets the batched rollout evaluate padded route tensors directly, with no
        per-row Python list conversion.
        """
        B, L = rt.shape
        if self.no_depot:
            return self._evaluate_cycle(rt, valid)
        # Fast fully-vectorized path (no O(L) Python scan) for the affine
        # single-depot family: distance + capacity + route_limit (+ open). TW,
        # backhaul, pickup-delivery and multi-depot keep the exact loop below.
        # Fully-vectorized path for every schema with a depot (single/multi
        # route, single/multi depot, all constraints, distance/prize/penalty
        # objectives). tsp/atsp (no depot) use the vectorized cycle path above.
        return self._evaluate_affine(rt, valid)
        depot_count = self.depot_count
        is_depot = (rt >= 0) & (rt < depot_count)
        is_cust = rt >= depot_count

        cost = torch.zeros(B, dtype=torch.float64, device=self.device)
        feas = torch.ones(B, dtype=torch.bool, device=self.device)
        viol = torch.zeros(B, dtype=torch.float64, device=self.device)

        # structural: starts at depot; last valid node is depot; every customer
        # visited exactly once; no two consecutive depots.
        feas &= is_depot[:, 0]
        last_idx = valid.sum(dim=1) - 1
        last_node = rt[torch.arange(B, device=self.device), last_idx]
        # multi-route representations end each route at a depot; single-route
        # problems (tsp/pdtsp) may omit the trailing depot -> close implicitly
        if self.multi_route:
            feas &= (last_node >= 0) & (last_node < depot_count)
        if self.has_visit_all:
            for cust in range(depot_count, self.n):
                feas &= (rt == cust).sum(dim=1) == 1

        # capacity via the C++ load-profile model (decoder.cpp:3205): per route
        # track cumulative load_delta (=-demand) and its running min/max, plus
        # whether the route has linehaul/backhaul; finalized at each route end.
        cload = torch.zeros(B, dtype=torch.float64, device=self.device)
        cmin = torch.zeros(B, dtype=torch.float64, device=self.device)
        cmax = torch.zeros(B, dtype=torch.float64, device=self.device)
        c_lh = torch.zeros(B, dtype=torch.bool, device=self.device)
        c_bh = torch.zeros(B, dtype=torch.bool, device=self.device)
        time = torch.zeros(B, dtype=torch.float64, device=self.device)
        route_dist = torch.zeros(B, dtype=torch.float64, device=self.device)
        seen_lh = torch.zeros(B, dtype=torch.bool, device=self.device)
        seen_bh = torch.zeros(B, dtype=torch.bool, device=self.device)
        # per-route set of pickups visited so far (reset at depot), for
        # pickup-delivery precedence + same-route
        picked = torch.zeros(B, self.n, dtype=torch.bool, device=self.device)
        prows = torch.arange(B, device=self.device)
        cur = rt[:, 0].clone()
        # the depot each route STARTED from -- multi-depot routes return to their
        # own start depot, so return legs target route_depot, not the next marker
        route_depot = rt[:, 0].clamp(min=0).clone()
        prev_was_depot = torch.ones(B, dtype=torch.bool, device=self.device)

        for t in range(1, L):
            nxt = rt[:, t]
            step = valid[:, t]  # rows that still have a node at position t
            depot_step = step & is_depot[:, t]
            cust_step = step & is_cust[:, t]
            # a step INTO a depot marker is physically the return to this route's
            # OWN start depot (route_depot), not travel to the marker node
            leg_tgt = torch.where(is_depot[:, t], route_depot, nxt.clamp(min=0))
            d = torch.where(
                step, self.dist[cur.clamp(min=0), leg_tgt], torch.zeros_like(cost)
            )
            # open routes don't pay the return-to-depot leg (verified vs C++)
            charged = step & ~(depot_step & self.open_route)
            cost = cost + torch.where(charged, d, torch.zeros_like(cost))

            # no consecutive depots
            feas &= ~(depot_step & prev_was_depot)

            if self.has_backhaul:
                # within a route: no linehaul (demand>0) after any backhaul
                # (demand<0). That is the ONLY backhaul_order rule (decoder.cpp:
                # 105-107 flags lhs.has_backhaul && rhs.has_linehaul); backhaul
                # before linehaul and backhaul-only routes are allowed.
                dem = self.demand[nxt.clamp(min=0)]
                is_lh = cust_step & (dem > FEASIBILITY_EPS)
                is_bh = cust_step & (dem < -FEASIBILITY_EPS)
                bad_bh = is_lh & seen_bh
                feas &= ~bad_bh
                viol = viol + torch.where(bad_bh, torch.ones_like(viol), torch.zeros_like(viol))
                seen_bh = seen_bh | is_bh

            if self.has_pd:
                # pickup c (local index < pd_k) then its delivery c+pd_k, same
                # route; a delivery is feasible only if its pickup was visited
                # earlier in THIS route (covers precedence AND same-route)
                node = nxt.clamp(min=0)
                local = node - depot_count
                is_pick = cust_step & (local < self.pd_k)
                is_deliv = cust_step & (local >= self.pd_k)
                picked[prows, node] = picked[prows, node] | is_pick
                paired = (node - self.pd_k).clamp(min=0)
                bad_pd = is_deliv & ~picked[prows, paired]
                feas &= ~bad_pd
                viol = viol + torch.where(bad_pd, torch.ones_like(viol), torch.zeros_like(viol))

            if self.has_capacity:
                demn = self.demand[nxt.clamp(min=0)]
                ld = torch.where(cust_step, -demn, torch.zeros_like(cload))
                cload = cload + ld
                cmin = torch.minimum(cmin, cload)
                cmax = torch.maximum(cmax, cload)
                c_lh = c_lh | (cust_step & (demn > FEASIBILITY_EPS))
                c_bh = c_bh | (cust_step & (demn < -FEASIBILITY_EPS))

            if self.has_tw:
                arrival = torch.maximum(time + d, self.tw_start[nxt.clamp(min=0)])
                late = (arrival - self.tw_end[nxt.clamp(min=0)]).clamp(min=0)
                # return leg is to this route's own depot (multi-depot aware)
                ret = arrival + self.service[nxt.clamp(min=0)] + self.dist[nxt.clamp(min=0), route_depot]
                late_depot = (ret - self.tw_end[route_depot]).clamp(min=0)
                if self.open_route:  # no return leg, so no return-to-depot deadline
                    late_depot = torch.zeros_like(late_depot)
                tw_over = late + late_depot
                bad = cust_step & (tw_over > FEASIBILITY_EPS)
                feas &= ~bad
                viol = viol + torch.where(cust_step, tw_over, torch.zeros_like(viol))
                time = torch.where(cust_step, arrival + self.service[nxt.clamp(min=0)], time)

            if self.has_route_limit:
                # per-route cumulative distance (incl. return leg) <= limit;
                # route_dist is monotone within a route so an over-limit is
                # caught as soon as it happens, violation booked once at route end
                route_dist = route_dist + torch.where(charged, d, torch.zeros_like(route_dist))
                rl_over = (route_dist - self.route_limit).clamp(min=0)
                feas &= ~(step & (rl_over > FEASIBILITY_EPS))
                viol = viol + torch.where(depot_step, rl_over, torch.zeros_like(viol))

            # depot reset
            if self.has_capacity:
                # finalize the route ending at this depot: initial load is cap
                # unless the route is backhaul-only (then 0), per decoder.cpp
                initial = torch.where(c_lh | ~c_bh, torch.full_like(cload, self.capacity),
                                      torch.zeros_like(cload))
                excess = torch.maximum(
                    torch.maximum(-(initial + cmin), initial + cmax - self.capacity),
                    torch.zeros_like(cload))
                feas &= ~(depot_step & (excess > FEASIBILITY_EPS))
                viol = viol + torch.where(depot_step, excess, torch.zeros_like(viol))
                z = torch.zeros_like(cload)
                cload = torch.where(depot_step, z, cload)
                cmin = torch.where(depot_step, z, cmin)
                cmax = torch.where(depot_step, z, cmax)
                c_lh = torch.where(depot_step, torch.zeros_like(c_lh), c_lh)
                c_bh = torch.where(depot_step, torch.zeros_like(c_bh), c_bh)
            if self.has_tw:
                time = torch.where(depot_step, torch.zeros_like(time), time)
            if self.has_route_limit:
                route_dist = torch.where(depot_step, torch.zeros_like(route_dist), route_dist)
            if self.has_backhaul:
                seen_lh = torch.where(depot_step, torch.zeros_like(seen_lh), seen_lh)
                seen_bh = torch.where(depot_step, torch.zeros_like(seen_bh), seen_bh)
            if self.has_pd:
                picked = torch.where(depot_step.unsqueeze(1), torch.zeros_like(picked), picked)

            # a depot marker starts a new route from that depot
            route_depot = torch.where(depot_step, nxt.clamp(min=0), route_depot)
            cur = torch.where(step, nxt, cur)
            prev_was_depot = torch.where(step, is_depot[:, t], prev_was_depot)

        if not self.multi_route and not self.open_route:
            # single closed route with implicit trailing depot: add the return
            # leg from the last node to the start depot when it ends at a customer
            arangeB = torch.arange(B, device=self.device)
            last_node = rt[arangeB, valid.sum(dim=1) - 1]
            need_close = last_node >= depot_count
            cost = cost + torch.where(
                need_close,
                self.dist[last_node.clamp(min=0), rt[:, 0].clamp(min=0)],
                torch.zeros_like(cost),
            )
        return self._finalize(rt, valid, cost, feas, viol, is_cust)

    def _finalize(self, rt, valid, cost, feas, viol, is_cust):
        """Apply tour_limit / prize_quota and turn the distance total into the
        schema's objective (distance / prize / distance+penalty). `cost` is the
        accumulated tour distance so far."""
        B = rt.shape[0]
        if self.has_tour_limit:  # total tour length <= budget (single route)
            over = (cost - self.tour_limit).clamp(min=0)
            feas &= over <= FEASIBILITY_EPS
            viol = viol + over
        obj = cost
        if self.objective in ("prize", "distance_plus_penalty") or self.has_prize_quota:
            # per-customer visited mask [B, n] (vectorized scatter, no L-loop)
            cust_mask = (valid & (rt >= self.depot_count)).double()
            counts = torch.zeros(B, self.n, dtype=torch.float64, device=self.device)
            counts.scatter_add_(1, rt.clamp(min=0), cust_mask)
            visited = counts > 0
            collected = (visited.to(torch.float64) * self.prize.unsqueeze(0)).sum(1)
            if self.has_prize_quota:
                short = (self.prize_quota - collected).clamp(min=0)
                feas &= short <= FEASIBILITY_EPS
                viol = viol + short
            if self.objective == "prize":
                obj = collected
            elif self.objective == "distance_plus_penalty":
                unvisited = (~visited).to(torch.float64)
                unvisited[:, : self.depot_count] = 0.0
                obj = cost + (unvisited * self.penalty.unsqueeze(0)).sum(1)
        # sentinel for infeasible: +inf if minimizing, -inf if maximizing (prize)
        bad_fill = float("-inf") if self.objective == "prize" else float("inf")
        obj = torch.where(feas, obj, torch.full_like(obj, bad_fill))
        return obj, feas, viol

    def _evaluate_cycle(self, rt, valid):
        """tsp/atsp: single Hamiltonian cycle over all nodes (no depot). Cost =
        consecutive legs + closing leg (last valid -> first). Feasible iff every
        node appears exactly once. Asymmetric handled by direct D[a,b]."""
        B, L = rt.shape
        dev = self.device
        rows = torch.arange(B, device=dev)
        node = rt.clamp(min=0)
        feas = torch.ones(B, dtype=torch.bool, device=dev)
        viol = torch.zeros(B, dtype=torch.float64, device=dev)
        # every node visited exactly once (vectorized bincount)
        counts = torch.zeros(B, self.n, dtype=torch.float64, device=dev)
        counts.scatter_add_(1, node, valid.double())
        feas &= (counts == 1).all(dim=1)
        # consecutive legs + closing leg (last valid -> first), no Python loop
        d = self.dist[node[:, :-1], node[:, 1:]]
        cost = torch.where(valid[:, 1:], d, torch.zeros_like(d)).sum(1)
        last = rt[rows, valid.sum(1) - 1].clamp(min=0)
        cost = cost + self.dist[last, node[:, 0]]
        return self._finalize(rt, valid, cost, feas, viol, rt >= 0)

    def _insertion_candidates(self, partial, m, removed):
        """Materialize every gap-insertion of `removed` into `partial`.

        Returns (cand [B,G,L], cand_valid [B,G,L], gap_valid [B,G]) where gap g
        is "insert `removed` after partial position g". Shared by
        insertion_feasible / insertion_eval so feasibility and cost come from one
        construction.
        """
        B, L = partial.shape
        G = L - 1
        dev = self.device
        ar = torch.arange(L, device=dev)
        g = torch.arange(G, device=dev)
        at = ar.unsqueeze(0) == (g + 1).unsqueeze(1)  # [G, L]
        after = ar.unsqueeze(0) > (g + 1).unsqueeze(1)
        src = torch.where(after, (ar - 1).clamp(min=0).unsqueeze(0), ar.unsqueeze(0))  # [G,L]
        src = src.unsqueeze(0).expand(B, G, L)
        gathered = torch.gather(partial.unsqueeze(1).expand(B, G, L), 2, src)
        cand = torch.where(
            at.unsqueeze(0), removed.view(B, 1, 1).expand(B, G, L), gathered
        )
        cand_valid = (ar.view(1, 1, L) < (m.view(B, 1, 1) + 1)).expand(B, G, L)
        cand = torch.where(cand_valid, cand, torch.full_like(cand, -1))
        gap_valid = g.unsqueeze(0) < (m - 1).unsqueeze(1)  # [B, G]
        return cand, cand_valid, gap_valid

    def insertion_eval(self, partial, m, removed):
        """(cost [B,G], feasible [B,G]) for every gap-insertion of `removed`.

        Materializes each candidate route and scores it with the exact evaluate
        scan, so cost and feasibility are exact for all supported constraints.
        Infeasible / out-of-range gaps get cost=inf.
        """
        B, L = partial.shape
        G = L - 1
        cand, cand_valid, gap_valid = self._insertion_candidates(partial, m, removed)
        cost, feas, _ = self.evaluate_padded(
            cand.reshape(B * G, L), cand_valid.reshape(B * G, L)
        )
        cost = cost.view(B, G)
        feas = feas.view(B, G) & gap_valid
        cost = torch.where(feas, cost, torch.full_like(cost, float("inf")))
        return cost, feas

    def insertion_feasible(self, partial, m, removed):
        """Which gaps yield a feasible full route when reinserting `removed`.

        partial: long [B, L] (pad=-1), a route with one customer removed;
        m: [B] number of valid nodes in each partial; removed: [B] the node to
        reinsert. Returns bool [B, L-1] where entry g means "insert `removed`
        after partial position g keeps the route feasible". This makes the
        recreate action space feasible-by-construction -- the CaR/SRR property
        my free-sampling relocate was missing.
        """
        _, feas = self.insertion_eval(partial, m, removed)
        return feas

    def _partial_after_removal(self, rt, valid, p):
        """Route `rt` with position `p` deleted, order preserved. p: [B] long.

        Returns (partial [B,L] pad=-1, m [B]) where m = valid length - 1.
        """
        B, L = rt.shape
        dev = self.device
        rows = torch.arange(B, device=dev)
        keep = valid.clone()
        keep[rows, p] = False
        m = valid.sum(1) - 1
        dest = keep.cumsum(1) - 1
        partial = torch.full((B, L), -1, dtype=torch.long, device=dev)
        rr = rows.unsqueeze(1).expand(B, L)
        partial[rr[keep], dest[keep]] = rt[keep]
        return partial, m

    def best_relocate(self, rt, valid):
        """Exact best-improving single relocate over the whole batch.

        For every rollout, tries removing each customer and reinserting it in
        every feasible gap, returns the move with the lowest resulting cost. This
        is the best-improvement local-search teacher used to warm-start the
        neural refiner (behaviour cloning), computed entirely in torch. Returns a
        dict with new_rt/new_valid/new_cost and the teacher targets
        rm_pos [B] (removed customer's position in `rt`) and gap [B] (insert-after
        position in the post-removal partial); improved [B] flags rows where a
        strictly-better feasible move exists (else the move is identity).
        """
        rt = rt.long()
        B, L = rt.shape
        dev = self.device
        base_cost, _, _ = self.evaluate_padded(rt, valid)
        is_cust = valid & (rt >= self.depot_count)
        best_cost = base_cost.clone()
        best_p = torch.zeros(B, dtype=torch.long, device=dev)
        best_g = torch.zeros(B, dtype=torch.long, device=dev)
        found = torch.zeros(B, dtype=torch.bool, device=dev)
        # Build every (removal position p) x (gap g) candidate once and evaluate
        # them in a SINGLE batched scan -- looping p with a per-p evaluate is
        # O(L) separate scans and is dominated by kernel-launch overhead.
        G = L - 1
        cands, cand_valids, ok_cols = [], [], []
        for p in range(L):
            removed = rt[:, p]
            partial, m = self._partial_after_removal(rt, valid, torch.full((B,), p, device=dev))
            cand, cand_valid, gap_valid = self._insertion_candidates(partial, m, removed)
            # invalidate gaps for rows where p is not a customer
            gap_valid = gap_valid & is_cust[:, p:p + 1]
            cands.append(cand)
            cand_valids.append(cand_valid)
            ok_cols.append(gap_valid)
        cand = torch.cat(cands, dim=1)          # [B, L*G, L]
        cand_valid = torch.cat(cand_valids, 1)
        gap_ok = torch.cat(ok_cols, dim=1)      # [B, L*G]
        cost, feas, _ = self.evaluate_padded(
            cand.reshape(B * L * G, L), cand_valid.reshape(B * L * G, L)
        )
        cost = cost.view(B, L * G)
        cost = torch.where(feas.view(B, L * G) & gap_ok, cost, torch.full_like(cost, float("inf")))
        flat_best_cost, flat_idx = cost.min(dim=1)
        take = flat_best_cost < best_cost - FEASIBILITY_EPS
        best_cost = torch.where(take, flat_best_cost, best_cost)
        best_p = torch.where(take, flat_idx // G, best_p)
        best_g = torch.where(take, flat_idx % G, best_g)
        found = take
        # build the winning route (identity where no improving move)
        partial, m = self._partial_after_removal(rt, valid, best_p)
        removed = rt[torch.arange(B, device=dev), best_p]
        cand, cand_valid, _ = self._insertion_candidates(partial, m, removed)
        rows = torch.arange(B, device=dev)
        new_rt = torch.where(found.view(B, 1), cand[rows, best_g], rt)
        new_valid = torch.where(found.view(B, 1), cand_valid[rows, best_g], valid)
        new_cost = torch.where(found, best_cost, base_cost)
        return {
            "new_rt": new_rt, "new_valid": new_valid, "new_cost": new_cost,
            "rm_pos": best_p, "gap": best_g, "improved": found,
        }

    def segment_partial_rows(self, rt, valid, p, s):
        """Per-row segment removal: delete [p_row, p_row+s-1] where p is [B].
        Returns (partial [B,L] pad=-1, m [B]=len-s, block [B,s])."""
        B, L = rt.shape
        dev = self.device
        rows = torch.arange(B, device=dev)
        keep = valid.clone()
        for k in range(s):
            keep[rows, (p + k).clamp(max=L - 1)] = False
        m = valid.sum(1) - s
        dest = keep.cumsum(1) - 1
        partial = torch.full((B, L), -1, dtype=torch.long, device=dev)
        rr = rows.unsqueeze(1).expand(B, L)
        partial[rr[keep], dest[keep]] = rt[keep]
        idx = (p.view(B, 1) + torch.arange(s, device=dev).view(1, s)).clamp(max=L - 1)
        block = torch.gather(rt, 1, idx)  # [B, s]
        return partial, m, block

    def best_relocate_local(self, rt, valid, nbr, cur_cost=None):
        """Scalable best-improving relocate: INCREMENTAL Δ-cost, restricted to
        candidate neighbours (nbr [N,K] node ids, -1 pad). O(L*K) not O(L^2), and
        NO full route re-evaluation (never calls evaluate_padded) -> works at
        n=1000. Distance objective + capacity feasibility (same-route moves always
        ok; cross-route checked against the target route load). `cur_cost` [B] is
        the running cost (tracked incrementally); if None, new_cost holds the
        signed delta. Returns the same dict shape as best_relocate."""
        rt = rt.long()
        B, L = rt.shape
        dev = self.device
        rows = torch.arange(B, device=dev)
        D = self.dist
        dc = self.depot_count
        is_cust = valid & (rt >= dc)
        c = rt.clamp(min=0)  # node at each position [B,L]
        prevn = torch.cat([c[:, :1], c[:, :-1]], 1)
        nextn = torch.cat([c[:, 1:], c[:, -1:]], 1)
        remove_gain = D[prevn, c] + D[c, nextn] - D[prevn, nextn]  # [B,L]

        # node -> position in the route
        pos = torch.full((B, self.n), -1, dtype=torch.long, device=dev)
        ar = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        pos[rows.unsqueeze(1).expand(B, L)[valid], c[valid]] = ar[valid]

        # route id + per-route load for capacity
        if self.has_capacity:
            is_depot = valid & (rt < dc)
            route_id = torch.cumsum(is_depot.long(), dim=1)  # [B,L]
            R = int(route_id.max().item()) + 1
            demand_c = self.demand[c]  # [B,L]
            route_load = torch.zeros(B, R, dtype=torch.float64, device=dev)
            route_load.scatter_add_(
                1, route_id, torch.where(is_cust, demand_c, torch.zeros_like(demand_c))
            )

        K = nbr.shape[1]
        cn = nbr[c]  # [B,L,K] candidate-neighbour node ids of the removed node
        cn_pos = torch.where(
            cn >= 0, pos[rows.view(B, 1, 1).expand(B, L, K), cn.clamp(min=0)],
            torch.full_like(cn, -1),
        )
        u = cn.clamp(min=0)
        vpos = (cn_pos.clamp(min=0) + 1).clamp(max=L - 1)
        v = torch.gather(c, 1, vpos.reshape(B, -1)).reshape(B, L, K)
        cc = c.unsqueeze(-1)
        insert_cost = D[u, cc] + D[cc, v] - D[u, v]
        delta = insert_cost - remove_gain.unsqueeze(-1)  # [B,L,K]

        i_idx = ar.unsqueeze(-1)  # removed position
        last_pos = (valid.sum(1) - 1).view(B, 1, 1)  # gap must have a successor
        # removing the SOLE customer of a route (both neighbours are depots)
        # would leave two consecutive depots (empty route) -> forbid.
        sole = ((prevn < dc) & (nextn < dc)).unsqueeze(-1)
        ok = (is_cust.unsqueeze(-1) & (cn >= 0) & (cn_pos >= 0)
              & (cn != cc) & (cn_pos != i_idx) & (cn_pos != (i_idx - 1))
              & (cn_pos < last_pos) & ~sole)
        if self.has_capacity:
            src_route = route_id.unsqueeze(-1)
            tgt_route = torch.gather(
                route_id, 1, cn_pos.clamp(min=0).reshape(B, -1)
            ).reshape(B, L, K)
            tgt_load = torch.gather(route_load, 1, tgt_route.reshape(B, -1)).reshape(B, L, K)
            cross = tgt_route != src_route
            cap_ok = (~cross) | (tgt_load + self.demand[cc] <= self.capacity + FEASIBILITY_EPS)
            ok = ok & cap_ok
        delta = torch.where(ok, delta, torch.full_like(delta, float("inf")))

        flat = delta.reshape(B, L * K)
        best_delta, flat_idx = flat.min(dim=1)
        found = best_delta < -FEASIBILITY_EPS
        rm_pos = torch.where(found, flat_idx // K, torch.zeros_like(flat_idx))
        kk = flat_idx % K
        ins_after = torch.where(  # original position of the neighbour u to insert after
            found, cn_pos[rows, rm_pos, kk], torch.zeros_like(flat_idx)
        )
        # gap in the post-removal partial: positions after rm_pos shift down by 1
        gap = torch.where(ins_after > rm_pos, ins_after - 1, ins_after)

        partial, m = self._partial_after_removal(rt, valid, rm_pos)
        removed = rt[rows, rm_pos]
        cand, cand_valid, _ = self._insertion_candidates(partial, m, removed)
        gi = gap.clamp(0, cand.shape[1] - 1)
        new_rt = torch.where(found.view(B, 1), cand[rows, gi], rt)
        new_valid = torch.where(found.view(B, 1), cand_valid[rows, gi], valid)
        if cur_cost is None:
            new_cost = torch.where(found, best_delta, torch.zeros_like(best_delta))
        else:
            new_cost = torch.where(found, cur_cost + best_delta, cur_cost)
        return {
            "new_rt": new_rt, "new_valid": new_valid, "new_cost": new_cost,
            "rm_pos": rm_pos, "gap": gap, "improved": found,
        }

    def best_relocate_restricted(self, rt, valid, nbr):
        """Best-improving relocate restricted to candidate neighbours but scored
        with the EXACT evaluate scan (so it is correct for EVERY schema: tw,
        route_limit, backhaul, pd, multi-depot, open). Builds only the K-neighbour
        gap candidates per removal (O(L*K) not O(L^2)) -> much faster than the
        all-gaps teacher and no OOM at n=1000. Same dict shape as best_relocate."""
        rt = rt.long()
        B, L = rt.shape
        dev = self.device
        rows = torch.arange(B, device=dev)
        dc = self.depot_count
        is_cust = valid & (rt >= dc)
        c_node = rt.clamp(min=0)
        prevn = torch.cat([c_node[:, :1], c_node[:, :-1]], 1)
        nextn = torch.cat([c_node[:, 1:], c_node[:, -1:]], 1)
        # node -> position
        pos = torch.full((B, self.n), -1, dtype=torch.long, device=dev)
        arL = torch.arange(L, device=dev)
        ar = arL.unsqueeze(0).expand(B, L)
        pos[rows.unsqueeze(1).expand(B, L)[valid], c_node[valid]] = ar[valid]
        K = nbr.shape[1]
        cn = nbr[c_node]                                   # [B,L,K] neighbour ids
        cn_pos = torch.where(
            cn >= 0, pos[rows.view(B, 1, 1).expand(B, L, K), cn.clamp(min=0)],
            torch.full_like(cn, -1),
        )
        ii = arL.view(1, L, 1)
        last_pos = (valid.sum(1) - 1).view(B, 1, 1)
        sole = ((prevn < dc) & (nextn < dc)).unsqueeze(-1)
        ok = (is_cust.unsqueeze(-1) & (cn >= 0) & (cn_pos >= 0) & (cn != c_node.unsqueeze(-1))
              & (cn_pos != ii) & (cn_pos != (ii - 1)) & (cn_pos < last_pos) & ~sole)

        # partial (remove each position i) for the whole batch, vectorized
        src_rm = torch.where(arL.view(1, L) < arL.view(L, 1), arL.view(1, L),
                             (arL.view(1, L) + 1).clamp(max=L - 1))  # [L,L]
        partial_all = rt[:, src_rm]                        # [B,L,L]
        m_all = (valid.sum(1, keepdim=True) - 1)           # [B,1]
        # insertion gap in the partial for neighbour at original position j
        j = cn_pos.clamp(min=0)
        g = torch.where(cn_pos < ii, j, j - 1).clamp(min=0)  # [B,L,K]
        p_ax = arL.view(1, 1, 1, L)
        gk = g.unsqueeze(-1)
        at = p_ax == (gk + 1)
        after = p_ax > (gk + 1)
        srcp = torch.where(after, (p_ax - 1).clamp(min=0), p_ax).expand(B, L, K, L)
        pa = partial_all.unsqueeze(2).expand(B, L, K, L)
        gathered = torch.gather(pa, 3, srcp)
        cand = torch.where(at, c_node.view(B, L, 1, 1).expand(B, L, K, L), gathered)
        cand_valid = p_ax < (m_all.view(B, 1, 1, 1) + 1)
        cand = torch.where(cand_valid, cand, torch.full_like(cand, -1))
        cand_valid = cand_valid.expand(B, L, K, L)

        M = L * K
        cost, feas, _ = self.evaluate_padded(
            cand.reshape(B * M, L), cand_valid.reshape(B * M, L)
        )
        cost = cost.view(B, M)
        cost = torch.where(feas.view(B, M) & ok.reshape(B, M), cost, torch.full_like(cost, float("inf")))
        base_cost, _, _ = self.evaluate_padded(rt, valid)  # current route (B rows)
        flat_best, flat_idx = cost.min(dim=1)
        found = flat_best < base_cost - FEASIBILITY_EPS  # only STRICTLY improving
        rm_pos = torch.where(found, flat_idx // K, torch.zeros_like(flat_idx))
        kk = flat_idx % K
        gap = torch.where(found, g[rows, rm_pos, kk], torch.zeros_like(flat_idx))
        chosen = cand.reshape(B, M, L)[rows, flat_idx]
        chosen_v = cand_valid.reshape(B, M, L)[rows, flat_idx]
        new_rt = torch.where(found.view(B, 1), chosen, rt)
        new_valid = torch.where(found.view(B, 1), chosen_v, valid)
        return {
            "new_rt": new_rt, "new_valid": new_valid,
            "new_cost": torch.where(found, flat_best, base_cost),
            "rm_pos": rm_pos, "gap": gap, "improved": found,
        }

    def _segment_partial(self, rt, valid, p, s):
        """Route with the contiguous position block [p, p+s-1] deleted (uniform
        p/s across batch). Returns (partial [B,L] pad=-1, m [B]=len-s)."""
        B, L = rt.shape
        dev = self.device
        keep = valid.clone()
        keep[:, p:p + s] = False
        m = valid.sum(1) - s
        dest = keep.cumsum(1) - 1
        partial = torch.full((B, L), -1, dtype=torch.long, device=dev)
        rr = torch.arange(B, device=dev).unsqueeze(1).expand(B, L)
        partial[rr[keep], dest[keep]] = rt[keep]
        return partial, m

    def _segment_insertion_candidates(self, partial, m, block):
        """Insert a length-s `block` [B,s] after each gap of `partial`.
        Returns (cand [B,G,L], cand_valid [B,G,L], gap_valid [B,G])."""
        B, L = partial.shape
        s = block.shape[1]
        G = L
        dev = self.device
        ar = torch.arange(L, device=dev)
        g = torch.arange(G, device=dev)
        in_block = (ar.unsqueeze(0) > g.unsqueeze(1)) & (
            ar.unsqueeze(0) <= (g + s).unsqueeze(1)
        )  # [G,L]
        block_idx = (ar.unsqueeze(0) - g.unsqueeze(1) - 1).clamp(0, s - 1)  # [G,L]
        after = ar.unsqueeze(0) > (g + s).unsqueeze(1)
        src = torch.where(after, (ar.unsqueeze(0) - s).clamp(min=0), ar.unsqueeze(0))
        src = src.clamp(0, L - 1).unsqueeze(0).expand(B, G, L)
        gathered = torch.gather(partial.unsqueeze(1).expand(B, G, L), 2, src)
        block_e = torch.gather(
            block.unsqueeze(1).expand(B, G, s), 2, block_idx.unsqueeze(0).expand(B, G, L)
        )
        cand = torch.where(in_block.unsqueeze(0), block_e, gathered)
        cand_valid = (ar.view(1, 1, L) < (m + s).view(B, 1, 1)).expand(B, G, L)
        cand = torch.where(cand_valid, cand, torch.full_like(cand, -1))
        gap_valid = g.unsqueeze(0) < m.unsqueeze(1)  # [B,G]
        return cand, cand_valid, gap_valid

    def _segment_insertion_eval(self, partial, m, block):
        B, L = partial.shape
        G = L
        cand, cand_valid, gap_valid = self._segment_insertion_candidates(partial, m, block)
        cost, feas, _ = self.evaluate_padded(
            cand.reshape(B * G, L), cand_valid.reshape(B * G, L)
        )
        cost = cost.view(B, G)
        feas = feas.view(B, G) & gap_valid
        return torch.where(feas, cost, torch.full_like(cost, float("inf"))), feas

    def best_oropt(self, rt, valid, max_seg=3, allow_reverse=True):
        """Exact best-improving OR-OPT move (segment relocate) over the batch:
        remove a contiguous customer segment of length 1..max_seg and reinsert it
        (optionally reversed) at the best feasible gap. Generalizes best_relocate
        (the seg_len=1 case) to the richer SRR-style neighborhood, still computed
        entirely in torch. Teacher targets: seg_pos [B], seg_len [B], gap [B],
        rev [B]; improved [B]."""
        rt = rt.long()
        B, L = rt.shape
        dev = self.device
        base_cost, _, _ = self.evaluate_padded(rt, valid)
        is_cust = valid & (rt >= self.depot_count)
        best_cost = base_cost.clone()
        best_p = torch.zeros(B, dtype=torch.long, device=dev)
        best_s = torch.ones(B, dtype=torch.long, device=dev)
        best_g = torch.zeros(B, dtype=torch.long, device=dev)
        best_rev = torch.zeros(B, dtype=torch.bool, device=dev)
        found = torch.zeros(B, dtype=torch.bool, device=dev)
        # Batched scoring: build every (segment len s, start p, orientation) x gap
        # candidate and evaluate them in ONE scan (a per-group evaluate is O(S*L)
        # separate scans, dominated by kernel-launch overhead).
        G = L
        cands, cand_valids, gap_oks, metas = [], [], [], []
        for s in range(1, max_seg + 1):
            for p in range(L - s + 1):
                seg_ok = is_cust[:, p:p + s].all(dim=1)  # segment inside one route
                if not bool(seg_ok.any()):
                    continue
                partial, m = self._segment_partial(rt, valid, p, s)
                seg = rt[:, p:p + s]
                for rev in ([False, True] if (allow_reverse and s > 1) else [False]):
                    block = seg.flip(1) if rev else seg
                    cand, cand_valid, gap_valid = self._segment_insertion_candidates(partial, m, block)
                    cands.append(cand)
                    cand_valids.append(cand_valid)
                    gap_oks.append(gap_valid & seg_ok.unsqueeze(1))
                    metas.append((s, p, rev))
        cand = torch.cat(cands, dim=1)             # [B, K*G, L]
        cand_valid = torch.cat(cand_valids, dim=1)
        gap_ok = torch.cat(gap_oks, dim=1)         # [B, K*G]
        KG = cand.shape[1]
        cost, feas, _ = self.evaluate_padded(
            cand.reshape(B * KG, L), cand_valid.reshape(B * KG, L)
        )
        cost = cost.view(B, KG)
        cost = torch.where(feas.view(B, KG) & gap_ok, cost, torch.full_like(cost, float("inf")))
        flat_best, flat_idx = cost.min(dim=1)
        found = flat_best < best_cost - FEASIBILITY_EPS
        best_cost = torch.where(found, flat_best, best_cost)
        grp = flat_idx // G  # which (s,p,rev) group the winning gap belongs to
        best_g = torch.where(found, flat_idx % G, best_g)
        meta_s = torch.tensor([m[0] for m in metas], device=dev, dtype=torch.long)[grp]
        meta_p = torch.tensor([m[1] for m in metas], device=dev, dtype=torch.long)[grp]
        meta_r = torch.tensor([m[2] for m in metas], device=dev, dtype=torch.bool)[grp]
        best_s = torch.where(found, meta_s, best_s)
        best_p = torch.where(found, meta_p, best_p)
        best_rev = torch.where(found, meta_r, best_rev)
        # build winning route per row (segment length varies -> loop the few s)
        new_rt = rt.clone()
        new_valid = valid.clone()
        rows = torch.arange(B, device=dev)
        for s in range(1, max_seg + 1):
            sel = found & (best_s == s)
            if not bool(sel.any()):
                continue
            # rebuild for rows selecting length s at their own best_p/rev/gap
            # (uniform p per iteration is not possible; gather per-row instead)
            for p in range(L - s + 1):
                selp = sel & (best_p == p)
                if not bool(selp.any()):
                    continue
                partial, m = self._segment_partial(rt, valid, p, s)
                seg = rt[:, p:p + s]
                for rev in ([False, True] if (allow_reverse and s > 1) else [False]):
                    selr = selp & (best_rev == rev)
                    if not bool(selr.any()):
                        continue
                    block = seg.flip(1) if rev else seg
                    cand, cand_valid, _ = self._segment_insertion_candidates(partial, m, block)
                    g = best_g.clamp(0, cand.shape[1] - 1)
                    chosen = cand[rows, g]
                    chosen_v = cand_valid[rows, g]
                    new_rt = torch.where(selr.view(B, 1), chosen, new_rt)
                    new_valid = torch.where(selr.view(B, 1), chosen_v, new_valid)
        new_cost = torch.where(found, best_cost, base_cost)
        return {
            "new_rt": new_rt, "new_valid": new_valid, "new_cost": new_cost,
            "seg_pos": best_p, "seg_len": best_s, "gap": best_g, "rev": best_rev,
            "improved": found,
        }

    def node_state(self, route):
        """Per-node dynamic features [N, C] for one route (policy input).

        A torch stand-in for the C++ incumbent_live_state: for each customer,
        the resource state when it is served -- load consumed so far in its trip
        (fraction of capacity) and, with TW, its normalized arrival time. Depot
        and any unvisited node get zeros. This does NOT need C++ parity (it is a
        feature, not the reward); the reward is evaluate_batch, which does.
        """
        cdim = 1 + (1 if self.has_tw else 0)
        state = torch.zeros(self.n, cdim, dtype=torch.float32, device=self.device)
        load = self.capacity
        time = 0.0
        cur = route[0]
        tw_depot = float(self.tw_end[self.depot]) if self.has_tw else 1.0
        for nxt in route[1:]:
            if nxt < 1:  # depot: reset trip
                load = self.capacity
                time = 0.0
                cur = nxt
                continue
            d = float(self.dist[cur, nxt])
            load = load - float(self.demand[nxt])
            col = [max(0.0, 1.0 - load / max(self.capacity, 1e-6))]
            if self.has_tw:
                time = max(time + d, float(self.tw_start[nxt])) + float(self.service[nxt])
                col.append(min(time / max(tw_depot, 1e-6), 1.0))
            state[nxt] = torch.tensor(col, dtype=torch.float32, device=self.device)
            cur = nxt
        return state

    def node_state_batch(self, rt, valid):
        """Batched per-node dynamic features [B, N, C] for routes [B, L].

        Vectorized version of node_state (no Python per-node loop): scans
        positions with the whole batch at once. rt: long [B, L] (pad=-1);
        valid: bool [B, L]. Rows are routes for THIS instance.
        """
        B, L = rt.shape
        dev = self.device
        has_c, has_tw, has_rl = self.has_capacity, self.has_tw, self.has_route_limit
        cdim = max(int(has_c) + int(has_tw) + int(has_rl), 1)
        state = torch.zeros(B, self.n, cdim, dtype=torch.float32, device=dev)
        rows = torch.arange(B, device=dev)
        load = torch.full((B,), self.capacity, dtype=torch.float64, device=dev)
        time = torch.zeros(B, dtype=torch.float64, device=dev)
        route_dist = torch.zeros(B, dtype=torch.float64, device=dev)
        cur = rt[:, 0].clone()
        cap = max(self.capacity, 1e-6)
        tw_depot = float(self.tw_end[self.depot]) if has_tw else 1.0
        rl = max(self.route_limit, 1e-6) if has_rl else 1.0
        for t in range(1, L):
            nxt = rt[:, t]
            step = valid[:, t]
            cust = step & (nxt >= self.depot_count)
            nc = nxt.clamp(min=0)
            d = self.dist[cur.clamp(min=0), nc]
            col = []
            if has_c:
                load = torch.where(cust, load - self.demand[nc], load)
                col.append((1.0 - load / cap).clamp(min=0.0).to(torch.float32))
            if has_tw:
                arrival = torch.maximum(time + d, self.tw_start[nc])
                col.append((arrival / max(tw_depot, 1e-6)).clamp(max=1.0).to(torch.float32))
                time = torch.where(cust, arrival + self.service[nc], time)
            if has_rl:
                route_dist = route_dist + torch.where(step, d, torch.zeros_like(route_dist))
                col.append((route_dist / rl).clamp(max=1.0).to(torch.float32))
            if not col:
                col = [torch.zeros(B, dtype=torch.float32, device=dev)]
            colv = torch.stack(col, dim=1)  # [B, C]
            existing = state[rows, nc]
            state[rows, nc] = torch.where(cust[:, None], colv, existing)
            # depot reset
            if has_c:
                load = torch.where(step & (nxt < self.depot_count), torch.full_like(load, self.capacity), load)
            if has_tw:
                time = torch.where(step & (nxt < self.depot_count), torch.zeros_like(time), time)
            if has_rl:
                route_dist = torch.where(step & (nxt < self.depot_count), torch.zeros_like(route_dist), route_dist)
            cur = torch.where(step, nxt, cur)
        return state
