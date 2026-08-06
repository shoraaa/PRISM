#!/usr/bin/env python3
"""Training hook for the learned remove-and-reinsert refinement operator.

Path A (CaR-faithful): the C++ decoder is used only for construction bootstrap
and as the feasibility+cost oracle (`evaluate`); the refinement *moves* are
emitted by the torch `RefinementDecoder` policy and trained with REINFORCE over
a small step budget, mirroring CaR's ~5-20 improvement steps.

This is deliberately standalone so it can be exercised in isolation before being
folded into train.py's SMDP loop. Wiring note at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from net import RefinementDecoder, build_decoder_data, schema_vector
from route_eval import RouteEvaluator


@dataclass
class RefineStep:
    logp: torch.Tensor  # scalar, grad-carrying
    entropy: torch.Tensor  # scalar, grad-carrying
    reward: float  # normalized improvement realized by this proposed move


def bootstrap_incumbent(decoder):
    """Install a feasible incumbent so incumbent_live_state is populated."""
    # Fields-off solve gives a feasible classical solution to start from.
    best = decoder.solve(1)
    if not best["feasible"]:
        raise RuntimeError("could not bootstrap a feasible incumbent")
    decoder.set_incumbent(best["route"])
    return decoder.best_solution


def _normalized_gain(old_cost, new_cost, scale):
    """Relative improvement; positive means the move helped."""
    return float(old_cost - new_cost) / max(abs(scale), 1e-6)


def _infeasible_reward(decoder, route, penalty, violation_scale):
    """Graded penalty by degree of constraint violation.

    A flat -penalty gives the policy no gradient toward feasibility, so most of
    an untrained rollout is a dead signal. Instead squash the total resource
    violation (from the C++ oracle) into (0, 1]: an almost-feasible route is
    penalized far less than a wildly infeasible one, densifying the reward.
    """
    ev = decoder.evaluate_resources(route)
    viol = float(np.abs(np.asarray(ev["violation"], dtype=np.float64)).sum())
    return -float(penalty) * (1.0 - math.exp(-viol / max(violation_scale, 1e-6)))


@torch.no_grad()
def _live_state(decoder, device):
    n = int(decoder.metadata["node_count"])
    r = int(decoder.metadata["resource_count"])
    flat = torch.as_tensor(
        decoder.incumbent_live_state, dtype=torch.float32, device=device
    )
    return flat.view(n, r)


@torch.no_grad()
def candidate_adjacency(decoder, device):
    """Dense [N, N] bool from the decoder's candidate graph (proximity/field
    neighborhood). Restricting reinsertion to these neighbors is what makes the
    improving-move density usable (~0.6% -> tens of percent)."""
    n = int(decoder.metadata["node_count"])
    ei = torch.as_tensor(
        np.asarray(decoder.edge_index), dtype=torch.long, device=device
    )  # [2, E], row 0 = from, row 1 = to
    adj = torch.zeros(n, n, dtype=torch.bool, device=device)
    adj[ei[0], ei[1]] = True
    return adj


def run_refine_episode(
    decoder,
    refine_model: RefinementDecoder,
    device="cpu",
    improve_steps=10,
    greedy=False,
    infeasible_penalty=1.0,
    violation_scale=1.0,
    walk_penalty=0.0,
    start_route=None,
):
    """One refinement rollout on a single instance (NeuOpt/DACT-style walk).

    The working state *walks*: every feasible proposed move is applied, whether
    or not it improves, so the policy can pass through worse states to reach an
    improving one (a single remove-reinsert improves the bootstrap only ~1% of
    the time -- confining the search to depth-1 from the bootstrap makes it
    unlearnable). Reward is the improvement of the *best-so-far* cost at each
    step; the episode return therefore equals the total improvement over the
    bootstrap, and return-to-go credit spreads it across the trajectory.

    start_route: if given, every rollout starts from this exact feasible route
    instead of re-bootstrapping. The POMO shared baseline needs k rollouts from
    an identical start so their returns are comparable.

    Returns (steps, best_cost). best_cost is the true best found (apples-to-apples
    with the hand-designed SRR).
    """
    depot_count = int(decoder.metadata["depot_count"])
    if start_route is None:
        incumbent = bootstrap_incumbent(decoder)
        start_route = list(incumbent["route"])
        base_cost = incumbent["objective"]
    else:
        start_route = list(start_route)
        base = decoder.evaluate(start_route)
        if not base["feasible"]:
            raise ValueError("start_route must be feasible")
        base_cost = base["objective"]
    scale = max(abs(base_cost), 1e-6)
    best_cost = base_cost
    current_route = list(start_route)
    steps: list[RefineStep] = []

    for _ in range(improve_steps):
        # Refresh live_state for the *current* walking state (set_incumbent keeps
        # the true best internally; it only overwrites it when strictly better).
        decoder.set_incumbent(current_route)
        graph = build_decoder_data(decoder, device=device)
        live = _live_state(decoder, device)
        adj = candidate_adjacency(decoder, device)

        new_route, logp, entropy = refine_model(
            graph, current_route, live, depot_count, greedy=greedy, adj=adj
        )
        candidate = decoder.evaluate(new_route)

        if candidate["feasible"]:
            current_route = list(candidate["route"])  # walk: always move
            new_best = max(best_cost - candidate["objective"], 0.0)
            best_cost = min(best_cost, candidate["objective"])
            # Reward new records; a small walk_penalty can discourage aimless
            # wandering (0 by default -- best-so-far already bounds regret).
            reward = new_best / scale - walk_penalty
        else:
            reward = _infeasible_reward(
                decoder, new_route, infeasible_penalty, violation_scale
            )

        steps.append(RefineStep(logp=logp, entropy=entropy, reward=reward))

    decoder.set_incumbent(current_route)
    return steps, best_cost


def run_refine_group(decoder, refine_model, group_size=8, **episode_kwargs):
    """k walking rollouts from one shared bootstrap start (POMO group).

    Returns (episodes, base_cost, best_cost) where episodes is a list of
    group_size step-lists that share an identical start -- the precondition for
    a per-instance shared baseline.
    """
    incumbent = bootstrap_incumbent(decoder)
    start_route = list(incumbent["route"])
    base_cost = incumbent["objective"]
    episodes, best_cost = [], base_cost
    for _ in range(group_size):
        steps, bc = run_refine_episode(
            decoder, refine_model, start_route=start_route, **episode_kwargs
        )
        episodes.append(steps)
        best_cost = min(best_cost, bc)
    return episodes, base_cost, best_cost


def run_refine_group_fast(
    decoder,
    problem,
    refine_model,
    group_size=8,
    improve_steps=10,
    device="cpu",
    infeasible_penalty=1.0,
    violation_scale=1.0,
):
    """Oracle-free POMO group: no per-step C++ call.

    The static GNN embedding and candidate adjacency are computed ONCE per
    instance (topology is incumbent-independent); every per-step cost -- move
    proposal, reward, and next-step live_state -- is torch via RouteEvaluator.
    This is the batched-training path; `decoder` is used only to bootstrap a
    feasible start and to build the static graph/adjacency once.
    """
    ev = RouteEvaluator(problem, device=device)
    incumbent = bootstrap_incumbent(decoder)
    start_route = list(incumbent["route"])
    base_cost = incumbent["objective"]
    depot_count = int(decoder.metadata["depot_count"])
    scale = max(abs(base_cost), 1e-6)

    graph = build_decoder_data(decoder, device=device)
    node_emb = refine_model.encode_static(graph)  # [N, units], once per instance
    adj = candidate_adjacency(decoder, device)  # static topology

    episodes, best_overall = [], base_cost
    for _ in range(group_size):
        cur = list(start_route)
        best = base_cost
        steps: list[RefineStep] = []
        for _ in range(improve_steps):
            live = ev.node_state(cur)  # [N, C] torch, no C++
            new_route, logp, entropy = refine_model(
                None, cur, live, depot_count, adj=adj, node_emb=node_emb
            )
            cost, feas, viol = ev.evaluate_batch([new_route])
            if bool(feas[0].item()):
                cur = list(new_route)
                new_best = max(best - float(cost[0].item()), 0.0)
                best = min(best, float(cost[0].item()))
                reward = new_best / scale
            else:
                v = float(viol[0].item())
                reward = -infeasible_penalty * (
                    1.0 - math.exp(-v / max(violation_scale, 1e-6))
                )
            steps.append(RefineStep(logp=logp, entropy=entropy, reward=reward))
        episodes.append(steps)
        best_overall = min(best_overall, best)
    return episodes, base_cost, best_overall


def run_batched_group(
    decoder,
    problem,
    model,
    group_size=8,
    improve_steps=10,
    device="cpu",
    infeasible_penalty=1.0,
    violation_scale=1.0,
    greedy=False,
    reward_mode="posdelta",
    accept_mode="improve",
    use_proximity=False,
    tabu_steps=True,
):
    """Fully-batched POMO group: k rollouts stepped together as [k, L] tensors.

    One BatchedRelocate forward proposes k moves; RouteEvaluator scores k routes
    and yields the next k live_states -- no Python per-rollout loop, no per-move
    sync. Static GNN embedding + adjacency computed once. Returns
    (logps[T,k], entropies[T,k], rewards[T,k], base_cost, best_cost).
    """
    ev = RouteEvaluator(problem, device=device)
    incumbent = bootstrap_incumbent(decoder)
    start = list(incumbent["route"])
    base = incumbent["objective"]
    depot_count = int(decoder.metadata["depot_count"])
    scale = max(abs(base), 1e-6)

    graph = build_decoder_data(decoder, device=device)
    node_emb = model.encode_static(graph)  # once per instance
    adj = candidate_adjacency(decoder, device) if use_proximity else None
    schema = schema_vector(problem, device=device)

    k, L = group_size, len(start)
    rt = torch.tensor(start, dtype=torch.long, device=device).unsqueeze(0).repeat(k, 1)
    valid = torch.ones(k, L, dtype=torch.bool, device=device)
    best = torch.full((k,), base, dtype=torch.float64, device=device)
    cur_cost = torch.full((k,), base, dtype=torch.float64, device=device)

    gap_feas_fn = ev.insertion_feasible

    logps, ents, rewards = [], [], []
    tabu = None  # last node removed per rollout -> can't be re-removed next step
    for _ in range(improve_steps):
        live = ev.node_state_batch(rt, valid)
        if getattr(model, "max_seg", 1) > 1:  # OR-OPT (richer neighborhood)
            new_rt, new_valid, logp, ent, removed = model.forward_oropt(
                ev, node_emb, live, rt, valid, depot_count, greedy=greedy,
                schema=schema,
            )
        else:
            new_rt, new_valid, logp, ent, removed = model(
                node_emb, live, rt, valid, adj, depot_count, greedy=greedy,
                gap_feas_fn=gap_feas_fn, tabu_node=tabu if tabu_steps else None,
                schema=schema,
            )
        if tabu_steps:
            tabu = removed
        cost, feas, viol = ev.evaluate_padded(new_rt, new_valid)
        if reward_mode == "posdelta":
            # dense but non-negative: the immediate improvement of a proposed
            # move, floored at 0. Rewarding the SIGNED delta punishes every
            # proposal near a local optimum (all moves worsen), collapsing the
            # policy; CaR sidesteps this by rewarding only best-so-far REDUCTION
            # (>=0, CVRPEnv.py:585). Flooring at 0 keeps the dense early signal
            # without the collapse -- a good state simply earns 0, not a penalty.
            improvement = ((cur_cost - cost) / scale).clamp(min=0.0)
            reward = torch.where(
                feas,
                improvement,
                -infeasible_penalty * (1.0 - torch.exp(-viol / max(violation_scale, 1e-6))),
            )
        elif reward_mode == "delta":
            # dense per-step SIGNED improvement of the accepted move; strong
            # gradient but biased near local optima (see posdelta).
            improvement = torch.where(feas, (cur_cost - cost) / scale, cur_cost.new_zeros(()))
            reward = torch.where(
                feas,
                improvement,
                -infeasible_penalty * (1.0 - torch.exp(-viol / max(violation_scale, 1e-6))),
            )
        else:  # "best": sparse best-so-far improvement
            new_best = torch.clamp(best - cost, min=0.0)
            reward = torch.where(
                feas,
                (new_best / scale),
                -infeasible_penalty * (1.0 - torch.exp(-viol / max(violation_scale, 1e-6))),
            )
        if accept_mode == "improve":
            # hill-climb: only step to strictly-better feasible routes, so the
            # greedy (argmax) policy descends monotonically instead of walking
            # into oscillation. This is how a refiner is deployed.
            accept = feas & (cost < cur_cost - 1e-9)
        else:  # "walk": accept every feasible move (NeuOpt-style navigation)
            accept = feas
        acc = accept.view(k, 1)
        rt = torch.where(acc, new_rt, rt)
        valid = torch.where(acc, new_valid, valid)
        cur_cost = torch.where(accept, cost, cur_cost)
        best = torch.where(feas, torch.minimum(best, cost), best)
        logps.append(logp)
        ents.append(ent)
        rewards.append(reward.to(logp.dtype))
    return (
        torch.stack(logps),
        torch.stack(ents),
        torch.stack(rewards),
        base,
        float(best.min().item()),
    )


@torch.no_grad()
def neural_refine_solve(
    decoder,
    problem,
    model,
    start_route=None,
    group_size=32,
    improve_steps=60,
    device="cpu",
    greedy=False,
):
    """Deploy the neural refiner IN PLACE of hand-designed SRR.

    EXPERIMENTAL: off by default (the default refinement path is C++ SRR); this
    may be removed or evolved in the future.

    The decoder should be built with use_srr=False so the C++ side only
    constructs the incumbent; refinement is done entirely by `model` (batched
    hill-climb with anti-cycling tabu). `start_route` is the already-constructed
    incumbent (the caller owns construction, which may be field-guided);
    if None, falls back to bootstrap_incumbent (classical decoders only).
    Returns a canonical solution dict {route, objective, feasible, direction,
    base_objective} -- the best route is re-checked with the C++ evaluate so
    downstream feasibility/cost match the native oracle exactly.
    """
    from net import schema_vector

    ev = RouteEvaluator(problem, device=device)
    if start_route is None:
        incumbent = bootstrap_incumbent(decoder)
        start = list(incumbent["route"])
    else:
        start = list(start_route)
        decoder.set_incumbent(start)
    base = float(decoder.evaluate(start)["objective"])
    depot_count = int(decoder.metadata["depot_count"])
    graph = build_decoder_data(decoder, device=device)
    node_emb = model.encode_static(graph)
    schema = schema_vector(problem, device=device)
    gap_feas_fn = ev.insertion_feasible

    k, L = group_size, len(start)
    rt = torch.tensor(start, dtype=torch.long, device=device).unsqueeze(0).repeat(k, 1)
    valid = torch.ones(k, L, dtype=torch.bool, device=device)
    cur_cost = torch.full((k,), base, dtype=torch.float64, device=device)
    best_cost = torch.full((k,), base, dtype=torch.float64, device=device)
    best_rt = rt.clone()
    best_valid = valid.clone()
    tabu = None
    for _ in range(improve_steps):
        live = ev.node_state_batch(rt, valid)
        if getattr(model, "max_seg", 1) > 1:  # OR-OPT (richer neighborhood)
            new_rt, new_valid, _, _, removed = model.forward_oropt(
                ev, node_emb, live, rt, valid, depot_count, greedy=greedy,
                schema=schema,
            )
        else:
            new_rt, new_valid, _, _, removed = model(
                node_emb, live, rt, valid, None, depot_count, greedy=greedy,
                gap_feas_fn=gap_feas_fn, tabu_node=tabu, schema=schema,
            )
        tabu = removed
        cost, feas, _ = ev.evaluate_padded(new_rt, new_valid)
        accept = feas & (cost < cur_cost - 1e-9)  # hill-climb
        acc = accept.view(k, 1)
        rt = torch.where(acc, new_rt, rt)
        valid = torch.where(acc, new_valid, valid)
        cur_cost = torch.where(accept, cost, cur_cost)
        improved = feas & (cost < best_cost)
        best_cost = torch.where(improved, cost, best_cost)
        best_rt = torch.where(improved.view(k, 1), new_rt, best_rt)
        best_valid = torch.where(improved.view(k, 1), new_valid, best_valid)

    bi = int(best_cost.argmin().item())
    route = [int(x) for x, v in zip(best_rt[bi].tolist(), best_valid[bi].tolist()) if v]
    checked = decoder.evaluate(route)
    return {
        "route": list(checked["route"]),
        "objective": float(checked["objective"]),
        "feasible": bool(checked["feasible"]),
        "direction": decoder.metadata.get("direction", "minimize"),
        "base_objective": base,
    }


def batched_pomo_loss(logps, ents, rewards, entropy_coef=0.01, loss_mode="traj"):
    """POMO loss for one batched group. rewards/logps/ents: [T, k].

    loss_mode="perstep" (CaR-style, Trainer.py:1637): each action a_t is credited
    with ITS OWN step's reward minus the POMO mean over the k rollouts at that
    step. Low variance, but the per-step baseline is only unbiased when the k
    rollouts share a state -- once they diverge, a rollout in a good (low
    headroom) state is punished for having no improving move, which collapses the
    policy under a dense reward.

    loss_mode="traj" (default): trajectory return R = sum_t r_t, baseline = mean
    over the k rollouts (unbiased -- all k share the bootstrap start), advantage
    A = R - mean(R) applied to the whole trajectory's log-prob. Higher variance
    per action but an unbiased objective, so it does not collapse; with a
    signed-delta walk reward R spreads across rollouts (different walks reach
    different finals), giving it real signal.
    """
    if loss_mode == "perstep":
        baseline = rewards.mean(dim=1, keepdim=True)  # [T, 1] POMO mean per step
        adv = rewards - baseline  # [T, k]
        return -(adv.detach() * logps).mean() - entropy_coef * ents.mean()
    R = rewards.sum(dim=0)  # [k] trajectory return
    adv = R - R.mean()  # [k], baseline is the shared-start POMO mean
    return -(adv.detach() * logps.sum(dim=0)).mean() - entropy_coef * ents.mean()


def refine_pomo_loss(groups, entropy_coef=0.01):
    """POMO shared-baseline REINFORCE (CaR-style).

    groups: list of groups; each group is a list of step-lists sharing one
    instance/start. Advantage is the trajectory return minus the *per-group*
    mean return, so the baseline cancels instance difficulty and leaves only
    "did this rollout beat the average rollout from the same start" -- the
    variance reduction my cross-instance baseline was missing.
    """
    losses = []
    entropies = []
    for episodes in groups:
        logps = torch.stack(
            [torch.stack([s.logp for s in steps]).sum() for steps in episodes]
        )
        returns = torch.tensor(
            [sum(s.reward for s in steps) for steps in episodes],
            dtype=logps.dtype,
            device=logps.device,
        )
        advantage = returns - returns.mean()
        losses.append(-(advantage * logps))
        entropies.extend(s.entropy for steps in episodes for s in steps)

    if not losses:
        return torch.zeros((), requires_grad=True)
    policy_loss = torch.cat(losses).mean()
    entropy = torch.stack(entropies).mean()
    return policy_loss - entropy_coef * entropy


def refine_reinforce_loss(episodes, gamma=1.0, entropy_coef=0.01):
    """REINFORCE with a batch-mean baseline over return-to-go.

    episodes: list of step-lists (one per instance). Returns a scalar loss.
    """
    all_logp = []
    all_entropy = []
    all_return = []
    for steps in episodes:
        future = 0.0
        returns = []
        for step in reversed(steps):
            future = step.reward + gamma * future
            returns.append(future)
        returns.reverse()
        for step, ret in zip(steps, returns):
            all_logp.append(step.logp)
            all_entropy.append(step.entropy)
            all_return.append(ret)

    if not all_logp:
        return torch.zeros((), requires_grad=True)

    logp = torch.stack(all_logp)
    entropy = torch.stack(all_entropy)
    returns = torch.tensor(all_return, dtype=logp.dtype, device=logp.device)
    baseline = returns.mean()
    advantage = returns - baseline
    policy_loss = -(logp * advantage).mean()
    return policy_loss - entropy_coef * entropy.mean()


# ---------------------------------------------------------------------------
# Minimal standalone training loop (demonstration / smoke test).
# ---------------------------------------------------------------------------
def train_refine(
    make_decoder,
    refine_model,
    optimizer,
    device="cpu",
    epochs=100,
    batch_instances=16,
    improve_steps=10,
):
    """`make_decoder()` returns a fresh Decoder for a sampled instance."""
    refine_model.train()
    for epoch in range(epochs):
        episodes = []
        for _ in range(batch_instances):
            decoder = make_decoder()
            steps, _ = run_refine_episode(
                decoder, refine_model, device=device, improve_steps=improve_steps
            )
            episodes.append(steps)
        loss = refine_reinforce_loss(episodes)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refine_model.parameters(), 1.0)
        optimizer.step()
        realized = sum(
            max(s.reward, 0.0) for steps in episodes for s in steps
        ) / max(len(episodes), 1)
        print(f"epoch {epoch:03d}  loss {loss.item():+.4f}  gain/inst {realized:.4f}")


# ---------------------------------------------------------------------------
# Integration into train.py (Path A, phase 2):
#
#   from net import RefinementDecoder
#   from refine import run_refine_episode, refine_reinforce_loss
#
#   # Share the field encoder (CaR unified_encoder):
#   refine_model = RefinementDecoder(units=args.units, rm_num=3,
#                                    emb_net=model.emb_net).to(args.device)
#
#   # Replace (or alternate with) the C++ perturb option: instead of
#   # decoder.sample_traced(**guidance), run one run_refine_episode() per
#   # instance and feed its steps' (logp, reward) into the existing advantage
#   # machinery -- an OptionStep whose "policy" is the refinement decoder rather
#   # than the field-guided C++ candidate sampler.
# ---------------------------------------------------------------------------
