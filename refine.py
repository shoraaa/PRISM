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

from net import RefinementDecoder, build_decoder_data


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


def run_refine_episode(
    decoder,
    refine_model: RefinementDecoder,
    device="cpu",
    improve_steps=10,
    greedy=False,
    infeasible_penalty=1.0,
    violation_scale=1.0,
):
    """One refinement rollout on a single instance.

    Returns (steps, best_cost). Accepts a move only if the C++ oracle says it is
    feasible and strictly improving -- the same accept rule as the hand-designed
    SRR, so the comparison is apples-to-apples.
    """
    incumbent = bootstrap_incumbent(decoder)
    depot_count = int(decoder.metadata["depot_count"])
    scale = max(abs(incumbent["objective"]), 1e-6)
    best_cost = incumbent["objective"]
    steps: list[RefineStep] = []

    for _ in range(improve_steps):
        graph = build_decoder_data(decoder, device=device)
        route = decoder.best_solution["route"]
        live = _live_state(decoder, device)

        new_route, logp, entropy = refine_model(
            graph, route, live, depot_count, greedy=greedy
        )
        candidate = decoder.evaluate(new_route)

        if candidate["feasible"]:
            gain = _normalized_gain(best_cost, candidate["objective"], scale)
            if gain > 0.0:
                decoder.set_incumbent(candidate["route"])
                best_cost = candidate["objective"]
                reward = gain
            else:
                reward = 0.0  # feasible but no improvement: neutral
        else:
            reward = _infeasible_reward(
                decoder, new_route, infeasible_penalty, violation_scale
            )

        steps.append(RefineStep(logp=logp, entropy=entropy, reward=reward))

    return steps, best_cost


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
