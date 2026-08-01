#!/usr/bin/env python3
"""Event-driven PPO training for the resource-field routing Decoder."""

from __future__ import annotations

import argparse
import copy
import gc
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil
import torch
import wandb
from torch.nn import functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prism_decoder
from net import (
    ConstraintFieldNet,
    build_decoder_data,
    load_constraint_field_state_dict,
)
from problem_data import (
    DEFAULT_DATASET_DIR,
    SavedProblems,
    VALIDATION_HELDOUT_VARIANTS,
    VariantCurriculum,
    generated_problem,
)
from utils import MetricsCollector, get_logger, init_logger


@dataclass
class OptionStep:
    graph: Any
    trace: dict
    old_logp: torch.Tensor
    decisions: torch.Tensor
    rewards: torch.Tensor
    resource_delta: Optional[torch.Tensor]
    binding_target: torch.Tensor
    duration: int
    decision_ants: Optional[torch.Tensor] = None
    field_enabled: bool = True
    risk_penalty: float = 0.0
    search_progress: float = 0.0
    transition_ant: Optional[int] = None
    temporal_advantage: float = 0.0
    old_value: float = 0.0
    value_target: Optional[float] = None


@dataclass
class InstanceRollout:
    variant: str
    steps: list[OptionStep]
    average_cost: float
    best_cost: float
    emissions: int
    improvements: int
    neural_seconds: float
    decoder_seconds: float


@dataclass
class OptionOutcome:
    steps: list[OptionStep]
    reward: torch.Tensor
    duration: int
    transition_reward: float = 0.0
    old_value: float = 0.0
    transition_step: Optional[OptionStep] = None
    winner_ant: Optional[int] = None


def replay_decision_logp_from_cpp_batch_trace(
    trace: dict,
    graph,
    output: dict,
    model: ConstraintFieldNet,
    beta: float,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay one log-probability per stochastic Decoder decision."""
    device = output["residual"].device
    current = torch.as_tensor(trace["current_nodes"], device=device).long()
    starts = torch.as_tensor(trace["starts"], device=device).long()
    stochastic = torch.as_tensor(trace["stochastic"], device=device).bool()
    chosen = torch.as_tensor(trace["chosen_indices"], device=device).long()
    states = torch.as_tensor(trace["live_state"], device=device).float()
    valid_offsets = torch.as_tensor(trace["valid_offsets"], device=device).long()
    valid_indices = torch.as_tensor(trace["valid_indices"], device=device).long()
    n_ants = int(starts.numel() - 1)
    counts = starts[1:] - starts[:-1]
    ant_index = torch.repeat_interleave(
        torch.arange(n_ants, device=device), counts
    )
    selected = stochastic & (chosen >= 0)
    decisions = torch.bincount(
        ant_index[selected], minlength=n_ants
    ).to(torch.int32)
    if current.numel() == 0 or not selected.any():
        empty_logp = output["residual"].new_empty(0)
        empty_ant = torch.empty(0, dtype=torch.long, device=device)
        return empty_logp, empty_ant, decisions

    lengths = valid_offsets[1:] - valid_offsets[:-1]
    maximum = int(lengths.max().item())
    valid = torch.zeros(
        (current.numel(), maximum), dtype=torch.bool, device=device
    )
    local = torch.zeros(
        (current.numel(), maximum), dtype=torch.long, device=device
    )
    decision_index = torch.repeat_interleave(
        torch.arange(current.numel(), device=device), lengths
    )
    rank = torch.arange(valid_indices.numel(), device=device)
    rank -= torch.repeat_interleave(valid_offsets[:-1], lengths)
    local[decision_index, rank] = valid_indices
    valid[decision_index, rank] = True

    edge_offsets = graph.edge_offsets.to(device)
    global_edge = edge_offsets[current].unsqueeze(1) + local
    global_edge = global_edge.clamp_max(output["residual"].shape[0] - 1)
    residual = output["residual"][global_edge]
    additive = output["additive"][global_edge]
    pressure = graph.raw_resource_pressure.to(device)[global_edge]
    scales = graph.resource_scales.to(device)
    objective = graph.objective_edge_costs.to(device)[global_edge]
    multiplier = model.couple(output, states)
    if not field_enabled:
        multiplier = torch.zeros_like(multiplier)
    risk_states = states.unsqueeze(1).expand(
        -1, global_edge.shape[1], -1
    )
    _, feasibility_risk = model.feasibility_for_state(
        output, global_edge, risk_states
    )
    # The classifier stays calibrated by supervised lookahead labels. PPO
    # controls only the separately bounded trust gate.
    feasibility_risk = feasibility_risk.detach()
    risk_gate = model.risk_gate_for_state(output, global_edge, risk_states)
    objective_scale = graph.objective_energy_scale.to(device).reshape(-1)[0]
    risk_energy = (
        float(risk_penalty)
        * objective_scale
        * feasibility_risk
        * risk_gate
    )
    energy = objective + (
        multiplier.unsqueeze(1)
        * torch.clamp_min(pressure * residual + scales * additive, 0.0)
    ).sum(dim=-1)
    energy = energy + risk_energy
    logits = (-float(beta) * energy).masked_fill(~valid, -torch.inf)

    chosen_edge = edge_offsets[current[selected]] + chosen[selected]
    chosen_energy = graph.objective_edge_costs.to(device)[chosen_edge]
    chosen_energy = chosen_energy + (
        multiplier[selected]
        * torch.clamp_min(
            graph.raw_resource_pressure.to(device)[chosen_edge]
            * output["residual"][chosen_edge]
            + scales * output["additive"][chosen_edge],
            0.0,
        )
    ).sum(dim=-1)
    _, chosen_risk = model.feasibility_for_state(
        output, chosen_edge, states[selected]
    )
    chosen_gate = model.risk_gate_for_state(
        output, chosen_edge, states[selected]
    )
    chosen_energy = chosen_energy + (
        float(risk_penalty)
        * objective_scale
        * chosen_risk.detach()
        * chosen_gate
    )
    step_logp = (
        -float(beta) * chosen_energy
        - torch.logsumexp(logits[selected], dim=1)
    )
    return step_logp, ant_index[selected], decisions


def replay_logp_from_cpp_batch_trace(
    trace: dict,
    graph,
    output: dict,
    model: ConstraintFieldNet,
    beta: float,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay summed per-ant log-probabilities for diagnostics."""
    decision_logp, decision_ants, decisions = (
        replay_decision_logp_from_cpp_batch_trace(
            trace,
            graph,
            output,
            model,
            beta,
            field_enabled=field_enabled,
            risk_penalty=risk_penalty,
        )
    )
    logp_sum = output["residual"].new_zeros(decisions.numel())
    logp_sum.scatter_add_(0, decision_ants, decision_logp)
    return logp_sum, decisions


def _guidance_numpy(
    output: dict,
    graph,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
) -> dict:
    multipliers = output["multipliers"][0]
    if not field_enabled:
        multipliers = torch.zeros_like(multipliers)
    return {
        "edge_field": output["residual"].detach().cpu().numpy(),
        "edge_additive": (
            output["additive"] * graph.resource_scales.unsqueeze(0)
        ).detach().cpu().numpy(),
        "multipliers": multipliers.detach().cpu().numpy(),
        "coupler_weights": output["coupler_weights"][0].detach().cpu().numpy(),
        "coupler_bias": output["coupler_bias"][0].detach().cpu().numpy(),
        "edge_risk": output["risk_guidance"].detach().cpu().numpy(),
        "risk_penalty": float(risk_penalty),
    }


def _neutral_guidance(decoder) -> dict:
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    return {
        "edge_field": np.ones(
            (decoder.metadata["edge_count"], channels), dtype=np.float32
        ),
        "edge_additive": np.zeros(
            (decoder.metadata["edge_count"], channels), dtype=np.float32
        ),
        "multipliers": np.zeros(channels, dtype=np.float32),
        "coupler_weights": np.zeros(
            (channels, prism_decoder.LIVE_STATE_FEATURE_COUNT), dtype=np.float32
        ),
        "coupler_bias": np.zeros(channels, dtype=np.float32),
        "edge_risk": np.zeros(
            (
                decoder.metadata["edge_count"],
                2 * (prism_decoder.LIVE_STATE_FEATURE_COUNT + 1),
            ),
            dtype=np.float32,
        ),
        "risk_penalty": 0.0,
    }


def _better(candidate: dict, incumbent: dict) -> bool:
    if not candidate["feasible"]:
        return False
    if not incumbent["feasible"]:
        return True
    if candidate["direction"] == "maximize":
        if abs(candidate["objective"] - incumbent["objective"]) > 1e-6:
            return candidate["objective"] > incumbent["objective"]
        return candidate["distance"] < incumbent["distance"]
    return candidate["objective"] < incumbent["objective"] - 1e-6


def _canonical_cost(solution: dict) -> float:
    value = float(solution["objective"])
    return -value if solution["direction"] == "maximize" else value


def _gain(incumbent: dict, candidate: dict, infeasible_penalty: float) -> float:
    if not candidate["feasible"]:
        return -float(infeasible_penalty)
    scale = max(abs(float(incumbent["objective"])), 1e-6)
    if incumbent["direction"] == "maximize":
        return (candidate["objective"] - incumbent["objective"]) / scale
    return (incumbent["objective"] - candidate["objective"]) / scale


def setup_decoder(
    problem: dict, args: argparse.Namespace, deterministic: bool = False
):
    decoder = prism_decoder.Decoder(
        problem,
        candidate_config={"max_candidates": args.candidates},
        search_config={
            "classical_behavior": False,
            "use_pheromone": False,
            "use_srr": True,
            "feasibility_lookahead_depth": getattr(
                args, "feasibility_lookahead_depth", 2
            ),
        },
        n_ants=args.n_ants,
        beta=args.beta,
    )
    decoder.seed(
        args.seed if deterministic else args.seed + random.randrange(1 << 30)
    )
    if deterministic:
        incumbent = decoder.sample_greedy(**_neutral_guidance(decoder))
        if incumbent["feasible"]:
            decoder.set_incumbent(incumbent["route"])
    else:
        incumbent = decoder.solve(1, **_neutral_guidance(decoder))
    if not incumbent["feasible"]:
        raise RuntimeError(f"bootstrap failed: {incumbent['error']}")
    return decoder, incumbent


def _resource_deltas(decoder, incumbent: dict, solutions: list[dict]) -> np.ndarray:
    base = decoder.evaluate_resources(incumbent["route"])
    base_binding = np.asarray(base["binding"], dtype=np.float32)
    labels = []
    for solution in solutions:
        if not solution["feasible"]:
            labels.append(np.ones_like(base_binding))
            continue
        evaluation = decoder.evaluate_resources(solution["route"])
        binding = np.asarray(evaluation["binding"], dtype=np.float32)
        violation = np.asarray(evaluation["violation"], dtype=np.float32)
        labels.append(np.clip(binding - base_binding + violation, 0.0, 1.0))
    return np.stack(labels)


def _assign_smdp_returns(
    outcomes: list[OptionOutcome], gamma: float, device: str | torch.device
) -> None:
    """Assign finite-horizon option returns G=R+gamma^tau G' in place."""
    future = torch.zeros((), dtype=torch.float32, device=device)
    for outcome in reversed(outcomes):
        option_return = outcome.reward + (gamma ** outcome.duration) * future
        for step in outcome.steps:
            step.rewards = option_return
            step.duration = outcome.duration
        future = option_return.mean()


def _assign_refresh_gae(
    outcomes: list[OptionOutcome], gamma: float, gae_lambda: float
) -> None:
    """Assign winner-gated SMDP advantages and refresh-value targets.

    The decoder transition is produced by exactly one ant.  Continuation
    credit therefore belongs to that ant's transition step rather than being
    broadcast as a constant that POMO centering would remove.
    """
    next_value = 0.0
    next_advantage = 0.0
    for outcome in reversed(outcomes):
        discount = float(gamma) ** outcome.duration
        delta = (
            outcome.transition_reward
            + discount * next_value
            - outcome.old_value
        )
        advantage = delta + discount * float(gae_lambda) * next_advantage
        if outcome.steps:
            value_step = outcome.steps[0]
            value_step.old_value = outcome.old_value
            value_step.value_target = outcome.old_value + advantage
        if (
            outcome.transition_step is not None
            and outcome.winner_ant is not None
        ):
            outcome.transition_step.transition_ant = outcome.winner_ant
            outcome.transition_step.temporal_advantage = advantage
        next_value = outcome.old_value
        next_advantage = advantage


def _winner_temporal_advantage(
    ant_count: int,
    winner_ant: int,
    advantage: float,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a zero-mean POMO contrast that retains winner continuation."""
    if ant_count < 1 or not 0 <= winner_ant < ant_count:
        raise ValueError("winner_ant must index a non-empty ant batch")
    contrast = torch.full(
        (ant_count,), -1.0 / ant_count, dtype=torch.float32, device=device
    )
    contrast[winner_ant] += 1.0
    return contrast * (float(advantage) / max(float(scale), 1e-8))


def collect_instance_rollout(
    model: ConstraintFieldNet,
    problem: dict,
    variant: str,
    args: argparse.Namespace,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
) -> InstanceRollout:
    model.train()
    decoder, incumbent = setup_decoder(problem, args)
    steps: list[OptionStep] = []
    emissions = 0
    improvements = 0
    neural_seconds = 0.0
    decoder_seconds = 0.0
    last_solutions = [incumbent]
    outcomes: list[OptionOutcome] = []
    cached_version = -1
    cached_graph = None
    cached_output = None
    cached_guidance = None
    cached_binding = None
    iteration = 0

    while iteration < args.search_iterations:
        version = int(decoder.graph_version)
        emitted = version != cached_version
        if emitted:
            graph = build_decoder_data(decoder, args.device)
            neural_start = time.perf_counter()
            with torch.no_grad():
                old_output = model(graph)
            neural_seconds += time.perf_counter() - neural_start
            guidance = _guidance_numpy(
                old_output,
                graph,
                field_enabled=field_enabled,
                risk_penalty=risk_penalty,
            )
            binding_target = torch.as_tensor(
                decoder.evaluate_resources(incumbent["route"])["binding"],
                dtype=torch.float32,
                device=args.device,
            )
            cached_version = version
            cached_graph = graph
            cached_output = old_output
            cached_guidance = guidance
            cached_binding = binding_target
            emissions += 1
        else:
            graph = cached_graph
            old_output = cached_output
            guidance = cached_guidance
            binding_target = cached_binding

        option_incumbent = incumbent
        option_steps: list[OptionStep] = []
        option_duration = 0
        option_progress = iteration / max(args.search_iterations, 1)
        with torch.no_grad():
            old_value = float(
                model.value(old_output, option_progress).reshape(-1)[0]
            )
        transition_reward = 0.0
        transition_step = None
        winner_ant = None
        for _ in range(args.option_max_steps):
            if iteration >= args.search_iterations:
                break
            decoder_start = time.perf_counter()
            batch = decoder.sample_traced(**guidance)
            decoder_seconds += time.perf_counter() - decoder_start
            solutions = list(batch["solutions"])
            last_solutions = solutions
            trace = batch["trace"]
            if field_enabled and args.rl_weight != 0.0:
                with torch.no_grad():
                    old_logp, decision_ants, decisions = (
                        replay_decision_logp_from_cpp_batch_trace(
                            trace,
                            graph,
                            old_output,
                            model,
                            args.beta,
                            field_enabled=field_enabled,
                            risk_penalty=risk_penalty,
                        )
                    )
            else:
                old_logp = torch.empty(0, device=args.device)
                decision_ants = torch.empty(
                    0, dtype=torch.long, device=args.device
                )
                decisions = torch.zeros(
                    args.n_ants, dtype=torch.long, device=args.device
                )
            resource_delta = None
            if trace["screened_edges"].size == 0:
                resource_delta = torch.as_tensor(
                    _resource_deltas(decoder, option_incumbent, solutions),
                    dtype=torch.float32,
                    device=args.device,
                )
            step = OptionStep(
                graph=graph,
                trace=trace,
                old_logp=old_logp.detach(),
                decisions=decisions.detach(),
                rewards=torch.zeros(args.n_ants, device=args.device),
                resource_delta=resource_delta,
                binding_target=binding_target,
                duration=0,
                decision_ants=decision_ants.detach(),
                field_enabled=field_enabled,
                risk_penalty=risk_penalty,
                search_progress=option_progress,
            )
            steps.append(step)
            option_steps.append(step)
            iteration += 1
            option_duration += 1

            iteration_best = option_incumbent
            iteration_winner = None
            for ant, solution in enumerate(solutions):
                if _better(solution, iteration_best):
                    iteration_best = solution
                    iteration_winner = ant
            normalized_gain = max(
                _gain(option_incumbent, iteration_best, 0.0), 0.0
            )
            if normalized_gain > args.improvement_epsilon:
                decoder.set_incumbent(iteration_best["route"])
                incumbent = iteration_best
                improvements += 1
                transition_reward = normalized_gain
                transition_step = step
                winner_ant = iteration_winner
                break

        terminal_reward = torch.tensor(
            [
                _gain(option_incumbent, solution, args.infeasible_penalty)
                for solution in last_solutions
            ],
            dtype=torch.float32,
            device=args.device,
        )
        # Bound the per-ant reward so a rare infeasible ant (-infeasible_penalty,
        # ~500x a typical feasible fractional gain) cannot dominate the
        # batch-pooled advantage scale and crush every feasible signal.
        if args.reward_clip > 0.0:
            terminal_reward = terminal_reward.clamp(
                -args.reward_clip, args.reward_clip
            )
        if emitted:
            terminal_reward -= args.neural_call_cost
            transition_reward -= args.neural_call_cost
        outcomes.append(
            OptionOutcome(
                option_steps,
                terminal_reward,
                option_duration,
                transition_reward=transition_reward,
                old_value=old_value,
                transition_step=transition_step,
                winner_ant=winner_ant,
            )
        )

    _assign_smdp_returns(outcomes, args.smdp_gamma, args.device)
    _assign_refresh_gae(
        outcomes,
        args.smdp_gamma,
        getattr(args, "gae_lambda", 1.0),
    )

    costs = [_canonical_cost(solution) for solution in last_solutions]
    return InstanceRollout(
        variant=variant,
        steps=steps,
        average_cost=float(np.mean(costs)),
        best_cost=_canonical_cost(incumbent),
        emissions=emissions,
        improvements=improvements,
        neural_seconds=neural_seconds,
        decoder_seconds=decoder_seconds,
    )


def _decision_ant_index(trace: dict, device: torch.device) -> torch.Tensor:
    starts = torch.as_tensor(trace["starts"], device=device).long()
    return torch.repeat_interleave(
        torch.arange(starts.numel() - 1, device=device), starts[1:] - starts[:-1]
    )


def _positive_class_weight(target: torch.Tensor) -> torch.Tensor:
    positive = (target > 0.5).sum().float()
    negative = target.numel() - positive
    if positive == 0 or negative == 0:
        return target.new_ones(())
    return negative / positive


def _rollout_class_weights(
    steps: list[OptionStep],
) -> dict[str, torch.Tensor]:
    """Estimate rare-event weights over the full mixed-variant PPO batch."""
    device = steps[0].binding_target.device
    binding_targets = []
    feasibility_targets = []
    for step in steps:
        active = step.graph.active_channels.to(device).reshape(-1).bool()
        binding = (step.binding_target.to(device) >= 0.95).float()
        binding_targets.append(binding[active])
        feasibility = torch.as_tensor(
            step.trace["feasibility_risk_labels"], device=device
        ).float()
        if feasibility.numel():
            feasibility_targets.append(feasibility)

    one = torch.ones((), device=device)
    binding_weight = (
        _positive_class_weight(torch.cat(binding_targets))
        if binding_targets and any(target.numel() for target in binding_targets)
        else one
    )
    feasibility_weight = (
        _positive_class_weight(torch.cat(feasibility_targets))
        if feasibility_targets
        else one
    )
    return {
        "binding": binding_weight.clamp_max(100.0).detach(),
        "feasibility": feasibility_weight.clamp_max(100.0).detach(),
    }


def _balanced_regression_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    positive = target > 1e-6
    positive_count = positive.sum().float()
    negative_count = target.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        return elementwise.mean()
    weights = torch.where(
        positive, negative_count / positive_count, target.new_ones(())
    )
    return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)


def _dual_loss(step: OptionStep, output: dict) -> torch.Tensor:
    device = output["residual"].device
    screened_edges = torch.as_tensor(
        step.trace.get("screened_edges", []), device=device
    ).long()
    if screened_edges.numel():
        target = torch.as_tensor(
            step.trace["screened_resource_delta"], device=device
        ).float()
        normalized_pressure = step.graph.edge_attr.to(device)[
            screened_edges, 1 : 1 + prism_decoder.FIELD_CHANNEL_COUNT
        ]
        prediction = torch.clamp_min(
            normalized_pressure * output["residual"][screened_edges]
            + output["additive"][screened_edges],
            0.0,
        )
        active = output["active_channels"][0].bool().expand_as(prediction)
        if active.any():
            return _balanced_regression_loss(
                prediction[active], target[active]
            )
        return prediction.sum() * 0.0

    # Construction-only instances have no SRR labels on their first option.
    # Retain the ant outcome as a lower-resolution supervision fallback.
    current = torch.as_tensor(step.trace["current_nodes"], device=device).long()
    chosen = torch.as_tensor(step.trace["chosen_indices"], device=device).long()
    stochastic = torch.as_tensor(step.trace["stochastic"], device=device).bool()
    if current.numel() == 0 or not stochastic.any():
        return output["residual"].sum() * 0.0
    ant_index = _decision_ant_index(step.trace, device)
    edge = step.graph.edge_offsets.to(device)[current] + chosen
    normalized_pressure = step.graph.edge_attr.to(device)[
        edge, 1 : 1 + prism_decoder.FIELD_CHANNEL_COUNT
    ]
    prediction = torch.clamp_min(
        normalized_pressure * output["residual"][edge]
        + output["additive"][edge],
        0.0,
    )
    if step.resource_delta is None:
        raise RuntimeError("missing fallback resource-delta labels")
    target = step.resource_delta.to(device)[ant_index]
    active = output["active_channels"][0].bool().expand_as(prediction)
    selected = stochastic.unsqueeze(1) & active
    if not selected.any():
        return prediction.sum() * 0.0
    return _balanced_regression_loss(
        prediction[selected], target[selected]
    )


def _feasibility_loss(
    model: ConstraintFieldNet,
    step: OptionStep,
    output: dict,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = output["feasibility_logits"].device
    labels = torch.as_tensor(
        step.trace["feasibility_risk_labels"], device=device
    ).float()
    edges = torch.as_tensor(
        step.trace["feasibility_edges"], device=device
    ).long()
    if labels.numel() == 0:
        return output["feasibility_logits"].sum() * 0.0
    if labels.shape != edges.shape:
        raise RuntimeError("feasibility labels are not aligned with edges")
    offsets = torch.as_tensor(
        step.trace["feasibility_offsets"], device=device
    ).long()
    states = torch.as_tensor(
        step.trace["feasibility_live_state"], device=device
    ).float()
    if offsets.ndim != 1 or offsets.numel() != states.shape[0] + 1:
        raise RuntimeError("feasibility offsets are not aligned with states")
    if offsets.numel() == 0 or offsets[0] != 0 or offsets[-1] != edges.numel():
        raise RuntimeError("feasibility offsets do not partition labels")
    lengths = offsets[1:] - offsets[:-1]
    if torch.any(lengths < 0):
        raise RuntimeError("feasibility offsets must be nondecreasing")
    label_states = torch.repeat_interleave(states, lengths, dim=0)
    target = labels
    logits, _ = model.feasibility_for_state(output, edges, label_states)
    if pos_weight is None:
        pos_weight = _positive_class_weight(target)
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight.to(device)
    )


def _binding_loss(
    step: OptionStep,
    output: dict,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    active = output["active_channels"][0].bool()
    if not active.any():
        return output["binding_logits"].sum() * 0.0
    target = step.binding_target.to(output["binding_logits"].device)
    target = (target >= 0.95).float()
    if pos_weight is None:
        pos_weight = _positive_class_weight(target[active])
    return F.binary_cross_entropy_with_logits(
        output["binding_logits"][0, active],
        target[active],
        pos_weight=pos_weight.to(target.device),
    )


def _price_loss(
    model: ConstraintFieldNet, step: OptionStep, output: dict
) -> torch.Tensor:
    active = output["active_channels"][0].bool()
    if not active.any():
        return output["multipliers"].sum() * 0.0
    binding = step.binding_target.to(output["multipliers"].device)
    base_loss = F.smooth_l1_loss(
        output["multipliers"][0, active], binding[active]
    )
    live_state = torch.as_tensor(
        step.trace["live_state"],
        dtype=torch.float32,
        device=output["multipliers"].device,
    )
    if live_state.numel() == 0:
        return base_loss
    dynamic_target = torch.maximum(live_state, binding.unsqueeze(0))
    coupled = model.couple(output, live_state)
    dynamic_active = active.unsqueeze(0).expand_as(coupled)
    return base_loss + F.smooth_l1_loss(
        coupled[dynamic_active], dynamic_target[dynamic_active]
    )


def _detached_output(
    output: dict,
) -> tuple[dict, list[tuple[torch.Tensor, torch.Tensor]]]:
    detached = dict(output)
    links = []
    for key in (
        "residual",
        "additive",
        "feasibility_logits",
        "feasibility_state_weights",
        "risk_gate_logits",
        "risk_gate_state_weights",
        "multipliers",
        "binding_logits",
        "coupler_weights",
        "coupler_bias",
        "value_context",
    ):
        value = output[key]
        if value.requires_grad:
            proxy = value.detach().requires_grad_(True)
            detached[key] = proxy
            links.append((value, proxy))
    return detached, links


def _step_loss(
    model: ConstraintFieldNet,
    step: OptionStep,
    output: dict,
    args: argparse.Namespace,
    rl_weight: float,
    auxiliary_scale: float,
    class_weights: Optional[dict[str, torch.Tensor]] = None,
    adv_scale: Optional[torch.Tensor] = None,
    temporal_adv_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    zero = output["residual"].sum() * 0.0
    temporal_credit_weight = float(
        getattr(args, "temporal_credit_weight", 0.0)
    )
    temporal_enabled = rl_weight != 0.0 and temporal_credit_weight != 0.0
    if rl_weight != 0.0:
        logp, decision_ants, decisions = (
            replay_decision_logp_from_cpp_batch_trace(
                step.trace,
                step.graph,
                output,
                model,
                args.beta,
                field_enabled=step.field_enabled,
                risk_penalty=step.risk_penalty,
            )
        )
        old_logp = step.old_logp.to(logp.device)
        if old_logp.shape != logp.shape:
            raise RuntimeError(
                "stored and replayed decision log-probs are misaligned"
            )
        if step.decision_ants is not None and not torch.equal(
            step.decision_ants.to(decision_ants.device), decision_ants
        ):
            raise RuntimeError(
                "stored and replayed decision-to-ant maps differ"
            )
    else:
        logp = output["residual"].new_empty(0)
        old_logp = logp
        decision_ants = torch.empty(
            0, dtype=torch.long, device=logp.device
        )
        decisions = torch.zeros(
            step.rewards.shape[0], dtype=torch.long, device=logp.device
        )

    if rl_weight != 0.0 and logp.numel():
        log_ratio = logp - old_logp
        ratio = torch.exp(log_ratio)
        ant_reward = step.rewards.to(logp.device)
        reward_std = ant_reward.std(unbiased=False)
        # POMO baseline: centre per option (all ants share one field, so the
        # ant mean is the correct control variate). _gain already divides by
        # |incumbent objective|, so the centred reward is a scale-invariant
        # fractional improvement -- per-option unit-std normalisation is thus
        # redundant and, when a field's ants land near-identical costs, only
        # amplifies RNG jitter into unit-scale spurious advantages. Normalise
        # by the batch-pooled scale instead so options with genuine improvement
        # variance dominate and near-degenerate options contribute little.
        ant_advantage = ant_reward - ant_reward.mean()
        if not args.no_adv_norm:
            scale = (
                adv_scale
                if adv_scale is not None
                else ant_advantage.std(unbiased=False) + 1e-8
            )
            ant_advantage = ant_advantage / scale
        temporal_ant_advantage = torch.zeros_like(ant_advantage)
        if temporal_enabled and step.transition_ant is not None:
            temporal_ant_advantage = _winner_temporal_advantage(
                ant_advantage.numel(),
                step.transition_ant,
                step.temporal_advantage,
                (
                    float(temporal_adv_scale)
                    if temporal_adv_scale is not None
                    else 1.0
                ),
                ant_advantage.device,
            )
            ant_advantage = (
                ant_advantage
                + temporal_credit_weight * temporal_ant_advantage
            )
        advantage = ant_advantage[decision_ants]
        temporal_advantage = temporal_ant_advantage[decision_ants]
        # Each ant contributes total weight one, independent of trace length.
        decision_weight = decisions[decision_ants].float().reciprocal()
        normalizer = decision_weight.sum().clamp_min(1.0)
        clipped_ratio = torch.clamp(
            ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
        )
        surrogate = torch.minimum(
            ratio * advantage, clipped_ratio * advantage
        )
        # With a single on-policy pass ratio is exactly one. POMO centering
        # then makes the scalar PPO surrogate cancel across ants, although its
        # gradient is nonzero through ratio. This score-function expression
        # has the same first-pass policy gradient but a useful diagnostic
        # value, so log it without changing the optimized PPO objective.
        rl_score_proxy = -(
            decision_weight * advantage * logp
        ).sum() / normalizer
        rl_loss = -(decision_weight * surrogate).sum() / normalizer
        policy_signal = (
            decision_weight * surrogate.abs()
        ).sum() / normalizer
        temporal_policy_signal = (
            decision_weight
            * (temporal_credit_weight * temporal_advantage).abs()
        ).sum() / normalizer
        entropy = -(decision_weight * logp).sum() / normalizer
        approx_kl = (
            decision_weight * (0.5 * log_ratio.square())
        ).sum() / normalizer
        clipped = (
            (ratio > 1 + args.ppo_clip) | (ratio < 1 - args.ppo_clip)
        ).float()
        clip_frac = (decision_weight * clipped).sum() / normalizer
        ratio_mean = (decision_weight * ratio).sum() / normalizer
        log_ratio_abs = (
            decision_weight * log_ratio.abs()
        ).sum() / normalizer
        advantage_abs = (
            decision_weight * advantage.abs()
        ).sum() / normalizer
        temporal_advantage_abs = (
            decision_weight * temporal_advantage.abs()
        ).sum() / normalizer
    else:
        log_ratio = logp
        ratio = logp
        rl_loss = zero
        entropy = zero
        approx_kl = zero
        clip_frac = zero
        ratio_mean = zero
        policy_signal = zero
        log_ratio_abs = zero
        rl_score_proxy = zero
        reward_std = zero
        advantage_abs = zero
        temporal_policy_signal = zero
        temporal_advantage_abs = zero

    critic_loss = zero
    value_prediction = zero
    value_target = zero
    critic_sample = 0.0
    if temporal_enabled and step.value_target is not None:
        value_prediction = model.value(output, step.search_progress).reshape(-1)[
            0
        ]
        value_target = value_prediction.new_tensor(step.value_target)
        critic_loss = F.smooth_l1_loss(value_prediction, value_target)
        critic_sample = 1.0

    dual = _dual_loss(step, output)
    feasibility = _feasibility_loss(
        model,
        step,
        output,
        None if class_weights is None else class_weights["feasibility"],
    )
    binding = _binding_loss(
        step,
        output,
        None if class_weights is None else class_weights["binding"],
    )
    price = _price_loss(model, step, output)
    auxiliary = (
        args.dual_weight * dual
        + args.feasibility_weight * feasibility
        + args.binding_weight * binding
        + args.price_weight * price
    )
    loss = (
        rl_weight * rl_loss
        + auxiliary_scale * auxiliary
        + float(getattr(args, "value_loss_weight", 0.0)) * critic_loss
        - args.entropy_weight * entropy
    )
    with torch.no_grad():
        risk_labels = torch.as_tensor(
            step.trace["feasibility_risk_labels"], device=logp.device
        ).float()
        risk_edges = torch.as_tensor(
            step.trace["feasibility_edges"], device=logp.device
        ).long()
        risk_offsets = torch.as_tensor(
            step.trace["feasibility_offsets"], device=logp.device
        ).long()
        risk_states = torch.as_tensor(
            step.trace["feasibility_live_state"], device=logp.device
        ).float()
        if risk_labels.numel():
            risk_lengths = risk_offsets[1:] - risk_offsets[:-1]
            label_states = torch.repeat_interleave(
                risk_states, risk_lengths, dim=0
            )
            _, observed_risk = model.feasibility_for_state(
                output, risk_edges, label_states
            )
            observed_gate = model.risk_gate_for_state(
                output, risk_edges, label_states
            )
            risk_energy_cap = (
                float(step.risk_penalty)
                * step.graph.objective_energy_scale.to(logp.device)
                .reshape(-1)[0]
            )
            observed_risk_energy = (
                risk_energy_cap * observed_risk * observed_gate
            )
        else:
            observed_risk = risk_labels
            observed_gate = risk_labels
            observed_risk_energy = risk_labels
            risk_energy_cap = 0.0
        screening_fast = float(
            step.trace.get("screening_fast_evaluations", 0)
        )
        screening_fallback = float(
            step.trace.get("screening_fallback_evaluations", 0)
        )
        screening_total = screening_fast + screening_fallback
        metrics = {
            "loss": loss.detach(),
            "rl_loss": rl_loss.detach(),
            "ppo_surrogate_loss": rl_loss.detach(),
            "rl_score_proxy": rl_score_proxy.detach(),
            "policy_signal": policy_signal.detach(),
            "temporal_policy_signal": temporal_policy_signal.detach(),
            "reward_std": reward_std.detach(),
            "advantage_abs": advantage_abs.detach(),
            "temporal_advantage_abs": temporal_advantage_abs.detach(),
            "critic_loss": critic_loss.detach(),
            "critic_sample": critic_sample,
            "value_prediction": value_prediction.detach(),
            "value_target": value_target.detach(),
            "dual_loss": dual.detach(),
            "feasibility_loss": feasibility.detach(),
            "binding_loss": binding.detach(),
            "price_loss": price.detach(),
            "auxiliary_loss": auxiliary.detach(),
            "auxiliary_scale": float(auxiliary_scale),
            "feasibility_labels": float(risk_labels.numel()),
            "feasibility_positive_rate": (
                risk_labels.mean().detach() if risk_labels.numel() else 0.0
            ),
            "feasibility_risk_mean": (
                observed_risk.mean().detach()
                if observed_risk.numel()
                else 0.0
            ),
            "risk_gate_mean": (
                observed_gate.mean().detach()
                if observed_gate.numel()
                else 0.0
            ),
            "risk_gate_closed_fraction": (
                (observed_gate < 0.1).float().mean().detach()
                if observed_gate.numel()
                else 0.0
            ),
            "risk_gate_open_fraction": (
                (observed_gate > 0.9).float().mean().detach()
                if observed_gate.numel()
                else 0.0
            ),
            "risk_energy_cap": risk_energy_cap,
            "risk_energy_mean": (
                observed_risk_energy.mean().detach()
                if observed_risk_energy.numel()
                else 0.0
            ),
            "risk_energy_max": (
                observed_risk_energy.max().detach()
                if observed_risk_energy.numel()
                else 0.0
            ),
            "screening_fast_evaluations": screening_fast,
            "screening_fallback_evaluations": screening_fallback,
            "screening_fast_fraction": (
                screening_fast / screening_total if screening_total else 0.0
            ),
            "entropy": entropy.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_frac": clip_frac.detach(),
            "ratio_mean": ratio_mean.detach(),
            "log_ratio_abs": log_ratio_abs.detach(),
            "decisions": decisions.float().mean().detach(),
        }
    return loss, metrics


def ppo_update(
    model: ConstraintFieldNet,
    optimizer: torch.optim.Optimizer,
    rollouts: list[InstanceRollout],
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    steps = [step for rollout in rollouts for step in rollout.steps]
    if not steps:
        return {}
    option_groups: list[list[OptionStep]] = []
    for step in steps:
        if not option_groups or option_groups[-1][0].graph is not step.graph:
            option_groups.append([])
        option_groups[-1].append(step)
    collector = MetricsCollector()
    pretraining = epoch < args.pretrain_epochs
    rl_weight = 0.0 if pretraining else args.rl_weight
    auxiliary_scale = (
        args.pretrain_aux_scale if pretraining else args.aux_rl_scale
    )
    class_weights = _rollout_class_weights(steps)
    # Batch-pooled advantage scale: centre each option, pool the residuals over
    # the whole mixed-variant PPO batch, and normalise by that single std. This
    # keeps degenerate low-variance options from being amplified to unit scale
    # (see _step_loss) while retaining a stable, cross-variant step size.
    adv_scale = None
    if not args.no_adv_norm:
        pooled = torch.cat(
            [step.rewards - step.rewards.mean() for step in steps]
        )
        adv_scale = (pooled.std(unbiased=False) + 1e-8).detach()
    temporal_advantages = [
        step.temporal_advantage
        for step in steps
        if step.transition_ant is not None
    ]
    temporal_adv_scale = None
    if temporal_advantages and not args.no_adv_norm:
        temporal_values = torch.as_tensor(
            temporal_advantages,
            dtype=torch.float32,
            device=steps[0].binding_target.device,
        )
        temporal_adv_scale = (
            temporal_values.square().mean().sqrt().clamp_min(1e-8)
        )
    timing = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    gradient_norms = []
    update_started = time.perf_counter()

    def synchronize() -> None:
        if not getattr(args, "profile_timing", False):
            return
        parameter = next(model.parameters())
        if parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)

    for _ in range(args.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        if args.smallvram:
            for group in option_groups:
                synchronize()
                phase_started = time.perf_counter()
                base = model(group[0].graph)
                output, links = _detached_output(base)
                group_losses = []
                for step in group:
                    loss, metrics = _step_loss(
                        model,
                        step,
                        output,
                        args,
                        rl_weight,
                        auxiliary_scale,
                        class_weights,
                        adv_scale,
                        temporal_adv_scale,
                    )
                    group_losses.append(loss)
                    collector.add_dict(metrics)
                synchronize()
                timing["forward"] += time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                (torch.stack(group_losses).sum() / len(steps)).backward()
                originals = [
                    original
                    for original, proxy in links
                    if proxy.grad is not None
                ]
                gradients = [
                    proxy.grad
                    for original, proxy in links
                    if proxy.grad is not None
                ]
                if originals:
                    torch.autograd.backward(originals, gradients)
                synchronize()
                timing["backward"] += time.perf_counter() - phase_started
                del base, output, links, group_losses
        else:
            synchronize()
            phase_started = time.perf_counter()
            losses = []
            for group in option_groups:
                output = model(group[0].graph)
                for step in group:
                    loss, metrics = _step_loss(
                        model,
                        step,
                        output,
                        args,
                        rl_weight,
                        auxiliary_scale,
                        class_weights,
                        adv_scale,
                        temporal_adv_scale,
                    )
                    losses.append(loss)
                    collector.add_dict(metrics)
            synchronize()
            timing["forward"] += time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            torch.stack(losses).mean().backward()
            synchronize()
            timing["backward"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        gradient_norms.append(float(gradient_norm))
        optimizer.step()
        synchronize()
        timing["optimizer"] += time.perf_counter() - phase_started

    result = collector.get_all_means()
    value_steps = [step for step in steps if step.value_target is not None]
    critic_explained_variance = 0.0
    if len(value_steps) > 1:
        targets = np.asarray(
            [step.value_target for step in value_steps], dtype=np.float64
        )
        predictions = np.asarray(
            [step.old_value for step in value_steps], dtype=np.float64
        )
        target_variance = float(np.var(targets))
        if target_variance > 1e-12:
            critic_explained_variance = 1.0 - float(
                np.var(targets - predictions) / target_variance
            )
    result.update(
        ppo_seconds=time.perf_counter() - update_started,
        ppo_forward_seconds=timing["forward"],
        ppo_backward_seconds=timing["backward"],
        ppo_optimizer_seconds=timing["optimizer"],
        ppo_reuse_passes=float(args.ppo_epochs),
        ppo_clipping_active=float(args.ppo_epochs > 1),
        binding_pos_weight=float(class_weights["binding"]),
        feasibility_pos_weight=float(class_weights["feasibility"]),
        gradient_norm=float(np.mean(gradient_norms)),
        advantage_scale=float(adv_scale) if adv_scale is not None else 0.0,
        temporal_advantage_scale=(
            float(temporal_adv_scale)
            if temporal_adv_scale is not None
            else 0.0
        ),
        temporal_transitions=float(len(temporal_advantages)),
        critic_explained_variance=critic_explained_variance,
    )
    return result


def train_instance_ppo(
    model: ConstraintFieldNet,
    optimizer: torch.optim.Optimizer,
    problem: dict,
    variant: str,
    args: argparse.Namespace,
    epoch: int = 0,
) -> tuple[float, float, dict[str, float]]:
    """Retained single-instance entry point, now backed by one SMDP rollout."""
    field_enabled = epoch >= args.pretrain_epochs
    rollout = collect_instance_rollout(
        model,
        problem,
        variant,
        args,
        field_enabled=field_enabled,
        risk_penalty=(
            args.feasibility_risk_penalty if field_enabled else 0.0
        ),
    )
    metrics = ppo_update(model, optimizer, [rollout], args, epoch)
    metrics.update(
        emissions=float(rollout.emissions),
        improvements=float(rollout.improvements),
        time_neural=rollout.neural_seconds,
        time_decoder=rollout.decoder_seconds,
    )
    return rollout.average_cost, rollout.best_cost, metrics


def train_epoch(
    model: ConstraintFieldNet,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    args: argparse.Namespace,
    curriculum: VariantCurriculum,
) -> tuple[int, float, float, float, float, dict[str, float]]:
    """Accumulate mixed-variant rollouts before each optimizer update."""
    logger = get_logger()
    costs = []
    neural_seconds = 0.0
    decoder_seconds = 0.0
    generation_seconds = 0.0
    rollout_seconds = 0.0
    ppo_seconds = 0.0
    ppo_forward_seconds = 0.0
    ppo_backward_seconds = 0.0
    ppo_optimizer_seconds = 0.0
    started = time.perf_counter()
    completed = 0
    variant_counts: dict[str, int] = {}
    variant_schedule = curriculum.schedule(
        epoch,
        args.epochs,
        args.steps_per_epoch,
        args.grad_accum_variants,
    )
    progress = tqdm(total=args.steps_per_epoch, desc="Epoch", leave=True)
    while completed < args.steps_per_epoch:
        group = min(args.grad_accum_variants, args.steps_per_epoch - completed)
        rollouts = []
        field_enabled = epoch >= args.pretrain_epochs
        risk_penalty = (
            args.feasibility_risk_penalty if field_enabled else 0.0
        )
        for variant in variant_schedule[completed : completed + group]:
            phase_started = time.perf_counter()
            problem = generated_problem(variant, args.n_node, args.capacity)
            generation_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            rollout = collect_instance_rollout(
                model,
                problem,
                variant,
                args,
                field_enabled=field_enabled,
                risk_penalty=risk_penalty,
            )
            rollout_seconds += time.perf_counter() - phase_started
            rollouts.append(rollout)
            costs.append(rollout.average_cost)
            neural_seconds += rollout.neural_seconds
            decoder_seconds += rollout.decoder_seconds
        metrics = ppo_update(model, optimizer, rollouts, args, epoch)
        ppo_seconds += metrics.get("ppo_seconds", 0.0)
        ppo_forward_seconds += metrics.get("ppo_forward_seconds", 0.0)
        ppo_backward_seconds += metrics.get("ppo_backward_seconds", 0.0)
        ppo_optimizer_seconds += metrics.get("ppo_optimizer_seconds", 0.0)
        # These metrics describe the complete mixed-variant optimizer update.
        # Log them once rather than falsely attributing the same values to each
        # rollout in the group.
        logger.log_metrics(
            {
                name: value
                for name, value in metrics.items()
                if name
                not in {
                    "auxiliary_scale",
                    "ppo_reuse_passes",
                    "ppo_clipping_active",
                }
            },
            prefix="train/update/",
            step=global_step,
        )
        for rollout in rollouts:
            logger.set_step(global_step)
            row = dict(
                emissions=rollout.emissions,
                improvements=rollout.improvements,
            )
            logger.debug(f"training variant={rollout.variant}")
            logger.log_train_step(
                rollout.variant,
                rollout.average_cost,
                rollout.best_cost,
                epoch,
                row,
                global_step,
            )
            variant_counts[rollout.variant] = (
                variant_counts.get(rollout.variant, 0) + 1
            )
            global_step += 1
            completed += 1
            progress.update(1)
    progress.close()
    logger.log_metrics(
        {
            f"variants/{variant}/count": count
            for variant, count in sorted(variant_counts.items())
        },
        prefix="train_epoch/",
        step=global_step,
    )
    epoch_seconds = time.perf_counter() - started
    rollout_other_seconds = max(
        rollout_seconds - neural_seconds - decoder_seconds, 0.0
    )
    unaccounted_seconds = max(
        epoch_seconds - generation_seconds - rollout_seconds - ppo_seconds,
        0.0,
    )
    return (
        global_step,
        float(np.mean(costs)),
        neural_seconds,
        decoder_seconds,
        epoch_seconds,
        {
            "generation_epoch": generation_seconds,
            "neural_epoch": neural_seconds,
            "decoder_epoch": decoder_seconds,
            "rollout_other_epoch": rollout_other_seconds,
            "ppo_epoch": ppo_seconds,
            "ppo_forward_epoch": ppo_forward_seconds,
            "ppo_backward_epoch": ppo_backward_seconds,
            "ppo_optimizer_epoch": ppo_optimizer_seconds,
            "unaccounted_epoch": unaccounted_seconds,
        },
    )


@torch.no_grad()
def infer_instance(
    model: Optional[ConstraintFieldNet],
    problem: dict,
    args: argparse.Namespace,
) -> tuple[float, dict, dict[str, float]]:
    # Inference must use the decoder's real stochastic n-ant ILS (solve), the
    # same search regime training optimises. The old deterministic single-ant
    # greedy loop stalls at the first local optimum -- its gap is flat from a
    # handful of iterations and ~2-10x worse than solve -- so it made the field
    # look untrainable (no field can move a stalled greedy descent). solve is
    # near-optimal (~0-3%) and the field's contribution is visible there.
    if args.search_iterations < 1:
        raise ValueError("search_iterations must be positive")
    if model is None:
        decoder = prism_decoder.Decoder(
            problem,
            candidate_config={"max_candidates": args.candidates},
            search_config={"use_pheromone": False},
            n_ants=args.n_ants,
            beta=args.beta,
        )
        decoder.seed(args.seed)
        best = decoder.solve(args.search_iterations)
        return _canonical_cost(best), best, {"emissions": 0.0}

    decoder = prism_decoder.Decoder(
        problem,
        candidate_config={"max_candidates": args.candidates},
        search_config={
            "classical_behavior": False,
            "use_pheromone": False,
            "use_srr": True,
            "feasibility_lookahead_depth": getattr(
                args, "feasibility_lookahead_depth", 2
            ),
        },
        n_ants=args.n_ants,
        beta=args.beta,
    )
    decoder.seed(args.seed)
    model.eval()
    dynamic = not getattr(args, "static_field", False)
    risk_penalty = args.feasibility_risk_penalty

    def _refresh_guidance() -> dict:
        graph = build_decoder_data(decoder, args.device)
        output = model(graph)
        return _guidance_numpy(output, graph, risk_penalty=risk_penalty)

    neural_seconds = 0.0
    decoder_seconds = 0.0
    net_evals = 0

    start = time.perf_counter()
    guidance = _refresh_guidance()
    net_evals += 1
    neural_seconds += time.perf_counter() - start

    if not dynamic:
        # Frozen-field ablation and compatibility path: preserve the original
        # one-shot solve exactly, including its random-number consumption.
        start = time.perf_counter()
        best = decoder.solve(args.search_iterations, **guidance)
        decoder_seconds += time.perf_counter() - start
    else:
        # An incumbent improvement rebuilds the candidate graph and bumps its
        # version. Advance one search iteration at a time so newly introduced
        # edges receive guidance before the next construction/refinement pass.
        version = int(decoder.graph_version)
        start = time.perf_counter()
        best = decoder.solve(1, **guidance)
        decoder_seconds += time.perf_counter() - start
        for _ in range(max(args.search_iterations - 1, 0)):
            if int(decoder.graph_version) != version:
                start = time.perf_counter()
                guidance = _refresh_guidance()
                net_evals += 1
                neural_seconds += time.perf_counter() - start
                version = int(decoder.graph_version)
            start = time.perf_counter()
            best = decoder.solve(1, **guidance)
            decoder_seconds += time.perf_counter() - start
    return (
        _canonical_cost(best),
        best,
        {
            "emissions": float(net_evals),
            "time_neural": neural_seconds,
            "time_decoder": decoder_seconds,
            "net_evals": float(net_evals),
        },
    )


def _gap(solution: dict, reference: float) -> float:
    if solution["direction"] == "maximize":
        return (reference - solution["objective"]) / max(abs(reference), 1e-9) * 100
    return (solution["objective"] - reference) / max(abs(reference), 1e-9) * 100


def _validation_rank(
    metrics: dict[str, float], macro_gap: float
) -> tuple[float, float, float, float, float]:
    """Lower-is-better feasibility, coverage, then variant-macro quality."""
    return (
        1.0 - float(metrics["worst_variant_feasibility_rate"]),
        1.0 - float(metrics["feasibility_rate"]),
        1.0 - float(metrics["baseline_improvement_coverage"]),
        -float(metrics["macro_score"]),
        float(macro_gap),
    )


def validation(
    model: Optional[ConstraintFieldNet],
    dataset: list[dict],
    args: argparse.Namespace,
    *,
    capture_classical_baseline: bool = False,
) -> tuple[float, float, float, dict[str, float]]:
    collector = MetricsCollector()
    average_costs = []
    best_costs = []
    gaps = []
    split_gaps: dict[str, list[float]] = {"seen": [], "heldout": []}
    variant_gaps: dict[str, list[float]] = {}
    baseline_improvements = []
    split_baseline_improvements: dict[str, list[float]] = {
        "seen": [],
        "heldout": [],
    }
    variant_baseline_improvements: dict[str, list[float]] = {}
    split_totals: dict[str, int] = {"seen": 0, "heldout": 0}
    split_feasible: dict[str, int] = {"seen": 0, "heldout": 0}
    variant_totals: dict[str, int] = {}
    variant_feasible: dict[str, int] = {}
    variant_objectives: dict[str, list[float]] = {}
    variant_best_costs: dict[str, list[float]] = {}
    for item in tqdm(dataset, desc="Validating", leave=False):
        average, best, metrics = infer_instance(model, item["problem"], args)
        collector.add_dict(metrics)
        split = item["split"]
        variant = item["variant"]
        split_totals[split] = split_totals.get(split, 0) + 1
        variant_totals[variant] = variant_totals.get(variant, 0) + 1

        best_cost = _canonical_cost(best)
        feasible = bool(best["feasible"]) and np.isfinite(best_cost)
        if capture_classical_baseline:
            item["classical_baseline"] = {
                "objective": float(best["objective"]),
                "direction": best["direction"],
                "feasible": feasible,
            }
            if item.get("reference_source") == "classical" and feasible:
                item["reference"] = float(best["objective"])
        if not feasible:
            continue

        average_costs.append(average)
        best_costs.append(best_cost)
        split_feasible[split] = split_feasible.get(split, 0) + 1
        variant_feasible[variant] = variant_feasible.get(variant, 0) + 1
        variant_objectives.setdefault(variant, []).append(
            float(best["objective"])
        )
        variant_best_costs.setdefault(variant, []).append(best_cost)
        baseline = item.get("classical_baseline")
        if baseline is not None and baseline.get("feasible", False):
            baseline_objective = float(baseline["objective"])
            if np.isfinite(baseline_objective):
                improvement = -_gap(best, baseline_objective)
                baseline_improvements.append(improvement)
                split_baseline_improvements[split].append(improvement)
                variant_baseline_improvements.setdefault(variant, []).append(
                    improvement
                )
        if item["reference"] is not None:
            gap = _gap(best, item["reference"])
            gaps.append(gap)
            split_gaps[split].append(gap)
            variant_gaps.setdefault(variant, []).append(gap)
    result = collector.get_all_means()
    total = len(dataset)
    feasible_total = len(best_costs)
    result.update(
        instances=float(total),
        variants=float(len(variant_totals)),
        feasible_instances=float(feasible_total),
        feasibility_rate=(feasible_total / total if total else 0.0),
        gap_instances=float(len(gaps)),
        gap_coverage=(len(gaps) / total if total else 0.0),
    )
    for split, split_total in split_totals.items():
        feasible_count = split_feasible.get(split, 0)
        result[f"{split}_instances"] = float(split_total)
        result[f"{split}_feasible_instances"] = float(feasible_count)
        result[f"{split}_feasibility_rate"] = (
            feasible_count / split_total if split_total else 0.0
        )
        values = split_gaps[split]
        result[f"{split}_gap_instances"] = float(len(values))
        result[f"{split}_gap_coverage"] = (
            len(values) / split_total if split_total else 0.0
        )
        if values:
            result[f"{split}_gap"] = float(np.mean(values))
        baseline_values = split_baseline_improvements[split]
        result[f"{split}_baseline_improvement_instances"] = float(
            len(baseline_values)
        )
        result[f"{split}_baseline_improvement_coverage"] = (
            len(baseline_values) / split_total if split_total else 0.0
        )
        if baseline_values:
            result[f"{split}_baseline_improvement_percent"] = float(
                np.mean(baseline_values)
            )
    for variant, variant_total in variant_totals.items():
        feasible_count = variant_feasible.get(variant, 0)
        prefix = f"variants/{variant}"
        result[f"{prefix}/instances"] = float(variant_total)
        result[f"{prefix}/feasible_instances"] = float(feasible_count)
        result[f"{prefix}/feasibility_rate"] = feasible_count / variant_total
        if variant in variant_objectives:
            result[f"{prefix}/objective"] = float(
                np.mean(variant_objectives[variant])
            )
            result[f"{prefix}/best_cost"] = float(
                np.mean(variant_best_costs[variant])
            )
        if variant in variant_gaps:
            result[f"{prefix}/gap_instances"] = float(
                len(variant_gaps[variant])
            )
            result[f"{prefix}/gap"] = float(np.mean(variant_gaps[variant]))
        if variant in variant_baseline_improvements:
            values = variant_baseline_improvements[variant]
            result[f"{prefix}/baseline_improvement_instances"] = float(
                len(values)
            )
            result[f"{prefix}/baseline_improvement_percent"] = float(
                np.mean(values)
            )

    variant_gap_means = [
        float(np.mean(values)) for values in variant_gaps.values()
    ]
    baseline_variant_means = [
        float(np.mean(values))
        for values in variant_baseline_improvements.values()
    ]
    macro_gap = (
        float(np.mean(variant_gap_means))
        if variant_gap_means
        else float("inf")
    )
    result.update(
        instance_weighted_gap=(
            float(np.mean(gaps)) if gaps else float("inf")
        ),
        macro_gap=macro_gap,
        worst_variant_gap=(
            max(variant_gap_means) if variant_gap_means else float("inf")
        ),
        worst_variant_feasibility_rate=(
            min(
                variant_feasible.get(variant, 0) / variant_total
                for variant, variant_total in variant_totals.items()
            )
            if variant_totals
            else 0.0
        ),
        baseline_improvement_instances=float(len(baseline_improvements)),
        baseline_improvement_coverage=(
            len(baseline_improvements) / total if total else 0.0
        ),
        instance_weighted_baseline_improvement_percent=(
            float(np.mean(baseline_improvements))
            if baseline_improvements
            else float("-inf")
        ),
        macro_baseline_improvement_percent=(
            float(np.mean(baseline_variant_means))
            if baseline_variant_means
            else float("-inf")
        ),
        worst_variant_baseline_improvement_percent=(
            min(baseline_variant_means)
            if baseline_variant_means
            else float("-inf")
        ),
    )
    # Checkpoint quality is a true variant macro; feasibility and paired
    # baseline coverage are separate lexicographically prior gates.
    result["macro_score"] = result[
        "macro_baseline_improvement_percent"
    ]
    return (
        float(np.mean(average_costs)) if average_costs else float("inf"),
        float(np.mean(best_costs)) if best_costs else float("inf"),
        macro_gap,
        result,
    )


def build_validation_data(
    args: argparse.Namespace, curriculum: VariantCurriculum
) -> list[dict]:
    if args.val_size < 1:
        raise ValueError("val_size must be at least one instance per problem")
    if args.val_seen is not None and args.val_seen < 0:
        raise ValueError("val_seen must be nonnegative or None")
    if args.val_heldout < 0:
        raise ValueError("val_heldout must be nonnegative")
    saved = SavedProblems(args.n_node, args.dataset_dir)
    seen = (
        curriculum.variants
        if args.val_seen is None
        else curriculum.variants[: args.val_seen]
    )
    heldout_set = set(curriculum.held_out)
    heldout_candidates = [
        variant
        for variant in VALIDATION_HELDOUT_VARIANTS
        if variant in heldout_set
    ]
    heldout_candidates.extend(
        variant
        for variant in curriculum.held_out
        if variant not in set(heldout_candidates)
    )
    heldout = heldout_candidates[: args.val_heldout]
    dataset = []
    logger = get_logger()
    missing = []
    for split, variants in (("seen", seen), ("heldout", heldout)):
        for variant in variants:
            for index in range(args.val_size):
                try:
                    problem, reference = saved.load(variant, index=index)
                except Exception as exc:  # missing data dir/file for this variant
                    missing.append((variant, index, str(exc)))
                    continue
                reference_source = "saved" if reference is not None else "classical"
                dataset.append(
                    {
                        "variant": variant,
                        "instance_index": index,
                        "split": split,
                        "problem": problem,
                        "reference": reference,
                        "reference_source": reference_source,
                    }
                )
    if missing:
        details = "; ".join(
            f"{variant}[{index}]: {error}"
            for variant, index, error in missing[:5]
        )
        message = (
            f"missing {len(missing)} validation instances from the fixed "
            f"manifest ({details})"
        )
        if not getattr(args, "allow_missing_validation", False):
            raise RuntimeError(message)
        logger.warning(message)
    return dataset


def save_checkpoint(
    model: ConstraintFieldNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    path: Path,
    val_gap: Optional[float] = None,
    *,
    validation_rank: Optional[tuple[float, ...]] = None,
    best_validation_rank: Optional[tuple[float, ...]] = None,
    global_step: int = 0,
    validation_manifest: Optional[tuple[tuple[str, int, str], ...]] = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": vars(args),
        "global_step": int(global_step),
    }
    if val_gap is not None:
        payload["val_gap"] = val_gap
    if validation_rank is not None:
        payload["validation_rank"] = tuple(validation_rank)
    if best_validation_rank is not None:
        payload["best_validation_rank"] = tuple(best_validation_rank)
        payload["checkpoint_rank_metric"] = "variant_macro_score"
    if validation_manifest is not None:
        payload["validation_manifest"] = validation_manifest
    torch.save(payload, path)


def _load_optimizer_state_compat(
    optimizer: torch.optim.Optimizer, state_dict: dict
) -> int:
    """Load optimizer state while initializing newly appended parameters.

    The refresh value head is appended after all legacy model parameters, so
    their optimizer slots retain the same order.  Returns the number of new
    parameters initialized without saved optimizer state.
    """
    current = optimizer.state_dict()
    saved = copy.deepcopy(state_dict)
    if len(saved["param_groups"]) != len(current["param_groups"]):
        raise ValueError("optimizer checkpoint has a different group count")
    added = 0
    for saved_group, current_group in zip(
        saved["param_groups"], current["param_groups"]
    ):
        saved_count = len(saved_group["params"])
        current_count = len(current_group["params"])
        if saved_count > current_count:
            raise ValueError("optimizer checkpoint has more parameters")
        added += current_count - saved_count
        saved_group["params"] = current_group["params"]
    optimizer.load_state_dict(saved)
    return added


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the event-driven resource-field routing Decoder"
    )
    parser.add_argument("--n-node", type=int, default=100)
    parser.add_argument(
        "--val-size",
        type=int,
        default=8,
        help="Number of saved instances to validate for each selected problem",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Benchmark dataset root. Defaults to PRISM_DATASET_DIR, then "
            "baselines/URS/dataset"
        ),
    )
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--n-ants", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--grad-accum-variants", type=int, default=4)
    parser.add_argument("--search-iterations", type=int, default=16)
    parser.add_argument(
        "--static-field",
        action="store_true",
        help=(
            "Disable inference-time field refinement and keep one frozen "
            "field for the complete search budget"
        ),
    )
    parser.add_argument("--option-max-steps", type=int, default=4)
    parser.add_argument("--improvement-epsilon", type=float, default=0.0)
    parser.add_argument("--smdp-gamma", type=float, default=0.99)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=1.0,
        help=(
            "SMDP trace parameter for refresh-level temporal credit; the "
            "default 1 uses the complete Monte Carlo reward-to-go"
        ),
    )
    parser.add_argument(
        "--temporal-credit-weight",
        type=float,
        default=0.1,
        help=(
            "Weight of winner-gated refresh continuation in the PPO advantage; "
            "0 restores local POMO credit only"
        ),
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the optional refresh-state critic loss; the default 0 "
            "keeps temporal credit critic-free"
        ),
    )
    parser.add_argument("--neural-call-cost", type=float, default=0.0)
    parser.add_argument("--infeasible-penalty", type=float, default=10.0)
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=1.0,
        help=(
            "Clip per-ant reward magnitude before advantage normalisation so "
            "rare infeasible ants cannot dominate the pooled scale; 0 disables"
        ),
    )
    parser.add_argument(
        "--pretrain-epochs",
        "--pretraining-epochs",
        dest="pretrain_epochs",
        type=int,
        default=3,
        help=(
            "Number of auxiliary-only epochs before PPO starts; use 0 to "
            "disable the pretraining phase (default: 3)"
        ),
    )
    parser.add_argument(
        "--pretrain-lr",
        "--pretraining-lr",
        dest="pretrain_lr",
        type=float,
        default=1e-4,
        help="Optimizer learning rate during auxiliary pretraining",
    )
    parser.add_argument(
        "--pretrain-aux-scale",
        "--pretraining-aux-scale",
        dest="pretrain_aux_scale",
        type=float,
        default=1.0,
        help="Scale applied to auxiliary losses during pretraining",
    )
    parser.add_argument("--rl-weight", type=float, default=1.0)
    parser.add_argument(
        "--aux-rl-scale",
        type=float,
        default=0.1,
        help="Auxiliary-loss scale after PPO fine-tuning starts",
    )
    parser.add_argument("--dual-weight", type=float, default=1.0)
    parser.add_argument("--feasibility-weight", type=float, default=1.0)
    parser.add_argument("--binding-weight", type=float, default=0.25)
    parser.add_argument(
        "--price-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the multiplier->binding-indicator supervision. Default 0:"
            " pinning multipliers to the binding (feasibility) target injects"
            " harmful ranking distortion (the penalty prices nothing in the"
            " objective-gated SRR), measured net-negative on distance variants."
            " Let RL shape the multipliers from search progress instead."
        ),
    )
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help=(
            "Full-graph PPO passes per rollout. With the scalable one-pass "
            "default, pre-update KL and the centered surrogate value are "
            "expected near zero; monitor policy_signal and rl_score_proxy."
        ),
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Synchronize accelerator phases for exact timing metrics",
    )
    parser.add_argument("--ppo-clip", type=float, default=0.1)
    parser.add_argument("--no-adv-norm", action="store_true")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument(
        "--feasibility-lookahead-depth", type=int, default=2
    )
    parser.add_argument(
        "--feasibility-risk-penalty",
        type=float,
        default=1.0,
        help=(
            "Maximum state-conditioned feasibility-risk contribution in "
            "objective-scale units (default: 1.0)"
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--smallvram",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use detached graph outputs; enabled by default on HIP/ROCm",
    )
    parser.add_argument(
        "--grad-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recompute GNN activations; enabled by default from n=1000",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--val-seen",
        type=int,
        default=None,
        help="Number of training variants to validate (default: all)",
    )
    parser.add_argument(
        "--val-heldout",
        type=int,
        default=16,
        help="Number of variants from the fixed stratified held-out manifest",
    )
    parser.add_argument(
        "--allow-missing-validation",
        action="store_true",
        help="Warn and continue instead of requiring the complete manifest",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("pretrained"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="prism-decoder")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--run-name")
    return parser.parse_args()


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.smdp_gamma <= 1.0:
        raise ValueError("smdp_gamma must lie in [0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1]")
    if args.temporal_credit_weight < 0.0:
        raise ValueError("temporal_credit_weight must be nonnegative")
    if args.value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be nonnegative")
    if args.smallvram is None:
        args.smallvram = (
            torch.version.hip is not None
            and str(args.device).startswith(("cuda", "hip"))
        )
    if args.grad_checkpointing is None:
        args.grad_checkpointing = args.n_node >= 1000
    setup_seeds(args.seed)
    args.threads = args.threads or psutil.cpu_count(logical=True) or 1
    prism_decoder.set_num_threads(args.threads)
    logger = init_logger(
        use_wandb=not args.no_wandb,
        log_dir=args.save_dir / "logs",
        verbose=True,
    )
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    curriculum = VariantCurriculum.default(args.seed)
    model = ConstraintFieldNet(
        grad_checkpointing=args.grad_checkpointing
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=args.device, weights_only=False
        )
        upgraded = load_constraint_field_state_dict(
            model, checkpoint["model_state_dict"]
        )
        added_optimizer_parameters = _load_optimizer_state_compat(
            optimizer, checkpoint["optimizer_state_dict"]
        )
        if upgraded:
            logger.warning(
                "resumed a legacy checkpoint; initialized missing value/risk "
                "heads at neutral defaults"
            )
        if added_optimizer_parameters:
            logger.warning(
                "initialized optimizer state for "
                f"{added_optimizer_parameters} newly appended parameters"
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(
            checkpoint.get("global_step", start_epoch * args.steps_per_epoch)
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    validation_data = (
        [] if args.skip_validation else build_validation_data(args, curriculum)
    )
    validation_manifest = tuple(
        (
            item["variant"],
            int(item["instance_index"]),
            item["split"],
        )
        for item in validation_data
    )
    best_validation_rank = (float("inf"),) * 5
    if checkpoint is not None:
        stored_rank = checkpoint.get("best_validation_rank")
        if (
            checkpoint.get("checkpoint_rank_metric")
            == "variant_macro_score"
            and checkpoint.get("validation_manifest") == validation_manifest
            and stored_rank is not None
            and len(stored_rank) == 5
        ):
            best_validation_rank = tuple(float(value) for value in stored_rank)
        elif stored_rank is not None:
            logger.warning(
                "resetting best validation rank because the checkpoint used "
                "a different metric or validation manifest"
            )
    if validation_data:
        (
            _baseline_average,
            _baseline_best,
            baseline_gap,
            _baseline_metrics,
        ) = validation(
            None,
            validation_data,
            args,
            capture_classical_baseline=True,
        )
        logger.log_baseline(baseline_gap)
    for epoch in range(start_epoch, args.epochs):
        phase_lr = args.pretrain_lr if epoch < args.pretrain_epochs else args.lr
        for group in optimizer.param_groups:
            group["lr"] = phase_lr
        (
            global_step,
            train_cost,
            _neural_time,
            _decoder_time,
            epoch_time,
            epoch_timing,
        ) = train_epoch(model, optimizer, global_step, epoch, args, curriculum)
        val_best = 0.0
        val_gap = None
        val_metrics = {}
        validation_rank = None
        if validation_data:
            val_average, val_best, val_gap, val_metrics = validation(
                model, validation_data, args
            )
            logger.log_validation(
                val_average,
                val_best,
                val_gap,
                epoch,
                val_metrics,
                timing=epoch_timing,
                step=global_step,
            )
            validation_rank = _validation_rank(val_metrics, val_gap)
            if validation_rank < best_validation_rank:
                best_validation_rank = validation_rank
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    args,
                    args.save_dir / "best.pt",
                    val_gap,
                    validation_rank=validation_rank,
                    best_validation_rank=best_validation_rank,
                    global_step=global_step,
                    validation_manifest=validation_manifest,
                )
        logger.log_epoch_summary(
            epoch,
            train_cost,
            val_best,
            val_gap,
            val_metrics.get("feasibility_rate") if validation_data else None,
        )
        epoch_timing["epoch_seconds"] = epoch_time
        if args.profile_timing:
            logger.info(
                "Profile "
                + " ".join(
                    f"{name}={seconds:.4f}s"
                    for name, seconds in epoch_timing.items()
                )
            )
        logger.log_metrics(epoch_timing, prefix="time/", step=global_step)
        save_checkpoint(
            model,
            optimizer,
            epoch,
            args,
            args.save_dir / "last.pt",
            val_gap,
            validation_rank=validation_rank,
            best_validation_rank=best_validation_rank,
            global_step=global_step,
            validation_manifest=validation_manifest,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
