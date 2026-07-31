#!/usr/bin/env python3
"""Event-driven PPO training for the resource-field routing Decoder."""

from __future__ import annotations

import argparse
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
from net import ConstraintFieldNet, build_decoder_data
from problem_data import SavedProblems, VariantCurriculum, generated_problem
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


def replay_decision_logp_from_cpp_batch_trace(
    trace: dict,
    graph,
    output: dict,
    model: ConstraintFieldNet,
    beta: float,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
    return_entropy: bool = False,
):
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
        if return_entropy:
            return empty_logp, empty_ant, decisions, empty_logp
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
    feasibility_risk = output["feasibility_risk"].detach()
    energy = objective + (
        multiplier.unsqueeze(1)
        * torch.clamp_min(pressure * residual + scales * additive, 0.0)
    ).sum(dim=-1)
    energy = energy + float(risk_penalty) * feasibility_risk[global_edge]
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
    chosen_energy = (
        chosen_energy
        + float(risk_penalty) * feasibility_risk[chosen_edge]
    )
    selected_logits = logits[selected]
    selected_valid = valid[selected]
    selected_log_probs = torch.log_softmax(selected_logits, dim=1)
    step_logp = -float(beta) * chosen_energy - torch.logsumexp(
        selected_logits, dim=1
    )
    if return_entropy:
        probabilities = torch.softmax(selected_logits, dim=1)
        safe_log_probs = selected_log_probs.masked_fill(~selected_valid, 0.0)
        entropy_terms = probabilities * safe_log_probs
        decision_entropy = -entropy_terms.sum(dim=1)
        return step_logp, ant_index[selected], decisions, decision_entropy
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
        "edge_risk": output["feasibility_risk"].detach().cpu().numpy(),
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
        "edge_risk": np.zeros(decoder.metadata["edge_count"], dtype=np.float32),
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
            "srr_candidate_limit": getattr(args, "srr_candidate_limit", 0),
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
    # The neutral bootstrap in setup_decoder consumes the first primitive
    # search iteration. Keep --search-iterations as one total budget shared by
    # training, validation, and the classical reference path.
    learned_iterations = max(args.search_iterations - 1, 0)

    while iteration < learned_iterations:
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
        normalized_gain = 0.0
        for _ in range(args.option_max_steps):
            if iteration >= learned_iterations:
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
            )
            steps.append(step)
            option_steps.append(step)
            iteration += 1
            option_duration += 1

            iteration_best = option_incumbent
            for solution in solutions:
                if _better(solution, iteration_best):
                    iteration_best = solution
            normalized_gain = max(
                _gain(option_incumbent, iteration_best, 0.0), 0.0
            )
            if normalized_gain > args.improvement_epsilon:
                decoder.set_incumbent(iteration_best["route"])
                incumbent = iteration_best
                improvements += 1
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
        outcomes.append(
            OptionOutcome(option_steps, terminal_reward, option_duration)
        )

    _assign_smdp_returns(outcomes, args.smdp_gamma, args.device)

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
    target = labels
    logits = output["feasibility_logits"][edges]
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
        "multipliers",
        "binding_logits",
        "coupler_weights",
        "coupler_bias",
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
    class_weights: Optional[dict[str, torch.Tensor]] = None,
    adv_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    zero = output["residual"].sum() * 0.0
    if rl_weight != 0.0:
        logp, decision_ants, decisions, decision_entropy = (
            replay_decision_logp_from_cpp_batch_trace(
                step.trace,
                step.graph,
                output,
                model,
                args.beta,
                field_enabled=step.field_enabled,
                risk_penalty=step.risk_penalty,
                return_entropy=True,
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
        decision_entropy = logp

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
        advantage = ant_advantage[decision_ants]
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
        entropy = (decision_weight * decision_entropy).sum() / normalizer
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

    dual = _dual_loss(step, output)
    feasibility = _feasibility_loss(
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
    auxiliary_scale = 1.0 if rl_weight == 0.0 else args.aux_rl_scale
    loss = (
        rl_weight * rl_loss
        + auxiliary_scale * auxiliary
        - args.entropy_weight * entropy
    )
    with torch.no_grad():
        risk_labels = torch.as_tensor(
            step.trace["feasibility_risk_labels"], device=logp.device
        ).float()
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
            "reward_std": reward_std.detach(),
            "advantage_abs": advantage_abs.detach(),
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
    rl_weight = 0.0 if epoch < args.pretrain_epochs else args.rl_weight
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
                synchronize()
                timing["forward"] += time.perf_counter() - phase_started

                # A graph version can remain unchanged for the entire search,
                # so `group` is not bounded by option_max_steps.  Retaining all
                # decision-replay losses until one backward makes peak memory
                # grow with search_iterations * n_ants.  Backpropagate through
                # the detached output leaves one step at a time instead.  The
                # proxy gradients still sum exactly to the gradient of the
                # original mean loss, and are sent through the GNN once below.
                for step in group:
                    synchronize()
                    phase_started = time.perf_counter()
                    loss, metrics = _step_loss(
                        model,
                        step,
                        output,
                        args,
                        rl_weight,
                        class_weights,
                        adv_scale,
                    )
                    collector.add_dict(metrics)
                    synchronize()
                    timing["forward"] += time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    (loss / len(steps)).backward()
                    synchronize()
                    timing["backward"] += time.perf_counter() - phase_started
                    del loss

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
                    synchronize()
                    phase_started = time.perf_counter()
                    torch.autograd.backward(originals, gradients)
                    synchronize()
                    timing["backward"] += (
                        time.perf_counter() - phase_started
                    )
                del base, output, links, originals, gradients
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
                        class_weights,
                        adv_scale,
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
    improved_count = 0
    progress = tqdm(total=args.steps_per_epoch, desc="Epoch", leave=True)
    while completed < args.steps_per_epoch:
        group = min(args.grad_accum_variants, args.steps_per_epoch - completed)
        rollouts = []
        field_enabled = epoch >= args.pretrain_epochs
        risk_penalty = (
            args.feasibility_risk_penalty if field_enabled else 0.0
        )
        for _ in range(group):
            variant = curriculum.sample(epoch, args.epochs)
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
        # PPO metrics describe this optimizer update over the whole mixed
        # rollout group. Log them once; repeating them on every variant creates
        # a false per-variant attribution in W&B.
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
            global_step += 1
            completed += 1
            improved_count += rollout.improvements
            progress.set_postfix(
                improved_count=improved_count, refresh=False
            )
            progress.update(1)
    progress.close()
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
            "srr_candidate_limit": getattr(args, "srr_candidate_limit", 0),
        },
        n_ants=args.n_ants,
        beta=args.beta,
    )
    decoder.seed(args.seed)
    bootstrap = decoder.solve(1, **_neutral_guidance(decoder))
    if not bootstrap["feasible"]:
        raise RuntimeError(f"validation bootstrap failed: {bootstrap['error']}")
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
    learned_iterations = max(args.search_iterations - 1, 0)
    best = bootstrap
    if learned_iterations == 0:
        return (
            _canonical_cost(best),
            best,
            {
                "emissions": 0.0,
                "time_neural": 0.0,
                "time_decoder": 0.0,
                "net_evals": 0.0,
            },
        )

    start = time.perf_counter()
    guidance = _refresh_guidance()
    net_evals += 1
    neural_seconds += time.perf_counter() - start

    if not dynamic:
        start = time.perf_counter()
        best = decoder.solve(learned_iterations, **guidance)
        decoder_seconds += time.perf_counter() - start
    else:
        # Dynamic heatmap: every incumbent improvement rebuilds the candidate
        # graph (graph_version bumps) and leaves newly introduced edges with an
        # inert default field. Drive the search one iteration at a time and
        # re-run the net on the fresh graph whenever the incumbent moved, so the
        # heatmap always covers the edges the next construction/refinement pass
        # will actually consume.
        version = decoder.graph_version
        start = time.perf_counter()
        best = decoder.solve(1, **guidance)
        decoder_seconds += time.perf_counter() - start
        for _ in range(max(learned_iterations - 1, 0)):
            if decoder.graph_version != version:
                start = time.perf_counter()
                guidance = _refresh_guidance()
                net_evals += 1
                neural_seconds += time.perf_counter() - start
                version = decoder.graph_version
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
) -> tuple[float, float, float]:
    """Lower-is-better rank, led by feasibility then mean baseline gain."""
    return (
        1.0 - float(metrics["feasibility_rate"]),
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
        if capture_classical_baseline:
            item["classical_baseline"] = {
                "objective": float(best["objective"]),
                "direction": best["direction"],
                "feasible": True,
            }
            if (
                item.get("reference_source") == "classical"
                and item["reference"] is None
            ):
                # Generated variants without an external oracle reuse the
                # baseline-pass result instead of executing the decoder twice.
                item["reference"] = float(best["objective"])
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
    macro_gap = (
        float(np.mean([np.mean(values) for values in variant_gaps.values()]))
        if variant_gaps
        else float("inf")
    )
    result["instance_weighted_gap"] = (
        float(np.mean(gaps)) if gaps else float("inf")
    )
    result["macro_gap"] = macro_gap
    result["worst_variant_gap"] = (
        max(float(np.mean(values)) for values in variant_gaps.values())
        if variant_gaps
        else float("inf")
    )
    result["worst_variant_feasibility_rate"] = (
        min(
            variant_feasible.get(variant, 0) / variant_total
            for variant, variant_total in variant_totals.items()
        )
        if variant_totals
        else 0.0
    )
    baseline_variant_means = [
        float(np.mean(values))
        for values in variant_baseline_improvements.values()
    ]
    result["baseline_improvement_instances"] = float(
        len(baseline_improvements)
    )
    result["baseline_improvement_coverage"] = (
        len(baseline_improvements) / total if total else 0.0
    )
    result["instance_weighted_baseline_improvement_percent"] = (
        float(np.mean(baseline_improvements))
        if baseline_improvements
        else float("-inf")
    )
    result["average_baseline_improvement_percent"] = result[
        "instance_weighted_baseline_improvement_percent"
    ]
    result["macro_baseline_improvement_percent"] = (
        float(np.mean(baseline_variant_means))
        if baseline_variant_means
        else float("-inf")
    )
    # Scalar checkpoint score: mean paired improvement over validation
    # instances. Feasibility remains a separate, lexicographically prior gate.
    result["macro_score"] = result[
        "average_baseline_improvement_percent"
    ]
    result["worst_variant_baseline_improvement_percent"] = (
        min(baseline_variant_means)
        if baseline_variant_means
        else float("-inf")
    )
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
    saved = SavedProblems(args.n_node, getattr(args, "dataset_dir", None) or None)
    seen = curriculum.variants[: args.val_seen]
    heldout = curriculum.held_out[: args.val_heldout]
    dataset = []
    logger = get_logger()
    for split, variants in (("seen", seen), ("heldout", heldout)):
        for variant in variants:
            for index in range(args.val_size):
                try:
                    problem, reference = saved.load(variant, index=index)
                except Exception as exc:  # missing data dir/file for this variant
                    logger.warning(
                        f"skipping validation variant {variant} [{index}]: {exc}"
                    )
                    continue
                reference_source = "saved" if reference is not None else "none"
                if variant == "vrptw" and reference is None:
                    # Capacity-free VRPTW is generated deterministically because
                    # the legacy benchmark set has no TW-only multi-route task.
                    # Anchor its validation gap to the same-budget non-neural
                    # decoder captured by the baseline pass.
                    reference_source = "classical"
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
    curriculum: Optional[VariantCurriculum] = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": vars(args),
        "global_step": int(global_step),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if curriculum is not None:
        payload["curriculum_rng_state"] = curriculum.rng.getstate()
    if val_gap is not None:
        payload["val_gap"] = val_gap
    if validation_rank is not None:
        payload["validation_rank"] = tuple(validation_rank)
    if best_validation_rank is not None:
        payload["best_validation_rank"] = tuple(best_validation_rank)
        payload["checkpoint_rank_metric"] = "macro_score"
    torch.save(payload, path)


def restore_training_state(
    checkpoint: dict, curriculum: VariantCurriculum
) -> None:
    """Restore stochastic streams so resume matches uninterrupted training."""
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if "curriculum_rng_state" in checkpoint:
        curriculum.rng.setstate(checkpoint["curriculum_rng_state"])


def load_model_checkpoint_state(
    model: ConstraintFieldNet, state_dict: dict[str, torch.Tensor]
) -> None:
    """Load checkpoints written before stateless graph BatchNorm.

    Only obsolete running-statistic buffers may be discarded; parameter or
    architecture mismatches remain hard errors.
    """
    filtered = {
        name: value
        for name, value in state_dict.items()
        if not name.endswith(
            ("running_mean", "running_var", "num_batches_tracked")
        )
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint architecture mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


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
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--n-ants", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--grad-accum-variants", type=int, default=4)
    parser.add_argument("--search-iterations", type=int, default=16)
    parser.add_argument(
        "--srr-candidate-limit",
        type=int,
        default=0,
        help=(
            "Cap on customer candidates SRR screens per anchor, in learned rank "
            "order (0 = screen all). Small values (e.g. 16-32) trade a little "
            "quality for a large, N-scaling speedup by trusting the field's "
            "ranking to propose the best moves."
        ),
    )
    parser.add_argument(
        "--static-field",
        action="store_true",
        help=(
            "Disable dynamic heatmap recomputation at inference; keep the "
            "single-shot field for the whole search budget."
        ),
    )
    parser.add_argument("--option-max-steps", type=int, default=4)
    parser.add_argument("--improvement-epsilon", type=float, default=0.0)
    parser.add_argument("--smdp-gamma", type=float, default=0.99)
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
        type=int,
        default=0,
        help=(
            "Auxiliary-only warm-up epochs before learned guidance and RL are "
            "enabled (default: 0, disabled)."
        ),
    )
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
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
        "--feasibility-risk-penalty", type=float, default=10.0
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
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Benchmark dataset root. Defaults to PRISM_DATASET_DIR, then the "
            "currently bundled benchmark artifact."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--val-seen",
        type=int,
        default=None,
        help="Number of training variants to validate (default: all)",
    )
    parser.add_argument("--val-heldout", type=int, default=16)
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
    best_validation_rank = (float("inf"),) * 3
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=args.device, weights_only=False
        )
        load_model_checkpoint_state(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(
            checkpoint.get("global_step", start_epoch * args.steps_per_epoch)
        )
        stored_rank = checkpoint.get("best_validation_rank")
        if (
            checkpoint.get("checkpoint_rank_metric")
            in {"average_baseline_improvement_percent", "macro_score"}
            and stored_rank is not None
            and len(stored_rank) == 3
        ):
            best_validation_rank = tuple(
                float(value) for value in stored_rank
            )
        restore_training_state(checkpoint, curriculum)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    validation_data = (
        [] if args.skip_validation else build_validation_data(args, curriculum)
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
                    curriculum=curriculum,
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
            curriculum=curriculum,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
