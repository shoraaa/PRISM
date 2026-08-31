#!/usr/bin/env python3
"""Event-driven PPO training for the resource-field routing Decoder."""

from __future__ import annotations

import argparse
import copy
import gc
import math
import random
import sys
import time
from contextlib import contextmanager, nullcontext
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
    MODEL_SCHEMA,
    build_decoder_data,
    load_constraint_field_state_dict,
)
from problem_data import (
    ALL_VARIANTS,
    DEFAULT_DATASET_DIR,
    SavedProblems,
    TRAIN_VARIANTS,
    VariantCurriculum,
    channel_balanced_weights,
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
    decision_rollouts: Optional[torch.Tensor] = None
    field_enabled: bool = True
    risk_penalty: float = 0.0
    search_progress: float = 0.0
    transition_rollout: Optional[int] = None
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
    winner_rollout: Optional[int] = None


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
    n_rollouts = int(starts.numel() - 1)
    counts = starts[1:] - starts[:-1]
    rollout_index = torch.repeat_interleave(
        torch.arange(n_rollouts, device=device), counts
    )
    selected = stochastic & (chosen >= 0)
    decisions = torch.bincount(
        rollout_index[selected], minlength=n_rollouts
    ).to(torch.int32)
    if current.numel() == 0 or not selected.any():
        empty_logp = output["residual"].new_empty(0)
        empty_rollout = torch.empty(0, dtype=torch.long, device=device)
        return empty_logp, empty_rollout, decisions

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
    objective_energy_scale = graph.objective_energy_scale.to(device).reshape(-1)
    if objective_energy_scale.numel() != 1:
        raise ValueError("PPO trace replay expects exactly one decoder graph")
    objective = (
        graph.objective_edge_costs.to(device)[global_edge]
        / objective_energy_scale[0]
    )
    channels = output["active_channels"].shape[-1]
    multiplier = model.couple(output, states)
    field_multiplier = multiplier[:, :channels]
    objective_weight = multiplier[:, channels]
    if not field_enabled:
        field_multiplier = torch.zeros_like(field_multiplier)
        objective_weight = torch.ones_like(objective_weight)
    feasibility_risk = output["feasibility_risk"].detach()
    # Match the native dimensionless, signed, zero-neutral energy contract.
    field_term = residual + additive
    objective_residual = output["objective_residual"][global_edge]
    if not field_enabled:
        objective_residual = torch.zeros_like(objective_residual)
    energy = objective_weight.unsqueeze(1) * (
        objective + objective_residual
    ) + (
        field_multiplier.unsqueeze(1) * field_term
    ).sum(dim=-1)
    energy = energy + float(risk_penalty) * feasibility_risk[global_edge]
    logits = (-float(beta) * energy).masked_fill(~valid, -torch.inf)

    chosen_edge = edge_offsets[current[selected]] + chosen[selected]
    chosen_field = (
        output["residual"][chosen_edge] + output["additive"][chosen_edge]
    )
    chosen_objective_residual = output["objective_residual"][chosen_edge]
    if not field_enabled:
        chosen_objective_residual = torch.zeros_like(chosen_objective_residual)
    chosen_energy = objective_weight[selected] * (
        graph.objective_edge_costs.to(device)[chosen_edge]
        / objective_energy_scale[0]
        + chosen_objective_residual
    ) + (field_multiplier[selected] * chosen_field).sum(dim=-1)
    chosen_energy = (
        chosen_energy
        + float(risk_penalty) * feasibility_risk[chosen_edge]
    )
    step_logp = (
        -float(beta) * chosen_energy
        - torch.logsumexp(logits[selected], dim=1)
    )
    return step_logp, rollout_index[selected], decisions


def replay_logp_from_cpp_batch_trace(
    trace: dict,
    graph,
    output: dict,
    model: ConstraintFieldNet,
    beta: float,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay summed per-rollout log-probabilities for diagnostics."""
    decision_logp, decision_rollouts, decisions = (
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
    logp_sum.scatter_add_(0, decision_rollouts, decision_logp)
    return logp_sum, decisions


def _guidance_numpy(
    output: dict,
    graph,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
) -> dict:
    multipliers = output["multipliers"][0]
    if not field_enabled:
        # Zero the resource-field intensities but keep the objective weight
        # (slot FIELD_CHANNEL_COUNT) so neutral guidance is the plain objective.
        multipliers = multipliers.clone()
        multipliers[:-1] = 0.0
    return {
        "objective_residual": (
            output["objective_residual"]
            if field_enabled
            else torch.zeros_like(output["objective_residual"])
        ).detach().cpu().numpy(),
        "edge_field": output["residual"].detach().cpu().numpy(),
        "edge_additive": output["additive"].detach().cpu().numpy(),
        "multipliers": multipliers.detach().cpu().numpy(),
        "coupler_weights": output["coupler_weights"][0].detach().cpu().numpy(),
        "coupler_bias": output["coupler_bias"][0].detach().cpu().numpy(),
        "edge_risk": output["feasibility_risk"].detach().cpu().numpy(),
        "risk_penalty": float(risk_penalty),
    }


def _field_guidance(model, decoder, device, risk_penalty: float = 0.0) -> dict:
    """One no-grad model evaluation of the current decoder graph -> guidance.

    Used to *construct* the bootstrap incumbent from the learned field (not just
    to refine it), so the field is load-bearing from the first solution. It is a
    no-grad evaluation on purpose: construction is a bootstrap, kept out of the
    trained refinement policy batch (see collect_instance_rollout), exactly as
    inference computes its guidance.
    """
    graph = build_decoder_data(decoder, device)
    with torch.no_grad():
        output = model(graph)
    return _guidance_numpy(output, graph, risk_penalty=risk_penalty)


def _neutral_guidance(decoder) -> dict:
    channels = int(decoder.metadata["resource_count"])
    multipliers = int(decoder.metadata["multiplier_count"])
    # Zero resource intensities but set the objective weight (final slot) to 1,
    # so neutral guidance reduces the energy to the plain objective edge cost.
    multiplier_values = np.zeros(multipliers, dtype=np.float32)
    multiplier_values[channels] = 1.0
    return {
        "objective_residual": np.zeros(
            decoder.metadata["edge_count"], dtype=np.float32
        ),
        "edge_field": np.zeros(
            (decoder.metadata["edge_count"], channels), dtype=np.float32
        ),
        "edge_additive": np.zeros(
            (decoder.metadata["edge_count"], channels), dtype=np.float32
        ),
        "multipliers": multiplier_values,
        "coupler_weights": np.zeros(
            (multipliers, channels),
            dtype=np.float32,
        ),
        "coupler_bias": np.zeros(multipliers, dtype=np.float32),
        "edge_risk": np.zeros(decoder.metadata["edge_count"], dtype=np.float32),
        "risk_penalty": 0.0,
    }


def _constant_guidance(decoder) -> dict:
    """Return the constant-energy control with identical candidate scores."""
    guidance = _neutral_guidance(decoder)
    # Zeroing the objective multiplier makes the concrete constant immaterial to
    # construction sampling and SRR ranking.
    guidance["multipliers"][:] = 0.0
    return guidance


def _distance_guidance(decoder, problem: dict) -> dict:
    """Return guidance whose complete candidate energy is normalized distance."""
    edge_index = np.asarray(decoder.edge_index, dtype=np.int64)
    if "distance" in problem:
        distance = np.asarray(problem["distance"], dtype=np.float64)
        edge_distance = distance[edge_index[0], edge_index[1]]
    else:
        coordinates = np.asarray(problem["coordinates"], dtype=np.float64)
        edge_distance = np.linalg.norm(
            coordinates[edge_index[0]] - coordinates[edge_index[1]], axis=1
        )
    offsets = np.asarray(decoder.edge_offsets, dtype=np.int64)

    squared_difference = 0.0
    difference_count = 0
    for begin, end in zip(offsets[:-1], offsets[1:]):
        if begin == end:
            continue
        row = edge_distance[begin:end]
        squared_difference += float(np.square(row - row.mean()).sum())
        difference_count += int(row.size)
    scale = (
        math.sqrt(squared_difference / difference_count)
        if difference_count
        else 0.0
    )
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = float(np.mean(np.abs(edge_distance))) if edge_distance.size else 1.0
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0

    guidance = _neutral_guidance(decoder)
    objective_scale = float(decoder.metadata["objective_energy_scale"])
    objective_energy = (
        np.asarray(decoder.objective_edge_costs, dtype=np.float64)
        / objective_scale
    )
    guidance["objective_residual"] = (
        edge_distance / scale - objective_energy
    ).astype(np.float32)
    return guidance


def _random_guidance(decoder, seed: int) -> dict:
    """Return a stable random energy keyed by seed and directed edge."""
    guidance = _neutral_guidance(decoder)
    objective_scale = float(decoder.metadata["objective_energy_scale"])
    objective_energy = (
        np.asarray(decoder.objective_edge_costs, dtype=np.float64)
        / objective_scale
    )
    edge_index = np.asarray(decoder.edge_index, dtype=np.uint64)
    # Vectorized SplitMix64: stable across Python processes and candidate-graph
    # rebuilds, unlike hash() or an advancing RNG. Mapping the upper 53 bits to
    # a centered unit-variance uniform gives distinct, scale-controlled energy.
    bits = (
        np.uint64(seed)
        ^ (edge_index[0] << np.uint64(32))
        ^ edge_index[1]
    )
    bits = bits + np.uint64(0x9E3779B97F4A7C15)
    bits = (bits ^ (bits >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    bits = (bits ^ (bits >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    bits ^= bits >> np.uint64(31)
    unit = ((bits >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
    random_energy = (unit - 0.5) * math.sqrt(12.0)
    # Cancel the analytic objective term so the complete energy, not merely an
    # additive perturbation, is random. This activates the same bounded SRR
    # exploration mechanism without carrying task-specific guidance.
    guidance["objective_residual"] = (
        random_energy - objective_energy
    ).astype(np.float32)
    return guidance


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


def _best_feasible_solution(
    solutions: list[dict], *, context: str
) -> tuple[dict, int]:
    best: dict = {"feasible": False}
    winner = -1
    for rollout, solution in enumerate(solutions):
        if _better(solution, best):
            best = solution
            winner = rollout
    if winner < 0:
        errors = sorted(
            {
                str(solution.get("error", "unknown construction failure"))
                for solution in solutions
            }
        )
        detail = "; ".join(errors[:3])
        raise RuntimeError(f"{context} failed: no feasible rollout ({detail})")
    return best, winner


def _gain(incumbent: dict, candidate: dict, infeasible_penalty: float) -> float:
    if not candidate["feasible"]:
        return -float(infeasible_penalty)
    scale = max(abs(float(incumbent["objective"])), 1e-6)
    if incumbent["direction"] == "maximize":
        return (candidate["objective"] - incumbent["objective"]) / scale
    return (incumbent["objective"] - candidate["objective"]) / scale


def _new_decoder(
    problem: dict, args: argparse.Namespace, deterministic: bool = False,
    use_srr: bool = True,
):
    decoder = prism_decoder.Decoder(
        problem,
        candidate_config={"max_candidates": args.candidates},
        search_config={
            "use_srr": use_srr,
            "min_changed_edges": getattr(args, "min_changed_edges", 8),
            "random_escape": getattr(args, "random_escape", False),
            "feasibility_lookahead_depth": getattr(
                args, "feasibility_lookahead_depth", 2
            ),
            # Guided exploration is inert for constant energy unless the
            # explicit seeded-random baseline escape control is enabled.
            "srr_exploration_budget": getattr(
                args, "srr_exploration_budget", 0
            ),
        },
        n_rollouts=args.n_rollouts,
        beta=args.beta,
    )
    decoder.seed(
        args.seed if deterministic else args.seed + random.randrange(1 << 30)
    )
    return decoder


def _inference_decoder_args(
    args: argparse.Namespace, model: Optional[ConstraintFieldNet]
) -> argparse.Namespace:
    """Isolate learned exploration from the opt-in random baseline control."""
    decoder_args = copy.copy(args)
    if model is None:
        decoder_args.random_escape = bool(
            getattr(args, "random_escape", False)
        )
        if not decoder_args.random_escape:
            decoder_args.srr_exploration_budget = 0
    else:
        decoder_args.random_escape = False
    return decoder_args


def setup_decoder(
    problem: dict, args: argparse.Namespace, deterministic: bool = False,
    model: Optional[ConstraintFieldNet] = None,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
):
    decoder = _new_decoder(problem, args, deterministic=deterministic)
    # Field-constructed bootstrap: once the field is live, build the initial
    # incumbent from the field itself so training and inference share the same
    # construction path (a no-grad bootstrap, kept out of the refinement policy
    # batch). During pretrain (field_enabled=False) or when no model is supplied,
    # fall back to neutral distance construction.
    if model is not None and field_enabled:
        guidance = _field_guidance(
            model, decoder, args.device, risk_penalty=risk_penalty
        )
    else:
        guidance = _neutral_guidance(decoder)
    if deterministic:
        incumbent = decoder.sample_greedy(**guidance)
        if incumbent["feasible"]:
            decoder.set_incumbent(incumbent["route"])
    else:
        incumbent = decoder.solve(1, **guidance)
    if not incumbent["feasible"]:
        raise RuntimeError(f"bootstrap failed: {incumbent['error']}")
    return decoder, incumbent


def setup_decoder_resampling(
    problem: dict,
    variant: str,
    args: argparse.Namespace,
    deterministic: bool = False,
    max_attempts: int = 8,
    model: Optional[ConstraintFieldNet] = None,
    field_enabled: bool = True,
    risk_penalty: float = 0.0,
):
    """Bootstrap a decoder, resampling the instance on infeasible generation.

    A handful of constraint intersections (e.g. asymmetric multi-depot backhaul
    with a route limit *and* time windows) occasionally emit a generated
    instance that has no feasible solution at all -- even the all-singleton
    routing violates a resource. That is a property of the sampled instance, not
    the solver, so the correct response is to draw a fresh instance rather than
    abort a multi-hour run. URS does the same for its own time-window generator.
    """
    last_error = ""
    for _ in range(max_attempts):
        try:
            decoder, incumbent = setup_decoder(
                problem, args, deterministic=deterministic,
                model=model, field_enabled=field_enabled,
                risk_penalty=risk_penalty,
            )
            return decoder, incumbent, problem
        except RuntimeError as exc:
            if "bootstrap failed" not in str(exc):
                raise
            last_error = str(exc)
            problem = generated_problem(
                variant,
                args.n_node,
                args.capacity,
                randomize_resource_program=getattr(
                    args, "randomize_resource_programs", False
                ),
            )
    raise RuntimeError(
        f"bootstrap failed for {variant} after {max_attempts} resamples: "
        f"{last_error}"
    )


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

    The decoder transition is produced by exactly one rollout.  Continuation
    credit therefore belongs to that rollout's transition step rather than being
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
            and outcome.winner_rollout is not None
        ):
            outcome.transition_step.transition_rollout = outcome.winner_rollout
            outcome.transition_step.temporal_advantage = advantage
        next_value = outcome.old_value
        next_advantage = advantage


def _winner_temporal_advantage(
    rollout_count: int,
    winner_rollout: int,
    advantage: float,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a zero-mean POMO contrast that retains winner continuation."""
    if rollout_count < 1 or not 0 <= winner_rollout < rollout_count:
        raise ValueError("winner_rollout must index a non-empty rollout batch")
    contrast = torch.full(
        (rollout_count,), -1.0 / rollout_count, dtype=torch.float32, device=device
    )
    contrast[winner_rollout] += 1.0
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
    # Match the successful refinement-only pipeline: build a neutral feasible
    # incumbent first, then train the learned field on post-bootstrap search
    # decisions. Native construction mixes greedy rollouts (which have no
    # policy log-probabilities) with stochastic rollouts and has a much larger
    # reward scale; including it in the pooled POMO batch suppresses the
    # refinement signal that rh1gudc1 learned from.
    decoder, incumbent, problem = setup_decoder_resampling(
        problem, variant, args,
        model=model, field_enabled=field_enabled, risk_penalty=risk_penalty,
    )
    steps: list[OptionStep] = []
    emissions = 0
    improvements = 0
    neural_seconds = 0.0
    decoder_seconds = 0.0
    last_solutions: list[dict] = [incumbent]
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
        winner_rollout = None
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
                    old_logp, decision_rollouts, decisions = (
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
                decision_rollouts = torch.empty(
                    0, dtype=torch.long, device=args.device
                )
                decisions = torch.zeros(
                    args.n_rollouts, dtype=torch.long, device=args.device
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
                rewards=torch.zeros(args.n_rollouts, device=args.device),
                resource_delta=resource_delta,
                binding_target=binding_target,
                duration=0,
                decision_rollouts=decision_rollouts.detach(),
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
            for rollout, solution in enumerate(solutions):
                if _better(solution, iteration_best):
                    iteration_best = solution
                    iteration_winner = rollout
            normalized_gain = max(
                _gain(option_incumbent, iteration_best, 0.0), 0.0
            )
            if normalized_gain > args.improvement_epsilon:
                decoder.set_incumbent(iteration_best["route"])
                incumbent = iteration_best
                improvements += 1
                transition_reward = normalized_gain
                transition_step = step
                winner_rollout = iteration_winner
                break

        terminal_reward = torch.tensor(
            [
                _gain(option_incumbent, solution, args.infeasible_penalty)
                for solution in last_solutions
            ],
            dtype=torch.float32,
            device=args.device,
        )
        # Bound the per-rollout reward so a rare infeasible rollout (-infeasible_penalty,
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
                winner_rollout=winner_rollout,
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


def _decision_rollout_index(trace: dict, device: torch.device) -> torch.Tensor:
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
        prediction = (
            output["residual"][screened_edges]
            + output["additive"][screened_edges]
        )
        active = output["active_channels"][0].bool().expand_as(prediction)
        if active.any():
            return _balanced_regression_loss(
                prediction[active], target[active]
            )
        return prediction.sum() * 0.0

    # Construction-only instances have no SRR labels on their first option.
    # Retain the rollout outcome as a lower-resolution supervision fallback.
    current = torch.as_tensor(step.trace["current_nodes"], device=device).long()
    chosen = torch.as_tensor(step.trace["chosen_indices"], device=device).long()
    stochastic = torch.as_tensor(step.trace["stochastic"], device=device).bool()
    if current.numel() == 0 or not stochastic.any():
        return output["residual"].sum() * 0.0
    rollout_index = _decision_rollout_index(step.trace, device)
    edge = step.graph.edge_offsets.to(device)[current] + chosen
    prediction = output["residual"][edge] + output["additive"][edge]
    if step.resource_delta is None:
        raise RuntimeError("missing fallback resource-delta labels")
    target = step.resource_delta.to(device)[rollout_index]
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
    channels = output["active_channels"].shape[-1]
    active = output["active_channels"][0].bool()
    if not active.any():
        return output["multipliers"].sum() * 0.0
    # The objective weight slot (index FIELD_CHANNEL_COUNT) is not a resource
    # intensity, so the binding supervision only applies to the field channels.
    field_multipliers = output["multipliers"][0, :channels]
    binding = step.binding_target.to(output["multipliers"].device)
    base_loss = F.smooth_l1_loss(
        field_multipliers[active], binding[active]
    )
    live_state = torch.as_tensor(
        step.trace["live_state"],
        dtype=torch.float32,
        device=output["multipliers"].device,
    )
    if live_state.numel() == 0:
        return base_loss
    dynamic_target = torch.maximum(live_state, binding.unsqueeze(0))
    coupled = model.couple(output, live_state)[:, :channels]
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
        "objective_residual",
        "residual",
        "additive",
        "feasibility_logits",
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


def _objective_residual_loss(step: OptionStep, output: dict) -> torch.Tensor:
    """Anchor policy-relevant dimensionless objective-residual differences.

    A constant added to every outgoing edge cancels in the decoder softmax, so
    center each source row before penalizing the learned residual.
    """
    residuals = output["objective_residual"]
    offsets = step.graph.edge_offsets.to(residuals.device)
    counts = offsets[1:] - offsets[:-1]
    source = torch.repeat_interleave(
        torch.arange(counts.numel(), device=residuals.device), counts
    )
    sums = residuals.new_zeros(counts.numel()).scatter_add(0, source, residuals)
    means = sums / counts.to(residuals.dtype).clamp_min(1.0)
    centered = residuals - means[source]
    return centered.square().mean()


def _disable_objective_residual(model: ConstraintFieldNet) -> None:
    """Fix the objective-energy residual at its neutral zero value."""
    with torch.no_grad():
        # Zeroing the final layer makes the shared head output identically zero
        # for every objective coefficient vector.
        model.objective_energy_residual_head[-1].weight.zero_()
        model.objective_energy_residual_head[-1].bias.zero_()
    for parameter in model.objective_energy_residual_head.parameters():
        parameter.requires_grad_(False)


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
        logp, decision_rollouts, decisions = (
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
        if step.decision_rollouts is not None and not torch.equal(
            step.decision_rollouts.to(decision_rollouts.device), decision_rollouts
        ):
            raise RuntimeError(
                "stored and replayed decision-to-rollout maps differ"
            )
    else:
        logp = output["residual"].new_empty(0)
        old_logp = logp
        decision_rollouts = torch.empty(
            0, dtype=torch.long, device=logp.device
        )
        decisions = torch.zeros(
            step.rewards.shape[0], dtype=torch.long, device=logp.device
        )

    if rl_weight != 0.0 and logp.numel():
        log_ratio = logp - old_logp
        ratio = torch.exp(log_ratio)
        rollout_reward = step.rewards.to(logp.device)
        reward_std = rollout_reward.std(unbiased=False)
        # POMO baseline: centre per option (all rollouts share one field, so the
        # rollout mean is the correct control variate). _gain already divides by
        # |incumbent objective|, so the centred reward is a scale-invariant
        # fractional improvement -- per-option unit-std normalisation is thus
        # redundant and, when a field's rollouts land near-identical costs, only
        # amplifies RNG jitter into unit-scale spurious advantages. Normalise
        # by the batch-pooled scale instead so options with genuine improvement
        # variance dominate and near-degenerate options contribute little.
        rollout_advantage = rollout_reward - rollout_reward.mean()
        if not args.no_adv_norm:
            base_scale = (
                adv_scale
                if adv_scale is not None
                else rollout_advantage.std(unbiased=False) + 1e-8
            )
            rollout_advantage = rollout_advantage / base_scale
        temporal_rollout_advantage = torch.zeros_like(rollout_advantage)
        if temporal_enabled and step.transition_rollout is not None:
            temporal_rollout_advantage = _winner_temporal_advantage(
                rollout_advantage.numel(),
                step.transition_rollout,
                step.temporal_advantage,
                (
                    float(temporal_adv_scale)
                    if temporal_adv_scale is not None
                    else 1.0
                ),
                rollout_advantage.device,
            )
            rollout_advantage = (
                rollout_advantage
                + temporal_credit_weight * temporal_rollout_advantage
            )
        advantage = rollout_advantage[decision_rollouts]
        temporal_advantage = temporal_rollout_advantage[decision_rollouts]
        # Each rollout contributes total weight one, independent of trace length.
        decision_weight = decisions[decision_rollouts].float().reciprocal()
        normalizer = decision_weight.sum().clamp_min(1.0)
        clipped_ratio = torch.clamp(
            ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
        )
        surrogate = torch.minimum(
            ratio * advantage, clipped_ratio * advantage
        )
        # With a single on-policy pass ratio is exactly one. POMO centering
        # then makes the scalar PPO surrogate cancel across rollouts, although its
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
    objective_residual_loss = _objective_residual_loss(step, output)
    auxiliary = (
        args.dual_weight * dual
        + args.feasibility_weight * feasibility
        + args.binding_weight * binding
        + args.price_weight * price
    )
    loss = (
        rl_weight * rl_loss
        + auxiliary_scale * auxiliary
        + float(getattr(args, "objective_residual_l2", 0.1)) * objective_residual_loss
        + float(getattr(args, "value_loss_weight", 0.0)) * critic_loss
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
            "objective_residual_loss": objective_residual_loss.detach(),
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
        if step.transition_rollout is not None
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


def _training_variant_schedule(
    curriculum: VariantCurriculum,
    args: argparse.Namespace,
    epoch: int,
) -> list[str]:
    """Return the epoch schedule, with curriculum phasing only when requested."""
    # Without --curriculum, make progress equal to 1.0 so every selected
    # variant is eligible from epoch 0. Keeping epoch in both arguments retains
    # deterministic epoch-local reshuffling. Validation never uses this phase
    # filter; it always covers args.variants directly.
    schedule_epochs = args.epochs if args.curriculum else epoch + 1
    eligible = curriculum.eligible(epoch, schedule_epochs) or curriculum.variants
    group_size = min(args.grad_accum_variants, len(eligible))
    weights = (
        channel_balanced_weights(curriculum.variants)
        if getattr(args, "channel_balanced_sampling", False)
        else None
    )
    return curriculum.schedule(
        epoch,
        schedule_epochs,
        args.steps_per_epoch,
        group_size,
        weights=weights,
    )


def _training_accumulation_size(
    curriculum: VariantCurriculum,
    args: argparse.Namespace,
    epoch: int,
) -> int:
    """Return rollout instances pooled into one PPO optimizer update.

    Variant diversity and optimizer batch size are separate concerns.  In
    particular, a single-variant run still needs more than one independently
    generated instance per update; capping this value by the number of eligible
    variants silently reduced CVRP-only training to batch size one.
    """
    del curriculum, epoch
    if args.grad_accum_variants < 1:
        raise ValueError("rollouts_per_update must be at least one")
    return min(args.grad_accum_variants, args.steps_per_epoch)


def train_epoch(
    model: ConstraintFieldNet,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    args: argparse.Namespace,
    curriculum: VariantCurriculum,
    ema: Optional["WeightEMA"] = None,
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
    variant_schedule = _training_variant_schedule(curriculum, args, epoch)
    accumulation_size = _training_accumulation_size(curriculum, args, epoch)
    progress = tqdm(total=args.steps_per_epoch, desc="Epoch", leave=True)
    while completed < args.steps_per_epoch:
        group = min(accumulation_size, args.steps_per_epoch - completed)
        rollouts = []
        field_enabled = epoch >= args.pretrain_epochs
        risk_penalty = (
            args.feasibility_risk_penalty if field_enabled else 0.0
        )
        for variant in variant_schedule[completed : completed + group]:
            phase_started = time.perf_counter()
            problem = generated_problem(
                variant,
                args.n_node,
                args.capacity,
                randomize_resource_program=getattr(
                    args, "randomize_resource_programs", False
                ),
            )
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
        if ema is not None:
            ema.update(model)
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
                    "imitation_scale",
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


def infer_instance(
    model: Optional[ConstraintFieldNet],
    problem: dict,
    args: argparse.Namespace,
    initial_route: Optional[np.ndarray] = None,
    baseline: str = "constant",
) -> tuple[float, dict, dict[str, float]]:
    # An externally guaranteed route, when supplied, is installed identically
    # for learned guidance and every native control, bypassing only construction.
    if args.search_iterations < 1:
        raise ValueError("search_iterations must be positive")
    if model is None and baseline not in {"constant", "distance", "random"}:
        raise ValueError(f"unknown inference baseline: {baseline}")
    decoder_args = _inference_decoder_args(args, model)
    decoder = _new_decoder(
        problem,
        decoder_args,
        deterministic=True,
    )
    # Construction and refinement use the same selected guidance mode.
    risk_penalty = args.feasibility_risk_penalty
    net_evals = 0
    def _baseline_guidance() -> dict:
        if baseline == "constant":
            return _constant_guidance(decoder)
        if baseline == "distance":
            return _distance_guidance(decoder, problem)
        if baseline == "random":
            return _random_guidance(decoder, args.seed)
        return {}

    if initial_route is None:
        if model is None:
            construct_guidance = _baseline_guidance()
        else:
            model.eval()
            construct_guidance = _field_guidance(
                model, decoder, args.device, risk_penalty=risk_penalty
            )
            net_evals += 1
        initial = list(decoder.sample(**construct_guidance))
        incumbent, _winner = _best_feasible_solution(
            initial, context="field bootstrap"
        )
    else:
        incumbent = decoder.evaluate(
            np.asarray(initial_route, dtype=np.int32)
        )
        if not incumbent["feasible"]:
            raise RuntimeError(
                "provided bootstrap route is infeasible: "
                + incumbent.get("error", "unknown route error")
            )
    decoder.set_incumbent(incumbent["route"])

    if model is not None:
        model.eval()

    if model is None:
        baseline_started = time.perf_counter()
        if baseline in {"distance", "random"}:
            # Incumbent changes can rebuild the candidate graph. Refresh the
            # aligned control energy between iterations just as the neural
            # branch refreshes its aligned edge outputs. Random energy is keyed
            # by directed edge, so preserved edges retain identical values.
            guidance = _baseline_guidance()
            version = int(decoder.graph_version)
            best = decoder.solve(1, **guidance)
            for _ in range(max(args.search_iterations - 1, 0)):
                if int(decoder.graph_version) != version:
                    guidance = _baseline_guidance()
                    version = int(decoder.graph_version)
                best = decoder.solve(1, **guidance)
        else:
            best = decoder.solve(
                args.search_iterations, **_baseline_guidance()
            )
        decoder_seconds = time.perf_counter() - baseline_started
        return (
            _canonical_cost(best),
            best,
            {
                "emissions": 0.0,
                "time_neural": 0.0,
                "time_decoder": decoder_seconds,
                "net_evals": 0.0,
            },
        )

    dynamic = not getattr(args, "static_field", False)

    def _refresh_guidance() -> tuple[dict, int]:
        graph = build_decoder_data(decoder, args.device)
        output = model(graph)
        evaluations = 1
        return (
            _guidance_numpy(output, graph, risk_penalty=risk_penalty),
            evaluations,
        )

    neural_seconds = 0.0
    decoder_seconds = 0.0
    # net_evals already counts the field-construction evaluation above.

    start = time.perf_counter()
    guidance, refresh_evals = _refresh_guidance()
    net_evals += refresh_evals
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
                guidance, refresh_evals = _refresh_guidance()
                net_evals += refresh_evals
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
    _metrics: dict[str, float], average_best_cost: float
) -> tuple[float]:
    """Rank checkpoints by the mean canonical best cost on validation."""
    return (float(average_best_cost),)


CHECKPOINT_RANK_METRIC = "mean_best_canonical_cost_v1"


def _validation_cost_groups(problem: dict) -> tuple[str, ...]:
    """Return overlapping semantic groups used for validation cost summaries."""
    constraints = set(problem.get("constraints", ()))
    distance = problem.get("distance")
    if "symmetric" in problem:
        symmetric = bool(problem["symmetric"])
    elif distance is None:
        # Coordinate-only routing problems use Euclidean, hence symmetric, cost.
        symmetric = "coordinates" in problem
    else:
        matrix = np.asarray(distance)
        symmetric = bool(
            matrix.ndim == 2
            and matrix.shape[0] == matrix.shape[1]
            and np.allclose(matrix, matrix.T, rtol=1.0e-5, atol=1.0e-7)
        )

    groups = [
        "symmetric" if symmetric else "asymmetric",
        "multi_route" if problem.get("multi_route", False) else "single_route",
        "open_route" if problem.get("open_route", False) else "closed_route",
    ]

    depot_count = int(problem.get("depot_count", 1))
    groups.append(
        "no_depot"
        if depot_count == 0
        else "single_depot"
        if depot_count == 1
        else "multi_depot"
    )
    groups.append(
        "visit_all" if "visit_all" in constraints else "optional_visits"
    )

    objective = str(problem.get("objective", "distance"))
    groups.append(f"objective_{objective}")

    constraint_group_names = {
        "capacity": "capacity",
        "backhaul_order": "backhaul_order",
        "pickup_delivery": "pickup_delivery",
        "route_limit": "route_limit",
        "time_windows": "time_window",
        "tour_limit": "tour_limit",
        "prize_quota": "prize_quota",
    }
    groups.extend(
        group
        for constraint, group in constraint_group_names.items()
        if constraint in constraints
    )

    # Backhaul instances include both unrestricted mixed pickup/delivery demand
    # (*b) and the ordered linehaul-before-backhaul form (*bp).
    demand = problem.get("demand")
    if "backhaul_order" in constraints or (
        "pickup_delivery" not in constraints
        and demand is not None
        and bool(np.any(np.asarray(demand) < 0))
    ):
        groups.append("backhaul")
    return tuple(groups)


def _parse_variants(value: str) -> list[str]:
    """Parse the variants shared by training and validation."""
    if value == "train":
        return list(TRAIN_VARIANTS)
    if value == "all":
        return list(ALL_VARIANTS)
    variants = [name.strip() for name in value.split(",") if name.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("--variants selected no variants")
    unknown = sorted(set(variants) - set(ALL_VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown variants: " + ", ".join(unknown)
        )
    if len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError("--variants contains duplicates")
    return variants


def validation(
    model: Optional[ConstraintFieldNet],
    dataset: list[dict],
    args: argparse.Namespace,
    *,
    capture_paired_baseline: bool = False,
) -> tuple[float, float, float, dict[str, float]]:
    validation_args = copy.copy(args)
    validation_args.n_rollouts = (
        args.n_rollouts
        if getattr(args, "val_n_rollouts", None) is None
        else args.val_n_rollouts
    )
    collector = MetricsCollector()
    average_costs = []
    best_costs = []
    gaps = []
    variant_gaps: dict[str, list[float]] = {}
    baseline_improvements = []
    variant_baseline_improvements: dict[str, list[float]] = {}
    variant_totals: dict[str, int] = {}
    variant_feasible: dict[str, int] = {}
    variant_objectives: dict[str, list[float]] = {}
    variant_best_costs: dict[str, list[float]] = {}
    group_best_costs: dict[str, list[float]] = {}
    for item in tqdm(dataset, desc="Validating", leave=False):
        average, best, metrics = infer_instance(
            model, item["problem"], validation_args
        )
        collector.add_dict(metrics)
        variant = item["variant"]
        variant_totals[variant] = variant_totals.get(variant, 0) + 1

        best_cost = _canonical_cost(best)
        feasible = bool(best["feasible"]) and np.isfinite(best_cost)
        if capture_paired_baseline:
            item["paired_baseline"] = {
                "objective": float(best["objective"]),
                "direction": best["direction"],
                "feasible": feasible,
            }
        if not feasible:
            continue

        average_costs.append(average)
        best_costs.append(best_cost)
        variant_feasible[variant] = variant_feasible.get(variant, 0) + 1
        variant_objectives.setdefault(variant, []).append(
            float(best["objective"])
        )
        variant_best_costs.setdefault(variant, []).append(best_cost)
        for group in _validation_cost_groups(item["problem"]):
            group_best_costs.setdefault(group, []).append(best_cost)
        baseline = item.get("paired_baseline")
        if baseline is not None and baseline.get("feasible", False):
            baseline_objective = float(baseline["objective"])
            if np.isfinite(baseline_objective):
                improvement = -_gap(best, baseline_objective)
                baseline_improvements.append(improvement)
                variant_baseline_improvements.setdefault(variant, []).append(
                    improvement
                )
        if item["reference"] is not None:
            gap = _gap(best, item["reference"])
            gaps.append(gap)
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
        saved_reference_instances=float(
            sum(item.get("reference_source") == "saved" for item in dataset)
        ),
        missing_reference_instances=float(
            sum(item.get("reference_source") == "missing" for item in dataset)
        ),
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
    # Keep paired-baseline improvement and oracle-only gap as diagnostics.
    # Checkpoint selection uses the mean canonical best cost instead.
    result["macro_score"] = result[
        "macro_baseline_improvement_percent"
    ]
    result.update(
        {
            f"group_cost/{group}": float(np.mean(costs))
            for group, costs in sorted(group_best_costs.items())
        }
    )
    return (
        float(np.mean(average_costs)) if average_costs else float("inf"),
        float(np.mean(best_costs)) if best_costs else float("inf"),
        macro_gap,
        result,
    )


def build_validation_data(args: argparse.Namespace) -> list[dict]:
    if args.val_size < 1:
        raise ValueError("val_size must be at least one instance per problem")
    if getattr(args, "val_generated", False):
        # Checkpoint selection is the averaged solution cost, so no oracle data
        # is required. But KEEP saved references where they exist so macrogap is
        # still reported as the gap over the reference-having subset; only fill
        # the missing slots with (seeded, held-out) generated instances.
        saved = SavedProblems(args.n_node, args.dataset_dir)
        dataset = []
        rng_state = np.random.get_state()
        np.random.seed(args.seed + 999_983)
        for variant in args.variants:
            for index in range(args.val_size):
                problem = reference = None
                try:
                    problem, reference = saved.load(variant, index=index)
                except Exception:
                    problem, reference = None, None
                if problem is None:
                    problem = generated_problem(
                        variant, args.n_node, args.capacity
                    )
                    source = "generated"
                else:
                    source = "saved" if reference is not None else "missing"
                dataset.append(
                    {
                        "variant": variant,
                        "instance_index": index,
                        "problem": problem,
                        "reference": reference,
                        "reference_source": source,
                    }
                )
        np.random.set_state(rng_state)
        return dataset
    saved = SavedProblems(args.n_node, args.dataset_dir)
    dataset = []
    logger = get_logger()
    missing = []
    for variant in args.variants:
        for index in range(args.val_size):
            try:
                problem, reference = saved.load(variant, index=index)
            except Exception as exc:  # missing data dir/file for this variant
                missing.append((variant, index, str(exc)))
                continue
            reference_source = "saved" if reference is not None else "missing"
            dataset.append(
                {
                    "variant": variant,
                    "instance_index": index,
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
            f"missing {len(missing)} validation instances selected by "
            f"--variants ({details})"
        )
        if not getattr(args, "allow_missing_validation", False):
            raise RuntimeError(message)
        logger.warning(message)
    return dataset


class WeightEMA:
    """Polyak (exponential moving average) shadow of the model parameters.

    The online weights random-walk within the converged basin every PPO step,
    and the validation macro-gap -- being the cost of a discrete search over the
    field -- is a jagged function of those weights, so the raw val curve jitters
    with no trend. Evaluating and checkpointing a slowly-moving average of the
    weights turns that jitter into a near-monotone descent and makes best.pt a
    meaningful selection rather than the luckiest noise trough. Buffers are not
    tracked (the field net carries no running statistics); only parameters move.
    """

    def __init__(self, model: ConstraintFieldNet, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: ConstraintFieldNet) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay
            )

    @contextmanager
    def applied(self, model: ConstraintFieldNet):
        """Temporarily swap the EMA weights into the model, then restore."""
        backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        try:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in self.shadow.items():
            if name in state:
                tensor.copy_(state[name].to(tensor.device))


def _epoch_lr(args: argparse.Namespace, epoch: int) -> float:
    """Learning rate for an epoch, honouring the pretrain phase and schedule."""
    if epoch < args.pretrain_epochs:
        return args.pretrain_lr
    if getattr(args, "lr_schedule", "constant") != "cosine":
        return args.lr
    # Cosine anneal from args.lr to args.lr_min across the post-pretrain epochs
    # so late updates shrink toward zero and the val curve settles instead of
    # wandering. The final epoch lands at lr_min.
    span = max(args.epochs - args.pretrain_epochs - 1, 1)
    progress = min(max(epoch - args.pretrain_epochs, 0), span) / span
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (
        1.0 + math.cos(math.pi * progress)
    )


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
    ema_state: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    payload = {
        "model_schema": MODEL_SCHEMA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": vars(args),
        "global_step": int(global_step),
    }
    if ema_state is not None:
        payload["ema_state_dict"] = ema_state
    if val_gap is not None:
        payload["val_gap"] = val_gap
    if validation_rank is not None:
        payload["validation_rank"] = tuple(validation_rank)
    if best_validation_rank is not None:
        payload["best_validation_rank"] = tuple(best_validation_rank)
        payload["checkpoint_rank_metric"] = CHECKPOINT_RANK_METRIC
    if validation_manifest is not None:
        payload["validation_manifest"] = validation_manifest
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the event-driven resource-field routing Decoder"
    )
    parser.add_argument("--n-node", type=int, default=100)
    parser.add_argument(
        "--variants",
        "--variant",
        type=_parse_variants,
        default=list(TRAIN_VARIANTS),
        metavar="NAMES",
        help=(
            "Comma-separated variants used for both training and validation; "
            "also accepts 'train' (default set) or 'all'"
        ),
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Phase the selected --variants by resource count. By default all "
            "selected variants train from epoch 0."
        ),
    )
    parser.add_argument(
        "--channel-balanced-sampling",
        action="store_true",
        help=(
            "Weight variant sampling by inverse constraint-channel coverage so "
            "singleton channels (tour_limit via op, prize_quota via pctsp) are "
            "not starved next to capacity. Off by default (uniform per-variant)."
        ),
    )
    parser.add_argument(
        "--randomize-resource-programs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append one anonymous resource program sampled in final term "
            "coordinates to every generated training instance. Off by default: "
            "the sampled bound is sized from a singleton route "
            "(append_random_resource_program), which no route in pctsp, pdtsp, "
            "pdcvrp, opdcvrp or amdocvrp can satisfy, so construction has no "
            "feasible solution and the run aborts at the first epoch. Turn it "
            "back on once the bound is derived from a real route rather than a "
            "constant leg factor."
        ),
    )
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
            "datasets/benchmarks"
        ),
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help=(
            "override the vehicle capacity. Defaults to the benchmark value "
            "for each variant (20 for pickup-delivery, 50 otherwise), which is "
            "what the saved evaluation instances use."
        ),
    )
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--n-rollouts", type=int, default=32)
    parser.add_argument(
        "--val-n-rollouts",
        type=int,
        default=None,
        help=(
            "Rollouts per validation instance (default: inherit "
            "--n-rollouts)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument(
        "--rollouts-per-update",
        "--grad-accum-variants",
        dest="grad_accum_variants",
        type=int,
        default=None,
        help=(
            "Independently generated problem rollouts pooled into each PPO "
            "optimizer update. By default this is chosen to preserve the "
            "legacy 128 sampled solutions/update (4 instances at 32 "
            "rollouts), so --n-rollouts 10 uses 13 instances/update. The legacy "
            "--grad-accum-variants spelling is retained as an alias."
        ),
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=4,
        help="post-bootstrap perturbation/SRR iterations (default: 16)",
    )
    parser.add_argument(
        "--min-changed-edges",
        type=int,
        default=8,
        help=(
            "Minimum number of route edges each perturbation tries to change "
            "before SRR refinement (default: 8)"
        ),
    )
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
            "EXPERIMENTAL (off by default; may be removed or evolved in the "
            "future). Weight of the optional refresh-state critic loss; the "
            "default 0 keeps temporal credit critic-free"
        ),
    )
    parser.add_argument("--neural-call-cost", type=float, default=0.0)
    parser.add_argument("--infeasible-penalty", type=float, default=10.0)
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=1.0,
        help=(
            "Clip per-rollout reward magnitude before advantage normalisation so "
            "rare infeasible rollouts cannot dominate the pooled scale; 0 disables"
        ),
    )
    parser.add_argument(
        "--pretrain-epochs",
        "--pretraining-epochs",
        dest="pretrain_epochs",
        type=int,
        default=0,
        help=(
            "Number of auxiliary-only epochs before PPO starts; use 0 to "
            "disable the pretraining phase"
        ),
    )
    parser.add_argument(
        "--pretrain-lr",
        "--pretraining-lr",
        dest="pretrain_lr",
        type=float,
        default=5e-6,
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
        default=0.0,
        help="EXPERIMENTAL (off by default; may be removed or evolved in the "
        "future). Auxiliary-loss scale after PPO fine-tuning starts; default 0 "
        "carries no auxiliary-head loss into RL",
    )
    parser.add_argument("--dual-weight", type=float, default=1.0)
    parser.add_argument("--feasibility-weight", type=float, default=1.0)
    parser.add_argument("--binding-weight", type=float, default=1.0)
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
    parser.add_argument(
        "--gate-multipliers-by-binding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Gate resource multipliers by the binding classifier (the stable"
            " legacy training behavior; --no-gate-multipliers-by-binding is"
            " available as an ablation)."
        ),
    )
    parser.add_argument(
        "--couple-resource-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Couple the per-resource tokens with resource-token attention so"
            " each constraint's learned intensity depends on the full active"
            " composition (default). --no-couple-resource-tokens is the"
            " compositional-attention ablation: tokens become an independent"
            " per-resource encoding with no cross-resource coupling."
        ),
    )
    parser.add_argument(
        "--couple-state-multipliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Modulate each resource multiplier by the live search state at every"
            " decision (default). --no-couple-state-multipliers is the"
            " static-intensity ablation: multipliers are frozen to their"
            " per-refresh GNN value, isolating live-state-modulated intensity"
            " (static vs. per-decision lambda_r)."
        ),
    )
    parser.add_argument(
        "--linear-objective-residual-head",
        action="store_true",
        help=(
            "Design-choice ablation for the signed objective-energy residual:"
            " replace the coefficient-conditioned MLP with a single linear head"
            " over [edge_state, coeffs]. Tests the claim that a purely linear"
            " coefficient term collapses into a per-row constant that downstream"
            " row-centering deletes, leaving the residual unable to specialize"
            " per objective. Mutually exclusive with"
            " --unconditioned-objective-residual-head."
        ),
    )
    parser.add_argument(
        "--unconditioned-objective-residual-head",
        action="store_true",
        help=(
            "Design-choice ablation for the signed objective-energy residual:"
            " drop the declared coefficient vector from the head input so one"
            " shared MLP logit serves every objective. Tests whether the"
            " unconditioned shared head suffers cross-objective negative transfer"
            " that coefficient conditioning resolves. Mutually exclusive with"
            " --linear-objective-residual-head."
        ),
    )
    parser.add_argument(
        "--index-embedded-resources",
        action="store_true",
        help=(
            "Semantic-factorization ablation: replace the algebra-derived"
            " resource descriptor with a learned embedding of the resource's"
            " registry POSITION (same width, same unit-interval range, so only"
            " the semantics change). The field can then only memorize resource"
            " identities seen in training; an appended schema row falls onto a"
            " single shared cold row. Tests the claim that semantic"
            " factorization -- not token capacity -- is what carries transfer to"
            " unseen compositions."
        ),
    )
    parser.add_argument(
        "--no-normalize-projections",
        dest="normalize_projections",
        action="store_false",
        help=(
            "Restore the pre-fix forward pass, in which edge_projection and"
            " graph_projection feed tanh unnormalized. Their activations reach"
            " |40| after the residual GNN, saturating every consuming head, so"
            " the per-edge field collapses to 1-3 distinct values and the"
            " feasibility-risk head becomes constant. Kept as a flag rather"
            " than deleted so the cost of the bug can be measured against a"
            " matched run; there is no reason to train with it."
        ),
    )
    parser.add_argument(
        "--program-blind-resources",
        action="store_true",
        help=(
            "Parsimony ablation, one rung below --index-embedded-resources:"
            " give every resource the SAME constant type vector, so neither the"
            " executable row/term properties nor the resource's identity reach"
            " the model. Per-resource factorization is kept -- each row still"
            " has its own token, multiplier and field head, and still receives"
            " live state, node attributes and candidate-conditioned effects"
            " u_r(e, t). Tests whether the hand-designed property maps supply"
            " anything the executed effects do not. Mutually exclusive with"
            " --index-embedded-resources and --monolithic-resource-field."
        ),
    )
    parser.add_argument(
        "--monolithic-resource-field",
        action="store_true",
        help=(
            "Factorization ablation (coarse end of the ladder): collapse every"
            " active resource onto one shared token, so a single"
            " undifferentiated penalty intensity serves the whole composition"
            " and two constraints can no longer be priced differently. Unlike"
            " --index-embedded-resources this degrades in-distribution as well,"
            " and it makes resource-token coupling vacuous, so it is an"
            " everything-off reference point rather than a clean separation of"
            " factorization from contextualization. Mutually exclusive with"
            " --index-embedded-resources."
        ),
    )
    parser.add_argument(
        "--resource-pooling",
        "--node-resource-pooling",
        dest="resource_pooling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Give the GNN encoder a pooled, descriptor-keyed summary of the"
            " per-resource quantities the decoder publishes: node attributes"
            " (demand, window bounds, service time) plus live incumbent state,"
            " and per-edge pressure and reset events. Without it only the seven"
            " compiled channels have named columns, so a declared row reaches"
            " the model at the field head alone and never enters the node or"
            " edge embeddings the feasibility, binding, coupler and"
            " objective-residual heads are built from. Pooling is masked by the"
            " active rows and shared across resources, so the width is"
            " independent of the registry and nothing reads a registry"
            " position. --no-resource-pooling is the ablation; it narrows both"
            " encoder inputs, so an ablated run must train from scratch."
        ),
    )
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument(
        "--objective-residual",
        "--edge-logit",
        dest="objective_residual_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the learned signed objective-energy residual (default: "
            "enabled). Use --no-objective-residual for the neutral ablation; "
            "--no-edge-logit remains a compatibility alias."
        ),
    )
    parser.add_argument(
        "--objective-residual-l2",
        "--edge-logit-l2",
        dest="objective_residual_l2",
        type=float,
        default=0.1,
        help=(
            "L2 anchor on source-centered objective-energy residuals "
            "(default: 0.1); prevents PPO random walk from overwhelming the "
            "exact objective energy while leaving row-constant corrections "
            "unpenalized"
        ),
    )
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
    parser.add_argument(
        "--srr-exploration-budget",
        type=int,
        default=0,
        help=(
            "Bounded number of objective-worsening but guided-energy-descending "
            "moves the SRR descent may accept per invocation, letting the learned "
            "field steer uphill to escape local optima (champion tracking keeps "
            "the best solution). Constant energy has no gradient, so the budget "
            "is inert for that control. 0 disables (default)."
        ),
    )
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
            "Weight of the detached feasibility-risk classifier in decoder "
            "ranking (default: 1). "
            "Use 0 for the measured risk-guidance ablation."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        dest="lr_schedule",
        choices=("constant", "cosine"),
        default="constant",
        help=(
            "Post-pretrain learning-rate schedule (default: constant). Use "
            "'cosine' to anneal lr to --lr-min."
        ),
    )
    parser.add_argument(
        "--lr-min",
        dest="lr_min",
        type=float,
        default=0.0,
        help="Floor learning rate for --lr-schedule cosine (default: 0.0)",
    )
    parser.add_argument(
        "--val-ema-decay",
        dest="val_ema_decay",
        type=float,
        default=0.0,
        help=(
            "If >0, keep an exponential moving average of the weights (decay "
            "per optimizer update) and run validation + save best.pt from it. "
            "Smooths the jagged val macro-gap into a near-monotone curve. "
            "Default: 0 (validate raw online weights); enable explicitly"
        ),
    )
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
        "--allow-missing-validation",
        action="store_true",
        help="Warn and continue when a selected validation instance is missing",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--val-generated",
        action="store_true",
        help="validate on freshly generated held-out instances (no saved oracle "
        "data); checkpoint selection is purely the best averaged cost",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("pretrained"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="prism-decoder")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if args.n_rollouts < 1:
        parser.error("--n-rollouts must be positive")
    if args.min_changed_edges < 1:
        parser.error("--min-changed-edges must be positive")
    if args.grad_accum_variants is None:
        legacy_rollout_batch = 4 * 32
        args.grad_accum_variants = max(
            1, math.ceil(legacy_rollout_batch / args.n_rollouts)
        )
    return args


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_seed(seed: int, epoch: int) -> int:
    """Derive an epoch-local RNG seed that ignores where the process started.

    Training instances and rollout sampling draw from the process-global RNGs,
    which setup_seeds fixes once at startup. Reseeding them from (seed, epoch)
    at the top of every epoch is what keeps a resumed run from replaying the
    stream from its first epoch: without it, epoch 43 of a resumed run trains
    on exactly the problems a fresh run sees at epoch 0. Mirrors the
    epoch-local scheduler seed in VariantCurriculum.schedule; the value is
    masked to 32 bits because np.random.seed rejects anything wider.
    """
    return random.Random((int(seed) << 32) ^ int(epoch)).randrange(1 << 32)


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
    if args.objective_residual_l2 < 0.0:
        raise ValueError("objective_residual_l2 must be nonnegative")
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

    curriculum = VariantCurriculum(
        list(args.variants), random.Random(args.seed), args.seed
    )
    model = ConstraintFieldNet(
        grad_checkpointing=args.grad_checkpointing,
        gate_multipliers_by_binding=args.gate_multipliers_by_binding,
        couple_resource_tokens=args.couple_resource_tokens,
        couple_state_multipliers=args.couple_state_multipliers,
        linear_objective_residual_head=args.linear_objective_residual_head,
        unconditioned_objective_residual_head=(
            args.unconditioned_objective_residual_head
        ),
        index_embedded_resources=args.index_embedded_resources,
        monolithic_resource_field=args.monolithic_resource_field,
        program_blind_resources=args.program_blind_resources,
        normalize_projections=args.normalize_projections,
        pool_node_resources=args.resource_pooling,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=args.device, weights_only=False
        )
        if checkpoint.get("model_schema") != MODEL_SCHEMA:
            raise RuntimeError(
                "resume checkpoint declares model schema "
                f"{checkpoint.get('model_schema')!r}, but this model is "
                f"{MODEL_SCHEMA!r}"
            )
        load_constraint_field_state_dict(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(
            checkpoint.get("global_step", start_epoch * args.steps_per_epoch)
        )

    if not args.objective_residual_enabled:
        _disable_objective_residual(model)

    ema = (
        WeightEMA(model, args.val_ema_decay)
        if args.val_ema_decay and args.val_ema_decay > 0.0
        else None
    )
    if ema is not None and checkpoint is not None:
        stored_ema = checkpoint.get("ema_state_dict")
        if stored_ema is not None:
            ema.load_state_dict(stored_ema)
        else:
            logger.warning(
                "resumed checkpoint has no EMA state; seeding the EMA shadow "
                "from the online weights"
            )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    validation_data = (
        [] if args.skip_validation else build_validation_data(args)
    )
    missing_references = sum(
        item.get("reference_source") == "missing" for item in validation_data
    )
    if missing_references:
        logger.info(
            f"Skipping oracle gap for {missing_references} validation "
            "instances without saved oracle references"
        )
    validation_manifest = tuple(
        (
            item["variant"],
            int(item["instance_index"]),
            item["reference_source"],
        )
        for item in validation_data
    )
    best_validation_rank = (float("inf"),)
    if checkpoint is not None:
        stored_rank = checkpoint.get("best_validation_rank")
        if (
            checkpoint.get("checkpoint_rank_metric")
            == CHECKPOINT_RANK_METRIC
            and checkpoint.get("validation_manifest") == validation_manifest
            and stored_rank is not None
            and len(stored_rank) == 1
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
            capture_paired_baseline=True,
        )
        logger.log_baseline(baseline_gap)
    for epoch in range(start_epoch, args.epochs):
        setup_seeds(_epoch_seed(args.seed, epoch))
        phase_lr = _epoch_lr(args, epoch)
        for group in optimizer.param_groups:
            group["lr"] = phase_lr
        (
            global_step,
            train_cost,
            _neural_time,
            _decoder_time,
            epoch_time,
            epoch_timing,
        ) = train_epoch(
            model, optimizer, global_step, epoch, args, curriculum, ema=ema
        )
        val_best = 0.0
        val_gap = None
        val_metrics = {}
        validation_rank = None
        is_best = False
        if validation_data:
            # Evaluate and select the checkpoint from the EMA weights (when
            # enabled) so the reported curve and best.pt track the smoothed
            # model rather than the jittery online one. A fresh context per use
            # -- ema.applied() is single-entry.
            def _eval_context():
                return ema.applied(model) if ema is not None else nullcontext()

            with _eval_context():
                val_average, val_best, val_gap, val_metrics = validation(
                    model, validation_data, args
                )
            validation_rank = _validation_rank(val_metrics, val_best)
            if validation_rank < best_validation_rank:
                best_validation_rank = validation_rank
                with _eval_context():
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
                is_best = True
                logger.info(
                    f"Saved BEST checkpoint: {args.save_dir / 'best.pt'} "
                    f"ValCost={val_best:.4f}"
                )
            logger.log_validation(
                val_average,
                val_best,
                val_gap,
                epoch,
                val_metrics,
                is_best=is_best,
                timing=epoch_timing,
                step=global_step,
            )
        logger.log_epoch_summary(
            epoch,
            train_cost,
            val_best,
            val_gap,
            val_metrics.get("feasibility_rate") if validation_data else None,
            is_best=is_best,
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
        # last.pt stores the online weights (resume-correct with the optimizer
        # state) plus the EMA shadow so a resumed run keeps averaging.
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
            ema_state=ema.state_dict() if ema is not None else None,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
