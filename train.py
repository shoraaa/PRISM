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
    BatchedRelocate,
    ConstraintFieldNet,
    MODEL_SCHEMA,
    build_decoder_data,
    load_constraint_field_state_dict,
    schema_vector,
)
from route_eval import RouteEvaluator
from refine import neural_refine_solve, candidate_adjacency
from problem_data import (
    ALL_VARIANTS,
    DEFAULT_DATASET_DIR,
    SavedProblems,
    TRAIN_VARIANTS,
    VariantCurriculum,
    channel_balanced_weights,
    generated_problem,
    problem_schema,
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
    quota_counts: Optional[torch.Tensor] = None
    old_quota_logp: Optional[torch.Tensor] = None
    # Algebra descriptor [SCHEMA_FEATURE_DIM] for the schema-conditioned advantage
    # scale g_phi(schema); populated in ppo_update only when --schema-adv-scale.
    schema: Optional[torch.Tensor] = None


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
    # Joint-training refiner supervision (CaR unified-encoder): a best-improvement
    # teacher trajectory whose imitation loss is added to the same backward, so
    # the shared GNN encoder is trained by construction RL AND refinement CE.
    refiner: Optional["RefinerSupervision"] = None


@dataclass
class RefinerSupervision:
    graph: Any  # decoder graph (build_decoder_data) at the bootstrap state
    live: torch.Tensor  # [T, N, C] per-state node resource features
    rt: torch.Tensor  # [T, L] teacher-visited route states
    valid: torch.Tensor  # [T, L] bool
    rm_pos: torch.Tensor  # [T] teacher removal / segment-start position
    gap: torch.Tensor  # [T] teacher reinsertion gap
    schema: torch.Tensor  # [SCHEMA_FEATURE_DIM] schema descriptor
    depot_count: int
    evaluator: Any  # RouteEvaluator (holds gap_feas_fn)
    oropt: bool = False  # or-opt teacher (segment) vs relocate-1
    seg_len: Optional[torch.Tensor] = None  # [T] segment length (or-opt)
    rev: Optional[torch.Tensor] = None  # [T] reversal flag (or-opt)


# The relocate refiner only models the min-distance visit-all family; op/pctsp
# (subset/maximize) need an add/drop operator, so joint supervision skips them.
def _refiner_supported(problem: dict) -> bool:
    return (
        problem.get("objective", "distance") == "distance"
        and "visit_all" in problem.get("constraints", [])
        and problem.get("multi_route", False)
    )


def build_refiner_supervision(
    decoder, problem: dict, incumbent: dict, args: argparse.Namespace,
    max_steps: int = 40,
) -> Optional[RefinerSupervision]:
    """Best-improvement teacher trajectory from the bootstrap incumbent, computed
    entirely in torch. Uses OR-OPT (best_oropt: segment relocate) when
    --refiner-max-seg>1, else relocate-1 (best_relocate). None if the schema is
    unsupported or the bootstrap is already a local optimum."""
    if not _refiner_supported(problem):
        return None
    device = args.device
    try:
        ev = RouteEvaluator(problem, device=device)
    except NotImplementedError:
        return None
    max_seg = getattr(args, "refiner_max_seg", 1)
    oropt = max_seg > 1
    # Candidate-restricted teachers (O(L*K) not O(L^2)) -- far faster than the
    # exact all-gaps teacher and no OOM at n=1000. `local` is the fully
    # incremental Δ path (no evaluate) for the cvrp family; `restricted` uses the
    # exact scan (correct for tw/route_limit/backhaul/pd/md/open) on the same
    # candidate set. Both need the candidate neighbour lists.
    use_local = (
        not oropt
        and int(problem.get("depot_count", 1)) == 1
        and not problem.get("open_route", False)
        and set(problem.get("constraints", [])) <= {"visit_all", "capacity"}
    )
    nbr = None
    if not oropt:
        adj = candidate_adjacency(decoder, device)  # [N,N] bool
        Kc = min(int(adj.sum(1).max().item()) if adj.numel() else 1, adj.shape[1])
        Kc = max(Kc, 1)
        vals, idx = adj.float().topk(Kc, dim=1)
        nbr = torch.where(vals > 0, idx, torch.full_like(idx, -1))
    start = list(incumbent["route"])
    L = len(start)
    rt = torch.tensor(start, dtype=torch.long, device=device).view(1, L)
    valid = torch.ones(1, L, dtype=torch.bool, device=device)
    s_rt, s_valid, s_rm, s_gap, s_len, s_rev = [], [], [], [], [], []
    for _ in range(max_steps):
        res = (ev.best_oropt(rt, valid, max_seg=max_seg) if oropt
               else ev.best_relocate_local(rt, valid, nbr) if use_local
               else ev.best_relocate_restricted(rt, valid, nbr))
        if not bool(res["improved"][0]):
            break
        s_rt.append(rt)
        s_valid.append(valid)
        s_gap.append(res["gap"])
        if oropt:
            s_rm.append(res["seg_pos"])
            s_len.append(res["seg_len"])
            s_rev.append(res["rev"])
        else:
            s_rm.append(res["rm_pos"])
        rt, valid = res["new_rt"], res["new_valid"]
    if not s_rt:
        return None
    rt_b = torch.cat(s_rt)
    valid_b = torch.cat(s_valid)
    return RefinerSupervision(
        graph=build_decoder_data(decoder, device=device),
        live=ev.node_state_batch(rt_b, valid_b),
        rt=rt_b,
        valid=valid_b,
        rm_pos=torch.cat(s_rm),
        gap=torch.cat(s_gap),
        schema=schema_vector(problem, device=device),
        depot_count=int(decoder.metadata["depot_count"]),
        evaluator=ev,
        oropt=oropt,
        seg_len=torch.cat(s_len) if oropt else None,
        rev=torch.cat(s_rev) if oropt else None,
    )


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


def _fields_off_guidance(decoder) -> dict:
    """Fields-off baseline guidance: a flat, identical field (E = 1 everywhere).

    Unlike neutral guidance, this also zeros the objective multiplier, so the
    objective term drops out and edge_energy collapses to the same constant on
    every edge. The baseline therefore ranks moves with no objective or distance
    signal at all -- "PRISM without fields" is uniform guidance, not distance
    guidance. It differs from PRISM by the field alone: the shared neutral
    bootstrap, candidate graph, seed, and budget stay identical, so the paired
    comparison still isolates the learned field.
    """
    guidance = _neutral_guidance(decoder)
    # Zero the objective weight (final multiplier slot) too. A constant energy is
    # identical for every candidate, so the concrete value is immaterial to every
    # ranking (softmax and the SRR edge sort); this is the E = 1 flat field.
    guidance["multipliers"][:] = 0.0
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
    use_srr: bool = True, neutral_ranking: bool = False,
):
    decoder = prism_decoder.Decoder(
        problem,
        candidate_config={
            "max_candidates": args.candidates,
            "candidate_mode": getattr(args, "candidate_mode", "schema"),
        },
        search_config={
            "classical_behavior": False,
            "use_srr": use_srr,
            # The fields-off baseline runs a flat, identical field (E = 1);
            # breaking SRR ties by edge index keeps it from silently inheriting
            # the classical_proximity ordering. Guided search leaves this off.
            "neutral_ranking": neutral_ranking,
            "feasibility_lookahead_depth": getattr(
                args, "feasibility_lookahead_depth", 2
            ),
            # Field-gated exploration budget. A flat field (fields-off) has no
            # energy gradient, so no exploration move ever qualifies -- the
            # budget is inert for the baseline and benefits only the learned
            # field, an unambiguous field-driven advantage.
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
            problem = generated_problem(variant, args.n_node, args.capacity)
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
    # Joint-training: capture the refiner teacher trajectory from the bootstrap
    # state (before the C++ search perturbs the incumbent / decoder features).
    refiner_sup = None
    if getattr(args, "joint_refiner", False):
        refiner_sup = build_refiner_supervision(
            decoder, problem, incumbent, args,
            max_steps=getattr(args, "refiner_teacher_steps", 40),
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
    cached_quota_counts = None
    cached_quota_logp = None
    cached_quota_fractions = None
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
            if (
                getattr(args, "learned_candidate_quotas", False)
                and field_enabled
            ):
                quota_policy = torch.distributions.Multinomial(
                    total_count=args.candidates,
                    logits=old_output["candidate_quota_logits"][0],
                )
                cached_quota_counts = quota_policy.sample()
                cached_quota_logp = quota_policy.log_prob(cached_quota_counts)
                cached_quota_fractions = (
                    cached_quota_counts[:-1] / float(args.candidates)
                ).detach().cpu().numpy().astype(np.float32)
            else:
                cached_quota_counts = None
                cached_quota_logp = None
                cached_quota_fractions = None
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
                if cached_quota_fractions is not None:
                    decoder.set_candidate_resource_quotas(
                        cached_quota_fractions
                    )
                    step.quota_counts = cached_quota_counts.detach()
                    step.old_quota_logp = cached_quota_logp.detach()
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
        refiner=refiner_sup,
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
        "candidate_quota_logits",
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
        model.objective_energy_residual_head.weight.zero_()
        model.objective_energy_residual_head.bias.zero_()
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
    schema_scale_loss = zero
    schema_scale_pred = zero
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
            if getattr(args, "schema_adv_scale", False) and step.schema is not None:
                # g_phi(schema) is a per-schema RELATIVE multiplier on the
                # batch-pooled scale, not an absolute dispersion predictor. The
                # pooled scale supplies the stationary global magnitude; g_phi only
                # corrects each schema's dispersion relative to the batch. It is
                # initialised at 1.0, so scale == base_scale and enabling this is an
                # exact no-op until fitted -- an absolute-dispersion head instead
                # starts ~1/base_scale off and ramps the effective step size by
                # orders of magnitude over training, which diverges. The divisor is
                # detached (the policy cannot game it) and depends only on the
                # schema, so degenerate low-variance options are not amplified.
                eps = rollout_advantage.new_tensor(1e-6)
                pred_scale = model.reward_scale(
                    step.schema.to(rollout_advantage.device)
                )
                # Bound the target and the applied multiplier to a sane band: a
                # degenerate near-zero-dispersion option must not drag g_phi toward
                # zero and re-amplify advantages. The relative correction lives in
                # [0.25, 4]x of the pooled scale; anything outside is a mis-fit, not
                # signal, so the effective step size can never diverge.
                target_rel = (reward_std.detach() / base_scale).clamp(0.1, 10.0)
                schema_scale_loss = F.smooth_l1_loss(
                    pred_scale.clamp_min(eps).log(), target_rel.log()
                )
                schema_scale_pred = pred_scale.detach()
                scale = base_scale * pred_scale.detach().clamp(0.25, 4.0)
            else:
                scale = base_scale
            rollout_advantage = rollout_advantage / scale
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

    quota_rl_loss = zero
    quota_ratio = zero
    quota_entropy = zero
    if (
        temporal_enabled
        and step.quota_counts is not None
        and step.old_quota_logp is not None
    ):
        counts = step.quota_counts.to(output["candidate_quota_logits"].device)
        quota_policy = torch.distributions.Multinomial(
            total_count=int(counts.sum().item()),
            logits=output["candidate_quota_logits"][0],
        )
        quota_logp = quota_policy.log_prob(counts)
        old_quota_logp = step.old_quota_logp.to(quota_logp.device)
        quota_ratio = torch.exp(quota_logp - old_quota_logp)
        quota_advantage = quota_logp.new_tensor(step.temporal_advantage)
        if temporal_adv_scale is not None:
            quota_advantage = quota_advantage / temporal_adv_scale
        clipped_quota_ratio = torch.clamp(
            quota_ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
        )
        quota_rl_loss = -torch.minimum(
            quota_ratio * quota_advantage,
            clipped_quota_ratio * quota_advantage,
        )
        quota_entropy = quota_policy.entropy()

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
        + rl_weight * temporal_credit_weight * quota_rl_loss
        + auxiliary_scale * auxiliary
        + float(getattr(args, "objective_residual_l2", 0.1)) * objective_residual_loss
        + float(getattr(args, "value_loss_weight", 0.0)) * critic_loss
        + float(getattr(args, "schema_scale_weight", 0.1)) * schema_scale_loss
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
            "quota_rl_loss": quota_rl_loss.detach(),
            "quota_ratio": quota_ratio.detach(),
            "quota_entropy": quota_entropy.detach(),
            "reward_std": reward_std.detach(),
            "schema_scale_loss": schema_scale_loss.detach(),
            "schema_scale_pred": schema_scale_pred.detach()
            if torch.is_tensor(schema_scale_pred)
            else schema_scale_pred,
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


def joint_parameters(model, refiner):
    """Deduplicated parameter list over field + refiner (shared emb_net counted
    once) for the optimizer and grad clipping."""
    params = list(model.parameters())
    if refiner is not None:
        seen = {id(p) for p in params}
        params += [p for p in refiner.parameters() if id(p) not in seen]
    return params


def _refiner_batch_loss(refiner, rollouts, args):
    """Mean imitation CE over the batch's teacher trajectories, forwarding
    through the SHARED encoder so the joint backward trains it. None if no
    rollout carried refiner supervision."""
    losses = []
    for rollout in rollouts:
        sup = getattr(rollout, "refiner", None)
        if sup is None:
            continue
        node_emb = refiner.encode_static(sup.graph)  # shared emb_net forward
        if sup.oropt:
            losses.append(
                refiner.oropt_imitation_loss(
                    node_emb, sup.live, sup.rt, sup.valid, sup.evaluator,
                    seg_pos=sup.rm_pos, seg_len=sup.seg_len, gap=sup.gap,
                    rev=sup.rev, depot_count=sup.depot_count, schema=sup.schema,
                )
            )
        else:
            losses.append(
                refiner.imitation_loss(
                    node_emb, sup.live, sup.rt, sup.valid, adj=None,
                    rm_pos=sup.rm_pos, gap_target=sup.gap,
                    depot_count=sup.depot_count,
                    gap_feas_fn=sup.evaluator.insertion_feasible,
                    schema=sup.schema,
                )
            )
    if not losses:
        return None
    return torch.stack(losses).mean()


def ppo_update(
    model: ConstraintFieldNet,
    optimizer: torch.optim.Optimizer,
    rollouts: list[InstanceRollout],
    args: argparse.Namespace,
    epoch: int,
    refiner: Optional[BatchedRelocate] = None,
) -> dict[str, float]:
    steps = [step for rollout in rollouts for step in rollout.steps]
    if not steps:
        return {}
    if getattr(args, "schema_adv_scale", False):
        # Attach the per-variant algebra descriptor to each step so _step_loss can
        # normalise advantages by g_phi(schema). Cached per variant per update.
        schema_cache: dict[str, torch.Tensor] = {}
        for rollout in rollouts:
            descriptor = schema_cache.get(rollout.variant)
            if descriptor is None:
                descriptor = schema_vector(
                    problem_schema(rollout.variant), device=args.device
                )
                schema_cache[rollout.variant] = descriptor
            for step in rollout.steps:
                step.schema = descriptor
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

        # Joint refiner loss: accumulate its gradient onto the SAME step, so one
        # optimizer.step() trains the shared encoder with construction RL + the
        # refinement CE (CaR unified-encoder joint objective).
        if refiner is not None:
            refiner_ce = _refiner_batch_loss(refiner, rollouts, args)
            if refiner_ce is not None:
                (args.refiner_ce_weight * refiner_ce).backward()
                collector.add_dict({"refiner_ce": float(refiner_ce.detach())})

        phase_started = time.perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            joint_parameters(model, refiner), args.grad_clip
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
    refiner: Optional[BatchedRelocate] = None,
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
        metrics = ppo_update(model, optimizer, rollouts, args, epoch, refiner=refiner)
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


def train_refiner_only_epoch(
    model, refiner, optimizer, global_step, epoch, args, curriculum,
    ema=None,
):
    """Refiner-only epoch: NO field RL / C++ search. Per accumulation group,
    bootstrap a cheap construction, build the best-improvement teacher trajectory,
    and take one imitation step (which also trains the shared encoder). Much
    faster than joint since it skips collect_instance_rollout + ppo_update."""
    logger = get_logger()
    started = time.perf_counter()
    variant_schedule = _training_variant_schedule(curriculum, args, epoch)
    accum = _training_accumulation_size(curriculum, args, epoch)
    completed = 0
    ces = []
    progress = tqdm(total=args.steps_per_epoch, desc="Epoch(refiner)", leave=True)
    while completed < args.steps_per_epoch:
        group = min(accum, args.steps_per_epoch - completed)
        rollouts = []
        for variant in variant_schedule[completed: completed + group]:
            problem = generated_problem(variant, args.n_node, args.capacity)
            try:
                decoder, incumbent = setup_decoder(problem, args, deterministic=True)
            except RuntimeError:
                continue
            sup = build_refiner_supervision(
                decoder, problem, incumbent, args,
                max_steps=args.refiner_teacher_steps,
            )
            if sup is not None:
                rollouts.append(InstanceRollout(
                    variant, [], 0.0, 0.0, 0, 0, 0.0, 0.0, refiner=sup))
        optimizer.zero_grad(set_to_none=True)
        ce = _refiner_batch_loss(refiner, rollouts, args)
        if ce is not None:
            (args.refiner_ce_weight * ce).backward()
            torch.nn.utils.clip_grad_norm_(joint_parameters(model, refiner), args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            ces.append(float(ce.detach()))
            logger.log_metrics({"refiner_ce": ces[-1]}, step=global_step)
        completed += group
        global_step += 1
        progress.update(group)
    progress.close()
    epoch_time = time.perf_counter() - started
    mean_ce = float(np.mean(ces)) if ces else 0.0
    return global_step, mean_ce, 0.0, 0.0, epoch_time, {"refiner_ce": mean_ce}


@torch.no_grad()
def infer_instance(
    model: Optional[ConstraintFieldNet],
    problem: dict,
    args: argparse.Namespace,
    refiner: Optional[BatchedRelocate] = None,
) -> tuple[float, dict, dict[str, float]]:
    # Both branches bootstrap from the same neutral rollout batch. Training also
    # installs a neutral feasible incumbent before exposing the model to search,
    # so inference must not ask an untrained field to construct from an empty
    # graph. This also makes fields-off a paired ablation with an identical
    # starting incumbent.
    if args.search_iterations < 1:
        raise ValueError("search_iterations must be positive")
    # Neural-refinement deployment: REPLACE hand-designed SRR. Build the decoder
    # with SRR off (C++ only constructs) and refine with the trained refiner.
    if getattr(args, "neural_refine", False) and refiner is not None:
        if _refiner_supported(problem):
            srr_decoder = _new_decoder(
                problem, args, deterministic=True, use_srr=False
            )
            # construct the incumbent (SRR off) from the neutral bootstrap, the
            # same start the SRR path uses; refinement is then purely neural.
            initial = list(srr_decoder.sample(**_neutral_guidance(srr_decoder)))
            incumbent, _ = _best_feasible_solution(
                initial, context="neutral bootstrap"
            )
            solution = neural_refine_solve(
                srr_decoder, problem, refiner,
                start_route=incumbent["route"],
                group_size=args.neural_refine_group,
                improve_steps=args.neural_refine_steps,
                device=args.device,
            )
            return _canonical_cost(solution), solution, {"emissions": 0.0}
        # op/pctsp etc.: refiner has no operator yet -> fall through to SRR.
    # The fields-off baseline (model is None) breaks equal-energy SRR ties by edge
    # index so its flat E = 1 field never inherits classical_proximity. The flag
    # only changes tie resolution under a constant energy.
    decoder = _new_decoder(
        problem, args, deterministic=True, neutral_ranking=(model is None)
    )
    # Field-constructed bootstrap: the initial incumbent is built from the SAME
    # field that drives search, so the learned field is load-bearing from the
    # first solution rather than only during refinement. The fields-off baseline
    # constructs from the identical flat E = 1 matrix -- no distance or objective
    # signal -- so ablating the field now also ablates construction. Both still
    # share the candidate graph, seed, and budget, so the paired comparison
    # isolates the field's total (construct + refine) contribution.
    risk_penalty = args.feasibility_risk_penalty
    net_evals = 0
    if model is None:
        construct_guidance = _fields_off_guidance(decoder)
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
    decoder.set_incumbent(incumbent["route"])

    if model is None:
        # Fields-off baseline: the identical decoder and SRR driven by a flat,
        # identical field (E = 1 on every edge) for BOTH construction and search.
        # PRISM and this baseline share the same candidate graph, seed, and
        # budget, so they differ only by the learned field -- a clean ablation of
        # PRISM's own guidance rather than a neural-vs-heuristic contest.
        best = decoder.solve(
            args.search_iterations, **_fields_off_guidance(decoder)
        )
        return _canonical_cost(best), best, {"emissions": 0.0}

    dynamic = not getattr(args, "static_field", False)

    def _refresh_guidance() -> tuple[dict, int]:
        graph = build_decoder_data(decoder, args.device)
        output = model(graph)
        evaluations = 1
        if getattr(args, "learned_candidate_quotas", False):
            decoder.set_candidate_resource_quotas(
                output["candidate_quota"][0].detach().cpu().numpy()
            )
            decoder.set_incumbent(decoder.best_solution["route"])
            graph = build_decoder_data(decoder, args.device)
            output = model(graph)
            evaluations += 1
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
    refiner: Optional[BatchedRelocate] = None,
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
    for item in tqdm(dataset, desc="Validating", leave=False):
        average, best, metrics = infer_instance(
            model, item["problem"], validation_args, refiner=refiner
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
    refiner: Optional[BatchedRelocate] = None,
) -> None:
    payload = {
        "model_schema": MODEL_SCHEMA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": vars(args),
        "global_step": int(global_step),
    }
    if refiner is not None:
        payload["refiner_state_dict"] = refiner.state_dict()
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


def _load_optimizer_state_compat(
    optimizer: torch.optim.Optimizer, state_dict: dict
) -> int:
    """Load optimizer state while initializing newly appended parameters.

    New compatibility heads are appended after all legacy model parameters, so
    their optimizer slots retain the same order. Returns the number of new
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
    parser.add_argument(
        "--candidate-mode",
        choices=["schema", "geometric"],
        default="geometric",
        help=(
            "Candidate-graph construction. 'schema' (default) admits resource "
            "candidates by the schema-derived relevance, using a uniform "
            "equal-share prior over active resources when no learned quota is "
            "installed, so any declared resource is covered without per-variant "
            "tuning. 'geometric' is an explicit ablation that keeps only the "
            "distance neighborhood. Currently geometric show more consistent improvement."
        ),
    )
    parser.add_argument(
        "--learned-candidate-quotas",
        action="store_true",
        help=(
            "EXPERIMENTAL (off by default; may be removed or evolved in the "
            "future). Let the typed multinomial quota policy reweight the schema "
            "candidate allocation (schema mode only). Without it the allocation "
            "stays at the uniform equal-share prior; with it the learned "
            "fractions replace that prior. No effect under --candidate-mode "
            "geometric."
        ),
    )
    parser.add_argument(
        "--typed-noninferiority-margin",
        type=float,
        default=0.25,
        help=(
            "Maximum allowed known-resource macro degradation in percentage "
            "points before a typed-neighborhood checkpoint can become best.pt "
            "(default: 0.25)."
        ),
    )
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
        "--joint-refiner",
        action="store_true",
        help="EXPERIMENTAL (off by default; may be removed or evolved in the "
        "future). Jointly train a BatchedRelocate refiner sharing the field GNN "
        "encoder; its best-improvement imitation loss is added to each PPO step "
        "(CaR unified-encoder joint training)",
    )
    parser.add_argument(
        "--refiner-ce-weight",
        type=float,
        default=1.0,
        help="weight of the joint refiner imitation CE in the combined loss",
    )
    parser.add_argument(
        "--refiner-only",
        action="store_true",
        help="EXPERIMENTAL (off by default; may be removed or evolved in the "
        "future). Train ONLY the neural refiner (imitation): skip the field RL / C++ "
        "search rollout entirely; per step just bootstrap a construction, build "
        "the teacher trajectory, and take an imitation step. Much faster; the "
        "field/construction stays neutral (pair with --neural-refine to deploy).",
    )
    parser.add_argument(
        "--refiner-teacher-steps",
        type=int,
        default=40,
        help="max best-improvement teacher trajectory length per instance",
    )
    parser.add_argument(
        "--refiner-max-seg",
        type=int,
        default=1,
        help="OR-OPT neighborhood: relocate contiguous segments of length "
        "1..max_seg (with reversal for len>1). 1 = single-node relocate.",
    )
    parser.add_argument(
        "--neural-refine",
        action="store_true",
        help="EXPERIMENTAL (off by default; may be removed or evolved in the "
        "future). DEPLOY the trained refiner in place of C++ SRR: validation/"
        "inference build the decoder with use_srr=False (C++ only constructs) "
        "and refine with the neural model",
    )
    parser.add_argument(
        "--neural-refine-group",
        type=int,
        default=32,
        help="stochastic rollouts per instance for neural refinement",
    )
    parser.add_argument(
        "--neural-refine-steps",
        type=int,
        default=60,
        help="neural refinement improvement steps per instance",
    )
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
            "the best solution). A flat fields-off field has no energy gradient, "
            "so the budget is inert for the baseline. 0 disables (default)."
        ),
    )
    parser.add_argument("--no-adv-norm", action="store_true")
    parser.add_argument(
        "--schema-adv-scale",
        action="store_true",
        help=(
            "Normalise policy-gradient advantages by a learned schema-conditioned "
            "scale g_phi(schema) instead of a single batch-pooled std. g_phi reads "
            "the algebra descriptor and is fitted to per-schema reward dispersion, "
            "so the normaliser generalises to held-out compositions with no "
            "per-objective grouping. Off by default (pooled std)."
        ),
    )
    parser.add_argument(
        "--schema-scale-weight",
        type=float,
        default=0.1,
        help=(
            "Weight on the log-space regression fitting g_phi(schema) to observed "
            "reward dispersion (used only with --schema-adv-scale)."
        ),
    )
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
    if args.typed_noninferiority_margin < 0.0:
        raise ValueError("typed_noninferiority_margin must be nonnegative")
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
    ).to(args.device)
    # Joint refiner (CaR unified-encoder): EXPERIMENTAL, off by default (may be
    # removed or evolved in the future). Shares model.emb_net so its
    # imitation gradient trains the same GNN. units MUST match the encoder width.
    # Build the refiner when jointly TRAINING it (--joint-refiner) OR DEPLOYING
    # it in place of SRR (--neural-refine); deployment must not require joint
    # training (a refiner can be loaded via --resume).
    refiner = None
    if args.joint_refiner or args.neural_refine or args.refiner_only:
        refiner = BatchedRelocate(
            units=model.emb_net.units, emb_net=model.emb_net,
            grad_checkpointing=args.grad_checkpointing,
            max_seg=args.refiner_max_seg,
        ).to(args.device)
        if args.neural_refine and not args.joint_refiner and not args.resume:
            get_logger().warning(
                "--neural-refine without --joint-refiner or --resume: the "
                "refiner is UNTRAINED; validation/inference will be poor"
            )
    optimizer = torch.optim.AdamW(joint_parameters(model, refiner), lr=args.lr)
    start_epoch = 0
    global_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=args.device, weights_only=False
        )
        if checkpoint.get("model_schema") != MODEL_SCHEMA:
            raise RuntimeError(
                "resume checkpoint is not a typed-resource v5 scale-equivariant-energy "
                "checkpoint"
            )
        upgraded = load_constraint_field_state_dict(
            model, checkpoint["model_state_dict"]
        )
        if refiner is not None and checkpoint.get("refiner_state_dict") is not None:
            refiner.load_state_dict(checkpoint["refiner_state_dict"])
        added_optimizer_parameters = _load_optimizer_state_compat(
            optimizer, checkpoint["optimizer_state_dict"]
        )
        if upgraded:
            logger.warning(
                "resumed a checkpoint without objective-conditioned edge "
                "logits; initialized the new signed heads at zero"
            )
        if added_optimizer_parameters:
            logger.warning(
                "initialized optimizer state for "
                f"{added_optimizer_parameters} new model parameters"
            )
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
        ) = (
            train_refiner_only_epoch(
                model, refiner, optimizer, global_step, epoch, args, curriculum,
                ema=ema,
            )
            if args.refiner_only
            else train_epoch(
                model, optimizer, global_step, epoch, args, curriculum, ema=ema,
                refiner=refiner,
            )
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
                    model, validation_data, args, refiner=refiner
                )
            typed_gate_pass = (
                not args.learned_candidate_quotas
                or (
                    val_metrics.get("feasibility_rate", 0.0) == 1.0
                    and val_metrics.get(
                        "macro_baseline_improvement_percent", float("-inf")
                    )
                    >= -float(args.typed_noninferiority_margin)
                )
            )
            val_metrics["typed_neighborhood_gate_pass"] = float(
                typed_gate_pass
            )
            validation_rank = _validation_rank(val_metrics, val_best)
            if typed_gate_pass and validation_rank < best_validation_rank:
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
                        refiner=refiner,
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
