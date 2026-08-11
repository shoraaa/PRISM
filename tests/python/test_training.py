import copy
import random
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import prism_decoder
from net import ConstraintFieldNet, build_decoder_data
from problem_data import (
    BENCHMARK_VARIANTS,
    HIGHER_ORDER_BENCHMARK_VARIANTS,
    PROGRAM_TRAIN_MAX_RESOURCE_ORDER,
    TRAIN_VARIANTS,
    VariantCurriculum,
    problem_schema,
    resource_count,
)
from train import (
    OptionOutcome,
    OptionStep,
    _assign_refresh_gae,
    _assign_smdp_returns,
    _best_feasible_solution,
    _disable_objective_residual,
    _distance_guidance,
    _neutral_guidance,
    _new_decoder,
    _dual_loss,
    _epoch_lr,
    _objective_residual_loss,
    _random_guidance,
    _feasibility_loss,
    _load_optimizer_state_compat,
    _positive_class_weight,
    _rollout_class_weights,
    _training_accumulation_size,
    _training_variants_by_order,
    _training_variant_schedule,
    _validation_cost_groups,
    _validation_rank,
    _winner_temporal_advantage,
    build_validation_data,
    collect_instance_rollout,
    infer_instance,
    parse_args,
    ppo_update,
    replay_decision_logp_from_cpp_batch_trace,
    setup_decoder,
    validation,
)


def _args() -> Namespace:
    return Namespace(
        candidates=64,
        variants=list(TRAIN_VARIANTS),
        curriculum=False,
        val_size=1,
        n_rollouts=2,
        val_n_rollouts=None,
        beta=2.0,
        seed=404,
        device="cpu",
        search_iterations=2,
        option_max_steps=4,
        infeasible_penalty=10.0,
        reward_clip=1.0,
        smdp_gamma=0.99,
        neural_call_cost=0.0,
        improvement_epsilon=0.0,
        ppo_epochs=1,
        pretrain_epochs=1,
        pretrain_aux_scale=1.0,
        gae_lambda=0.95,
        temporal_credit_weight=0.1,
        value_loss_weight=0.5,
        rl_weight=1.0,
        aux_rl_scale=0.1,
        dual_weight=1.0,
        feasibility_weight=1.0,
        binding_weight=0.25,
        price_weight=0.25,
        entropy_weight=0.001,
        no_adv_norm=False,
        ppo_clip=0.1,
        grad_clip=1.0,
        smallvram=False,
        feasibility_lookahead_depth=2,
        feasibility_risk_penalty=10.0,
    )


def test_program_training_is_order_two_and_has_higher_order_holdouts() -> None:
    assert PROGRAM_TRAIN_MAX_RESOURCE_ORDER == 2
    assert max(resource_count(name) for name in TRAIN_VARIANTS) == 2
    assert all(
        resource_count(name) <= PROGRAM_TRAIN_MAX_RESOURCE_ORDER
        for name in TRAIN_VARIANTS
    )
    assert set(HIGHER_ORDER_BENCHMARK_VARIANTS) <= set(BENCHMARK_VARIANTS)
    assert all(
        resource_count(name) > PROGRAM_TRAIN_MAX_RESOURCE_ORDER
        for name in HIGHER_ORDER_BENCHMARK_VARIANTS
    )
    assert any(resource_count(name) == 4 for name in HIGHER_ORDER_BENCHMARK_VARIANTS)
    selected = _training_variants_by_order(list(BENCHMARK_VARIANTS), 2)
    assert set(selected).isdisjoint(HIGHER_ORDER_BENCHMARK_VARIANTS)
    assert all(resource_count(name) <= 2 for name in selected)
    with pytest.raises(ValueError, match="excludes every selected"):
        _training_variants_by_order(list(HIGHER_ORDER_BENCHMARK_VARIANTS), 0)


def test_setup_decoder_installs_a_neutral_greedy_incumbent() -> None:
    rng = np.random.default_rng(403)
    coordinates = rng.random((20, 2), dtype=np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
        "demand": np.r_[0.0, rng.uniform(0.01, 0.05, 19)].astype(np.float32),
        "capacity": 0.5,
    }
    problem = problem_schema("cvrp") | problem
    args = _args()

    decoder, incumbent = setup_decoder(problem, args, deterministic=True)

    assert incumbent["feasible"]
    assert decoder.best_solution["feasible"]
    assert decoder.best_solution["route"].size > 0
    graph = build_decoder_data(decoder)
    assert torch.count_nonzero(graph.x[:, 12]) > 0


@pytest.mark.parametrize("smallvram", [False, True])
def test_event_driven_option_rollout_and_pretrain_update(
    smallvram: bool,
) -> None:
    rng = np.random.default_rng(405)
    coordinates = rng.random((24, 2), dtype=np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
        "demand": np.r_[0.0, rng.uniform(0.01, 0.07, 23)].astype(np.float32),
        "capacity": 0.5,
    }
    problem = problem_schema("cvrp") | problem
    args = _args()
    args.smallvram = smallvram
    args.pretrain_aux_scale = 0.25
    model = ConstraintFieldNet(depth=1, units=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    binding_before = model.binding_head.weight.detach().clone()
    multiplier_before = model.multiplier_head.weight.detach().clone()
    coupler_before = model.coupler_query_head.weight.detach().clone()
    coupler_bias_before = model.coupler_bias_head.weight.detach().clone()
    feasibility_before = model.feasibility_head.weight.detach().clone()

    rollout = collect_instance_rollout(model, problem, "cvrp", args)
    metrics = ppo_update(model, optimizer, [rollout], args, epoch=0)

    assert 1 <= rollout.emissions <= args.search_iterations
    assert len(rollout.steps) == args.search_iterations
    # Match the successful pipeline: PPO sees only incumbent-conditioned
    # refinement states after a neutral feasible bootstrap.
    assert all(
        torch.count_nonzero(step.graph.x[:, 12]) > 0
        for step in rollout.steps
    )
    assert all(
        1 <= step.duration <= args.search_iterations for step in rollout.steps
    )
    assert all(
        step.old_logp.numel() == int(step.decisions.sum())
        for step in rollout.steps
    )
    transition_steps = [
        step for step in rollout.steps if step.transition_rollout is not None
    ]
    assert len(transition_steps) == rollout.improvements
    assert any(step.value_target is not None for step in rollout.steps)
    assert all(0.0 <= step.search_progress < 1.0 for step in rollout.steps)
    assert {
        "dual_loss",
        "feasibility_loss",
        "binding_loss",
        "price_loss",
        "approx_kl",
        "clip_frac",
    } <= metrics.keys()
    assert metrics["auxiliary_scale"] == pytest.approx(0.25)
    assert not torch.equal(binding_before, model.binding_head.weight.detach())
    assert not torch.equal(multiplier_before, model.multiplier_head.weight.detach())
    assert not torch.equal(
        feasibility_before, model.feasibility_head.weight.detach()
    )
    assert not (
        torch.equal(coupler_before, model.coupler_query_head.weight.detach())
        and torch.equal(
            coupler_bias_before, model.coupler_bias_head.weight.detach()
        )
    )


def test_typed_candidate_quota_receives_winner_gated_ppo_gradient() -> None:
    rng = np.random.default_rng(406)
    coordinates = rng.random((20, 2), dtype=np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
        "demand": np.r_[0.0, rng.uniform(0.01, 0.05, 19)].astype(np.float32),
        "capacity": 0.5,
    }
    problem = problem_schema("cvrp") | problem
    args = _args()
    model = ConstraintFieldNet(depth=1, units=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rollout = collect_instance_rollout(model, problem, "cvrp", args)
    step = rollout.steps[0]
    with torch.no_grad():
        output = model(step.graph)
        policy = torch.distributions.Multinomial(
            total_count=args.candidates,
            logits=output["candidate_quota_logits"][0],
        )
        counts = policy.sample()
        step.quota_counts = counts
        step.old_quota_logp = policy.log_prob(counts)
    step.transition_rollout = 0
    step.temporal_advantage = 1.0
    before = model.candidate_quota_head.weight.detach().clone()

    metrics = ppo_update(model, optimizer, [rollout], args, epoch=1)

    assert "quota_rl_loss" in metrics
    assert not torch.equal(
        before, model.candidate_quota_head.weight.detach()
    )


@pytest.mark.parametrize(
    ("smallvram", "temporal_credit_weight"),
    [(False, 0.1), (True, 0.1), (False, 0.0)],
)
def test_decision_level_ppo_moves_policy_without_auxiliary_losses(
    smallvram: bool, temporal_credit_weight: float
) -> None:
    rng = np.random.default_rng(408)
    coordinates = rng.random((24, 2), dtype=np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
        "demand": np.r_[0.0, rng.uniform(0.01, 0.07, 23)].astype(np.float32),
        "capacity": 0.5,
    }
    problem = problem_schema("cvrp") | problem
    args = _args()
    args.pretrain_epochs = 0
    args.ppo_epochs = 1
    args.smallvram = smallvram
    args.temporal_credit_weight = temporal_credit_weight
    args.dual_weight = 0.0
    args.feasibility_weight = 0.0
    args.binding_weight = 0.0
    args.price_weight = 0.0
    args.entropy_weight = 0.0
    model = ConstraintFieldNet(depth=1, units=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rollout = collect_instance_rollout(model, problem, "cvrp", args)
    for step in rollout.steps:
        step.rewards = torch.tensor([-1.0, 1.0])
    policy_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "objective_energy_residual_head",
                "field_head",
                "additive_head",
                "multiplier_head",
            )
        )
    }
    value_before = model.value_head.weight.detach().clone()

    metrics = ppo_update(model, optimizer, [rollout], args, epoch=0)

    replay_drift = []
    with torch.no_grad():
        for step in rollout.steps:
            output = model(step.graph)
            new_logp, _, _ = replay_decision_logp_from_cpp_batch_trace(
                step.trace,
                step.graph,
                output,
                model,
                args.beta,
                field_enabled=step.field_enabled,
                risk_penalty=step.risk_penalty,
            )
            if new_logp.numel():
                replay_drift.append(
                    float((new_logp - step.old_logp).abs().mean())
                )

    assert any(
        not torch.equal(policy_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
        if name in policy_before
    )
    # First inner epoch: the policy is unchanged, so approx_kl is zero up to
    # float32 C++/torch replay roundoff (~1e-8 now that the objective residual is
    # a small non-zero learned term rather than exactly zero).
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-5)
    assert np.isfinite(metrics["rl_loss"])
    assert abs(metrics["rl_score_proxy"]) > 1e-6
    assert metrics["policy_signal"] > 0.0
    assert metrics["gradient_norm"] > 0.0
    assert rollout.improvements > 0
    assert metrics["temporal_transitions"] == rollout.improvements
    if temporal_credit_weight:
        assert metrics["temporal_policy_signal"] > 0.0
        assert metrics["critic_loss"] > 0.0
        assert not torch.equal(value_before, model.value_head.weight.detach())
    else:
        assert metrics["temporal_policy_signal"] == 0.0
        assert metrics["critic_loss"] == 0.0
        assert torch.equal(value_before, model.value_head.weight.detach())
    assert replay_drift and max(replay_drift) > 1e-7
    assert metrics["ppo_reuse_passes"] == 1.0
    assert metrics["ppo_clipping_active"] == 0.0
    assert metrics["auxiliary_scale"] == pytest.approx(args.aux_rl_scale)


def test_tsp_refinement_transition_updates_edge_logit_head() -> None:
    rng = np.random.default_rng(409)
    coordinates = rng.random((20, 2), dtype=np.float32)
    problem = problem_schema("tsp") | {
        "name": "tsp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
    }
    args = _args()
    args.pretrain_epochs = 0
    args.ppo_epochs = 1
    args.temporal_credit_weight = 0.0
    args.dual_weight = 0.0
    args.feasibility_weight = 0.0
    args.binding_weight = 0.0
    args.price_weight = 0.0
    args.entropy_weight = 0.0
    model = ConstraintFieldNet(depth=1, units=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rollout = collect_instance_rollout(model, problem, "tsp", args)
    refinement = rollout.steps[0]
    assert torch.count_nonzero(refinement.graph.x[:, 12]) > 0
    assert refinement.old_logp.numel() > 0
    refinement.rewards = torch.tensor([-1.0, 1.0])
    rollout.steps = [refinement]
    before = (
        model.objective_energy_residual_head[-1].weight.detach().clone()
    )

    ppo_update(model, optimizer, [rollout], args, epoch=0)

    assert not torch.equal(
        before, model.objective_energy_residual_head[-1].weight.detach()
    )


def test_binding_weight_is_computed_across_rollout_steps() -> None:
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    graph = SimpleNamespace(
        active_channels=torch.tensor(
            [[1.0] + [0.0] * (channels - 1)]
        )
    )
    steps = []
    for target in (0.0, 0.0, 0.0, 1.0):
        steps.append(
            OptionStep(
                graph=graph,
                trace={
                    "feasibility_risk_labels": np.empty(0, dtype=np.float32)
                },
                old_logp=torch.empty(0),
                decisions=torch.zeros(1, dtype=torch.int32),
                rewards=torch.zeros(1),
                resource_delta=None,
                binding_target=torch.tensor(
                    [target] + [0.0] * (channels - 1)
                ),
                duration=1,
            )
        )

    weights = _rollout_class_weights(steps)

    assert weights["binding"] == 3.0


def test_smdp_returns_discount_across_variable_duration_options() -> None:
    first_steps = [SimpleNamespace(), SimpleNamespace()]
    second_steps = [SimpleNamespace()]
    outcomes = [
        OptionOutcome(first_steps, torch.tensor([1.0, 2.0]), 2),
        OptionOutcome(second_steps, torch.tensor([3.0, 5.0]), 1),
    ]

    _assign_smdp_returns(outcomes, gamma=0.5, device="cpu")

    assert torch.equal(second_steps[0].rewards, torch.tensor([3.0, 5.0]))
    assert torch.equal(first_steps[0].rewards, torch.tensor([2.0, 3.0]))
    assert torch.equal(first_steps[1].rewards, torch.tensor([2.0, 3.0]))
    assert all(step.duration == 2 for step in first_steps)
    assert second_steps[0].duration == 1


def test_refresh_gae_credits_only_the_transition_winner() -> None:
    first_value_step = SimpleNamespace()
    first_transition_step = SimpleNamespace()
    second_step = SimpleNamespace()
    outcomes = [
        OptionOutcome(
            [first_value_step, first_transition_step],
            torch.zeros(2),
            2,
            transition_reward=1.0,
            old_value=0.2,
            transition_step=first_transition_step,
            winner_rollout=1,
        ),
        OptionOutcome(
            [second_step],
            torch.zeros(2),
            1,
            transition_reward=3.0,
            old_value=0.4,
            transition_step=second_step,
            winner_rollout=0,
        ),
    ]

    _assign_refresh_gae(outcomes, gamma=0.5, gae_lambda=0.8)

    assert second_step.temporal_advantage == pytest.approx(2.6)
    assert second_step.value_target == pytest.approx(3.0)
    assert first_transition_step.temporal_advantage == pytest.approx(1.42)
    assert first_transition_step.transition_rollout == 1
    assert first_value_step.value_target == pytest.approx(1.62)
    assert not hasattr(first_value_step, "transition_rollout")


def test_winner_temporal_advantage_is_non_cancelling_pomo_contrast() -> None:
    advantage = _winner_temporal_advantage(
        rollout_count=4,
        winner_rollout=2,
        advantage=3.0,
        scale=1.0,
        device=torch.device("cpu"),
    )

    assert torch.allclose(
        advantage, torch.tensor([-0.75, -0.75, 2.25, -0.75])
    )
    assert float(advantage.sum()) == pytest.approx(0.0)
    assert float(advantage.abs().sum()) > 0.0


def test_optimizer_state_initializes_appended_objective_logit_head() -> None:
    model = ConstraintFieldNet(depth=1, units=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    legacy = copy.deepcopy(optimizer.state_dict())
    removed = legacy["param_groups"][0]["params"][-2:]
    legacy["param_groups"][0]["params"] = legacy["param_groups"][0][
        "params"
    ][:-2]
    for parameter_id in removed:
        legacy["state"].pop(parameter_id)
    restored = ConstraintFieldNet(depth=1, units=8)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)

    added = _load_optimizer_state_compat(restored_optimizer, legacy)

    assert added == 2
    assert len(restored_optimizer.param_groups[0]["params"]) == len(
        list(restored.parameters())
    )
    assert len(restored_optimizer.state) == len(list(restored.parameters())) - 2


def test_positive_class_weight_balances_rare_events() -> None:
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])

    assert _positive_class_weight(target) == 3.0
    assert _positive_class_weight(torch.zeros(4)) == 1.0


def test_edge_logit_anchor_ignores_row_constants() -> None:
    step = SimpleNamespace(
        graph=SimpleNamespace(edge_offsets=torch.tensor([0, 2, 5]))
    )
    constant_rows = torch.tensor(
        [2.0, 2.0, -3.0, -3.0, -3.0], requires_grad=True
    )
    varying_rows = torch.tensor(
        [1.0, 3.0, -4.0, -3.0, -2.0], requires_grad=True
    )

    constant_loss = _objective_residual_loss(
        step, {"objective_residual": constant_rows}
    )
    varying_loss = _objective_residual_loss(
        step, {"objective_residual": varying_rows}
    )

    assert constant_loss == 0.0
    assert varying_loss > 0.0


def test_feasibility_auxiliary_retains_legacy_unconstrained_supervision() -> None:
    logits = torch.tensor([0.5, -0.5], requires_grad=True)
    step = SimpleNamespace(
        trace={
            "feasibility_risk_labels": np.array([1.0, 0.0], dtype=np.float32),
            "feasibility_edges": np.array([0, 1], dtype=np.int32),
        }
    )
    output = {
        "feasibility_logits": logits,
        "active_channels": torch.zeros(1, prism_decoder.FIELD_CHANNEL_COUNT),
    }

    loss = _feasibility_loss(step, output)
    loss.backward()

    assert loss > 0.0
    assert logits.grad is not None
    assert logits.grad.norm() > 0.0


def test_direct_dual_heads_learn_when_analytic_pressure_is_zero() -> None:
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    edge_features = torch.zeros(1, prism_decoder.EDGE_FEATURE_COUNT)
    graph = SimpleNamespace(
        edge_attr=edge_features,
        edge_offsets=torch.tensor([0, 1], dtype=torch.long),
    )
    trace = {
        "screened_edges": np.array([0], dtype=np.int32),
        "screened_resource_delta": np.array(
            [[1.0] + [0.0] * (channels - 1)], dtype=np.float32
        ),
    }
    step = OptionStep(
        graph=graph,
        trace=trace,
        old_logp=torch.zeros(1),
        decisions=torch.ones(1),
        rewards=torch.zeros(1),
        resource_delta=None,
        binding_target=torch.zeros(channels),
        duration=1,
    )
    residual = torch.zeros(1, channels, requires_grad=True)
    additive = torch.zeros(1, channels, requires_grad=True)
    output = {
        "residual": residual,
        "additive": additive,
        "active_channels": torch.tensor(
            [[1.0] + [0.0] * (channels - 1)]
        ),
    }

    _dual_loss(step, output).backward()

    assert additive.grad[0, 0] != 0.0
    assert residual.grad[0, 0] != 0.0


def test_screened_dual_loss_is_zero_without_active_resources() -> None:
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    graph = SimpleNamespace(
        edge_attr=torch.zeros(1, prism_decoder.EDGE_FEATURE_COUNT),
        edge_offsets=torch.tensor([0, 1], dtype=torch.long),
    )
    step = OptionStep(
        graph=graph,
        trace={
            "screened_edges": np.array([0], dtype=np.int32),
            "screened_resource_delta": np.zeros(
                (1, channels), dtype=np.float32
            ),
        },
        old_logp=torch.zeros(1),
        decisions=torch.ones(1),
        rewards=torch.zeros(1),
        resource_delta=None,
        binding_target=torch.zeros(channels),
        duration=1,
    )
    output = {
        "residual": torch.ones(1, channels, requires_grad=True),
        "additive": torch.zeros(1, channels, requires_grad=True),
        "active_channels": torch.zeros(1, channels),
    }

    loss = _dual_loss(step, output)
    loss.backward()

    assert loss == 0.0


def test_stagnant_options_reuse_field_and_skip_fallback_labels() -> None:
    rng = np.random.default_rng(406)
    coordinates = rng.random((24, 2), dtype=np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
        "demand": np.r_[0.0, rng.uniform(0.01, 0.07, 23)].astype(np.float32),
        "capacity": 0.5,
    }
    problem = problem_schema("cvrp") | problem
    args = _args()
    args.search_iterations = 4
    args.improvement_epsilon = float("inf")

    rollout = collect_instance_rollout(
        ConstraintFieldNet(depth=1, units=8), problem, "cvrp", args
    )

    assert rollout.emissions == 1
    assert len(rollout.steps) == args.search_iterations
    assert len({id(step.graph) for step in rollout.steps}) == 1
    assert all(
        torch.count_nonzero(step.graph.x[:, 12]) > 0 for step in rollout.steps
    )
    assert all(
        step.trace["screened_edges"].size > 0 for step in rollout.steps
    )
    assert all(step.resource_delta is None for step in rollout.steps)
    assert all(step.duration == args.search_iterations for step in rollout.steps)


def test_inference_is_deterministic() -> None:
    rng = np.random.default_rng(407)
    coordinates = rng.random((20, 2), dtype=np.float32)
    problem = {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
        "demand": np.r_[0.0, rng.uniform(0.01, 0.05, 19)].astype(np.float32),
        "capacity": 0.5,
    }
    problem = problem_schema("cvrp") | problem
    args = _args()
    model = ConstraintFieldNet(depth=1, units=8).eval()

    first = infer_instance(model, problem, args)
    second = infer_instance(model, problem, args)

    assert first[0] == second[0]
    assert np.array_equal(first[1]["route"], second[1]["route"])
    assert first[1]["objective"] == second[1]["objective"]
    assert first[2]["net_evals"] >= 2.0
    assert first[2]["emissions"] == first[2]["net_evals"]
    assert second[2]["emissions"] == second[2]["net_evals"]

    args.static_field = True
    static = infer_instance(model, problem, args)

    # Construction now runs the field too (field-constructed bootstrap), so even
    # the frozen-incumbent static path evaluates the model twice: once to build
    # the initial incumbent, once for the frozen search field.
    assert static[2]["net_evals"] == 2.0
    assert static[2]["emissions"] == 2.0


def test_zero_neutral_model_reproduces_plain_objective_search() -> None:
    rng = np.random.default_rng(1407)
    coordinates = rng.random((18, 2), dtype=np.float32)
    problem = problem_schema("cvrp") | {
        "name": "cvrp",
        "coordinates": coordinates,
        "distance": np.linalg.norm(
            coordinates[:, None] - coordinates[None, :], axis=-1
        ).astype(np.float32),
        "demand": np.r_[0.0, rng.uniform(0.01, 0.05, 17)].astype(np.float32),
        "capacity": 0.5,
    }
    args = _args()
    model = ConstraintFieldNet(depth=1, units=8).eval()

    # A fresh model preserves plain-objective quality; equal-cost route
    # orientations may differ under the small v6 objective-residual initializer.
    decoder = _new_decoder(problem, args, deterministic=True)
    initial = list(decoder.sample(**_neutral_guidance(decoder)))
    incumbent, _ = _best_feasible_solution(initial, context="neutral bootstrap")
    decoder.set_incumbent(incumbent["route"])
    plain = decoder.solve(args.search_iterations, **_neutral_guidance(decoder))

    zero_neural = infer_instance(model, problem, args)
    assert zero_neural[1]["objective"] == plain["objective"]
    assert zero_neural[1]["feasible"]

    for baseline in ("constant", "distance", "random"):
        result = infer_instance(None, problem, args, baseline=baseline)
        assert result[1]["feasible"]
        assert result[2]["net_evals"] == 0.0


def test_distance_guidance_replaces_objective_energy_with_distance() -> None:
    rng = np.random.default_rng(212)
    coordinates = rng.random((12, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    problem = problem_schema("op") | {
        "name": "op",
        "coordinates": coordinates,
        "distance": distance,
        "prize": np.r_[0.0, rng.uniform(0.1, 1.0, 11)].astype(np.float32),
        "tour_limit": 4.0,
    }
    decoder = _new_decoder(problem, _args(), deterministic=True)
    guidance = _distance_guidance(decoder, problem)
    edge_index = np.asarray(decoder.edge_index)
    edge_distance = distance[edge_index[0], edge_index[1]]
    energy = (
        np.asarray(decoder.objective_edge_costs)
        / float(decoder.metadata["objective_energy_scale"])
        + guidance["objective_residual"]
    )

    assert np.corrcoef(edge_distance, energy)[0, 1] == pytest.approx(1.0)
    assert np.asarray(guidance["multipliers"])[-1] == 1.0
    assert np.count_nonzero(guidance["multipliers"][:-1]) == 0


def test_random_guidance_replaces_objective_energy_reproducibly() -> None:
    rng = np.random.default_rng(213)
    coordinates = rng.random((12, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    problem = problem_schema("tsp") | {
        "name": "tsp",
        "coordinates": coordinates,
        "distance": distance,
    }
    decoder = _new_decoder(problem, _args(), deterministic=True)
    seed = 909
    guidance = _random_guidance(decoder, seed)
    energy = (
        np.asarray(decoder.objective_edge_costs)
        / float(decoder.metadata["objective_energy_scale"])
        + guidance["objective_residual"]
    )
    repeated = _random_guidance(decoder, seed)
    different = _random_guidance(decoder, seed + 1)

    np.testing.assert_array_equal(
        guidance["objective_residual"], repeated["objective_residual"]
    )
    assert not np.array_equal(
        guidance["objective_residual"], different["objective_residual"]
    )
    assert np.std(energy) == pytest.approx(1.0, abs=0.2)
    assert np.asarray(guidance["multipliers"])[-1] == 1.0
    assert np.count_nonzero(guidance["multipliers"][:-1]) == 0

    before = {
        tuple(edge): value
        for edge, value in zip(np.asarray(decoder.edge_index).T, energy)
    }
    samples = list(decoder.sample(**_neutral_guidance(decoder)))
    incumbent, _ = _best_feasible_solution(samples, context="random stability")
    decoder.set_incumbent(incumbent["route"])
    rebuilt = _random_guidance(decoder, seed)
    rebuilt_energy = (
        np.asarray(decoder.objective_edge_costs)
        / float(decoder.metadata["objective_energy_scale"])
        + rebuilt["objective_residual"]
    )
    after = {
        tuple(edge): value
        for edge, value in zip(np.asarray(decoder.edge_index).T, rebuilt_energy)
    }
    shared = before.keys() & after.keys()
    assert shared
    np.testing.assert_allclose(
        [before[edge] for edge in shared],
        [after[edge] for edge in shared],
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_policy_replay_uses_direct_field_not_analytic_pressure() -> None:
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    trace = {
        "current_nodes": np.array([0], dtype=np.int32),
        "starts": np.array([0, 1], dtype=np.int32),
        "stochastic": np.array([1], dtype=np.uint8),
        "chosen_indices": np.array([0], dtype=np.int32),
        "live_state": np.zeros(
            (1, prism_decoder.LIVE_STATE_FEATURE_COUNT), dtype=np.float32
        ),
        "valid_offsets": np.array([0, 2], dtype=np.int32),
        "valid_indices": np.array([0, 1], dtype=np.int32),
    }
    graph = SimpleNamespace(
        edge_offsets=torch.tensor([0, 2], dtype=torch.long),
        objective_edge_costs=torch.zeros(2),
        objective_energy_scale=torch.ones(1, 1),
        resource_scales=torch.ones(channels),
        # Deliberately unequal: these must be input features only, not energy.
        raw_resource_pressure=torch.stack(
            (torch.full((channels,), 0.1), torch.full((channels,), 0.9))
        ),
    )
    active = torch.zeros(1, channels)
    active[0, 0] = 1.0
    residual = torch.zeros(2, channels)
    residual[:, 0] = 0.5
    output = {
        "residual": residual,
        "additive": torch.zeros_like(residual),
        "objective_residual": torch.zeros(2),
        "feasibility_risk": torch.zeros(2),
        "active_channels": active,
    }

    class UnitCoupler:
        @staticmethod
        def couple(_output, states):
            return torch.ones(states.shape[0], channels + 1)

    logp, _, _ = replay_decision_logp_from_cpp_batch_trace(
        trace, graph, output, UnitCoupler(), beta=1.0
    )

    assert logp.item() == pytest.approx(-np.log(2.0))


def test_policy_replay_is_objective_scale_and_resource_unit_invariant() -> None:
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    trace = {
        "current_nodes": np.array([0], dtype=np.int32),
        "starts": np.array([0, 1], dtype=np.int32),
        "stochastic": np.array([1], dtype=np.uint8),
        "chosen_indices": np.array([1], dtype=np.int32),
        "live_state": np.zeros(
            (1, prism_decoder.LIVE_STATE_FEATURE_COUNT), dtype=np.float32
        ),
        "valid_offsets": np.array([0, 2], dtype=np.int32),
        "valid_indices": np.array([0, 1], dtype=np.int32),
    }
    active = torch.ones(1, channels)
    residual = torch.arange(2 * channels, dtype=torch.float32).reshape(
        2, channels
    ) / 10.0
    output = {
        "residual": residual,
        "additive": torch.flip(residual, dims=(1,)) / 5.0,
        "objective_residual": torch.tensor([0.2, -0.1]),
        "feasibility_risk": torch.tensor([0.05, 0.15]),
        "active_channels": active,
    }
    field_multipliers = torch.linspace(0.1, 0.7, channels)

    class FixedCoupler:
        @staticmethod
        def couple(_output, states):
            weights = torch.cat((field_multipliers, torch.ones(1)))
            return weights.unsqueeze(0).expand(states.shape[0], -1)

    def replay(objective_factor: float, resource_factor: float) -> torch.Tensor:
        graph = SimpleNamespace(
            edge_offsets=torch.tensor([0, 2], dtype=torch.long),
            objective_edge_costs=(
                torch.tensor([2.0, 6.0]) * objective_factor
            ),
            objective_energy_scale=torch.tensor([[2.0 * objective_factor]]),
            # Physical resource units are model inputs only and must not alter
            # a dimensionless learned energy after inference.
            resource_scales=torch.full((channels,), resource_factor),
        )
        return replay_decision_logp_from_cpp_batch_trace(
            trace,
            graph,
            output,
            FixedCoupler(),
            beta=1.7,
            risk_penalty=0.4,
        )[0]

    reference = replay(1.0, 1.0)
    assert torch.allclose(reference, replay(100.0, 1.0), atol=1e-6)
    assert torch.allclose(reference, replay(1.0, 1000.0), atol=1e-6)


def test_policy_replay_resource_energy_is_channel_permutation_invariant() -> None:
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    permutation = torch.arange(channels - 1, -1, -1)
    trace = {
        "current_nodes": np.array([0], dtype=np.int32),
        "starts": np.array([0, 1], dtype=np.int32),
        "stochastic": np.array([1], dtype=np.uint8),
        "chosen_indices": np.array([0], dtype=np.int32),
        "live_state": np.zeros(
            (1, prism_decoder.LIVE_STATE_FEATURE_COUNT), dtype=np.float32
        ),
        "valid_offsets": np.array([0, 2], dtype=np.int32),
        "valid_indices": np.array([0, 1], dtype=np.int32),
    }
    graph = SimpleNamespace(
        edge_offsets=torch.tensor([0, 2], dtype=torch.long),
        objective_edge_costs=torch.tensor([1.0, 3.0]),
        objective_energy_scale=torch.tensor([[2.0]]),
        resource_scales=torch.ones(channels),
    )
    residual = torch.arange(2 * channels, dtype=torch.float32).reshape(
        2, channels
    ) / 10.0
    output = {
        "residual": residual,
        "additive": torch.zeros_like(residual),
        "objective_residual": torch.zeros(2),
        "feasibility_risk": torch.zeros(2),
        "active_channels": torch.ones(1, channels),
    }
    multipliers = torch.linspace(0.1, 0.7, channels)

    class FixedCoupler:
        def __init__(self, values):
            self.values = values

        def couple(self, _output, states):
            weights = torch.cat((self.values, torch.ones(1)))
            return weights.unsqueeze(0).expand(states.shape[0], -1)

    reference = replay_decision_logp_from_cpp_batch_trace(
        trace, graph, output, FixedCoupler(multipliers), beta=1.0
    )[0]
    permuted_output = dict(output)
    permuted_output["residual"] = residual[:, permutation]
    permuted_output["additive"] = output["additive"][:, permutation]
    permuted_output["active_channels"] = output["active_channels"][:, permutation]
    permuted = replay_decision_logp_from_cpp_batch_trace(
        trace,
        graph,
        permuted_output,
        FixedCoupler(multipliers[permutation]),
        beta=1.0,
    )[0]

    assert torch.allclose(reference, permuted, atol=1e-6)


def test_validation_size_loads_each_requested_instance(monkeypatch) -> None:
    loaded_sizes = []
    loaded_instances = []

    class FakeSavedProblems:
        def __init__(self, size: int, dataset_dir) -> None:
            loaded_sizes.append(size)

        def load(self, variant: str, index: int = 0) -> tuple[dict, float]:
            loaded_instances.append((variant, index))
            return {"name": variant, "index": index}, 1.0

    args = _args()
    args.n_node = 20
    args.dataset_dir = "dataset-root"
    args.val_size = 2
    args.variants = ["tsp", "cvrp"]
    monkeypatch.setattr("train.SavedProblems", FakeSavedProblems)

    data = build_validation_data(args)

    assert loaded_sizes == [20]
    assert loaded_instances == [
        ("tsp", 0),
        ("tsp", 1),
        ("cvrp", 0),
        ("cvrp", 1),
    ]
    assert [(item["variant"], item["instance_index"]) for item in data] == [
        ("tsp", 0),
        ("tsp", 1),
        ("cvrp", 0),
        ("cvrp", 1),
    ]
    assert all("split" not in item for item in data)


def test_validation_size_defaults_to_eight_instances(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = parse_args()

    assert args.val_size == 8
    assert args.val_n_rollouts is None
    assert args.variants == TRAIN_VARIANTS
    assert args.max_train_resource_order == PROGRAM_TRAIN_MAX_RESOURCE_ORDER
    assert args.curriculum is False
    assert args.allow_missing_validation is False
    assert args.static_field is False
    assert args.gae_lambda == pytest.approx(1.0)
    assert args.temporal_credit_weight == pytest.approx(0.1)
    assert args.value_loss_weight == pytest.approx(0.0)
    assert args.epochs == 100
    assert args.pretrain_epochs == 0
    assert args.option_max_steps == 4
    assert args.smdp_gamma == pytest.approx(0.99)
    assert args.infeasible_penalty == pytest.approx(10.0)
    assert args.feasibility_risk_penalty == pytest.approx(1.0)
    assert args.price_weight == pytest.approx(0.0)
    assert args.grad_accum_variants == 4
    assert args.aux_rl_scale == pytest.approx(0.0)
    assert args.objective_residual_enabled is True
    assert args.objective_residual_l2 == pytest.approx(0.1)
    assert args.val_ema_decay == pytest.approx(0.0)
    assert args.lr_schedule == "constant"


def test_no_objective_residual_cli_zeros_and_freezes_objective_head(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--no-objective-residual"])

    args = parse_args()
    model = ConstraintFieldNet(depth=1, units=8)
    final_layer = model.objective_energy_residual_head[-1]
    with torch.no_grad():
        final_layer.weight.fill_(1.0)
        final_layer.bias.fill_(1.0)

    _disable_objective_residual(model)

    assert args.objective_residual_enabled is False
    assert torch.count_nonzero(final_layer.weight) == 0
    assert torch.count_nonzero(final_layer.bias) == 0
    assert all(
        not parameter.requires_grad
        for parameter in model.objective_energy_residual_head.parameters()
    )


def test_rollout_accumulation_preserves_legacy_sample_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--n-rollouts", "10"]
    )

    args = parse_args()

    assert args.grad_accum_variants == 13

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--n-rollouts",
            "10",
            "--rollouts-per-update",
            "4",
        ],
    )
    explicit = parse_args()
    assert explicit.grad_accum_variants == 4


def test_default_cosine_lr_decays_to_zero() -> None:
    args = SimpleNamespace(
        pretrain_epochs=0,
        epochs=5,
        lr=1e-4,
        lr_min=0.0,
        lr_schedule="cosine",
    )

    rates = [_epoch_lr(args, epoch) for epoch in range(args.epochs)]

    assert rates[0] == pytest.approx(args.lr)
    assert rates[-1] == pytest.approx(args.lr_min)
    assert all(left >= right for left, right in zip(rates, rates[1:]))


@pytest.mark.parametrize(
    ("val_n_rollouts", "expected_rollouts"), [(None, 3), (7, 7)]
)
def test_validation_rollouts_can_override_training_rollouts(
    monkeypatch, val_n_rollouts, expected_rollouts
) -> None:
    args = _args()
    args.n_rollouts = 3
    args.val_n_rollouts = val_n_rollouts
    observed_rollouts = []

    def fake_infer(_model, _problem, inference_args, **_kwargs):
        observed_rollouts.append(inference_args.n_rollouts)
        return (
            1.0,
            {
                "objective": 1.0,
                "direction": "minimize",
                "feasible": True,
            },
            {},
        )

    monkeypatch.setattr("train.infer_instance", fake_infer)
    dataset = [
        {
            "variant": "tsp",
            "problem": {"name": "tsp"},
            "reference": None,
        }
    ]

    validation(None, dataset, args)

    assert observed_rollouts == [expected_rollouts]
    assert args.n_rollouts == 3


def test_validation_rollout_cli_override_is_parsed(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--n-rollouts", "3", "--val-n-rollouts", "7"],
    )

    args = parse_args()

    assert args.n_rollouts == 3
    assert args.val_n_rollouts == 7


def test_variants_and_optional_curriculum_are_parsed(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--variants", "tsp,cvrp", "--curriculum"],
    )

    args = parse_args()

    assert args.variants == ["tsp", "cvrp"]
    assert args.curriculum is True


def test_singular_variant_alias_is_parsed(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--variant", "tsp,cvrp"],
    )

    args = parse_args()

    assert args.variants == ["tsp", "cvrp"]


def test_missing_references_are_marked_missing_without_classical_cache(
    monkeypatch,
) -> None:
    class FakeSavedProblems:
        def __init__(self, size: int, dataset_dir) -> None:
            pass

        def load(self, variant: str, index: int = 0):
            return {"name": variant}, None

    args = _args()
    args.n_node = 20
    args.dataset_dir = "dataset-root"
    args.val_size = 1
    args.variants = ["op"]
    args.allow_missing_validation = False
    monkeypatch.setattr("train.SavedProblems", FakeSavedProblems)

    data = build_validation_data(args)

    assert len(data) == 1
    assert data[0]["reference"] is None
    assert data[0]["reference_source"] == "missing"


def test_pretraining_hyperparameters_are_configurable(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--pretraining-epochs",
            "7",
            "--pretraining-lr",
            "0.0002",
            "--pretraining-aux-scale",
            "0.4",
        ],
    )

    args = parse_args()

    assert args.pretrain_epochs == 7
    assert args.pretrain_lr == pytest.approx(2e-4)
    assert args.pretrain_aux_scale == pytest.approx(0.4)


def test_curriculum_schedule_is_balanced_and_group_distinct() -> None:
    curriculum = VariantCurriculum.default(seed=1234)

    schedule = curriculum.schedule(
        epoch=0, epochs=100, steps=32, group_size=4
    )
    repeated = curriculum.schedule(
        epoch=0, epochs=100, steps=32, group_size=4
    )
    counts = {
        variant: schedule.count(variant)
        for variant in curriculum.eligible(0, 100)
    }

    assert schedule == repeated
    assert set(schedule) == set(curriculum.eligible(0, 100))
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(
        len(set(schedule[start : start + 4])) == 4
        for start in range(0, len(schedule), 4)
    )


def test_curriculum_phasing_is_opt_in() -> None:
    curriculum = VariantCurriculum.default(seed=1234)
    args = SimpleNamespace(
        curriculum=False,
        epochs=100,
        steps_per_epoch=32,
        grad_accum_variants=4,
    )

    unphased = _training_variant_schedule(curriculum, args, epoch=0)
    args.curriculum = True
    phased = _training_variant_schedule(curriculum, args, epoch=0)

    assert set(unphased) == set(curriculum.variants)
    assert set(phased) == set(curriculum.eligible(0, args.epochs))


def test_single_variant_still_accumulates_multiple_instances() -> None:
    curriculum = VariantCurriculum(
        ["tsp", "cvrp"], random.Random(1234), seed=1234
    )
    args = SimpleNamespace(
        curriculum=False,
        epochs=100,
        steps_per_epoch=6,
        grad_accum_variants=4,
    )

    schedule = _training_variant_schedule(curriculum, args, epoch=0)
    group_size = _training_accumulation_size(curriculum, args, epoch=0)

    assert group_size == 4
    assert len(schedule) == args.steps_per_epoch
    assert all(
        set(schedule[start : start + group_size]) == {"tsp", "cvrp"}
        for start in range(0, len(schedule), group_size)
    )

    one_variant = VariantCurriculum(
        ["cvrp"], random.Random(1234), seed=1234
    )
    one_schedule = _training_variant_schedule(one_variant, args, epoch=0)
    one_group_size = _training_accumulation_size(one_variant, args, epoch=0)
    assert one_group_size == 4
    assert one_schedule == ["cvrp"] * args.steps_per_epoch


def test_validation_uses_only_selected_variants(monkeypatch) -> None:
    loaded = []

    class FakeSavedProblems:
        def __init__(self, size: int, dataset_dir) -> None:
            pass

        def load(self, variant: str, index: int = 0) -> tuple[dict, float]:
            loaded.append((variant, index))
            return {"name": variant}, 1.0

    args = _args()
    args.n_node = 20
    args.dataset_dir = "dataset-root"
    args.val_size = 1
    args.variants = ["tsp", "cvrptw"]
    args.allow_missing_validation = False
    monkeypatch.setattr("train.SavedProblems", FakeSavedProblems)

    data = build_validation_data(args)

    assert loaded == [("tsp", 0), ("cvrptw", 0)]
    assert [item["variant"] for item in data] == args.variants
    assert all(item["reference_source"] == "saved" for item in data)


def test_validation_manifest_is_strict_by_default(monkeypatch) -> None:
    class MissingSavedProblems:
        def __init__(self, size: int, dataset_dir) -> None:
            pass

        def load(self, variant: str, index: int = 0):
            raise FileNotFoundError("not found")

    args = _args()
    args.n_node = 20
    args.dataset_dir = "dataset-root"
    args.val_size = 1
    args.variants = ["tsp"]
    args.allow_missing_validation = False
    monkeypatch.setattr("train.SavedProblems", MissingSavedProblems)

    with pytest.raises(RuntimeError, match="selected by --variants"):
        build_validation_data(args)


def test_validation_macro_score_weights_variants_equally(monkeypatch) -> None:
    dataset = [
        {
            "variant": "tsp",
            "problem": {"name": "tsp", "index": 0},
            "reference": 10.0,
            "paired_baseline": {
                "objective": 10.0,
                "direction": "minimize",
                "feasible": True,
            },
        },
        {
            "variant": "tsp",
            "problem": {"name": "tsp", "index": 1},
            "reference": 10.0,
            "paired_baseline": {
                "objective": 10.0,
                "direction": "minimize",
                "feasible": True,
            },
        },
        {
            "variant": "cvrp",
            "problem": {"name": "cvrp"},
            "reference": 10.0,
            "paired_baseline": {
                "objective": 10.0,
                "direction": "minimize",
                "feasible": True,
            },
        },
    ]
    solutions = iter(
        [
            (9.0, {"objective": 9.0, "direction": "minimize", "feasible": True}, {}),
            (9.0, {"objective": 9.0, "direction": "minimize", "feasible": True}, {}),
            (7.0, {"objective": 7.0, "direction": "minimize", "feasible": True}, {}),
        ]
    )
    monkeypatch.setattr("train.infer_instance", lambda *_, **__: next(solutions))

    _, _, macro_gap, metrics = validation(None, dataset, _args())

    assert macro_gap == pytest.approx(-20.0)
    assert metrics["instance_weighted_gap"] == pytest.approx(-50 / 3)
    assert metrics["macro_baseline_improvement_percent"] == pytest.approx(20.0)
    assert metrics["macro_score"] == pytest.approx(20.0)
    assert metrics[
        "instance_weighted_baseline_improvement_percent"
    ] == pytest.approx(50 / 3)


def test_validation_macro_gap_uses_saved_references_only(monkeypatch) -> None:
    dataset = [
        {
            "variant": "tsp",
            "instance_index": 0,
            "problem": {"name": "tsp", "index": 0},
            "reference": 10.0,
            "reference_source": "saved",
        },
        {
            "variant": "tsp",
            "instance_index": 1,
            "problem": {"name": "tsp", "index": 1},
            "reference": None,
            "reference_source": "missing",
        },
    ]
    solutions = iter(
        [
            (11.0, {"objective": 11.0, "direction": "minimize", "feasible": True}, {}),
            (99.0, {"objective": 99.0, "direction": "minimize", "feasible": True}, {}),
        ]
    )
    monkeypatch.setattr("train.infer_instance", lambda *_, **__: next(solutions))

    _, _, macro_gap, metrics = validation(None, dataset, _args())

    assert macro_gap == pytest.approx(10.0)
    assert metrics["gap_instances"] == 1.0
    assert metrics["gap_coverage"] == pytest.approx(0.5)
    assert metrics["saved_reference_instances"] == 1.0
    assert metrics["missing_reference_instances"] == 1.0


def test_validation_reports_average_cost_by_semantic_group(monkeypatch) -> None:
    symmetric = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    asymmetric = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    dataset = [
        {
            "variant": "tsp",
            "problem": problem_schema("tsp") | {"distance": symmetric},
            "reference": None,
        },
        {
            "variant": "atsp",
            "problem": problem_schema("atsp") | {"distance": asymmetric},
            "reference": None,
        },
        {
            "variant": "cvrpbptw",
            "problem": problem_schema("cvrpbptw")
            | {
                "distance": symmetric,
                "demand": np.array([0.0, -1.0], dtype=np.float32),
            },
            "reference": None,
        },
    ]
    solutions = iter(
        [
            (4.0, {"objective": 4.0, "direction": "minimize", "feasible": True}, {}),
            (6.0, {"objective": 6.0, "direction": "minimize", "feasible": True}, {}),
            (10.0, {"objective": 10.0, "direction": "minimize", "feasible": True}, {}),
        ]
    )
    monkeypatch.setattr("train.infer_instance", lambda *_, **__: next(solutions))

    _, _, _, metrics = validation(None, dataset, _args())

    assert metrics["group_cost/symmetric"] == pytest.approx(7.0)
    assert metrics["group_cost/asymmetric"] == pytest.approx(6.0)
    assert metrics["group_cost/single_route"] == pytest.approx(5.0)
    assert metrics["group_cost/multi_route"] == pytest.approx(10.0)
    assert metrics["group_cost/backhaul"] == pytest.approx(10.0)
    assert metrics["group_cost/backhaul_order"] == pytest.approx(10.0)
    assert metrics["group_cost/time_window"] == pytest.approx(10.0)
    pickup_delivery = problem_schema("pdcvrp") | {
        "demand": np.array([0.0, 1.0, -1.0], dtype=np.float32)
    }
    assert "backhaul" not in _validation_cost_groups(pickup_delivery)


def test_validation_rank_uses_mean_best_cost() -> None:
    complete = {
        "worst_variant_feasibility_rate": 1.0,
        "feasibility_rate": 1.0,
        "baseline_improvement_coverage": 1.0,
        "macro_score": 2.0,
    }

    # Lower mean best cost wins regardless of gap, feasibility, or baseline metrics.
    assert _validation_rank({**complete, "macro_score": 3.0}, 5.0) > (
        _validation_rank(complete, 4.0)
    )
    assert _validation_rank({**complete, "macro_score": 3.0}, 4.0) == (
        _validation_rank(complete, 4.0)
    )
    assert _validation_rank(
        {**complete, "worst_variant_feasibility_rate": 0.99}, -100.0
    ) < _validation_rank(complete, 100.0)
    assert _validation_rank(
        {**complete, "baseline_improvement_coverage": 0.99}, -100.0
    ) < _validation_rank(complete, 100.0)


def test_validation_mean_best_cost_canonicalizes_maximize_objectives(
    monkeypatch,
) -> None:
    dataset = [
        {
            "variant": "tsp",
            "instance_index": 0,
            "problem": {"name": "tsp"},
            "reference": None,
            "reference_source": "missing",
        },
        {
            "variant": "op",
            "instance_index": 0,
            "problem": {"name": "op"},
            "reference": None,
            "reference_source": "missing",
        },
    ]
    solutions = iter(
        [
            (5.0, {"objective": 5.0, "direction": "minimize", "feasible": True}, {}),
            (-12.0, {"objective": 12.0, "direction": "maximize", "feasible": True}, {}),
        ]
    )
    monkeypatch.setattr("train.infer_instance", lambda *_, **__: next(solutions))

    _, average_best_cost, _, _ = validation(None, dataset, _args())

    assert average_best_cost == pytest.approx((5.0 - 12.0) / 2.0)
