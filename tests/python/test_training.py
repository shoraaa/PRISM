import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import prism_decoder
from net import ConstraintFieldNet
from train import (
    OptionOutcome,
    OptionStep,
    _assign_smdp_returns,
    _dual_loss,
    _feasibility_loss,
    _positive_class_weight,
    _rollout_class_weights,
    build_validation_data,
    collect_instance_rollout,
    infer_instance,
    parse_args,
    ppo_update,
    replay_decision_logp_from_cpp_batch_trace,
)


def _args() -> Namespace:
    return Namespace(
        candidates=64,
        val_size=1,
        n_ants=2,
        beta=2.0,
        seed=404,
        device="cpu",
        search_iterations=2,
        option_max_steps=2,
        infeasible_penalty=10.0,
        smdp_gamma=0.99,
        neural_call_cost=0.0,
        improvement_epsilon=0.0,
        ppo_epochs=1,
        pretrain_epochs=1,
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
    args = _args()
    args.smallvram = smallvram
    model = ConstraintFieldNet(depth=1, units=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    binding_before = model.binding_head.weight.detach().clone()
    multiplier_before = model.multiplier_head.weight.detach().clone()
    coupler_before = model.coupler_head.weight.detach().clone()
    coupler_bias_before = model.coupler_bias_head.weight.detach().clone()
    feasibility_before = model.feasibility_head.weight.detach().clone()

    rollout = collect_instance_rollout(model, problem, "cvrp", args)
    metrics = ppo_update(model, optimizer, [rollout], args, epoch=0)

    assert 1 <= rollout.emissions <= args.search_iterations
    assert len(rollout.steps) == args.search_iterations
    assert all(
        1 <= step.duration <= args.option_max_steps for step in rollout.steps
    )
    assert all(
        step.old_logp.numel() == int(step.decisions.sum())
        for step in rollout.steps
    )
    assert {
        "dual_loss",
        "feasibility_loss",
        "binding_loss",
        "price_loss",
        "approx_kl",
        "clip_frac",
    } <= metrics.keys()
    assert not torch.equal(binding_before, model.binding_head.weight.detach())
    assert not torch.equal(multiplier_before, model.multiplier_head.weight.detach())
    assert not torch.equal(
        feasibility_before, model.feasibility_head.weight.detach()
    )
    assert not (
        torch.equal(coupler_before, model.coupler_head.weight.detach())
        and torch.equal(
            coupler_bias_before, model.coupler_bias_head.weight.detach()
        )
    )


@pytest.mark.parametrize("smallvram", [False, True])
def test_decision_level_ppo_moves_policy_without_auxiliary_losses(
    smallvram: bool,
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
    args = _args()
    args.pretrain_epochs = 0
    args.ppo_epochs = 1
    args.smallvram = smallvram
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
        if name.startswith(("field_head", "additive_head", "multiplier_head"))
    }

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
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-10)
    assert abs(metrics["rl_loss"]) < 1e-5
    assert abs(metrics["rl_score_proxy"]) > 1e-6
    assert metrics["policy_signal"] > 0.0
    assert metrics["gradient_norm"] > 0.0
    assert replay_drift and max(replay_drift) > 1e-7
    assert metrics["ppo_reuse_passes"] == 1.0
    assert metrics["ppo_clipping_active"] == 0.0
    assert metrics["auxiliary_scale"] == pytest.approx(args.aux_rl_scale)


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


def test_positive_class_weight_balances_rare_events() -> None:
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])

    assert _positive_class_weight(target) == 3.0
    assert _positive_class_weight(torch.zeros(4)) == 1.0


def test_additive_dual_head_learns_when_analytic_pressure_is_zero() -> None:
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
    residual = torch.ones(1, channels, requires_grad=True)
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
    assert residual.grad[0, 0] == 0.0


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
    args = _args()
    args.search_iterations = 4
    args.improvement_epsilon = float("inf")

    rollout = collect_instance_rollout(
        ConstraintFieldNet(depth=1, units=8), problem, "cvrp", args
    )

    assert rollout.emissions == 1
    assert len({id(step.graph) for step in rollout.steps}) == 1
    assert all(step.trace["screened_edges"].size > 0 for step in rollout.steps)
    assert all(step.resource_delta is None for step in rollout.steps)


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
    args = _args()
    model = ConstraintFieldNet(depth=1, units=8).eval()

    first = infer_instance(model, problem, args)
    second = infer_instance(model, problem, args)

    assert first[0] == second[0]
    assert np.array_equal(first[1]["route"], second[1]["route"])
    assert first[1]["objective"] == second[1]["objective"]


def test_validation_size_loads_each_requested_instance(monkeypatch) -> None:
    loaded_sizes = []
    loaded_instances = []

    class FakeSavedURS:
        def __init__(self, size: int) -> None:
            loaded_sizes.append(size)

        def load(self, variant: str, index: int = 0) -> tuple[dict, float]:
            loaded_instances.append((variant, index))
            return {"name": variant, "index": index}, 1.0

    args = _args()
    args.n_node = 20
    args.val_size = 2
    args.val_seen = 1
    args.val_heldout = 1
    curriculum = SimpleNamespace(variants=["tsp"], held_out=["cvrp"])
    monkeypatch.setattr("train.SavedURS", FakeSavedURS)

    data = build_validation_data(args, curriculum)

    assert loaded_sizes == [20]
    assert loaded_instances == [
        ("tsp", 0),
        ("tsp", 1),
        ("cvrp", 0),
        ("cvrp", 1),
    ]
    assert [
        (item["variant"], item["instance_index"], item["split"])
        for item in data
    ] == [
        ("tsp", 0, "seen"),
        ("tsp", 1, "seen"),
        ("cvrp", 0, "heldout"),
        ("cvrp", 1, "heldout"),
    ]


def test_validation_size_defaults_to_one_instance(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])

    assert parse_args().val_size == 1
