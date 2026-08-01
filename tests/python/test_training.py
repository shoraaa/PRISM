import copy
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import prism_decoder
from net import ConstraintFieldNet
from problem_data import VALIDATION_HELDOUT_VARIANTS, VariantCurriculum
from train import (
    OptionOutcome,
    OptionStep,
    _assign_refresh_gae,
    _assign_smdp_returns,
    _dual_loss,
    _feasibility_loss,
    _load_optimizer_state_compat,
    _positive_class_weight,
    _rollout_class_weights,
    _validation_rank,
    _winner_temporal_advantage,
    build_validation_data,
    collect_instance_rollout,
    infer_instance,
    parse_args,
    ppo_update,
    replay_decision_logp_from_cpp_batch_trace,
    validation,
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
    args.pretrain_aux_scale = 0.25
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
    transition_steps = [
        step for step in rollout.steps if step.transition_ant is not None
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
        torch.equal(coupler_before, model.coupler_head.weight.detach())
        and torch.equal(
            coupler_bias_before, model.coupler_bias_head.weight.detach()
        )
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
        if name.startswith(("field_head", "additive_head", "multiplier_head"))
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
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-10)
    assert abs(metrics["rl_loss"]) < 1e-5
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
            winner_ant=1,
        ),
        OptionOutcome(
            [second_step],
            torch.zeros(2),
            1,
            transition_reward=3.0,
            old_value=0.4,
            transition_step=second_step,
            winner_ant=0,
        ),
    ]

    _assign_refresh_gae(outcomes, gamma=0.5, gae_lambda=0.8)

    assert second_step.temporal_advantage == pytest.approx(2.6)
    assert second_step.value_target == pytest.approx(3.0)
    assert first_transition_step.temporal_advantage == pytest.approx(1.42)
    assert first_transition_step.transition_ant == 1
    assert first_value_step.value_target == pytest.approx(1.62)
    assert not hasattr(first_value_step, "transition_ant")


def test_winner_temporal_advantage_is_non_cancelling_pomo_contrast() -> None:
    advantage = _winner_temporal_advantage(
        ant_count=4,
        winner_ant=2,
        advantage=3.0,
        scale=1.0,
        device=torch.device("cpu"),
    )

    assert torch.allclose(
        advantage, torch.tensor([-0.75, -0.75, 2.25, -0.75])
    )
    assert float(advantage.sum()) == pytest.approx(0.0)
    assert float(advantage.abs().sum()) > 0.0


def test_legacy_optimizer_state_initializes_appended_value_head() -> None:
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
    assert first[2]["net_evals"] >= 2.0
    assert first[2]["emissions"] == first[2]["net_evals"]
    assert second[2]["emissions"] == second[2]["net_evals"]

    args.static_field = True
    static = infer_instance(model, problem, args)

    assert static[2]["net_evals"] == 1.0
    assert static[2]["emissions"] == 1.0


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
    args.val_seen = 1
    args.val_heldout = 1
    curriculum = SimpleNamespace(variants=["tsp"], held_out=["cvrp"])
    monkeypatch.setattr("train.SavedProblems", FakeSavedProblems)

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


def test_validation_size_defaults_to_eight_instances(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = parse_args()

    assert args.val_size == 8
    assert args.val_seen is None
    assert args.val_heldout == 16
    assert args.allow_missing_validation is False
    assert args.static_field is False
    assert args.gae_lambda == pytest.approx(1.0)
    assert args.temporal_credit_weight == pytest.approx(0.1)
    assert args.value_loss_weight == pytest.approx(0.0)


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


def test_default_validation_uses_all_seen_and_stratified_heldout(
    monkeypatch,
) -> None:
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
    args.val_seen = None
    args.val_heldout = 16
    args.allow_missing_validation = False
    curriculum = VariantCurriculum.default(seed=1234)
    monkeypatch.setattr("train.SavedProblems", FakeSavedProblems)

    data = build_validation_data(args, curriculum)

    assert [item["variant"] for item in data] == [
        *curriculum.variants,
        *VALIDATION_HELDOUT_VARIANTS,
    ]
    assert len(data) == 32
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
    args.val_seen = 1
    args.val_heldout = 0
    args.allow_missing_validation = False
    monkeypatch.setattr("train.SavedProblems", MissingSavedProblems)

    with pytest.raises(RuntimeError, match="fixed manifest"):
        build_validation_data(args, VariantCurriculum.default(seed=1234))


def test_validation_macro_score_weights_variants_equally(monkeypatch) -> None:
    dataset = [
        {
            "variant": "tsp",
            "split": "seen",
            "problem": {"name": "tsp", "index": 0},
            "reference": 10.0,
            "classical_baseline": {
                "objective": 10.0,
                "direction": "minimize",
                "feasible": True,
            },
        },
        {
            "variant": "tsp",
            "split": "seen",
            "problem": {"name": "tsp", "index": 1},
            "reference": 10.0,
            "classical_baseline": {
                "objective": 10.0,
                "direction": "minimize",
                "feasible": True,
            },
        },
        {
            "variant": "cvrp",
            "split": "heldout",
            "problem": {"name": "cvrp"},
            "reference": 10.0,
            "classical_baseline": {
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
    monkeypatch.setattr("train.infer_instance", lambda *_: next(solutions))

    _, _, macro_gap, metrics = validation(None, dataset, _args())

    assert macro_gap == pytest.approx(-20.0)
    assert metrics["instance_weighted_gap"] == pytest.approx(-50 / 3)
    assert metrics["macro_baseline_improvement_percent"] == pytest.approx(20.0)
    assert metrics["macro_score"] == pytest.approx(20.0)
    assert metrics[
        "instance_weighted_baseline_improvement_percent"
    ] == pytest.approx(50 / 3)


def test_validation_rank_is_feasibility_and_coverage_first() -> None:
    complete = {
        "worst_variant_feasibility_rate": 1.0,
        "feasibility_rate": 1.0,
        "baseline_improvement_coverage": 1.0,
        "macro_score": 2.0,
    }

    assert _validation_rank({**complete, "macro_score": 3.0}, 5.0) < (
        _validation_rank(complete, 4.0)
    )
    assert _validation_rank(
        {**complete, "worst_variant_feasibility_rate": 0.99}, -100.0
    ) > _validation_rank(complete, 100.0)
    assert _validation_rank(
        {**complete, "baseline_improvement_coverage": 0.99}, -100.0
    ) > _validation_rank(complete, 100.0)
