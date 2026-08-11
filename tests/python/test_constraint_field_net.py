import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import prism_decoder  # noqa: E402
from problem_data import problem_schema  # noqa: E402
from net import (  # noqa: E402
    EDGE_FEATURE_COUNT,
    FIELD_CHANNEL_COUNT,
    LIVE_STATE_FEATURE_COUNT,
    NODE_FEATURE_COUNT,
    OBJECTIVE_COEFF_DIM,
    ConstraintFieldNet,
    build_decoder_data,
    decode_iteration,
    encode_objective_coeffs,
    load_constraint_field_state_dict,
)
from train import (  # noqa: E402
    _guidance_numpy,
    _neutral_guidance,
    replay_decision_logp_from_cpp_batch_trace,
    replay_logp_from_cpp_batch_trace,
)


def make_decoder(problem: dict, *args, **kwargs):
    """Materialize explicit fixture semantics before calling the native API."""
    explicit = problem_schema(str(problem.get("name", "schema")))
    explicit.update(problem)
    return prism_decoder.Decoder(explicit, *args, **kwargs)


def test_constraint_field_net_uses_normalized_decoder_contract() -> None:
    rng = np.random.default_rng(301)
    coordinates = rng.random((30, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.06, 29)].astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.6,
        },
        n_rollouts=4,
    )
    decoder.seed(3001)
    decoder.solve(1)

    data = build_decoder_data(decoder)
    model = ConstraintFieldNet(depth=2, units=16).eval()
    with torch.no_grad():
        # Legacy objective-head parameters remain loadable but must not reopen
        # the graph-level temperature shortcut on constrained instances.
        model.objective_multiplier_head.weight.fill_(3.0)
        model.objective_multiplier_head.bias.fill_(3.0)
        model.objective_coupler_query_head.weight.fill_(3.0)
        model.objective_coupler_query_head.bias.fill_(3.0)
        model.objective_coupler_bias_head.weight.fill_(3.0)
        model.objective_coupler_bias_head.bias.fill_(3.0)
        output = model(data)

    edge_count = decoder.metadata["edge_count"]
    channel_count = prism_decoder.FIELD_CHANNEL_COUNT
    assert output["residual"].shape == (edge_count, channel_count)
    assert output["objective_residual"].shape == (edge_count,)
    # Small non-zero at init (near-neutral) so the head's hidden coefficient-
    # conditioning layer is not gradient-starved; not exactly zero like the
    # resource field/additive heads.
    assert output["objective_residual"].abs().max() < 0.1
    assert output["multipliers"].shape == (1, prism_decoder.MULTIPLIER_COUNT)
    assert output["binding_logits"].shape == (1, channel_count)
    assert output["feasibility_logits"].shape == (edge_count,)
    assert output["feasibility_risk"].shape == (edge_count,)
    assert output["value_context"].shape == (1, 16)
    assert output["candidate_quota"].shape == (1, channel_count)
    assert output["candidate_quota_logits"].shape == (1, channel_count + 1)
    assert torch.allclose(
        output["candidate_quota"].sum(dim=1)
        + torch.softmax(output["candidate_quota_logits"], dim=1)[:, -1],
        torch.ones(1),
    )
    assert model.value(output, 0.5).shape == (1,)
    assert torch.equal(model.value(output, 0.5), torch.zeros(1))
    assert torch.all(
        (output["feasibility_risk"] >= 0.0)
        & (output["feasibility_risk"] <= 1.0)
    )
    # Resource fields are signed and exactly zero-neutral at initialization.
    # Inactive channels remain masked to zero after learning.
    active = torch.as_tensor(decoder.metadata["field_channel_mask"]).bool()
    assert torch.equal(output["residual"], torch.zeros_like(output["residual"]))
    assert torch.equal(output["additive"], torch.zeros_like(output["additive"]))
    assert torch.all(output["residual"][:, ~active] == 0.0)
    assert torch.all(output["multipliers"] >= 0.0)
    # The objective weight slot is a fixed unit anchor; field channels learn
    # their strength relative to it.
    field_multipliers = output["multipliers"][0, :channel_count]
    assert torch.all(field_multipliers[~active] == 0.0)
    assert output["multipliers"][0, -1] == 1.0
    assert torch.equal(
        output["coupler_weights"][0, -1],
        torch.zeros_like(output["coupler_weights"][0, -1]),
    )
    assert output["coupler_bias"][0, -1] == 0.0
    live_state = torch.rand(5, channel_count)
    assert torch.equal(model.couple(output, live_state)[:, -1], torch.ones(5))
    assert _guidance_numpy(output, data, risk_penalty=10.0)[
        "risk_penalty"
    ] == 10.0


def test_v2_checkpoint_loader_is_strict_and_rejects_v1_identity() -> None:
    original = ConstraintFieldNet(depth=1, units=8)
    incomplete_v2 = {
        key: value
        for key, value in original.state_dict().items()
        if not key.startswith("value_head.")
    }
    restored = ConstraintFieldNet(depth=1, units=8)

    with pytest.raises(RuntimeError, match="v2 checkpoint"):
        load_constraint_field_state_dict(restored, incomplete_v2)
    v1 = dict(original.state_dict())
    v1["resource_types"] = torch.eye(prism_decoder.FIELD_CHANNEL_COUNT)
    with pytest.raises(RuntimeError, match="v1 checkpoint"):
        load_constraint_field_state_dict(restored, v1)
    assert load_constraint_field_state_dict(restored, original.state_dict()) is False

    pre_logit_v2 = {
        key: value
        for key, value in original.state_dict().items()
        if not key.startswith(
            ("edge_logit_head.", "objective_energy_residual_head.")
        )
    }
    assert load_constraint_field_state_dict(restored, pre_logit_v2) is True
    # A pre-objective checkpoint injects the model's own fresh head init: small
    # non-zero final weight (so the hidden layer is not gradient-starved) with a
    # zero bias so the residual starts near-neutral.
    final_layer = restored.objective_energy_residual_head[-1]
    assert final_layer.weight.abs().sum() > 0.0
    assert torch.equal(final_layer.bias, torch.zeros_like(final_layer.bias))


def test_python_dimensions_come_from_cpp_extension() -> None:
    assert FIELD_CHANNEL_COUNT == prism_decoder.FIELD_CHANNEL_COUNT
    assert LIVE_STATE_FEATURE_COUNT == prism_decoder.LIVE_STATE_FEATURE_COUNT
    assert NODE_FEATURE_COUNT == prism_decoder.NODE_FEATURE_COUNT
    assert EDGE_FEATURE_COUNT == prism_decoder.EDGE_FEATURE_COUNT


def test_coupler_supports_states_from_multiple_graphs() -> None:
    model = ConstraintFieldNet(depth=1, units=8)
    channels = prism_decoder.FIELD_CHANNEL_COUNT
    states = prism_decoder.LIVE_STATE_FEATURE_COUNT
    output = {
        "multipliers": torch.arange(2 * channels, dtype=torch.float32).view(
            2, channels
        ),
        "coupler_weights": torch.zeros(2, channels, states),
        "coupler_bias": torch.zeros(2, channels),
    }
    live_state = torch.zeros(3, states)
    graph_index = torch.tensor([0, 1, 1])

    coupled = model.couple(output, live_state, graph_index)

    assert torch.equal(coupled, output["multipliers"][graph_index])


def test_constraint_field_net_rejects_unnormalized_inputs() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=1,
    )
    data = build_decoder_data(decoder)
    data.edge_attr[0, 0] = 1.01

    with pytest.raises(ValueError, match="normalized to \\[0, 1\\]"):
        ConstraintFieldNet(depth=1, units=8).eval()(data)


def test_typed_field_accepts_unseen_runtime_resource_without_new_weights() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [0.8, 0.0], [0.4, 0.0]], dtype=np.float32
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)

    def make(resource_name: str):
        return make_decoder(
            {
                "name": "cvrp",
                "coordinates": coordinates,
                "distance": distance,
                "constraints": [],
                "multi_route": False,
                "resources": [
                    {
                        "name": resource_name,
                        "operator": "affine_accumulator",
                        "initial": 1.0,
                        "scale": 1.0,
                        "increment": {
                            "edge_attribute": "distance",
                            "coefficient": -1.0,
                        },
                        "bounds": [{"lower": 0.0}],
                    }
                ],
            },
            n_rollouts=1,
        )

    battery = make("battery")
    renamed = make("fuel_remaining")
    graph = build_decoder_data(battery)
    renamed_graph = build_decoder_data(renamed)
    assert torch.equal(graph.resource_descriptors, renamed_graph.resource_descriptors)

    model = ConstraintFieldNet(depth=1, units=8).eval()
    with torch.no_grad():
        output = model(graph)
        renamed_output = model(renamed_graph)
    resource_count = battery.metadata["resource_count"]
    assert output["residual"].shape == (battery.metadata["edge_count"], resource_count)
    assert output["multipliers"].shape == (1, resource_count + 1)
    assert output["coupler_weights"].shape == (
        1,
        resource_count + 1,
        resource_count,
    )
    assert torch.equal(output["residual"], renamed_output["residual"])
    assert torch.equal(output["multipliers"], renamed_output["multipliers"])
    battery.set_incumbent(np.array([0, 2, 0], dtype=np.int32))
    graph = build_decoder_data(battery)
    with torch.no_grad():
        output = model(graph)
    traced = battery.sample_traced(
        edge_field=output["residual"].numpy(),
        edge_additive=output["additive"].numpy(),
        multipliers=output["multipliers"][0].numpy(),
        coupler_weights=output["coupler_weights"][0].numpy(),
        coupler_bias=output["coupler_bias"][0].numpy(),
        edge_risk=output["feasibility_risk"].numpy(),
    )
    assert all(solution["feasible"] for solution in traced["solutions"])
    assert traced["trace"]["live_state"].shape[1] == resource_count


def test_objective_conditioning_reaches_descriptor_and_field() -> None:
    rng = np.random.default_rng(305)
    coordinates = rng.random((18, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    prize = np.r_[0.0, rng.uniform(0.05, 1.0, 17)].astype(np.float32)

    distance_problem = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=1,
    )
    prize_problem = make_decoder(
        {
            "name": "op",
            "coordinates": coordinates,
            "distance": distance,
            "prize": prize,
            "tour_limit": 4.0,
        },
        n_rollouts=1,
    )

    distance_data = build_decoder_data(distance_problem)
    prize_data = build_decoder_data(prize_problem)

    # The declared objective coefficient vector (not a one-hot type) and a
    # bounded per-graph scale reach the model.
    assert distance_data.objective_coeffs.shape == (1, OBJECTIVE_COEFF_DIM)
    assert torch.allclose(
        distance_data.objective_coeffs,
        encode_objective_coeffs(
            {"distance_coeff": 1.0, "visit_coeff": 0.0, "miss_coeff": 0.0,
             "distance_regularizer": 0.0, "sense": 1.0}
        ),
    )
    assert torch.allclose(
        prize_data.objective_coeffs,
        encode_objective_coeffs(
            {"distance_coeff": 0.0, "visit_coeff": 1.0, "miss_coeff": 0.0,
             "distance_regularizer": 1.0e-3, "sense": -1.0}
        ),
    )
    for data in (distance_data, prize_data):
        assert 0.0 <= float(data.objective_scale) <= 1.0

    # A shared model must produce different fields for different objectives.
    model = ConstraintFieldNet(depth=2, units=16).eval()
    with torch.no_grad():
        distance_output = model(distance_data)
        prize_output = model(prize_data)
    assert not torch.allclose(
        distance_output["multipliers"], prize_output["multipliers"]
    ) or not torch.allclose(
        distance_output["residual"].mean(0), prize_output["residual"].mean(0)
    )


def test_objective_residual_conditions_on_coeffs() -> None:
    """One shared head, conditioned on the declared coefficient vector. The head
    is zero-init (residual inert by design -- an actively-trained objective
    residual was measured net-harmful), but once given non-zero weights it must
    produce different residuals for different objective coefficients on the same
    graph (the conditioning is wired, not dead)."""
    rng = np.random.default_rng(315)
    coordinates = rng.random((18, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=1,
    )
    distance_data = build_decoder_data(decoder)
    prize_data = distance_data.clone()
    prize_data.objective_coeffs = encode_objective_coeffs(
        {"distance_coeff": 0.0, "visit_coeff": 1.0, "miss_coeff": 0.0,
         "distance_regularizer": 1.0e-3, "sense": -1.0}
    )
    model = ConstraintFieldNet(depth=2, units=16)

    # Head starts small non-zero (near-neutral) so it is trainable.
    with torch.no_grad():
        assert model(distance_data)["objective_residual"].abs().max() < 0.1

    # The hidden (coefficient-conditioning) layer must receive gradient from the
    # first step -- a zero-init final layer would freeze it. A linear-in-residual
    # signal mimics the PPO policy gradient.
    output = model(distance_data)
    residual = output["objective_residual"]
    (residual * torch.randn_like(residual)).sum().backward()
    head = model.objective_energy_residual_head
    assert head[0].weight.grad is not None and head[0].weight.grad.norm() > 0.0

    # A non-trivial head must respond to the coefficient conditioning.
    model.eval()
    with torch.no_grad():
        for parameter in model.objective_energy_residual_head[-1].parameters():
            torch.nn.init.normal_(parameter, std=0.5)
        distance_logits = model(distance_data)["objective_residual"]
        prize_logits = model(prize_data)["objective_residual"]

    assert distance_logits.std() > 0.0
    assert not torch.allclose(distance_logits, prize_logits)


def test_depot_conditioning_reaches_descriptor_and_field() -> None:
    import problem_data

    single = make_decoder(
        problem_data.generated_problem("ocvrp", 16), n_rollouts=1
    )
    multi = make_decoder(
        problem_data.generated_problem("mdocvrp", 16), n_rollouts=1
    )
    single_data = build_decoder_data(single)
    multi_data = build_decoder_data(multi)

    # The raw depot count is squashed to [0, 1): 1 -> 0.5, 3 -> 0.75.
    assert float(single_data.depot_scale) == pytest.approx(0.5)
    assert float(multi_data.depot_scale) == pytest.approx(0.75)
    assert float(single_data.multi_route) == 1.0
    assert float(multi_data.multi_route) == 1.0
    for data in (single_data, multi_data):
        assert 0.0 <= float(data.depot_scale) < 1.0

    # Neutralizing the depot conditioning on the same graph must move the field,
    # proving the signal reaches the shaped multipliers rather than being inert.
    model = ConstraintFieldNet(depth=2, units=16).eval()
    neutral_data = multi_data.clone()
    neutral_data.depot_scale = torch.zeros(1, 1)
    neutral_data.multi_route = torch.zeros(1, 1)
    with torch.no_grad():
        conditioned = model(multi_data)
        neutral = model(neutral_data)
    assert not torch.allclose(
        conditioned["multipliers"], neutral["multipliers"]
    ) or not torch.allclose(
        conditioned["residual"].mean(0), neutral["residual"].mean(0)
    )


def test_resource_attention_handles_no_active_constraint_tokens() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=1,
    )

    with torch.no_grad():
        output = ConstraintFieldNet(depth=1, units=8).eval()(
            build_decoder_data(decoder)
        )

    assert all(torch.isfinite(value).all() for value in output.values())
    # No active constraints -> every resource-field intensity is zero (the
    # always-on objective weight slot is excluded).
    assert torch.all(
        output["multipliers"][0, : prism_decoder.FIELD_CHANNEL_COUNT] == 0.0
    )


def test_model_output_drives_field_decoder_iteration() -> None:
    rng = np.random.default_rng(302)
    coordinates = rng.random((26, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.05, 25)].astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.5,
        },
        n_rollouts=4,
    )
    decoder.seed(3002)
    with pytest.raises(ValueError, match="requires a feasible installed incumbent"):
        decode_iteration(decoder, ConstraintFieldNet(depth=2, units=16).eval())
    incumbent = decoder.sample_greedy(**_neutral_guidance(decoder))
    assert incumbent["feasible"]
    decoder.set_incumbent(incumbent["route"])
    solution, _ = decode_iteration(
        decoder, ConstraintFieldNet(depth=2, units=16).eval()
    )

    assert solution["feasible"]
    assert decoder.evaluate(solution["route"])["feasible"]


def test_cpp_trace_replays_exact_state_dependent_policy() -> None:
    rng = np.random.default_rng(303)
    coordinates = rng.random((32, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.05, 31)].astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.5,
        },
        n_rollouts=4,
        beta=2.0,
    )
    decoder.seed(3003)
    model = ConstraintFieldNet(depth=2, units=16).eval()
    with torch.no_grad():
        torch.nn.init.normal_(
            model.objective_energy_residual_head[-1].weight, std=0.2
        )

    incumbent = decoder.sample_greedy(**_neutral_guidance(decoder))
    assert incumbent["feasible"]
    decoder.set_incumbent(incumbent["route"])

    graph = build_decoder_data(decoder)
    with torch.no_grad():
        output = model(graph)
    assert output["objective_residual"].std() > 0.0
    traced = decoder.sample_traced(
        edge_field=output["residual"].detach().numpy(),
        edge_additive=output["additive"].detach().numpy(),
        multipliers=output["multipliers"][0].detach().numpy(),
        coupler_weights=output["coupler_weights"][0].detach().numpy(),
        coupler_bias=output["coupler_bias"][0].detach().numpy(),
        objective_residual=output["objective_residual"].detach().numpy(),
        edge_risk=output["feasibility_risk"].detach().numpy(),
        risk_penalty=3.0,
    )
    trace = traced["trace"]
    replayed, decisions = replay_logp_from_cpp_batch_trace(
        trace, graph, output, model, beta=2.0, risk_penalty=3.0
    )
    decision_logp, decision_rollouts, decision_counts = (
        replay_decision_logp_from_cpp_batch_trace(
            trace, graph, output, model, beta=2.0, risk_penalty=3.0
        )
    )

    starts = trace["starts"]
    expected = np.zeros(4, dtype=np.float32)
    for rollout in range(4):
        selected = trace["stochastic"][starts[rollout] : starts[rollout + 1]].astype(bool)
        expected[rollout] = trace["log_probabilities"][
            starts[rollout] : starts[rollout + 1]
        ][selected].sum()
    assert np.allclose(replayed.detach().numpy(), expected, atol=2e-5)
    selected = trace["stochastic"].astype(bool) & (trace["chosen_indices"] >= 0)
    assert np.allclose(
        decision_logp.detach().numpy(),
        trace["log_probabilities"][selected],
        atol=2e-5,
    )
    expected_rollouts = np.repeat(np.arange(4), np.diff(starts))[selected]
    assert np.array_equal(decision_rollouts.detach().numpy(), expected_rollouts)
    assert torch.equal(decision_counts, decisions)
    assert np.all(decisions.detach().numpy() >= 0)
    assert trace["live_state"].shape[1] == prism_decoder.LIVE_STATE_FEATURE_COUNT
    assert np.all((trace["live_state"] >= 0.0) & (trace["live_state"] <= 1.0))
    assert trace["screened_edges"].size > 0
    assert trace["screened_resource_delta"].shape == (
        trace["screened_edges"].shape[0],
        prism_decoder.FIELD_CHANNEL_COUNT,
    )
    assert np.all(
        (trace["screened_resource_delta"] >= 0.0)
        & (trace["screened_resource_delta"] <= 1.0)
    )
    assert trace["feasibility_risk_labels"].shape == trace["feasibility_edges"].shape
    assert np.all(
        (trace["feasibility_risk_labels"] == 0.0)
        | (trace["feasibility_risk_labels"] == 1.0)
    )
    assert trace["feasibility_risk_labels"].size > 0


def test_tsp_edge_logit_receives_objective_policy_gradient() -> None:
    rng = np.random.default_rng(304)
    coordinates = rng.random((24, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=6,
        beta=2.0,
    )
    decoder.seed(3004)
    graph = build_decoder_data(decoder)
    model = ConstraintFieldNet(depth=1, units=8)
    output = model(graph)
    traced = decoder.sample_traced(
        edge_field=output["residual"].detach().numpy(),
        edge_additive=output["additive"].detach().numpy(),
        multipliers=output["multipliers"][0].detach().numpy(),
        coupler_weights=output["coupler_weights"][0].detach().numpy(),
        coupler_bias=output["coupler_bias"][0].detach().numpy(),
        objective_residual=output["objective_residual"].detach().numpy(),
        edge_risk=output["feasibility_risk"].detach().numpy(),
    )
    replayed, _ = replay_logp_from_cpp_batch_trace(
        traced["trace"], graph, output, model, beta=2.0
    )
    advantages = torch.linspace(-1.0, 1.0, replayed.numel())
    loss = -(replayed * advantages).mean()
    loss.backward()

    gradient = model.objective_energy_residual_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0.0
    assert not torch.as_tensor(
        decoder.metadata["field_channel_mask"]
    ).any()


def test_default_depth_tsp_edge_logit_does_not_saturate_constant() -> None:
    rng = np.random.default_rng(305)
    coordinates = rng.random((32, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=2,
    )
    graph = build_decoder_data(decoder)
    model = ConstraintFieldNet().eval()
    with torch.no_grad():
        torch.nn.init.normal_(
            model.objective_energy_residual_head[-1].weight, std=0.2
        )
        output = model(graph)

    assert output["objective_residual"].std() > 1e-4
    assert output["objective_residual"].amax() > output["objective_residual"].amin()
    assert output["multipliers"][0, -1] == 1.0
    assert torch.equal(
        output["coupler_weights"][0, -1],
        torch.zeros_like(output["coupler_weights"][0, -1]),
    )
    assert output["coupler_bias"][0, -1] == 0.0
    assert _guidance_numpy(output, graph, risk_penalty=10.0)[
        "risk_penalty"
    ] == 10.0


def test_objective_view_does_not_change_legacy_dynamic_batch_norm() -> None:
    rng = np.random.default_rng(1305)
    coordinates = rng.random((24, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.r_[0.0, rng.uniform(0.01, 0.06, 23)].astype(
                np.float32
            ),
            "capacity": 0.5,
        },
        n_rollouts=2,
    )
    incumbent = decoder.sample_greedy(**_neutral_guidance(decoder))
    decoder.set_incumbent(incumbent["route"])
    graph = build_decoder_data(decoder)
    model = ConstraintFieldNet(depth=2, units=16)
    legacy = ConstraintFieldNet(depth=2, units=16)
    legacy.load_state_dict(model.state_dict())
    embeddings = []
    hook = model.emb_net.register_forward_hook(
        lambda _module, _inputs, output: embeddings.append(output.detach())
    )

    model.train()
    model(graph)
    hook.remove()
    legacy.train()
    expected_dynamic = legacy.emb_net(
        graph.x, graph.edge_index, graph.edge_attr
    ).detach()

    assert len(embeddings) == 2
    assert embeddings[0].shape == expected_dynamic.shape
    assert embeddings[1].shape == expected_dynamic.shape
    assert torch.allclose(embeddings[0], expected_dynamic)
    for actual_layer, legacy_layer in zip(
        model.emb_net.layers, legacy.emb_net.layers
    ):
        for actual_norm, legacy_norm in (
            (actual_layer.v_bn.module, legacy_layer.v_bn.module),
            (actual_layer.e_bn.module, legacy_layer.e_bn.module),
        ):
            assert torch.equal(
                actual_norm.running_mean, legacy_norm.running_mean
            )
            assert torch.equal(
                actual_norm.running_var, legacy_norm.running_var
            )


def test_edge_logit_is_invariant_to_incumbent_state() -> None:
    rng = np.random.default_rng(306)
    coordinates = rng.random((24, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.r_[0.0, rng.uniform(0.01, 0.06, 23)].astype(
                np.float32
            ),
            "capacity": 0.5,
        },
        n_rollouts=2,
    )
    model = ConstraintFieldNet(depth=2, units=16).eval()
    with torch.no_grad():
        torch.nn.init.normal_(
            model.objective_energy_residual_head[-1].weight, std=0.2
        )
        model.field_head.weight.copy_(
            torch.linspace(-0.2, 0.2, 16).view(1, -1)
        )
        empty_graph = build_decoder_data(decoder)
        empty_output = model(empty_graph)
        incumbent = decoder.sample_greedy(**_neutral_guidance(decoder))
        decoder.set_incumbent(incumbent["route"])
        incumbent_graph = build_decoder_data(decoder)
        incumbent_output = model(incumbent_graph)

    assert torch.equal(empty_graph.edge_index, incumbent_graph.edge_index)
    assert torch.count_nonzero(incumbent_graph.x[:, 12]) > 0
    assert torch.count_nonzero(incumbent_graph.edge_attr[:, 8:10]) > 0
    assert torch.allclose(
        empty_output["objective_residual"], incumbent_output["objective_residual"]
    )
    # Resource guidance remains state-conditioned.
    assert not torch.equal(
        empty_output["residual"], incumbent_output["residual"]
    )
