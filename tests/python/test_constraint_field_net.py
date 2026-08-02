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
    ConstraintFieldNet,
    build_decoder_data,
    decode_iteration,
    load_constraint_field_state_dict,
)
from train import (  # noqa: E402
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
        output = model(data)

    edge_count = decoder.metadata["edge_count"]
    channel_count = prism_decoder.FIELD_CHANNEL_COUNT
    assert output["residual"].shape == (edge_count, channel_count)
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
    # The residual is now the direct per-edge field (no pressure gate): it is
    # non-negative and exactly zero on inactive channels.
    active = torch.as_tensor(decoder.metadata["field_channel_mask"]).bool()
    assert torch.all(output["residual"] >= 0.0)
    assert torch.all(output["residual"][:, ~active] == 0.0)
    assert torch.all(output["multipliers"] >= 0.0)
    # The objective weight slot (final) is always on; field channels are gated.
    field_multipliers = output["multipliers"][0, :channel_count]
    assert torch.all(field_multipliers[~active] == 0.0)


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
            search_config={"classical_behavior": False},
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
    scales = graph.resource_scales.unsqueeze(0)
    traced = battery.sample_traced(
        edge_field=(output["residual"] * scales).numpy(),
        edge_additive=(output["additive"] * scales).numpy(),
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

    # One-hot objective type and a bounded per-graph scale reach the model.
    assert distance_data.objective_type.shape == (1, 3)
    assert torch.equal(
        distance_data.objective_type, torch.tensor([[1.0, 0.0, 0.0]])
    )
    assert torch.equal(prize_data.objective_type, torch.tensor([[0.0, 1.0, 0.0]]))
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
        search_config={
            "classical_behavior": False,
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
        search_config={"classical_behavior": False},
        n_rollouts=4,
        beta=2.0,
    )
    decoder.seed(3003)
    model = ConstraintFieldNet(depth=2, units=16).eval()

    incumbent = decoder.sample_greedy(**_neutral_guidance(decoder))
    assert incumbent["feasible"]
    decoder.set_incumbent(incumbent["route"])

    graph = build_decoder_data(decoder)
    with torch.no_grad():
        output = model(graph)
    scales = graph.resource_scales.unsqueeze(0)
    traced = decoder.sample_traced(
        edge_field=(output["residual"] * scales).detach().numpy(),
        edge_additive=(output["additive"] * scales).detach().numpy(),
        multipliers=output["multipliers"][0].detach().numpy(),
        coupler_weights=output["coupler_weights"][0].detach().numpy(),
        coupler_bias=output["coupler_bias"][0].detach().numpy(),
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
