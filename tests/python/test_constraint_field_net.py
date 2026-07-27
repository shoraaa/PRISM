import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import prism_decoder  # noqa: E402
from net import (  # noqa: E402
    EDGE_FEATURE_COUNT,
    FIELD_CHANNEL_COUNT,
    LIVE_STATE_FEATURE_COUNT,
    NODE_FEATURE_COUNT,
    ConstraintFieldNet,
    build_decoder_data,
    decode_iteration,
)
from train import (  # noqa: E402
    replay_decision_logp_from_cpp_batch_trace,
    replay_logp_from_cpp_batch_trace,
)


def test_constraint_field_net_uses_normalized_decoder_contract() -> None:
    rng = np.random.default_rng(301)
    coordinates = rng.random((30, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.06, 29)].astype(np.float32)
    decoder = prism_decoder.Decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.6,
        },
        n_ants=4,
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
    assert output["multipliers"].shape == (1, channel_count)
    assert output["binding_logits"].shape == (1, channel_count)
    assert output["feasibility_logits"].shape == (edge_count,)
    assert output["feasibility_risk"].shape == (edge_count,)
    assert torch.all(
        (output["feasibility_risk"] >= 0.0)
        & (output["feasibility_risk"] <= 1.0)
    )
    assert torch.allclose(output["residual"], torch.ones_like(output["residual"]))
    assert torch.all(output["multipliers"] >= 0.0)
    active = torch.as_tensor(decoder.metadata["field_channel_mask"]).bool()
    assert torch.all(output["multipliers"][0, ~active] == 0.0)


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
    decoder = prism_decoder.Decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_ants=1,
    )
    data = build_decoder_data(decoder)
    data.edge_attr[0, 0] = 1.01

    with pytest.raises(ValueError, match="normalized to \\[0, 1\\]"):
        ConstraintFieldNet(depth=1, units=8).eval()(data)


def test_resource_attention_handles_no_active_constraint_tokens() -> None:
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = prism_decoder.Decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_ants=1,
    )

    with torch.no_grad():
        output = ConstraintFieldNet(depth=1, units=8).eval()(
            build_decoder_data(decoder)
        )

    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.all(output["multipliers"] == 0.0)


def test_model_output_drives_field_decoder_iteration() -> None:
    rng = np.random.default_rng(302)
    coordinates = rng.random((26, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.05, 25)].astype(np.float32)
    decoder = prism_decoder.Decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.5,
        },
        search_config={
            "classical_behavior": False,
            "use_pheromone": False,
        },
        n_ants=4,
    )
    decoder.seed(3002)
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
    decoder = prism_decoder.Decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.5,
        },
        search_config={"classical_behavior": False, "use_pheromone": False},
        n_ants=4,
        beta=2.0,
    )
    decoder.seed(3003)
    model = ConstraintFieldNet(depth=2, units=16).eval()

    initial_graph = build_decoder_data(decoder)
    with torch.no_grad():
        initial = model(initial_graph)
    decoder.solve(
        1,
        edge_field=initial["residual"].detach().numpy(),
        multipliers=initial["multipliers"][0].detach().numpy(),
        coupler_weights=initial["coupler_weights"][0].detach().numpy(),
        coupler_bias=initial["coupler_bias"][0].detach().numpy(),
    )

    graph = build_decoder_data(decoder)
    with torch.no_grad():
        output = model(graph)
    traced = decoder.sample_traced(
        edge_field=output["residual"].detach().numpy(),
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
    decision_logp, decision_ants, decision_counts = (
        replay_decision_logp_from_cpp_batch_trace(
            trace, graph, output, model, beta=2.0, risk_penalty=3.0
        )
    )

    starts = trace["starts"]
    expected = np.zeros(4, dtype=np.float32)
    for ant in range(4):
        selected = trace["stochastic"][starts[ant] : starts[ant + 1]].astype(bool)
        expected[ant] = trace["log_probabilities"][
            starts[ant] : starts[ant + 1]
        ][selected].sum()
    assert np.allclose(replayed.detach().numpy(), expected, atol=2e-5)
    selected = trace["stochastic"].astype(bool) & (trace["chosen_indices"] >= 0)
    assert np.allclose(
        decision_logp.detach().numpy(),
        trace["log_probabilities"][selected],
        atol=2e-5,
    )
    expected_ants = np.repeat(np.arange(4), np.diff(starts))[selected]
    assert np.array_equal(decision_ants.detach().numpy(), expected_ants)
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
