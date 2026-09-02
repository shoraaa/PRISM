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
    CORE_INTERFACE_LEVELS,
    MODEL_SCHEMA,
    LEGACY_NO_OBJECTIVE_CHANNEL_SCHEMA,
    LEGACY_NO_OBJECTIVE_RESIDUAL_SCHEMA,
    EDGE_FEATURE_COUNT,
    FIELD_CHANNEL_COUNT,
    EDGE_RESOURCE_FEATURE_COUNT,
    OBJECTIVE_NODE_TERM_COUNT,
    OBJECTIVE_SUMMARY_DIM,
    INCUMBENT_EDGE_FEATURE_END,
    INCUMBENT_EDGE_FEATURE_START,
    NODE_FEATURE_COUNT,
    NODE_RESOURCE_FEATURE_COUNT,
    RESOURCE_SUFFIX_FEATURE_COUNT,
    NODE_RESOURCE_SUMMARY_DIM,
    OBJECTIVE_COEFF_DIM,
    RESOURCE_ROW_PROPERTY_DIM,
    RESOURCE_TERM_PROPERTY_DIM,
    RESOURCE_TYPE_DIM,
    ResourceProgramEncoder,
    ResourcePool,
    ConstraintFieldNet,
    build_decoder_data,
    decode_iteration,
    encode_objective_coeffs,
    load_constraint_field_state_dict,
    migrate_constraint_field_optimizer_state_dict,
)
from train import (  # noqa: E402
    _field_guidance,
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


def test_resource_program_encoder_pools_term_sets_permutation_invariantly() -> None:
    torch.manual_seed(300)
    encoder = ResourceProgramEncoder(units=16).eval()
    rows = torch.rand(2, RESOURCE_ROW_PROPERTY_DIM)
    terms = torch.rand(5, RESOURCE_TERM_PROPERTY_DIM)
    counts = torch.tensor([2, 3])

    original = encoder(rows, terms, counts)
    permuted = encoder(rows, terms[torch.tensor([1, 0, 4, 2, 3])], counts)
    torch.testing.assert_close(original, permuted)
    assert original.shape == (2, RESOURCE_TYPE_DIM)

    extended = encoder(
        rows,
        torch.cat((terms[:2], terms[:1], terms[2:])),
        torch.tensor([3, 3]),
    )
    assert not torch.allclose(original[0], extended[0])


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
    # One column per registry row of this problem, not per compiled channel.
    channel_count = decoder.metadata["resource_count"]
    assert output["residual"].shape == (edge_count, channel_count)
    assert output["semantic_margin"].shape == (edge_count, channel_count)
    # The objective is a channel of the same field, but it keeps only the field
    # head: it has no bound, so no signed admissibility margin to supervise.
    assert output["objective_residual"].shape == (edge_count,)
    assert output["semantic_margin"].shape[1] == channel_count
    assert output["multipliers"].shape == (
        1,
        decoder.metadata["multiplier_count"],
    )
    assert output["binding_logits"].shape == (1, channel_count)
    assert output["value_context"].shape == (1, 16)
    assert model.value(output, 0.5).shape == (1,)
    assert torch.equal(model.value(output, 0.5), torch.zeros(1))
    assert "feasibility_risk" not in output
    assert "additive" not in output
    # Resource fields are signed and exactly zero-neutral at initialization.
    # Inactive channels remain masked to zero after learning.
    active = torch.as_tensor(decoder.metadata["field_channel_mask"]).bool()
    assert torch.equal(output["residual"], torch.zeros_like(output["residual"]))
    assert torch.equal(
        output["semantic_margin"],
        torch.zeros_like(output["semantic_margin"]),
    )
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
    # v14 deleted the risk channel: its head produced a value that was constant
    # within a graph, and a constant added to every candidate at a node cancels
    # in the comparison that picks one, so it could never change a decision.
    guidance = _guidance_numpy(output, data)
    assert "edge_risk" not in guidance and "edge_additive" not in guidance


def _pooling_fixture(seed: int = 907, name: str = "cvrptw"):
    """A decoder whose registry has more than one active resource."""
    rng = np.random.default_rng(seed)
    coordinates = rng.random((24, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    problem = {
        "name": name,
        "coordinates": coordinates,
        "distance": distance,
        "demand": np.r_[0.0, rng.uniform(0.01, 0.06, 23)].astype(np.float32),
        "capacity": 0.6,
        "tw_start": np.r_[0.0, rng.uniform(0.0, 1.0, 23)].astype(np.float32),
        "tw_end": np.r_[6.0, rng.uniform(2.0, 6.0, 23)].astype(np.float32),
        "service_time": np.full(24, 0.02, dtype=np.float32),
    }
    decoder = make_decoder(problem, n_rollouts=4)
    decoder.seed(9071)
    decoder.solve(1)
    return build_decoder_data(decoder)


def test_resource_pooling_widens_both_encoder_inputs_and_ablation_narrows_them() -> None:
    pooled = ConstraintFieldNet(depth=1, units=8)
    ablated = ConstraintFieldNet(depth=1, units=8, pool_node_resources=False)

    assert pooled.emb_net.v_lin0.weight.shape[1] == (
        NODE_FEATURE_COUNT + OBJECTIVE_SUMMARY_DIM + NODE_RESOURCE_SUMMARY_DIM
    )
    assert pooled.emb_net.e_lin0.weight.shape[1] == (
        EDGE_FEATURE_COUNT + NODE_RESOURCE_SUMMARY_DIM
    )
    assert ablated.emb_net.v_lin0.weight.shape[1] == NODE_FEATURE_COUNT
    assert ablated.emb_net.e_lin0.weight.shape[1] == EDGE_FEATURE_COUNT
    assert ablated.emb_net.node_resource_encoder is None
    assert ablated.emb_net.edge_resource_encoder is None
    assert ablated.emb_net.objective_node_encoder is None
    assert pooled.emb_net.node_resource_encoder.attr_proj.in_features == (
        NODE_RESOURCE_FEATURE_COUNT + 2 + RESOURCE_SUFFIX_FEATURE_COUNT
    )
    assert pooled.resource_edge_projection.in_features == (
        2 + NODE_RESOURCE_FEATURE_COUNT + 2 * 8
    )
    assert ablated.resource_edge_projection.in_features == (
        2 + NODE_RESOURCE_FEATURE_COUNT
    )
    # Both must still run: the ablation is a training configuration, not a
    # broken model.
    data = _pooling_fixture()
    with torch.no_grad():
        pooled.eval()(data)
        ablated.eval()(data)


def test_node_resource_summary_ignores_registry_order_and_inactive_rows() -> None:
    """The pooled summary must read semantics, not registry position."""
    model = ConstraintFieldNet(depth=1, units=8).eval()
    emb_net = model.emb_net
    torch.manual_seed(11)
    nodes, resources = 6, 4
    node_resource = torch.rand(nodes, resources, NODE_RESOURCE_FEATURE_COUNT)
    descriptors = torch.rand(nodes, resources, RESOURCE_TYPE_DIM)
    active = torch.tensor([1.0, 1.0, 0.0, 0.0]).expand(nodes, resources)
    x = torch.zeros(nodes, NODE_FEATURE_COUNT)

    live = torch.rand(nodes, resources)
    suffix = torch.rand(nodes, resources)
    suffix_features = torch.rand(
        nodes, resources, RESOURCE_SUFFIX_FEATURE_COUNT
    )
    objective = torch.rand(nodes, OBJECTIVE_NODE_TERM_COUNT)
    coeffs = torch.rand(nodes, OBJECTIVE_COEFF_DIM)
    with torch.no_grad():
        base = emb_net.augment_nodes(
            x, node_resource, live, suffix, suffix_features, objective, coeffs,
            descriptors, active,
        )
        # Permuting rows carries each descriptor with its attributes and its
        # live state, so a masked reduction over resources cannot notice.
        order = torch.tensor([1, 0, 3, 2])
        permuted = emb_net.augment_nodes(
            x,
            node_resource[:, order],
            live[:, order],
            suffix[:, order],
            suffix_features[:, order],
            objective,
            coeffs,
            descriptors[:, order],
            active[:, order],
        )
        # Rewriting an INACTIVE row must not move the summary either.
        disturbed = node_resource.clone()
        disturbed[:, 2:] = torch.rand_like(disturbed[:, 2:])
        masked = emb_net.augment_nodes(
            x, disturbed, live, suffix, suffix_features, objective, coeffs,
            descriptors, active
        )

    assert base.shape == (
        nodes,
        NODE_FEATURE_COUNT + OBJECTIVE_SUMMARY_DIM + NODE_RESOURCE_SUMMARY_DIM,
    )
    torch.testing.assert_close(base, permuted)
    torch.testing.assert_close(base, masked)
    # The appended blocks come after the published columns, which pass through.
    torch.testing.assert_close(base[:, :NODE_FEATURE_COUNT], x)


def test_node_resource_summary_is_neutral_with_no_active_resources() -> None:
    """An all-inactive registry reduces to zero rather than to a stray value."""
    model = ConstraintFieldNet(depth=1, units=8).eval()
    emb_net = model.emb_net
    torch.manual_seed(12)
    nodes, resources = 3, 5
    with torch.no_grad():
        summary = emb_net.augment_nodes(
            torch.zeros(nodes, NODE_FEATURE_COUNT),
            torch.rand(nodes, resources, NODE_RESOURCE_FEATURE_COUNT),
            torch.rand(nodes, resources),
            torch.rand(nodes, resources),
            torch.rand(nodes, resources, RESOURCE_SUFFIX_FEATURE_COUNT),
            torch.rand(nodes, OBJECTIVE_NODE_TERM_COUNT),
            torch.rand(nodes, OBJECTIVE_COEFF_DIM),
            torch.rand(nodes, resources, RESOURCE_TYPE_DIM),
            torch.zeros(nodes, resources),
        )
    assert torch.isfinite(summary).all()
    # Only the pooled resource block is forced to zero; the objective block is a
    # direct projection and is unaffected by an empty registry.
    torch.testing.assert_close(
        summary[:, NODE_FEATURE_COUNT + OBJECTIVE_SUMMARY_DIM :],
        torch.zeros(nodes, NODE_RESOURCE_SUMMARY_DIM),
    )


def test_resource_pool_is_equivariant_before_invariant_deepset_reduction() -> None:
    """The shared row map is equivariant and its single sum is invariant."""
    pool = ResourcePool(
        attr_feats=3, descriptor_dim=5, units=8, summary=6
    ).eval()
    assert pool.summary_head[0].in_features == 9

    torch.manual_seed(121)
    attributes = torch.rand(2, 4, 3)
    descriptors = torch.rand(2, 4, 5)
    active = torch.tensor([[1.0, 1.0, 1.0, 0.0]]).expand(2, -1)
    order = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        rows = pool.encode_rows(attributes, descriptors, active)
        original = pool(attributes, descriptors, active)
        permuted_rows = pool.encode_rows(
            attributes[:, order], descriptors[:, order], active[:, order]
        )
        permuted = pool(
            attributes[:, order], descriptors[:, order], active[:, order]
        )
    torch.testing.assert_close(
        rows[:, order], permuted_rows, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)


def test_resource_pool_exposes_active_cardinality() -> None:
    """Duplicating an identical row is not erased as it is by mean pooling."""
    pool = ResourcePool(
        attr_feats=1, descriptor_dim=1, units=4, summary=2
    ).eval()
    with torch.no_grad():
        for parameter in pool.parameters():
            parameter.zero_()
        pool.summary_head[0].weight[0, -1] = 1.0
        pool.summary_head[2].weight[0, 0] = 1.0
    attributes = torch.ones(2, 2, 1)
    descriptors = torch.ones(2, 2, 1)
    active = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    with torch.no_grad():
        summary = pool(attributes, descriptors, active)
    expected = torch.nn.functional.silu(torch.tensor([0.5, 2.0 / 3.0]))
    torch.testing.assert_close(summary[:, 0], expected)
    torch.testing.assert_close(summary[:, 1], torch.zeros(2))


def test_resource_pool_deepset_path_is_trainable_end_to_end() -> None:
    """The invariant summary trains both shared row map and reduction."""
    torch.manual_seed(122)
    pool = ResourcePool(
        attr_feats=2, descriptor_dim=3, units=8, summary=5
    )
    attributes = torch.rand(2, 3, 2, requires_grad=True)
    descriptors = torch.rand(2, 3, 3)
    active = torch.ones(2, 3)
    summary = pool(attributes, descriptors, active)
    assert summary.shape == (2, 5)
    assert summary.detach().std() > 1e-6
    summary.square().mean().backward()
    assert pool.summary_head[0].weight.grad is not None
    assert pool.summary_head[0].weight.grad.abs().sum() > 0
    assert pool.attr_proj.weight.grad is not None
    assert pool.attr_proj.weight.grad.abs().sum() > 0


def test_pooled_live_state_tracks_the_incumbent_and_can_be_masked() -> None:
    """The generic route state reaches the encoder and can be masked explicitly."""
    rng = np.random.default_rng(2291)
    coordinates = rng.random((20, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.r_[0.0, rng.uniform(0.01, 0.06, 19)].astype(np.float32),
            "capacity": 0.5,
        },
        n_rollouts=2,
    )
    model = ConstraintFieldNet(depth=1, units=8).eval()
    emb_net = model.emb_net

    first = decoder.sample_greedy(**_neutral_guidance(decoder))
    decoder.set_incumbent(first["route"])
    before = build_decoder_data(decoder)
    # A different incumbent gives every node a different per-resource state.
    decoder.set_incumbent(list(reversed(first["route"])))
    after = build_decoder_data(decoder)

    assert not torch.equal(before.node_live_state, after.node_live_state)
    with torch.no_grad():
        before_type = model._resource_type_rows(before, 1)
        after_type = model._resource_type_rows(after, 1)
        dynamic_before = emb_net.augment_from_graph(
            before, resource_type=before_type
        )
        dynamic_after = emb_net.augment_from_graph(after, resource_type=after_type)
        static_before = emb_net.augment_from_graph(
            before, resource_type=before_type, live_state=False
        )
        static_after = emb_net.augment_from_graph(
            after, resource_type=after_type, live_state=False
        )

    summary = slice(NODE_FEATURE_COUNT, None)
    # The pooled block moves with the incumbent...
    assert not torch.allclose(
        dynamic_before[:, summary], dynamic_after[:, summary]
    )
    # ...and is identical once the live state is explicitly zeroed.
    torch.testing.assert_close(
        static_before[:, summary], static_after[:, summary]
    )


def test_edge_pooling_appends_a_varying_block_and_keeps_published_columns() -> None:
    """Only the seven compiled channels have a named edge column, so the pooled
    block is the sole edge-side path for any other row."""
    rng = np.random.default_rng(4410)
    coordinates = rng.random((18, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.06, 17)].astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.5,
        },
        n_rollouts=2,
    )
    decoder.set_incumbent(decoder.sample_greedy(**_neutral_guidance(decoder))["route"])
    graph = build_decoder_data(decoder)
    model = ConstraintFieldNet(depth=1, units=8).eval()
    emb_net = model.emb_net

    with torch.no_grad():
        resource_type = model._resource_type_rows(graph, 1)
        augmented = emb_net.augment_edges(
            graph.edge_attr, graph, resource_type=resource_type
        )
    assert augmented.shape == (
        graph.edge_attr.shape[0],
        EDGE_FEATURE_COUNT + NODE_RESOURCE_SUMMARY_DIM,
    )
    torch.testing.assert_close(
        augmented[:, :EDGE_FEATURE_COUNT], graph.edge_attr
    )
    # The pooled block carries the per-resource pressure the named columns only
    # cover for compiled channels, so it must actually vary across edges.
    summary = augmented[:, EDGE_FEATURE_COUNT:]
    assert summary.std(dim=0).max() > 1e-6
    # The pool reads exactly the three per-(edge, resource) quantities the
    # decoder publishes, whatever the registry size happens to be.
    assert (
        emb_net.edge_resource_encoder.attr_proj.in_features
        == EDGE_RESOURCE_FEATURE_COUNT
    )


def test_prepooling_checkpoint_is_rejected_by_the_program_schema() -> None:
    """The term encoder is a clean model-input break, not a width upgrade."""
    torch.manual_seed(13)
    narrow = ConstraintFieldNet(depth=1, units=8, pool_node_resources=False)
    activate_field_heads(narrow, seed=13)
    state = narrow.state_dict()
    assert narrow.emb_net.v_lin0.weight.shape[1] == NODE_FEATURE_COUNT

    pooled = ConstraintFieldNet(depth=1, units=8)
    with pytest.raises(RuntimeError, match=MODEL_SCHEMA):
        load_constraint_field_state_dict(pooled, state)


def test_program_checkpoint_loader_is_strict() -> None:
    original = ConstraintFieldNet(depth=1, units=8)
    incomplete_v2 = {
        key: value
        for key, value in original.state_dict().items()
        if not key.startswith("value_head.")
    }
    restored = ConstraintFieldNet(depth=1, units=8)

    with pytest.raises(RuntimeError, match=MODEL_SCHEMA):
        load_constraint_field_state_dict(restored, incomplete_v2)
    v1 = dict(original.state_dict())
    v1["resource_types"] = torch.eye(prism_decoder.FIELD_CHANNEL_COUNT)
    with pytest.raises(RuntimeError, match=MODEL_SCHEMA):
        load_constraint_field_state_dict(restored, v1)
    assert load_constraint_field_state_dict(restored, original.state_dict()) is None


def test_v15_no_objective_residual_checkpoint_migrates_strictly() -> None:
    original = ConstraintFieldNet(depth=1, units=8, objective_channel=False)
    legacy = dict(original.state_dict())
    legacy.update(
        {
            "objective_energy_residual_head.0.weight": torch.randn(8, 16),
            "objective_energy_residual_head.0.bias": torch.randn(8),
            "objective_energy_residual_head.2.weight": torch.zeros(1, 8),
            "objective_energy_residual_head.2.bias": torch.zeros(1),
        }
    )
    config = {
        "objective_residual_enabled": False,
        "linear_objective_residual_head": False,
    }
    restored = ConstraintFieldNet(depth=1, units=8, objective_channel=False)
    load_constraint_field_state_dict(
        restored,
        legacy,
        model_schema=LEGACY_NO_OBJECTIVE_RESIDUAL_SCHEMA,
        config=config,
    )
    for key, value in original.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)

    nonneutral = dict(legacy)
    nonneutral["objective_energy_residual_head.2.weight"] = torch.ones(1, 8)
    with pytest.raises(RuntimeError, match="incompatible ConstraintFieldNet"):
        load_constraint_field_state_dict(
            restored,
            nonneutral,
            model_schema=LEGACY_NO_OBJECTIVE_RESIDUAL_SCHEMA,
            config=config,
        )
    with pytest.raises(RuntimeError, match="incompatible ConstraintFieldNet"):
        load_constraint_field_state_dict(
            restored,
            legacy,
            model_schema=LEGACY_NO_OBJECTIVE_RESIDUAL_SCHEMA,
            config={**config, "objective_residual_enabled": True},
        )


def test_v15_no_objective_residual_optimizer_group_migrates() -> None:
    model = ConstraintFieldNet(depth=1, units=8, objective_channel=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    current = optimizer.state_dict()
    names = [name for name, _ in model.named_parameters()]
    insertion = names.index("resource_index_embedding.weight")
    current_ids = current["param_groups"][0]["params"]
    dormant_ids = list(range(max(current_ids) + 1, max(current_ids) + 5))
    legacy = {
        **current,
        "param_groups": [dict(current["param_groups"][0])],
    }
    legacy["param_groups"][0]["params"] = (
        current_ids[:insertion] + dormant_ids + current_ids[insertion:]
    )
    migrated = migrate_constraint_field_optimizer_state_dict(
        model,
        legacy,
        model_schema=LEGACY_NO_OBJECTIVE_RESIDUAL_SCHEMA,
        config={
            "objective_residual_enabled": False,
            "linear_objective_residual_head": False,
        },
    )

    assert migrated["param_groups"][0]["params"] == current_ids
    optimizer.load_state_dict(migrated)


def test_objective_channel_is_the_only_learned_signal_on_a_bare_tour() -> None:
    """tsp/atsp declare no constraint, so this is their entire policy.

    Without the objective channel the registry is empty: the field stacks to
    [E, 0], the sole multiplier is the pinned objective constant, and every
    model output is discarded -- logp is constant in theta and the gradient is
    exactly zero. The channel is what makes those variants trainable at all.
    """
    rng = np.random.default_rng(11)
    coordinates = rng.random((12, 2)).astype(np.float32)
    distance = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {"name": "tsp", "coordinates": coordinates, "distance": distance},
        n_rollouts=1,
    )
    assert decoder.metadata["resource_count"] == 0
    graph = build_decoder_data(decoder)

    blind = ConstraintFieldNet(depth=2, units=16, objective_channel=False)
    blind_output = blind(graph)
    assert blind_output["residual"].shape[1] == 0
    # Nothing the module emits can move the energy on this problem.
    assert torch.equal(
        blind_output["objective_residual"],
        torch.zeros_like(blind_output["objective_residual"]),
    )
    # Not merely zero-valued: not a function of theta at all. Nothing this
    # module emits can move the energy on a problem with an empty registry.
    assert not blind_output["objective_residual"].requires_grad
    assert not blind_output["residual"].requires_grad

    model = ConstraintFieldNet(depth=2, units=16)
    output = model(graph)
    assert output["residual"].shape[1] == 0
    # Zero at initialization -- field_head is zero-init -- so the decoder energy
    # at step 0 is exactly the analytic objective it has always been.
    assert torch.equal(
        output["objective_residual"],
        torch.zeros_like(output["objective_residual"]),
    )
    # ...but no longer constant in theta, which is the whole point.
    output["objective_residual"].sum().backward()
    assert model.field_head.weight.grad.abs().sum() > 0.0


def test_objective_channel_is_load_bearing_in_the_native_energy() -> None:
    """It must reorder candidates, not merely appear in the output dict.

    A per-edge term that is constant within a source row cancels in the argmin
    that picks the next node, so "the field is non-zero" is not evidence that it
    can change a decision. Both halves are checked: the decoder honours the
    channel, and the model's own field varies WITHIN a candidate row.
    """
    rng = np.random.default_rng(4)
    coordinates = rng.random((40, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    problem = {"name": "tsp", "coordinates": coordinates, "distance": distance}

    def greedy(objective_residual):
        decoder = make_decoder(problem, n_rollouts=4)
        decoder.seed(99)
        guidance = _neutral_guidance(decoder)
        if objective_residual is not None:
            guidance["objective_residual"] = objective_residual(decoder)
        return tuple(decoder.sample_greedy(**guidance)["route"])

    baseline = greedy(None)
    perturbed = greedy(
        lambda decoder: (
            0.05
            * np.random.default_rng(1).standard_normal(
                int(decoder.metadata["edge_count"])
            )
        ).astype(np.float32)
    )
    # A 5% perturbation of a row-RMS-1 anchor already moves the tour, so the
    # channel's tanh range is far wider than it needs to be -- which is why its
    # zero initialization is the load-bearing safety property, not its bound.
    assert perturbed != baseline

    decoder = make_decoder(problem, n_rollouts=4)
    torch.manual_seed(7)
    model = ConstraintFieldNet(depth=4, units=16).eval()
    torch.nn.init.normal_(model.field_head.weight, std=1.0)
    field = _field_guidance(model, decoder, "cpu")["objective_residual"]
    offsets = np.asarray(decoder.edge_offsets, dtype=np.int64)
    within_row = [
        field[begin:end].std()
        for begin, end in zip(offsets[:-1], offsets[1:])
        if end > begin + 1
    ]
    assert np.mean(within_row) > 0.5 * field.std()


def test_objective_channel_shares_the_resource_field_parameters() -> None:
    """No new parameters: that is what keeps old checkpoints loadable."""
    with_channel = ConstraintFieldNet(depth=1, units=8)
    without = ConstraintFieldNet(depth=1, units=8, objective_channel=False)
    assert set(with_channel.state_dict()) == set(without.state_dict())


def test_v16_pre_objective_channel_checkpoint_requires_the_channel_off() -> None:
    trained = ConstraintFieldNet(depth=1, units=8, objective_channel=False)
    # A trained field head is exactly the danger: shared with the new channel.
    torch.nn.init.normal_(trained.field_head.weight, std=0.1)
    state = dict(trained.state_dict())

    restored = ConstraintFieldNet(depth=1, units=8, objective_channel=False)
    load_constraint_field_state_dict(
        restored, state, model_schema=LEGACY_NO_OBJECTIVE_CHANNEL_SCHEMA
    )
    for key, value in trained.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)

    with pytest.raises(RuntimeError, match="objective_channel=False"):
        load_constraint_field_state_dict(
            ConstraintFieldNet(depth=1, units=8),
            state,
            model_schema=LEGACY_NO_OBJECTIVE_CHANNEL_SCHEMA,
        )


def test_python_dimensions_come_from_cpp_extension() -> None:
    assert FIELD_CHANNEL_COUNT == prism_decoder.FIELD_CHANNEL_COUNT
    assert NODE_FEATURE_COUNT == prism_decoder.NODE_FEATURE_COUNT
    assert EDGE_FEATURE_COUNT == prism_decoder.EDGE_FEATURE_COUNT


def test_coupler_supports_states_from_multiple_graphs() -> None:
    model = ConstraintFieldNet(depth=1, units=8)
    # The coupler is generic in both widths; a registry of this size is just a
    # convenient synthetic one. Live state carries one value per registry row.
    channels = 3
    states = channels
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
    assert torch.equal(
        graph.resource_row_properties, renamed_graph.resource_row_properties
    )
    assert torch.equal(
        graph.resource_term_properties, renamed_graph.resource_term_properties
    )

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
        multipliers=output["multipliers"][0].numpy(),
        coupler_weights=output["coupler_weights"][0].numpy(),
        coupler_bias=output["coupler_bias"][0].numpy(),
    )
    assert all(solution["feasible"] for solution in traced["solutions"])
    assert traced["trace"]["live_state"].shape[1] == resource_count


def activate_field_heads(model: ConstraintFieldNet, seed: int) -> None:
    """Lift the energy and semantic heads off their zero init.

    A fresh ConstraintFieldNet emits an identically-zero residual by design (the
    neutral plain-objective policy), so any assertion comparing residuals on an
    untrained model is vacuous. Give the heads real weights first.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in (model.field_head, model.semantic_margin_head):
            head.weight.normal_(std=0.5, generator=generator)
            head.bias.normal_(std=0.5, generator=generator)


def test_core_interface_rungs_are_nested_and_mask_excluded_channels() -> None:
    """Each rung must be a true subset along every route into the model."""
    rng = np.random.default_rng(6441)
    coordinates = rng.random((18, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrptw",
            "coordinates": coordinates,
            "distance": distance,
            "demand": np.r_[0.0, rng.uniform(0.01, 0.04, 17)].astype(
                np.float32
            ),
            "capacity": 0.6,
            "time_window": np.column_stack(
                (np.zeros(18, dtype=np.float32), np.full(18, 10.0, dtype=np.float32))
            ),
            "service_time": np.r_[0.0, np.full(17, 0.01, dtype=np.float32)],
        },
        n_rollouts=2,
    )
    decoder.set_incumbent(
        decoder.sample_greedy(**_neutral_guidance(decoder))["route"]
    )
    graph = build_decoder_data(decoder)

    assert torch.count_nonzero(graph.node_live_state) > 0
    assert torch.count_nonzero(graph.resource_transition_mask) > 0
    assert torch.count_nonzero(graph.resource_events) > 0

    reference_layout = None
    for level in CORE_INTERFACE_LEVELS:
        model = ConstraintFieldNet(
            depth=1, units=8, core_interface=level
        ).eval()
        layout = {
            name: tuple(value.shape) for name, value in model.state_dict().items()
        }
        if reference_layout is None:
            reference_layout = layout
        else:
            assert layout == reference_layout

        node_resource, live, suffix, suffix_features = (
            model.emb_net.behavioral_node_inputs(graph)
        )
        pressure, raw_pressure, events, transition, mask = (
            model.emb_net.behavioral_edge_inputs(graph)
        )
        torch.testing.assert_close(live, graph.node_live_state)

        if level == "full":
            torch.testing.assert_close(node_resource, graph.node_resource)
            torch.testing.assert_close(suffix, graph.node_suffix_state)
            torch.testing.assert_close(
                suffix_features, graph.node_suffix_features
            )
            torch.testing.assert_close(pressure, graph.resource_features)
            torch.testing.assert_close(raw_pressure, graph.raw_resource_pressure)
        else:
            assert torch.count_nonzero(node_resource) == 0
            assert torch.count_nonzero(suffix) == 0
            assert torch.count_nonzero(suffix_features) == 0
            assert torch.count_nonzero(pressure) == 0
            assert torch.count_nonzero(raw_pressure) == 0

        post_state = level in ("post-state", "events", "margin", "full")
        has_events = level in ("events", "margin", "full")
        has_margin = level in ("margin", "full")
        if post_state:
            torch.testing.assert_close(
                transition[..., 0], graph.resource_transition_features[..., 0]
            )
            assert torch.equal(mask, graph.resource_transition_mask)
        else:
            assert torch.count_nonzero(transition[..., 0]) == 0
            assert torch.count_nonzero(mask) == 0
        if has_events:
            torch.testing.assert_close(events, graph.resource_events)
        else:
            assert torch.count_nonzero(events) == 0
        if has_margin:
            torch.testing.assert_close(
                transition[..., 1], graph.resource_transition_features[..., 1]
            )
        else:
            assert torch.count_nonzero(transition[..., 1]) == 0

    with pytest.raises(ValueError, match="core_interface"):
        ConstraintFieldNet(depth=1, units=8, core_interface="constraint-name")

    default_model = ConstraintFieldNet(depth=1, units=8).eval()
    explicit_full = ConstraintFieldNet(
        depth=1, units=8, core_interface="full"
    ).eval()
    explicit_full.load_state_dict(default_model.state_dict())
    activate_field_heads(default_model, 6443)
    explicit_full.load_state_dict(default_model.state_dict())
    with torch.no_grad():
        default_output = default_model(graph)
        full_output = explicit_full(graph)
    for key in default_output:
        assert torch.equal(default_output[key], full_output[key]), key


@pytest.mark.parametrize(
    "level", ("live-state", "post-state", "events", "margin")
)
def test_core_interface_excluded_channels_cannot_leak_into_outputs(level) -> None:
    """Perturb every excluded tensor; a lower-rung forward pass stays exact."""
    import problem_data

    decoder = prism_decoder.Decoder(problem_data.generated_problem("cvrptw", 20))
    decoder.solve(1)
    graph = build_decoder_data(decoder)
    altered = graph.clone()

    # Full-only operand, pressure, and suffix context.
    altered.node_resource = 1.0 - graph.node_resource
    altered.node_suffix_state = 1.0 - graph.node_suffix_state
    altered.node_suffix_features = 1.0 - graph.node_suffix_features
    altered.resource_features = 1.0 - graph.resource_features
    altered.raw_resource_pressure = graph.raw_resource_pressure + 0.37

    if level == "live-state":
        altered.resource_transition_features[..., 0] = (
            1.0 - graph.resource_transition_features[..., 0]
        )
        altered.resource_transition_mask = ~graph.resource_transition_mask
    if level in ("live-state", "post-state"):
        altered.resource_events = 1.0 - graph.resource_events
    if level in ("live-state", "post-state", "events"):
        altered.resource_transition_features[..., 1] = (
            -graph.resource_transition_features[..., 1] + 0.125
        )

    model = ConstraintFieldNet(
        depth=2, units=16, core_interface=level
    ).eval()
    activate_field_heads(model, 6442)
    with torch.no_grad():
        original = model(graph)
        perturbed = model(altered)
    for key in original:
        assert torch.equal(original[key], perturbed[key]), key


def test_index_embedded_resources_replaces_descriptor_semantics() -> None:
    """The ablation must read registry position only, never the algebra.

    Same graph, two different (valid) resource descriptors: the descriptor model
    must respond to the semantics, the ablated model must not -- that is the
    whole content of "semantic factorization".
    """
    rng = np.random.default_rng(311)
    coordinates = rng.random((12, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.06, 11)].astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrp",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.6,
        },
        n_rollouts=1,
    )
    graph = build_decoder_data(decoder)
    altered = build_decoder_data(decoder)
    # Flip the executable program properties to another point of the unit cube;
    # every behavioral input stays bit-identical.
    altered.resource_row_properties = 1.0 - graph.resource_row_properties
    altered.resource_term_properties = 1.0 - graph.resource_term_properties
    assert not torch.equal(
        graph.resource_row_properties, altered.resource_row_properties
    )

    # v14 makes program-blind the default, so the descriptor-conditioned model
    # this ablation is measured against has to ask for the rung above it.
    typed = ConstraintFieldNet(
        depth=1, units=8, program_blind_resources=False
    ).eval()
    activate_field_heads(typed, 4401)
    ablated = ConstraintFieldNet(
        depth=1, units=8, index_embedded_resources=True
    ).eval()
    # Identical weights in both models, so every difference below is the flag.
    ablated.load_state_dict(typed.state_dict())

    with torch.no_grad():
        assert not torch.equal(
            typed(graph)["multipliers"], typed(altered)["multipliers"]
        )
        assert not torch.equal(
            typed(graph)["residual"], typed(altered)["residual"]
        )
        assert not torch.equal(
            typed(graph)["semantic_margin"],
            typed(altered)["semantic_margin"],
        )
        base = ablated(graph)
        assert torch.equal(base["multipliers"], ablated(altered)["multipliers"])
        assert torch.equal(base["residual"], ablated(altered)["residual"])
        assert torch.equal(
            base["semantic_margin"],
            ablated(altered)["semantic_margin"],
        )
        # The ablation is a real change of policy, not a silent no-op.
        assert not torch.equal(base["multipliers"], typed(graph)["multipliers"])
        assert not torch.equal(base["residual"], typed(graph)["residual"])
        assert all(torch.isfinite(value).all() for value in base.values())


def test_monolithic_resource_field_prices_every_resource_alike() -> None:
    """The monolithic rung must collapse the factorization: one shared
    intensity for every active resource, and no new parameters."""
    rng = np.random.default_rng(312)
    coordinates = rng.random((14, 2), dtype=np.float32)
    distance = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    ).astype(np.float32)
    demand = np.r_[0.0, rng.uniform(0.01, 0.06, 13)].astype(np.float32)
    decoder = make_decoder(
        {
            "name": "cvrptw",
            "coordinates": coordinates,
            "distance": distance,
            "demand": demand,
            "capacity": 0.6,
        },
        n_rollouts=1,
    )
    graph = build_decoder_data(decoder)
    active = graph.active_channels.reshape(-1).bool()
    assert active.sum() >= 2, "need two active resources to compare pricing"

    typed = ConstraintFieldNet(depth=1, units=8).eval()
    activate_field_heads(typed, 4402)
    monolithic = ConstraintFieldNet(
        depth=1, units=8, monolithic_resource_field=True
    ).eval()
    # Pooling introduces no parameters, so the layout is unchanged by
    # construction and a checkpoint stays loadable across the ablation.
    assert list(typed.state_dict()) == list(monolithic.state_dict())
    monolithic.load_state_dict(typed.state_dict())

    with torch.no_grad():
        pooled = monolithic(graph)
        factorized = typed(graph)

    # Every active resource is priced identically...
    priced = pooled["multipliers"][0, : active.shape[0]][active]
    assert torch.allclose(priced, priced[0].expand_as(priced))
    # ...and the same field is applied on every active channel.
    columns = pooled["residual"][:, active]
    assert torch.allclose(columns, columns[:, :1].expand_as(columns))
    # The factorized model does differentiate, so this is a real ablation.
    reference = factorized["multipliers"][0, : active.shape[0]][active]
    assert not torch.allclose(reference, reference[0].expand_as(reference))
    assert all(torch.isfinite(value).all() for value in pooled.values())


def test_monolithic_resource_field_handles_no_active_resources() -> None:
    """Masked pooling must not divide by zero on a constraint-free schema."""
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
        output = ConstraintFieldNet(
            depth=1, units=8, monolithic_resource_field=True
        ).eval()(build_decoder_data(decoder))
    assert all(torch.isfinite(value).all() for value in output.values())
    # Pooling has nothing to pool: a problem with no declared constraint carries
    # no resource rows, so only the always-on objective slot remains.
    assert decoder.metadata["resource_count"] == 0
    assert output["multipliers"].shape == (1, 1)


def test_factorization_ablations_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConstraintFieldNet(
            depth=1,
            units=8,
            index_embedded_resources=True,
            monolithic_resource_field=True,
        )


def test_index_embedded_resources_keeps_current_checkpoint_layout() -> None:
    """Current typed and index-ablation models share one strict layout."""
    typed = ConstraintFieldNet(depth=1, units=8)
    ablated = ConstraintFieldNet(depth=1, units=8, index_embedded_resources=True)
    assert list(typed.state_dict()) == list(ablated.state_dict())
    # The table remains the last parameter so both current ablation modes have
    # an identical optimizer layout.
    assert [name for name, _ in typed.named_parameters()][-1] == (
        "resource_index_embedding.weight"
    )

    legacy = {
        key: value
        for key, value in typed.state_dict().items()
        if not key.startswith("resource_index_embedding.")
    }
    target = ConstraintFieldNet(depth=1, units=8)
    with pytest.raises(RuntimeError, match="incompatible ConstraintFieldNet"):
        load_constraint_field_state_dict(target, legacy)

    assert load_constraint_field_state_dict(target, typed.state_dict()) is None


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
    # No declared constraint -> no resource rows at all, so there is no
    # resource-field intensity to damp: the multiplier vector is the always-on
    # objective weight slot alone.
    assert decoder.metadata["resource_count"] == 0
    assert output["multipliers"].shape == (1, 1)


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

    incumbent = decoder.sample_greedy(**_neutral_guidance(decoder))
    assert incumbent["feasible"]
    decoder.set_incumbent(incumbent["route"])

    graph = build_decoder_data(decoder)
    with torch.no_grad():
        output = model(graph)
    traced = decoder.sample_traced(
        edge_field=output["residual"].detach().numpy(),
        multipliers=output["multipliers"][0].detach().numpy(),
        coupler_weights=output["coupler_weights"][0].detach().numpy(),
        coupler_bias=output["coupler_bias"][0].detach().numpy(),
    )
    trace = traced["trace"]
    replayed, decisions = replay_logp_from_cpp_batch_trace(
        trace, graph, output, model, beta=2.0
    )
    decision_logp, decision_rollouts, decision_counts = (
        replay_decision_logp_from_cpp_batch_trace(
            trace, graph, output, model, beta=2.0
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
    # Both are one column per registry row of THIS problem: the screening label
    # has to be as wide as the field head it supervises.
    registry = decoder.metadata["resource_count"]
    assert trace["live_state"].shape[1] == registry
    assert np.all((trace["live_state"] >= 0.0) & (trace["live_state"] <= 1.0))
    assert trace["screened_edges"].size > 0
    assert trace["screened_resource_delta"].shape == (
        trace["screened_edges"].shape[0],
        registry,
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


def test_tsp_model_uses_only_the_exact_objective() -> None:
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
        output = model(graph)

    # The objective channel is this problem's whole policy: the registry is
    # empty, so it is the only learned term in the decoder's energy.
    assert output["residual"].shape[1] == 0
    assert output["objective_residual"].shape == (
        decoder.metadata["edge_count"],
    )
    # The objective's INTENSITY stays a pinned unit anchor. A graph-level scalar
    # on the only active channel is exactly a temperature, and PPO took that
    # shortcut before (1.0 -> 3.2 on cvrp); the learned content is the per-edge
    # field, not the multiplier.
    assert output["multipliers"][0, -1] == 1.0
    assert torch.equal(
        output["coupler_weights"][0, -1],
        torch.zeros_like(output["coupler_weights"][0, -1]),
    )
    assert output["coupler_bias"][0, -1] == 0.0
    # v14 deleted the risk channel: its head produced a value that was constant
    # within a graph, and a constant added to every candidate at a node cancels
    # in the comparison that picks one, so it could never change a decision.
    assert "edge_risk" not in _guidance_numpy(output, graph)
    assert "edge_additive" not in _guidance_numpy(output, graph)
    # The channel has to reach the decoder, or it is not in the energy.
    guidance = _guidance_numpy(output, graph)
    assert guidance["objective_residual"].shape == (
        decoder.metadata["edge_count"],
    )
    # The fields-off control must carry no learned guidance at all.
    assert not _guidance_numpy(output, graph, field_enabled=False)[
        "objective_residual"
    ].any()


def test_model_runs_the_dynamic_gnn_once() -> None:
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
    embeddings = []
    hook = model.emb_net.register_forward_hook(
        lambda _module, _inputs, output: embeddings.append(output.detach())
    )

    model.train()
    model(graph)
    hook.remove()
    assert len(embeddings) == 1


def test_resource_field_remains_conditioned_on_incumbent_state() -> None:
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
    assert torch.count_nonzero(incumbent_graph.x[:, 5]) > 0
    assert (
        torch.count_nonzero(
            incumbent_graph.edge_attr[
                :, INCUMBENT_EDGE_FEATURE_START:INCUMBENT_EDGE_FEATURE_END
            ]
        )
        > 0
    )
    assert not torch.equal(
        empty_output["residual"], incumbent_output["residual"]
    )


def test_objective_node_terms_reach_the_encoder_without_a_named_column() -> None:
    """prize and penalty are node terms of the declared objective, not columns."""
    import problem_data

    names = None
    for variant, has_penalty in (("pctsp", True), ("op", False)):
        problem = problem_data.generated_problem(variant, 20)
        decoder = prism_decoder.Decoder(problem)
        names = decoder.metadata["node_feature_names"]
        terms = np.asarray(decoder.node_objective_features)
        assert terms.shape == (len(problem["coordinates"]), OBJECTIVE_NODE_TERM_COUNT)
        # Slot 0 is the term charged on visited nodes, slot 1 on unvisited ones.
        prize = np.asarray(problem["prize"])
        assert np.corrcoef(terms[1:, 0], prize[1:])[0, 1] == pytest.approx(
            1.0, abs=1e-4
        )
        if has_penalty:
            penalty = np.asarray(problem["penalty"])
            assert np.corrcoef(terms[1:, 1], penalty[1:])[0, 1] == pytest.approx(
                1.0, abs=1e-4
            )
        else:
            # An objective with no unvisited-node term leaves that slot inert.
            assert float(np.abs(terms[:, 1]).max()) == 0.0

        graph = build_decoder_data(decoder)
        model = ConstraintFieldNet(depth=1, units=8).eval()
        emb_net = model.emb_net
        with torch.no_grad():
            resource_type = model._resource_type_rows(
                graph, int(graph.active_channels.numel())
            )
            augmented = emb_net.augment_from_graph(
                graph, resource_type=resource_type
            )
        block = augmented[
            :, NODE_FEATURE_COUNT : NODE_FEATURE_COUNT + OBJECTIVE_SUMMARY_DIM
        ]
        assert block.shape[1] == OBJECTIVE_SUMMARY_DIM
        assert float(block.std(dim=0).max()) > 1e-6

    assert "prize" not in names and "penalty" not in names


def test_all_pre_program_pooling_layouts_are_rejected() -> None:
    """No old input width is widened or silently reinterpreted."""
    model = ConstraintFieldNet(depth=1, units=8)
    current = model.state_dict()

    prepooling = dict(current)
    prepooling["emb_net.e_lin0.weight"] = torch.randn(8, EDGE_FEATURE_COUNT)
    prepooling["emb_net.v_lin0.weight"] = torch.randn(8, NODE_FEATURE_COUNT)
    restored = ConstraintFieldNet(depth=1, units=8)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_constraint_field_state_dict(restored, prepooling)

    # Only a current, exact state dictionary loads.
    assert load_constraint_field_state_dict(restored, current) is None

    # A pre-v7 vector (one slot per constraint) is not silently reinterpreted.
    stale = dict(current)
    stale["emb_net.v_lin0.weight"] = torch.randn(8, NODE_FEATURE_COUNT + 8)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_constraint_field_state_dict(
            ConstraintFieldNet(depth=1, units=8), stale
        )


def test_metric_skew_reaches_the_resource_descriptor() -> None:
    """The asymmetry regime conditions the field, not just individual arcs."""
    import problem_data

    problem = problem_data.generated_problem("acvrp", 20)
    decoder = prism_decoder.Decoder(problem)
    data = build_decoder_data(decoder)
    assert data.metric_skew.shape == (1, 1)
    assert 0.0 < float(data.metric_skew) <= 1.0
    assert float(data.metric_skew) == pytest.approx(
        decoder.metadata["metric_skew"], abs=1e-6
    )

    model = ConstraintFieldNet(depth=1, units=8)
    model.eval()
    with torch.no_grad():
        asymmetric = model(data)["multipliers"]
        data.metric_skew = torch.zeros_like(data.metric_skew)
        as_symmetric = model(data)["multipliers"]
    assert not torch.allclose(asymmetric, as_symmetric)


def test_objective_conditioning_is_invariant_to_a_positive_coefficient_rescale() -> None:
    """c and 2c are the same problem, so they must condition the field alike.

    A positive rescale of the objective leaves the argmin untouched and is
    absorbed exactly by objective_energy_scale, so the policy is unchanged --
    the claim refresh_objective_energy_scale makes. Two model inputs used to
    break it anyway: the coefficient descriptor squashed absolute magnitudes,
    and objective_scale divided a cost that mixes travel with prize and penalty
    terms by a distance scale.
    """
    import problem_data

    base = problem_data.generated_problem("pctsp", 30)

    def make_solver(factor: float) -> prism_decoder.Decoder:
        problem = dict(base)
        problem["objective"] = {
            "distance_coeff": 1.0 * factor,
            "visit_coeff": 0.0,
            "miss_coeff": 1.0 * factor,
            "distance_regularizer": 0.0,
            "sense": 1.0,
        }
        return prism_decoder.Decoder(problem)

    reference = make_solver(1.0)
    scaled = make_solver(4.0)

    assert scaled.metadata["objective_scale"] == pytest.approx(
        reference.metadata["objective_scale"], rel=1e-5
    )
    assert scaled.metadata["objective_energy_scale"] == pytest.approx(
        4.0 * reference.metadata["objective_energy_scale"], rel=1e-5
    )
    assert torch.allclose(
        encode_objective_coeffs(reference.metadata["objective_coeffs"]),
        encode_objective_coeffs(scaled.metadata["objective_coeffs"]),
    )
    # The ratios between the primitives stay semantic: doubling only the miss
    # coefficient is a different objective and must encode differently.
    reweighted = dict(base)
    reweighted["objective"] = {
        "distance_coeff": 1.0,
        "visit_coeff": 0.0,
        "miss_coeff": 2.0,
        "distance_regularizer": 0.0,
        "sense": 1.0,
    }
    assert not torch.allclose(
        encode_objective_coeffs(reference.metadata["objective_coeffs"]),
        encode_objective_coeffs(
            prism_decoder.Decoder(reweighted).metadata["objective_coeffs"]
        ),
    )


def test_projection_normalization_is_live_and_checkpoint_compatible() -> None:
    """The saturation fix must change the forward pass but not the state dict.

    `edge_projection` and `graph_projection` emit values around |40| after the
    residual GNN, and every head consuming them applies a tanh. Unnormalized,
    that tanh saturates completely, so a head sees only the sign pattern of its
    input and the per-resource signals added to it (~0.5) cannot move any sign.
    A trained checkpoint's per-edge field then took *one* distinct value across
    every edge of a pdtsp instance.

    The fix is the parameter-free layer_norm the objective head already uses, so
    it adds no state and an existing checkpoint still loads -- which is exactly
    why the flag has to be recorded and read back rather than assumed.
    """
    normalized = ConstraintFieldNet()
    saturated = ConstraintFieldNet(normalize_projections="none")
    assert normalized.normalize_projections == "both", "the fix is the default"
    assert set(normalized.state_dict()) == set(saturated.state_dict()), (
        "layer_norm must stay parameter-free so checkpoints cross-load"
    )
    # At init the field and multiplier heads are zeroed, so both models emit
    # zeros whatever the projections do. Reproduce the trained regime instead:
    # non-degenerate heads, and projection weights large enough that the
    # consuming tanh saturates the way it does after the residual GNN.
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in normalized.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
        normalized.edge_projection.weight.mul_(20.0)
        normalized.graph_projection.weight.mul_(20.0)
    load_constraint_field_state_dict(saturated, normalized.state_dict())

    data = _pooling_fixture()
    with torch.no_grad():
        with_fix = normalized(data)
        without = saturated(data)
    # Both heads that read the projections must actually see a difference.
    assert not torch.allclose(
        with_fix["residual"], without["residual"], atol=1e-6
    ), "normalization did not reach the resource field"
    assert not torch.allclose(
        with_fix["multipliers"], without["multipliers"], atol=1e-6
    ), "normalization did not reach the multiplier head"
