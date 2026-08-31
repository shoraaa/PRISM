import math

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import prism_decoder


# The number of compiled fast paths, not a registry width: a problem's registry
# holds one row per constraint it declares plus whatever rows it appends, so it
# may be shorter or longer than this. Nothing in the model reads it -- a guard
# here used to reject any registry with fewer rows, which is exactly the
# assumption the model does not make.
FIELD_CHANNEL_COUNT = prism_decoder.FIELD_CHANNEL_COUNT
RESOURCE_ROW_PROPERTY_DIM = prism_decoder.RESOURCE_ROW_PROPERTY_DIM
RESOURCE_TERM_PROPERTY_DIM = prism_decoder.RESOURCE_TERM_PROPERTY_DIM
# Learned fixed-width type passed to every per-resource consumer after pooling
# the row's variable-length term set.
RESOURCE_TYPE_DIM = 32
NODE_FEATURE_COUNT = prism_decoder.NODE_FEATURE_COUNT
NODE_RESOURCE_FEATURE_COUNT = prism_decoder.NODE_RESOURCE_FEATURE_COUNT
RESOURCE_SUFFIX_FEATURE_COUNT = prism_decoder.RESOURCE_SUFFIX_FEATURE_COUNT
RESOURCE_TRANSITION_FEATURE_COUNT = (
    prism_decoder.RESOURCE_TRANSITION_FEATURE_COUNT
)
EDGE_FEATURE_COUNT = prism_decoder.EDGE_FEATURE_COUNT
# The native feature contract places instance-static node attributes first and
# incumbent replay state afterwards. Edge slots 8/9 mark incumbent/reverse
# incumbent arcs; all other edge slots are instance-static. The node contract
# grew a static tail (the in/out distance profiles) after the incumbent block,
# so the static view masks a middle range rather than a suffix.
# Width of the invariant resource-set summary appended to node/edge features.
# A shared descriptor-conditioned row map followed by one DeepSets sum reduces
# any number of declared resources to this fixed width for the ordinary GNN.
NODE_RESOURCE_SUMMARY_DIM = 32
# Node layout: x, y, is_depot | incumbent_served, route_position,
# forward_distance, backward_distance | mean_out_distance, mean_in_distance.
# Edge layout: distance | incumbent, reverse incumbent | objective edge term,
# reverse distance. Both incumbent blocks are a middle range, so the static
# view masks a span rather than a suffix.
# Per-(edge, resource) quantities: normalized pressure, raw pressure, event,
# exact post-transition state, signed feasibility margin, and a validity bit.
EDGE_RESOURCE_FEATURE_COUNT = 4 + RESOURCE_TRANSITION_FEATURE_COUNT
OBJECTIVE_NODE_TERM_COUNT = prism_decoder.OBJECTIVE_NODE_TERM_COUNT
# Width of the per-node objective summary. The objective language is closed, so
# this is a direct conditioned projection rather than a pooled reduction.
OBJECTIVE_SUMMARY_DIM = 8
STATIC_NODE_FEATURE_COUNT = 3
INCUMBENT_NODE_FEATURE_END = 7
INCUMBENT_EDGE_FEATURE_START = 1
INCUMBENT_EDGE_FEATURE_END = 3
# Registry-position table size for the --index-embedded-resources ablation.
# Rows 0..RESOURCE_INDEX_EMBEDDING_ROWS-2 address registry positions directly;
# the final row is a single shared out-of-table row, so every resource beyond
# the table (an appended schema row on a novel variant) collapses onto one cold
# embedding. That is the intended failure mode of the ablation, not a fallback:
# an identity-addressed model has nothing to say about a resource it never
# indexed during training.
RESOURCE_INDEX_EMBEDDING_ROWS = 16


# The objective is conditioned on its declared coefficient algebra, not a
# categorical type. Every objective is a signed linear combination over
# (total travel, prize on visited nodes, penalty on unvisited nodes); the
# encoder maps the five declared coefficients to a unit-interval vector so a
# single shared head learns per-primitive corrections that transfer across
# objectives (the well-trained distance primitive reaches prize/penalty
# objectives) and generalize to any future coefficient vector.
OBJECTIVE_COEFF_KEYS = (
    "distance_coeff",
    "visit_coeff",
    "miss_coeff",
    "distance_regularizer",
    "sense",
)
# 3 signed coeffs -> (magnitude, sign) each, plus regularizer magnitude and a
# sense sign bit.
OBJECTIVE_COEFF_DIM = 8
# Small non-zero init for the residual head's final layer so its hidden
# coefficient-conditioning layer is not gradient-starved (a zero-init output
# layer sends zero gradient to the layer below it).
OBJECTIVE_RESIDUAL_HEAD_INIT_STD = 0.1
MODEL_SCHEMA = "slack_energy_v14"


def _squash_magnitude(value: float) -> float:
    magnitude = abs(float(value))
    return magnitude / (1.0 + magnitude)


def _sign_bit(value: float) -> float:
    value = float(value)
    return 0.5 * (1.0 + (1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)))


def encode_objective_coeffs(coeffs: dict, device="cpu") -> torch.Tensor:
    """Map the declared objective coefficient triple to a [1, OBJECTIVE_COEFF_DIM]
    unit-interval descriptor. Signed coefficients become (squashed magnitude,
    sign) pairs so the encoding is bounded, sign-aware, and general to unseen
    coefficient values.

    The triple is normalized by its own largest magnitude first. Only the ratios
    between the primitives are semantic: c and 2*c are the same optimization
    problem, with the same argmin, and objective_energy_scale already absorbs
    the factor so the energy is unchanged. Encoding raw magnitudes made those
    two instances condition the field differently, which is the one thing a
    positive rescale must not do.
    """
    distance = float(coeffs.get("distance_coeff", 1.0))
    visit = float(coeffs.get("visit_coeff", 0.0))
    miss = float(coeffs.get("miss_coeff", 0.0))
    regularizer = float(coeffs.get("distance_regularizer", 0.0))
    sense = float(coeffs.get("sense", 1.0))
    norm = max(abs(distance), abs(visit), abs(miss))
    if norm > 0.0:
        distance /= norm
        visit /= norm
        miss /= norm
        # The regularizer is a term of the same objective, so it is measured
        # against the same norm rather than kept in absolute units.
        regularizer /= norm
    values = [
        _squash_magnitude(distance), _sign_bit(distance),
        _squash_magnitude(visit), _sign_bit(visit),
        _squash_magnitude(miss), _sign_bit(miss),
        _squash_magnitude(regularizer),
        _sign_bit(sense),
    ]
    return torch.tensor([values], dtype=torch.float32, device=device)


def _require_unit_interval(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if value.numel() and (value.min() < -1e-6 or value.max() > 1.0 + 1e-6):
        raise ValueError(f"{name} must be normalized to [0, 1]")


def _require_signed_unit_interval(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if value.numel() and (value.min() < -1.0 - 1e-6 or value.max() > 1.0 + 1e-6):
        raise ValueError(f"{name} must be normalized to [-1, 1]")


def _per_graph_descriptor(pyg, name, width, batch_size, reference, default=None):
    """Resolve a per-graph conditioning tensor to shape [batch_size, width]."""
    value = getattr(pyg, name, None)
    if value is None:
        if default is None:
            value = reference.new_zeros(batch_size, width)
        else:
            value = reference.new_tensor(default).view(1, width).expand(
                batch_size, -1
            )
    else:
        value = value.to(reference.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch_size != 1:
            value = value.expand(batch_size, -1)
        if value.shape != (batch_size, width):
            raise ValueError(f"{name} must have shape [num_graphs, {width}]")
    _require_unit_interval(name, value)
    return value


def build_decoder_data(decoder, device="cpu"):
    """Build the only supported GNN input contract from the C++ decoder."""
    x = torch.as_tensor(
        decoder.node_features, dtype=torch.float32, device=device
    )
    edge_attr = torch.as_tensor(
        decoder.edge_features, dtype=torch.float32, device=device
    )
    edge_index = torch.as_tensor(
        decoder.edge_index, dtype=torch.long, device=device
    )
    active_channels = torch.as_tensor(
        decoder.metadata["field_channel_mask"],
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    resource_count = active_channels.shape[1]
    open_route = torch.tensor(
        [[float(bool(decoder.metadata["open_route"]))]],
        dtype=torch.float32,
        device=device,
    )
    objective_coeffs = encode_objective_coeffs(
        decoder.metadata["objective_coeffs"], device=device
    )
    objective_scale = torch.tensor(
        [[float(decoder.metadata["objective_scale"])]],
        dtype=torch.float32,
        device=device,
    )
    objective_energy_scale = torch.tensor(
        [[float(decoder.metadata["objective_energy_scale"])]],
        dtype=torch.float32,
        device=device,
    )
    multi_route = torch.tensor(
        [[float(bool(decoder.metadata["multi_route"]))]],
        dtype=torch.float32,
        device=device,
    )
    # Mean normalized |d(i,j) - d(j,i)| over the instance. A per-arc reverse
    # distance says nothing about the regime the instance as a whole is in,
    # the same argument depot_scale makes for multi-depot; a scalar rather
    # than a bit so the degree of asymmetry survives, not just its presence.
    metric_skew = torch.tensor(
        [[min(max(float(decoder.metadata["metric_skew"]), 0.0), 1.0)]],
        dtype=torch.float32,
        device=device,
    )
    # Squash the raw depot count into [0, 1): 0->0, 1->0.5, 3->0.75. Gives the
    # field a graph-level multi-depot signal that node feature is_depot (a single
    # binary flag shared by every depot) and raw edge distances cannot convey.
    depot_count = float(decoder.metadata["depot_count"])
    depot_scale = torch.tensor(
        [[depot_count / (1.0 + depot_count)]],
        dtype=torch.float32,
        device=device,
    )
    raw_resource_pressure = torch.as_tensor(
        decoder.resource_pressure, dtype=torch.float32, device=device
    )
    resource_features = torch.as_tensor(
        decoder.resource_features, dtype=torch.float32, device=device
    )
    resource_events = torch.as_tensor(
        decoder.resource_events, dtype=torch.float32, device=device
    )
    # Per-node, per-resource attributes. The node-side counterpart of the
    # per-edge pressure: variable in resource_count with no constraint-named
    # column, so a declared row's node attributes reach the model through the
    # same shared projection that demand and time windows do.
    node_resource = torch.as_tensor(
        decoder.node_resource_features, dtype=torch.float32, device=device
    )
    # Per-resource route state along the incumbent, written by the C++ replay
    # from each row's own declaration. This is the generic counterpart of the
    # hand-written forward_load / forward_time / open_pickups node columns,
    # which only exist for three of the seven compiled kernels; a declared row
    # has no such column and reaches the encoder only through here.
    node_live_state = torch.as_tensor(
        decoder.incumbent_live_state, dtype=torch.float32, device=device
    )
    # Node terms of the DECLARED objective (charged on visited / on unvisited).
    # The coefficient vector says what each slot weighs and whether it is live,
    # so neither is a column named prize or penalty.
    node_objective = torch.as_tensor(
        decoder.node_objective_features, dtype=torch.float32, device=device
    )
    # Reverse counterpart: what the remaining route still spends on each row.
    # This is the generic replacement for the backward_load / backward_time /
    # backward_open_pickups columns, which only ever existed for three of the
    # seven compiled kernels.
    node_suffix_state = torch.as_tensor(
        decoder.incumbent_suffix_state, dtype=torch.float32, device=device
    )
    # Rich reverse-route statistics retain positive/negative workload and
    # departure contributions separately.  incumbent_suffix_state remains in
    # the contract as the legacy |signed total| view; keeping both makes the new
    # representation strictly richer without silently reinterpreting the old
    # scalar column.
    node_suffix_features = torch.as_tensor(
        decoder.incumbent_suffix_features, dtype=torch.float32, device=device
    )
    resource_transition_features = torch.as_tensor(
        decoder.incumbent_transition_features,
        dtype=torch.float32,
        device=device,
    )
    resource_transition_mask = torch.as_tensor(
        decoder.incumbent_transition_feature_mask,
        dtype=torch.bool,
        device=device,
    )
    resource_row_properties = torch.as_tensor(
        decoder.resource_row_properties, dtype=torch.float32, device=device
    ).view(resource_count, RESOURCE_ROW_PROPERTY_DIM)
    resource_term_properties = torch.as_tensor(
        decoder.resource_term_properties, dtype=torch.float32, device=device
    ).view(-1, RESOURCE_TERM_PROPERTY_DIM)
    resource_term_counts = torch.as_tensor(
        decoder.resource_term_counts, dtype=torch.long, device=device
    )
    objective_edge_costs = torch.as_tensor(
        decoder.objective_edge_costs, dtype=torch.float32, device=device
    )
    resource_scales = torch.as_tensor(
        decoder.resource_scales, dtype=torch.float32, device=device
    )
    edge_offsets = torch.as_tensor(
        decoder.edge_offsets, dtype=torch.long, device=device
    )
    if x.ndim != 2 or x.shape[1] != NODE_FEATURE_COUNT:
        raise ValueError(
            f"node_features must have shape [N, {NODE_FEATURE_COUNT}]"
        )
    if edge_attr.ndim != 2 or edge_attr.shape[1] != EDGE_FEATURE_COUNT:
        raise ValueError(
            f"edge_features must have shape [E, {EDGE_FEATURE_COUNT}]"
        )
    _require_unit_interval("node_features", x)
    _require_unit_interval("edge_features", edge_attr)
    _require_unit_interval("active_channels", active_channels)
    if not torch.isfinite(raw_resource_pressure).all():
        raise ValueError("raw_resource_pressure must contain only finite values")
    if not torch.isfinite(objective_edge_costs).all():
        raise ValueError("objective_edge_costs must contain only finite values")
    if (
        not torch.isfinite(objective_energy_scale).all()
        or torch.any(objective_energy_scale <= 0.0)
    ):
        raise ValueError(
            "objective_energy_scale must be finite and strictly positive"
        )
    if (
        resource_scales.shape != (resource_count,)
        or not torch.isfinite(resource_scales).all()
        or torch.any(resource_scales <= 0.0)
    ):
        raise ValueError("resource_scales must be finite and strictly positive")
    if resource_features.shape != (edge_attr.shape[0], resource_count):
        raise ValueError("resource_features must have shape [E, resource_count]")
    if raw_resource_pressure.shape != resource_features.shape:
        raise ValueError("raw_resource_pressure must match resource_features")
    if resource_events.shape != resource_features.shape:
        raise ValueError("resource_events must match resource_features")
    if node_resource.shape != (
        x.shape[0],
        resource_count,
        NODE_RESOURCE_FEATURE_COUNT,
    ):
        raise ValueError(
            "node_resource_features must have shape "
            "[N, resource_count, NODE_RESOURCE_FEATURE_COUNT]"
        )
    _require_unit_interval("resource_features", resource_features)
    _require_unit_interval("resource_events", resource_events)
    if node_live_state.shape != (x.shape[0], resource_count):
        raise ValueError(
            "incumbent_live_state must have shape [N, resource_count]"
        )
    _require_unit_interval("node_resource_features", node_resource)
    if node_suffix_state.shape != (x.shape[0], resource_count):
        raise ValueError(
            "incumbent_suffix_state must have shape [N, resource_count]"
        )
    _require_unit_interval("incumbent_live_state", node_live_state)
    _require_unit_interval("incumbent_suffix_state", node_suffix_state)
    if node_suffix_features.shape != (
        x.shape[0],
        resource_count,
        RESOURCE_SUFFIX_FEATURE_COUNT,
    ):
        raise ValueError(
            "incumbent_suffix_features must have shape "
            "[N, resource_count, RESOURCE_SUFFIX_FEATURE_COUNT]"
        )
    _require_unit_interval("incumbent_suffix_features", node_suffix_features)
    if resource_transition_features.shape != (
        edge_attr.shape[0],
        resource_count,
        RESOURCE_TRANSITION_FEATURE_COUNT,
    ):
        raise ValueError(
            "incumbent_transition_features must have shape "
            "[E, resource_count, RESOURCE_TRANSITION_FEATURE_COUNT]"
        )
    if resource_transition_mask.shape != (
        edge_attr.shape[0],
        resource_count,
    ):
        raise ValueError(
            "incumbent_transition_feature_mask must have shape "
            "[E, resource_count]"
        )
    _require_unit_interval(
        "incumbent_transition_features.next_state",
        resource_transition_features[..., 0],
    )
    _require_signed_unit_interval(
        "incumbent_transition_features.signed_margin",
        resource_transition_features[..., 1],
    )
    if node_objective.shape != (x.shape[0], OBJECTIVE_NODE_TERM_COUNT):
        raise ValueError(
            "node_objective_features must have shape "
            "[N, OBJECTIVE_NODE_TERM_COUNT]"
        )
    _require_unit_interval("node_objective_features", node_objective)
    _require_unit_interval("resource_row_properties", resource_row_properties)
    _require_unit_interval("resource_term_properties", resource_term_properties)
    if resource_term_counts.shape != (resource_count,):
        raise ValueError("resource_term_counts must have shape [resource_count]")
    if int(resource_term_counts.sum()) != resource_term_properties.shape[0]:
        raise ValueError("resource term counts do not match term properties")
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        active_channels=active_channels,
        open_route=open_route,
        objective_coeffs=objective_coeffs,
        objective_scale=objective_scale,
        objective_energy_scale=objective_energy_scale,
        multi_route=multi_route,
        depot_scale=depot_scale,
        metric_skew=metric_skew,
        raw_resource_pressure=raw_resource_pressure,
        resource_features=resource_features,
        resource_events=resource_events,
        node_resource=node_resource,
        node_live_state=node_live_state,
        node_suffix_state=node_suffix_state,
        node_suffix_features=node_suffix_features,
        resource_transition_features=resource_transition_features,
        resource_transition_mask=resource_transition_mask,
        node_objective=node_objective,
        resource_row_properties=resource_row_properties,
        resource_term_properties=resource_term_properties,
        resource_term_counts=resource_term_counts,
        objective_edge_costs=objective_edge_costs,
        resource_scales=resource_scales,
        edge_offsets=edge_offsets,
        graph_version=int(decoder.graph_version),
    )


@torch.no_grad()
def decode_iteration(
    decoder,
    model,
    device="cpu",
    risk_penalty=10.0,
):
    """Run one model-guided perturbation on an installed incumbent graph."""
    if not decoder.best_solution["feasible"]:
        raise ValueError(
            "decode_iteration requires a feasible installed incumbent; "
            "greedily bootstrap and call decoder.set_incumbent() first"
        )
    graph = build_decoder_data(decoder, device=device)
    output = model(graph)
    edge_field = output["residual"].detach().cpu().numpy()
    multipliers = output["multipliers"][0].detach().cpu().numpy()
    solution = decoder.solve(
        1,
        edge_field=edge_field,

        multipliers=multipliers,
        coupler_weights=output["coupler_weights"][0].detach().cpu().numpy(),
        coupler_bias=output["coupler_bias"][0].detach().cpu().numpy(),
        objective_residual=output["objective_residual"].detach().cpu().numpy(),
        risk_penalty=0.0,
    )
    return solution, output

# GNN for edge embeddings
# Single GNN layer for checkpointing
class GNNLayer(nn.Module):
    def __init__(self, units, act_fn, agg_fn):
        super().__init__()
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.v_lin1 = nn.Linear(units, units)
        self.v_lin2 = nn.Linear(units, units)
        self.v_lin3 = nn.Linear(units, units)
        self.v_lin4 = nn.Linear(units, units)
        self.v_bn = gnn.BatchNorm(units)
        self.e_lin0 = nn.Linear(units, units)
        self.e_bn = gnn.BatchNorm(units)

    def forward(self, x, w, edge_index):
        x0 = x
        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)
        w0 = w
        w1 = self.e_lin0(w0)
        w2 = torch.sigmoid(w0)
        x = x0 + self.act_fn(self.v_bn(x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w

def _unit_scale(projected):
    return F.layer_norm(projected, (projected.shape[-1],))


class ConditionedProjection(nn.Module):
    """Project attributes under a semantic descriptor without set reduction."""

    def __init__(self, attr_feats, descriptor_dim, units, output):
        super().__init__()
        self.attr_proj = nn.Linear(attr_feats, units)
        self.type_proj = nn.Linear(descriptor_dim, units)
        self.head = nn.Linear(units, output)

    def forward(self, attributes, descriptor):
        hidden = _unit_scale(self.attr_proj(attributes)) + _unit_scale(
            self.type_proj(descriptor)
        )
        return self.head(F.silu(hidden))


class ResourceProgramEncoder(nn.Module):
    """Encode fixed row properties plus a set of executable term properties.

    The shared term map and normalized sum are invariant to term order.  An
    explicit cardinality coordinate preserves multiplicity, so adding a term
    changes the semantic point without changing any tensor width.
    """

    def __init__(self, units, output=RESOURCE_TYPE_DIM):
        super().__init__()
        self.row_map = nn.Sequential(
            nn.Linear(RESOURCE_ROW_PROPERTY_DIM, units),
            nn.SiLU(),
            nn.Linear(units, units),
        )
        self.term_map = nn.Sequential(
            nn.Linear(RESOURCE_TERM_PROPERTY_DIM, units),
            nn.SiLU(),
            nn.Linear(units, units),
            nn.SiLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(2 * units + 1, units),
            nn.SiLU(),
            nn.Linear(units, output),
        )

    def forward(self, row_properties, term_properties, term_counts):
        if row_properties.ndim != 2 or row_properties.shape[-1] != (
            RESOURCE_ROW_PROPERTY_DIM
        ):
            raise ValueError("row properties must have shape [R, Dr]")
        if term_properties.ndim != 2 or term_properties.shape[-1] != (
            RESOURCE_TERM_PROPERTY_DIM
        ):
            raise ValueError("term properties must have shape [K, Dt]")
        if term_counts.shape != (row_properties.shape[0],):
            raise ValueError("term counts must have shape [R]")
        if int(term_counts.sum()) != term_properties.shape[0]:
            raise ValueError("term counts do not match term property rows")

        resources = row_properties.shape[0]
        row_hidden = self.row_map(row_properties)
        term_sum = row_hidden.new_zeros(resources, row_hidden.shape[-1])
        if term_properties.shape[0] > 0:
            term_hidden = self.term_map(term_properties)
            owners = torch.repeat_interleave(
                torch.arange(resources, device=term_counts.device), term_counts
            )
            term_sum.index_add_(0, owners, term_hidden)
            term_sum = term_sum / term_counts.to(term_sum.dtype).clamp_min(
                1.0
            ).sqrt().unsqueeze(-1)
        cardinality = term_counts.to(row_hidden.dtype).unsqueeze(-1)
        cardinality = cardinality / (1.0 + cardinality)
        return torch.sigmoid(
            self.combine(torch.cat((row_hidden, term_sum, cardinality), dim=-1))
        )


class ResourcePool(nn.Module):
    """Program-property-conditioned DeepSets encoder over resource rows.

    Shared ``phi`` weights produce one equivariant state per row. A normalized
    sum and explicit cardinality are then passed through ``rho`` for the
    fixed-width GNN summary. The unpooled states also go directly to the shared
    resource field, retaining the easy per-resource path that made v6 trainable.
    """

    def __init__(
        self,
        attr_feats,
        descriptor_dim,
        units,
        summary,
    ):
        super().__init__()
        self.attr_proj = nn.Linear(attr_feats, units)
        self.type_proj = nn.Linear(descriptor_dim, units)
        self.units = units
        self.row_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(units, units),
            nn.SiLU(),
        )
        self.summary_head = nn.Sequential(
            nn.Linear(units + 1, units),
            nn.SiLU(),
            nn.Linear(units, summary),
        )

    def encode_rows(self, attributes, descriptor, active):
        """Return permutation-equivariant per-resource states [B,R,U]."""
        if attributes.ndim != 3 or descriptor.ndim != 3 or active.ndim != 2:
            raise ValueError(
                "resource pooling expects attributes [B,R,A], descriptors "
                "[B,R,D], and active mask [B,R]"
            )
        if attributes.shape[:2] != descriptor.shape[:2] or (
            attributes.shape[:2] != active.shape
        ):
            raise ValueError("resource pooling tensors disagree on [B,R]")
        batch_size, resource_count = active.shape
        if resource_count == 0:
            return attributes.new_zeros(batch_size, 0, self.units)

        weights = active.to(attributes.dtype).unsqueeze(-1)
        hidden = self.row_mlp(
            _unit_scale(self.attr_proj(attributes))
            + _unit_scale(self.type_proj(descriptor))
        ) * weights
        return hidden

    def reduce_rows(self, hidden, active):
        """Reduce equivariant row states with one invariant DeepSets sum."""
        batch_size, resource_count = active.shape
        if resource_count == 0:
            return hidden.new_zeros(
                batch_size, self.summary_head[-1].out_features
            )
        active_count = active.to(hidden.dtype).sum(dim=1, keepdim=True)
        semantic_sum = hidden.sum(dim=1) / active_count.clamp_min(1.0).sqrt()
        cardinality = active_count / (1.0 + active_count)
        summary = self.summary_head(
            torch.cat((semantic_sum, cardinality), dim=-1)
        )
        no_resources = ~active.bool().any(dim=1)
        if no_resources.any():
            summary = summary.masked_fill(no_resources.unsqueeze(-1), 0.0)
        return summary

    def forward(self, attributes, descriptor, active):
        return self.reduce_rows(
            self.encode_rows(attributes, descriptor, active), active
        )


class EmbNet(nn.Module):
    def __init__(
        self,
        depth=12,
        feats=2,
        edge_feats=6,
        units=32,
        act_fn="silu",
        agg_fn="mean",
        grad_checkpointing=False,
        node_resource_feats=0,
        edge_resource_feats=0,
        objective_node_feats=0,
        objective_coeff_dim=0,
        resource_descriptor_dim=0,
        node_resource_summary=0,
        objective_summary=0,
    ):
        super().__init__()
        self.depth = depth
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.grad_checkpointing = grad_checkpointing
        # Per-node resource attributes (demand, window bounds, service time, and
        # whatever slots a declared row publishes) reach the node encoder as a
        # fixed-width pooled summary instead of as constraint-named columns.
        # phi is shared across resources and keyed by the semantic descriptor,
        # so an appended schema row contributes through the same weights, the
        # width does not depend on resource_count, and nothing here reads a
        # registry position. Without this the attributes reach only the
        # per-channel field head, leaving the feasibility, binding, coupler and
        # objective-residual heads with no view of demand or time windows.
        self.node_resource_summary = node_resource_summary
        if node_resource_summary:
            # Forward state, the legacy scalar suffix, and the richer split
            # suffix statistics are ordinary per-(node, resource) attributes.
            self.node_resource_encoder = ResourcePool(
                node_resource_feats + 2 + RESOURCE_SUFFIX_FEATURE_COUNT,
                resource_descriptor_dim,
                units,
                node_resource_summary,
            )
            feats = feats + node_resource_summary
            # The edge side has the same shape of problem: 7 of the 12 edge
            # columns are one-per-compiled-channel, so a declared row's live
            # pressure and reset events -- which the decoder already publishes
            # per resource -- never reach the GNN at all. Same pool, same
            # descriptor key, three per-(edge, resource) attributes instead of
            # six.
            self.edge_resource_encoder = ResourcePool(
                edge_resource_feats,
                resource_descriptor_dim,
                units,
                node_resource_summary,
            )
            edge_feats = edge_feats + node_resource_summary
            # The objective's node terms take the same two-branch conditioned
            # projection, keyed by the declared coefficient vector instead of a
            # resource descriptor. No reduction: the objective language is
            # closed at three declared quantities, so there is no set to pool.
            self.objective_node_encoder = ConditionedProjection(
                objective_node_feats,
                objective_coeff_dim,
                units,
                objective_summary,
            )
            feats = feats + objective_summary
        else:
            self.node_resource_encoder = None
            self.edge_resource_encoder = None
            self.objective_node_encoder = None
        self.feats = feats
        self.edge_feats = edge_feats

        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)
        
        self.layers = nn.ModuleList([
            GNNLayer(self.units, self.act_fn, self.agg_fn) for _ in range(self.depth)
        ])
        
    def augment_nodes(
        self,
        x,
        node_resource,
        node_live_state,
        node_suffix_state,
        node_suffix_features,
        node_objective,
        objective_coeffs,
        resource_type,
        resource_active,
        return_resource_rows=False,
    ):
        """Append the pooled per-node resource summary to node features.

        ``node_resource`` is [N, R, A] static attributes, ``node_live_state``
        and ``node_suffix_state`` the [N, R] route state before and after this
        node, and ``node_suffix_features`` the richer [N, R, S] split reverse
        statistics. ``resource_type`` is the
        matching [N, R, D] semantic descriptors and ``resource_active`` the
        [N, R] active mask. The reduction is masked and taken over resources, so
        the result is invariant to registry order and fixed-width for any
        registry. Returns ``x`` unchanged when pooling is disabled.
        """
        if self.node_resource_encoder is None:
            return (x, None) if return_resource_rows else x
        x = torch.cat(
            (x, self.objective_node_encoder(node_objective, objective_coeffs)),
            dim=-1,
        )
        return self._pool(
            self.node_resource_encoder,
            x,
            torch.cat(
                (
                    node_resource,
                    node_live_state.unsqueeze(-1),
                    node_suffix_state.unsqueeze(-1),
                    node_suffix_features,
                ),
                dim=-1,
            ),
            resource_type,
            resource_active,
            return_resource_rows=return_resource_rows,
        )

    @staticmethod
    def _pool(
        encoder,
        base,
        attributes,
        resource_type,
        resource_active,
        return_resource_rows=False,
    ):
        """Append a permutation-invariant learned resource-set summary.

        A registry with no active row reduces to zero, which is the same
        neutral value an all-inactive registry contributes anywhere else.
        """
        rows = encoder.encode_rows(attributes, resource_type, resource_active)
        summary = encoder.reduce_rows(rows, resource_active)
        augmented = torch.cat((base, summary), dim=-1)
        return (augmented, rows) if return_resource_rows else augmented

    def augment_from_graph(
        self,
        graph,
        x=None,
        resource_type=None,
        live_state=True,
        return_resource_rows=False,
    ):
        """augment_nodes for a decoder graph, broadcasting the per-graph rows.

        ``resource_type`` is the learned pooled program type produced one level
        up by ConstraintFieldNet. ``live_state=False`` substitutes zeros for the incumbent
        route state, keeping the width identical the way the incumbent node
        columns are zeroed rather than dropped -- that is what an
        incumbent-blind view needs.
        """
        if self.node_resource_encoder is None:
            base = graph.x if x is None else x
            return (base, None) if return_resource_rows else base
        node_count = graph.x.shape[0]
        batch = getattr(graph, "batch", None)

        def rows(tensor, rank):
            # active_channels arrives as [R] or [G, R], descriptors as [R, D] or
            # [G, R, D]; add the graph axis before broadcasting to one row per
            # node so a single-graph registry serves every node.
            while tensor.ndim < rank:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1 or batch is None:
                return tensor[0].unsqueeze(0).expand(node_count, *tensor.shape[1:])
            return tensor[batch]

        if resource_type is None:
            raise ValueError("augment_from_graph requires pooled resource_type")
        node_live_state = graph.node_live_state
        node_suffix_state = graph.node_suffix_state
        node_suffix_features = graph.node_suffix_features
        if not live_state:
            node_live_state = torch.zeros_like(node_live_state)
            node_suffix_state = torch.zeros_like(node_suffix_state)
            node_suffix_features = torch.zeros_like(node_suffix_features)
        return self.augment_nodes(
            graph.x if x is None else x,
            graph.node_resource,
            node_live_state,
            node_suffix_state,
            node_suffix_features,
            graph.node_objective,
            rows(graph.objective_coeffs, 2),
            rows(resource_type, 3),
            rows(graph.active_channels, 2),
            return_resource_rows=return_resource_rows,
        )

    def augment_edges(
        self,
        edge_attr,
        graph,
        resource_type=None,
        behavioral=True,
        return_resource_rows=False,
    ):
        """Append the pooled per-edge resource summary to edge features.

        Pressure/event features are joined by the exact normalized next state
        and signed admissibility margin produced by the native resource
        transition. The validity bit distinguishes sources absent from the
        incumbent rather than silently treating missing behavior as zero.
        """
        if self.edge_resource_encoder is None:
            return (edge_attr, None) if return_resource_rows else edge_attr
        edge_count = edge_attr.shape[0]
        batch = getattr(graph, "batch", None)
        edge_graph = None if batch is None else batch[graph.edge_index[0]]

        def rows(tensor, rank):
            while tensor.ndim < rank:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1 or edge_graph is None:
                return tensor[0].unsqueeze(0).expand(edge_count, *tensor.shape[1:])
            return tensor[edge_graph]

        if resource_type is None:
            raise ValueError("augment_edges requires pooled resource_type")
        transition_features = graph.resource_transition_features
        transition_mask = graph.resource_transition_mask.to(edge_attr.dtype)
        if not behavioral:
            transition_features = torch.zeros_like(transition_features)
            transition_mask = torch.zeros_like(transition_mask)
        attributes = torch.cat(
            (
                torch.stack(
                    (
                        graph.resource_features,
                        graph.raw_resource_pressure,
                        graph.resource_events,
                    ),
                    dim=-1,
                ),
                transition_features,
                transition_mask.unsqueeze(-1),
            ),
            dim=-1,
        )
        return self._pool(
            self.edge_resource_encoder,
            edge_attr,
            attributes,
            rows(resource_type, 3),
            rows(graph.active_channels, 2),
            return_resource_rows=return_resource_rows,
        )

    def forward(self, x, edge_index, edge_attr, return_nodes=False):
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)

        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                x, w = torch.utils.checkpoint.checkpoint(
                    layer, x, w, edge_index, use_reentrant=False
                )
            else:
                x, w = layer(x, w, edge_index)
        # Node embeddings x feed the refinement policy (CaR-style shared
        # encoder); the field net keeps consuming only the edge embeddings w.
        if return_nodes:
            return w, x
        return w

class ConstraintFieldNet(nn.Module):
    """Shared resource-token field over the Decoder candidate graph."""

    def __init__(
        self,
        depth=12,
        units=32,
        act_fn="silu",
        agg_fn="mean",
        grad_checkpointing=False,
        gate_multipliers_by_binding=True,
        couple_resource_tokens=True,
        linear_objective_residual_head=False,
        unconditioned_objective_residual_head=False,
        couple_state_multipliers=True,
        index_embedded_resources=False,
        monolithic_resource_field=False,
        # v14 default. The 14-row/20-term property maps are the most
        # hand-designed object in the system, and the executed quantities the
        # model reads -- the signed admissibility margin per candidate edge,
        # the live state, the per-node attributes -- already distinguish every
        # row in the benchmark. Descriptor conditioning stays available as the
        # ablation rung above this one.
        program_blind_resources=None,
        normalize_projections=True,
        pool_node_resources=True,
    ):
        super().__init__()
        # Monolithic ablation: remove the FACTORIZATION itself, not just its
        # semantics. Every active resource is collapsed onto one shared token,
        # so a single undifferentiated penalty intensity serves the whole
        # composition and the field can no longer price two constraints
        # differently. This is the coarse "everything off" end of the
        # factorization ladder (semantic descriptor -> identity-addressed ->
        # monolithic); unlike [[index_embedded_resources]] it degrades
        # in-distribution too, and it makes resource-token coupling vacuous, so
        # it cannot separate factorization from contextualization on its own.
        self.monolithic_resource_field = monolithic_resource_field
        # Semantic-factorization ablation. When True the algebra-derived type
        # descriptor is replaced by a learned embedding of the resource's
        # REGISTRY POSITION, keeping the descriptor's width and unit-interval
        # range so only the semantics change, not the capacity or the downstream
        # shapes. The model can then only memorize resource identities it saw in
        # training instead of reading what a resource *is*, which is the claim
        # under test: semantic factorization -- not sheer token capacity -- is
        # what carries the field to appended schema rows and unseen
        # compositions.
        self.index_embedded_resources = index_embedded_resources
        if index_embedded_resources and monolithic_resource_field:
            raise ValueError(
                "index_embedded_resources and monolithic_resource_field are "
                "mutually exclusive: pooling collapses every token, so the "
                "registry-position embedding it would read is erased"
            )
        # Program-blind ablation: the rung BELOW [[index_embedded_resources]] on
        # the same ladder. Identity-addressing still tells the model *which*
        # registry row it is looking at; this tells it nothing. Every resource
        # receives the same constant type vector, so no static description of
        # the constraint -- neither its executable properties nor its identity --
        # reaches the network. Per-resource factorization is untouched: each row
        # keeps its own token, multiplier and field head, and the live state,
        # node attributes and candidate-conditioned effects u_r(e, t) still
        # arrive per resource. The claim under test is the parsimony one: if the
        # row/term property maps earn their place, removing them must cost
        # something that the executed effects alone cannot supply.
        # The residual GNN's activations grow with depth, so `edge_projection`
        # and `graph_projection` emit values around |40| and |20|. Every head
        # that consumes them passes them through a tanh, which saturates
        # completely: the field, risk and multiplier heads then see only the
        # SIGN pattern of their input, and the per-resource signals added to it
        # (|token| ~ 0.2-1.1, |resource_edge| ~ 0.5) are ~50x too small to move
        # any sign. Measured consequence: the per-edge field takes 1 distinct
        # value across every edge of a pdtsp instance and 2 on cvrp, and the
        # feasibility-risk head is constant to 1e-15.
        #
        # This is the same failure the objective head already documents and
        # fixes below with a parameter-free per-edge layer_norm; these two
        # projections never received it. Normalizing adds no parameters, so a
        # checkpoint trained either way stays layout-compatible -- but the
        # forward result differs, so test.py must read the flag rather than
        # assume it.
        self.normalize_projections = normalize_projections
        # None means "take the v14 default", which is blind unless a higher rung
        # of the same ladder was asked for. Only an EXPLICIT True alongside
        # another rung is ambiguous, and that is what still raises.
        explicit_blind = program_blind_resources is True
        if program_blind_resources is None:
            program_blind_resources = not (
                index_embedded_resources or monolithic_resource_field
            )
        self.program_blind_resources = program_blind_resources
        if explicit_blind and (
            index_embedded_resources or monolithic_resource_field
        ):
            raise ValueError(
                "program_blind_resources is mutually exclusive with "
                "index_embedded_resources and monolithic_resource_field: all "
                "three overwrite the same resource type vector, so combining "
                "them measures whichever happens to run last"
            )
        # Objective-residual head parameterization. Default: a coefficient-
        # conditioned MLP (hidden layer mixes edge state with the declared
        # coefficients). Ablation (True): a single linear layer over
        # [edge_state, coeffs], so the coefficient contribution enters only
        # linearly -- the claim under test is that a purely linear coeff term
        # collapses into a per-row constant the downstream row-centering
        # removes, leaving the residual unable to specialize per objective.
        self.linear_objective_residual_head = linear_objective_residual_head
        # Unconditioned-head ablation (True): build the objective-energy residual
        # head over the edge state ALONE, dropping the declared coefficient vector
        # from its input. The head keeps the same depth/width as the default MLP so
        # the ablation isolates the *conditioning* (not capacity): one shared signed
        # logit must serve every objective. The claim under test is that this
        # unconditioned head suffers cross-objective negative transfer (e.g. a
        # distance correction fighting a prize correction) that coefficient
        # conditioning resolves. Mutually exclusive with the linear-head ablation.
        self.unconditioned_objective_residual_head = (
            unconditioned_objective_residual_head
        )
        if (
            linear_objective_residual_head
            and unconditioned_objective_residual_head
        ):
            raise ValueError(
                "linear_objective_residual_head and "
                "unconditioned_objective_residual_head are mutually exclusive"
            )
        # State-coupler ablation. When False, the per-decision live-state
        # modulation of the resource multipliers is disabled: forward emits zero
        # coupler weights/bias so both the Python couple() and the C++ decoder
        # leave each multiplier at its per-refresh GNN value (2*sigmoid(0)==1).
        # This isolates "live-state-modulated intensity" -- static per-refresh
        # lambda_r vs. per-decision lambda_r. The coupler heads stay registered
        # (unused, at their init) so the parameter layout is unchanged.
        self.couple_state_multipliers = couple_state_multipliers
        # Gate resource multipliers by the binding classifier so inactive/slack
        # resources begin damped. The signed edge field still receives policy
        # gradients through this gate; the ungated path remains an ablation.
        self.gate_multipliers_by_binding = gate_multipliers_by_binding
        # Resource-token attention couples the per-resource tokens so each
        # channel's learned intensity depends on the full active constraint
        # composition. Setting this False removes that cross-resource coupling
        # (the compositional-attention ablation); tokens then depend only on
        # their own descriptor. The attention parameters stay registered either
        # way so checkpoints remain layout-compatible across the ablation.
        self.couple_resource_tokens = couple_resource_tokens
        # Node-attribute pooling ablation. False drops the pooled per-node
        # resource summary, so demand, window bounds and service time reach only
        # the per-channel field head -- the routing that made the feasibility
        # and binding heads stall. The node encoder narrows accordingly, so an
        # ablated run is trained from scratch rather than resumed.
        self.pool_node_resources = pool_node_resources
        self.resource_program_encoder = ResourceProgramEncoder(units)
        self.emb_net = EmbNet(
            depth=depth,
            feats=NODE_FEATURE_COUNT,
            edge_feats=EDGE_FEATURE_COUNT,
            units=units,
            act_fn=act_fn,
            agg_fn=agg_fn,
            grad_checkpointing=grad_checkpointing,
            node_resource_feats=NODE_RESOURCE_FEATURE_COUNT,
            edge_resource_feats=EDGE_RESOURCE_FEATURE_COUNT,
            objective_node_feats=OBJECTIVE_NODE_TERM_COUNT,
            objective_coeff_dim=OBJECTIVE_COEFF_DIM,
            resource_descriptor_dim=RESOURCE_TYPE_DIM,
            node_resource_summary=(
                NODE_RESOURCE_SUMMARY_DIM if pool_node_resources else 0
            ),
            objective_summary=(
                OBJECTIVE_SUMMARY_DIM if pool_node_resources else 0
            ),
        )
        # Algebra-derived type descriptor + active flag + mean/max pressure +
        # graph context. No resource name or registry position reaches the
        # token encoder, so the same weights apply to appended schema rows.
        # (--index-embedded-resources deliberately breaks exactly this property;
        # see self.index_embedded_resources above.)
        descriptor_size = (
            RESOURCE_TYPE_DIM + 4 + OBJECTIVE_COEFF_DIM + 1 + 3
        )
        self.resource_encoder = nn.Sequential(
            nn.Linear(descriptor_size, units),
            nn.SiLU(),
            nn.Linear(units, units),
        )
        attention_heads = 4 if units % 4 == 0 else 1
        self.resource_attention = nn.MultiheadAttention(
            units, attention_heads, batch_first=True
        )
        self.edge_projection = nn.Linear(units, units)
        # (algebraic pressure, reset event) per edge, plus the descriptor-
        # conditioned node and edge row states. These states stay separated by
        # resource until this shared field projection, preserving the v6-style
        # per-column signal without depending on constraint names or registry
        # positions. The no-pooling ablation retains the old raw node block.
        resource_field_context = (
            NODE_RESOURCE_FEATURE_COUNT + 2 * units
            if pool_node_resources
            else NODE_RESOURCE_FEATURE_COUNT
        )
        self.resource_edge_projection = nn.Linear(
            2 + resource_field_context, units
        )
        self.token_projection = nn.Linear(units, units)
        self.graph_projection = nn.Linear(units, units)
        # v14: one head per resource channel. `additive_head` was a second
        # linear map of the SAME interaction vector whose output the decoder
        # simply added to this one, so the pair was one head with extra steps.
        # `feasibility_head` is gone with it: its output was constant within a
        # graph (measured std 1e-15), and a constant added to every candidate at
        # a node cancels in the comparison that picks one, so it could not
        # change a decision whatever its loss did.
        self.field_head = nn.Linear(units, 1)
        self.multiplier_head = nn.Linear(units, 1)
        self.binding_head = nn.Linear(units, 1)
        self.coupler_query_head = nn.Linear(units, units)
        self.coupler_key_head = nn.Linear(units, units)
        self.coupler_bias_head = nn.Linear(units, 1)
        self.value_head = nn.Linear(units + 1, 1)
        # Keep this final in module registration order. A single shared head
        # conditioned on the declared objective coefficient vector produces the
        # signed, dimensionless correction to normalized objective energy. The
        # coefficients modulate a per-edge correction nonlinearly (a plain linear
        # head would push their contribution into a per-row constant that the
        # downstream row-centering removes), so the distance primitive's learned
        # correction is shared across every objective while prize/penalty
        # objectives still specialize via their coefficients -- avoiding the
        # negative transfer of a single unconditioned logit and the cold
        # per-type columns of a one-hot head.
        residual_head_input = (
            units
            if self.unconditioned_objective_residual_head
            else units + OBJECTIVE_COEFF_DIM
        )
        if self.linear_objective_residual_head:
            self.objective_energy_residual_head = nn.Linear(
                residual_head_input, 1
            )
        else:
            self.objective_energy_residual_head = nn.Sequential(
                nn.Linear(residual_head_input, units),
                nn.SiLU(),
                nn.Linear(units, 1),
            )
        # Registered unconditionally (like resource_attention) so a property
        # model and an index-embedded ablation share one parameter
        # layout and one checkpoint format.
        self.resource_index_embedding = nn.Embedding(
            RESOURCE_INDEX_EMBEDDING_ROWS, RESOURCE_TYPE_DIM
        )
        nn.init.zeros_(self.field_head.weight)
        nn.init.zeros_(self.field_head.bias)
        # Small non-zero init on the final layer so the hidden (coefficient-
        # conditioning) layer receives gradient from step 0. Bias stays zero so a
        # row-constant output remains neutral after row-centering; the residual
        # starts small and is bounded by row-center + tanh + --objective-residual-l2.
        final_residual_layer = (
            self.objective_energy_residual_head
            if isinstance(self.objective_energy_residual_head, nn.Linear)
            else self.objective_energy_residual_head[-1]
        )
        nn.init.normal_(
            final_residual_layer.weight,
            std=OBJECTIVE_RESIDUAL_HEAD_INIT_STD,
        )
        nn.init.zeros_(final_residual_layer.bias)
        nn.init.zeros_(self.coupler_query_head.weight)
        nn.init.zeros_(self.coupler_query_head.bias)
        nn.init.zeros_(self.coupler_bias_head.weight)
        nn.init.zeros_(self.coupler_bias_head.bias)
        # A zero value is the neutral bootstrap and makes
        # enabling temporal credit leave the field policy unchanged initially.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        # Unit-variance logits so sigmoid() spreads the rows across (0, 1) and
        # distinct registry positions start distinguishable. Unused (and so
        # exactly neutral) unless --index-embedded-resources is set.
        nn.init.normal_(self.resource_index_embedding.weight, std=1.0)
        self.register_buffer(
            "unit_softplus_shift",
            torch.log(torch.expm1(torch.ones(()))),
        )
        # Retain these buffers so the parameter/buffer layout remains explicit
        # when diagnosing older checkpoints. V4 no longer uses either softplus
        # shift: both resource terms are signed and exactly zero at init.
        self.register_buffer(
            "additive_softplus_shift",
            torch.log(torch.expm1(torch.full((), 0.01))),
        )

    def _field_channel(
        self, edge_projection, token_projection, resource_edge, edge_batch
    ):
        token = (
            token_projection
            if token_projection.ndim == 1
            else token_projection[edge_batch]
        )
        interaction = torch.tanh(
            edge_projection + token + resource_edge
        )
        return self.field_head(interaction).squeeze(-1)

    def _resource_type_rows(self, pyg, resource_count):
        """Pool each resource's executable term set into [G, R, D]."""
        rows = pyg.resource_row_properties
        terms = pyg.resource_term_properties
        counts = pyg.resource_term_counts
        _require_unit_interval("resource_row_properties", rows)
        _require_unit_interval("resource_term_properties", terms)
        if self.program_blind_resources:
            # Same shape, same unit-interval range, and identical across rows so
            # the type vector carries no per-resource information at all. The
            # encoder stays registered (and unused, at its init) so a blind and
            # a descriptor-conditioned checkpoint remain layout-compatible.
            if rows.ndim not in (2, 3):
                raise ValueError("resource_row_properties must have rank 2 or 3")
            graphs = rows.shape[0] if rows.ndim == 3 else 1
            return rows.new_full((graphs, resource_count, RESOURCE_TYPE_DIM), 0.5)
        if rows.ndim == 2:
            resource_type = self.resource_program_encoder(
                rows, terms, counts.reshape(-1)
            ).unsqueeze(0)
        elif rows.ndim == 3:
            if counts.ndim != 2 or rows.shape[:2] != counts.shape:
                raise ValueError("batched row properties and term counts disagree")
            encoded = []
            offset = 0
            for graph_rows, graph_counts in zip(rows, counts):
                term_count = int(graph_counts.sum())
                encoded.append(
                    self.resource_program_encoder(
                        graph_rows,
                        terms[offset : offset + term_count],
                        graph_counts,
                    )
                )
                offset += term_count
            if offset != terms.shape[0]:
                raise ValueError("batched term counts do not match term rows")
            resource_type = torch.stack(encoded, dim=0)
        else:
            raise ValueError("resource_row_properties must have rank 2 or 3")
        if resource_type.shape[1] != resource_count:
            raise ValueError("resource program rows disagree with active channels")
        if self.index_embedded_resources:
            positions = torch.arange(
                resource_count, device=resource_type.device
            ).clamp_(max=RESOURCE_INDEX_EMBEDDING_ROWS - 1)
            resource_type = (
                torch.sigmoid(self.resource_index_embedding(positions))
                .unsqueeze(0)
                .expand(resource_type.shape[0], -1, -1)
            )
        return resource_type

    def forward(self, pyg):
        _require_unit_interval("node_features", pyg.x)
        _require_unit_interval("edge_features", pyg.edge_attr)
        active = pyg.active_channels
        if active.ndim == 1:
            active = active.unsqueeze(0)
        _require_unit_interval("active_channels", active)
        resource_count = active.shape[-1]

        # Objective guidance must not see the incumbent whose refinement it is
        # supposed to improve. Otherwise PPO can reduce its loss by recognizing
        # incumbent/reverse edges and assigning them positive logits, which
        # suppresses exploration without learning objective structure. Reuse
        # the same GNN weights on a static view: resource heads retain the full
        # state-conditioned embedding above, while the objective residual is
        # identical for the same instance before and after set_incumbent().
        static_x = torch.cat(
            (
                pyg.x[:, :STATIC_NODE_FEATURE_COUNT],
                torch.zeros_like(
                    pyg.x[
                        :,
                        STATIC_NODE_FEATURE_COUNT:INCUMBENT_NODE_FEATURE_END,
                    ]
                ),
                pyg.x[:, INCUMBENT_NODE_FEATURE_END:],
            ),
            dim=1,
        )
        static_edge_attr = torch.cat(
            (
                pyg.edge_attr[:, :INCUMBENT_EDGE_FEATURE_START],
                torch.zeros_like(
                    pyg.edge_attr[
                        :,
                        INCUMBENT_EDGE_FEATURE_START:INCUMBENT_EDGE_FEATURE_END,
                    ]
                ),
                pyg.edge_attr[:, INCUMBENT_EDGE_FEATURE_END:],
            ),
            dim=1,
        )
        # Preserve the legacy resource-field path exactly: its BatchNorm
        # statistics must be computed from the dynamic incumbent graph alone.
        # Concatenating the static objective graph here changes every resource
        # embedding even while the objective residual is identically zero, so
        # merely adding the objective head changes the policy being restored.
        edge_count = pyg.edge_attr.shape[0]
        # The pooled summary is a static instance property, so it is appended to
        # both views: the objective view stays blind to the incumbent because
        # only the incumbent columns of pyg.x are zeroed above.
        node_resource_type = self._resource_type_rows(pyg, resource_count)
        encoder_x, node_resource_rows = self.emb_net.augment_from_graph(
            pyg,
            resource_type=node_resource_type,
            return_resource_rows=True,
        )
        static_x = self.emb_net.augment_from_graph(
            pyg, x=static_x, resource_type=node_resource_type, live_state=False
        )
        encoder_edge_attr, edge_resource_rows = self.emb_net.augment_edges(
            pyg.edge_attr,
            pyg,
            resource_type=node_resource_type,
            return_resource_rows=True,
        )
        static_edge_attr = self.emb_net.augment_edges(
            static_edge_attr,
            pyg,
            resource_type=node_resource_type,
            behavioral=False,
        )
        edge_embedding = self.emb_net(
            encoder_x, pyg.edge_index, encoder_edge_attr
        )

        # The objective view must not update BatchNorm a second time. Reuse the
        # running statistics learned by the dynamic path while retaining
        # gradients through the shared GNN and affine BatchNorm parameters. Do
        # not checkpoint this pass: checkpoint recomputation happens after the
        # modules have returned to training mode and would use different BN
        # semantics from the original forward.
        batch_norms = [
            norm
            for layer in self.emb_net.layers
            for norm in (layer.v_bn, layer.e_bn)
        ]
        batch_norm_training = [norm.training for norm in batch_norms]
        checkpointing = self.emb_net.grad_checkpointing
        try:
            for norm in batch_norms:
                norm.eval()
            self.emb_net.grad_checkpointing = False
            objective_edge_embedding = self.emb_net(
                static_x, pyg.edge_index, static_edge_attr
            )
        finally:
            self.emb_net.grad_checkpointing = checkpointing
            for norm, training in zip(batch_norms, batch_norm_training):
                norm.train(training)
        batched = hasattr(pyg, "batch") and pyg.batch is not None
        if batched:
            edge_batch = pyg.batch[pyg.edge_index[0]]
            graph_embedding = gnn.global_mean_pool(
                edge_embedding, edge_batch
            )
        else:
            edge_batch = None
            graph_embedding = edge_embedding.mean(dim=0, keepdim=True)
        if active.shape[0] == 1 and graph_embedding.shape[0] != 1:
            active = active.expand(graph_embedding.shape[0], -1)
        if active.shape[0] != graph_embedding.shape[0]:
            raise ValueError("active_channels must have one row per graph")

        batch_size = graph_embedding.shape[0]
        open_route = _per_graph_descriptor(
            pyg, "open_route", 1, batch_size, active
        )
        objective_coeffs = _per_graph_descriptor(
            pyg,
            "objective_coeffs",
            OBJECTIVE_COEFF_DIM,
            batch_size,
            active,
            # Neutral default = the pure-distance objective (coeff (1, 0, 0),
            # sense +1) encoded through encode_objective_coeffs.
            default=[0.5, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 1.0],
        )
        objective_scale = _per_graph_descriptor(
            pyg, "objective_scale", 1, batch_size, active
        )
        multi_route = _per_graph_descriptor(
            pyg, "multi_route", 1, batch_size, active
        )
        depot_scale = _per_graph_descriptor(
            pyg, "depot_scale", 1, batch_size, active
        )
        metric_skew = _per_graph_descriptor(
            pyg, "metric_skew", 1, batch_size, active
        )

        normalized_resources = pyg.resource_features
        if normalized_resources.shape != (edge_embedding.shape[0], resource_count):
            raise ValueError(
                "resource_features must have shape [num_edges, resource_count]"
            )
        resource_events = pyg.resource_events
        if resource_events.shape != normalized_resources.shape:
            raise ValueError("resource_events must match resource_features")
        if batched:
            resource_mean = gnn.global_mean_pool(
                normalized_resources, edge_batch
            )
            resource_max = gnn.global_max_pool(
                normalized_resources, edge_batch
            )
        else:
            resource_mean = normalized_resources.mean(dim=0, keepdim=True)
            resource_max = normalized_resources.amax(dim=0, keepdim=True)
        resource_type = self._resource_type_rows(pyg, resource_count)
        if resource_type.shape != (
            batch_size,
            resource_count,
            RESOURCE_TYPE_DIM,
        ):
            raise ValueError(
                "pooled resource types must have shape "
                "[num_graphs, resource_count, RESOURCE_TYPE_DIM]"
            )
        _require_unit_interval("resource_types", resource_type)

        def _broadcast(column):
            return column.unsqueeze(1).expand(-1, resource_count, -1)

        descriptor = torch.cat(
            (
                resource_type,
                active.unsqueeze(-1),
                resource_mean.unsqueeze(-1),
                resource_max.unsqueeze(-1),
                _broadcast(open_route),
                _broadcast(objective_coeffs),
                _broadcast(objective_scale),
                _broadcast(multi_route),
                _broadcast(depot_scale),
                _broadcast(metric_skew),
            ),
            dim=-1,
        )
        _require_unit_interval("resource_token_inputs", descriptor)
        tokens = self.resource_encoder(descriptor)
        # Monolithic ablation: pool the per-resource tokens into one shared
        # token (mean over ACTIVE resources) and broadcast it back across every
        # channel. Pooling must happen here, after the encoder -- the descriptor
        # also carries per-resource mean/max pressure, so collapsing only its
        # type block would leave the tokens differentiated through pressure.
        # The decoder contract is untouched (still resource_count channels and
        # MULTIPLIER_COUNT multipliers), but each active resource necessarily
        # receives the same learned intensity. Resource-token coupling below
        # degenerates to an identity on identical tokens, which is exactly the
        # point: there is no composition left to attend over.
        if self.monolithic_resource_field:
            weights = active.unsqueeze(-1)
            pooled = (tokens * weights).sum(dim=1, keepdim=True) / (
                weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            tokens = pooled.expand(-1, resource_count, -1)
        # Cross-resource coupling. The --no-couple-resource-tokens ablation
        # skips this residual so every token is an independent per-resource
        # encoding of its own descriptor, isolating the contribution of
        # compositional attention. The attention parameters remain in the state
        # dict (unused, at their init) so a coupled and an ablated checkpoint
        # share an identical parameter layout.
        # An EMPTY registry is now reachable: a problem that declares no
        # constraint (a TSP) carries no resource rows at all, where it used to
        # carry seven inactive ones. There is no sequence to attend over, and
        # the all-masked workaround below has no row to unmask, so skip the
        # residual entirely -- it would be a no-op on zero tokens anyway.
        if self.couple_resource_tokens and resource_count > 0:
            padding_mask = ~active.bool()
            no_resources = ~active.bool().any(dim=1)
            if no_resources.any():
                # Attention over an all-masked row is undefined, so leave one
                # key visible; the tokens it mixes are inactive either way.
                padding_mask = padding_mask.clone()
                padding_mask[no_resources, 0] = False
            coupled_tokens, _ = self.resource_attention(
                tokens, tokens, tokens, key_padding_mask=padding_mask
            )
            tokens = tokens + coupled_tokens

        projected_edges = self.edge_projection(edge_embedding)
        if self.normalize_projections:
            # Same normalization, same reason, as the objective head below --
            # see the note in __init__ for what saturating here costs the
            # resource field and the feasibility-risk head.
            projected_edges = F.layer_norm(
                projected_edges, (projected_edges.shape[-1],)
            )
        # The residual GNN can have large eval-time activations after many
        # layers. Applying tanh directly here saturated every component and
        # made the nominal per-edge residual constant across all TSP edges. Use
        # a parameter-free per-edge normalization so the head retains objective
        # ordering information without adding checkpoint state.
        objective_projected_edges = self.edge_projection(
            objective_edge_embedding
        )
        objective_edge_state = F.layer_norm(
            objective_projected_edges,
            (objective_projected_edges.shape[-1],),
        )
        # Condition the shared head on the declared objective coefficients so a
        # single set of weights specializes per objective without siloing.
        edge_objective_coeffs = (
            objective_coeffs[edge_batch]
            if batched
            else objective_coeffs[0].expand(edge_count, -1)
        )
        # Unconditioned-head ablation: drop the coefficient vector so a single
        # shared logit serves every objective (isolating the value of
        # coefficient conditioning). Otherwise condition on the declared coeffs.
        if self.unconditioned_objective_residual_head:
            residual_head_input = objective_edge_state
        else:
            residual_head_input = torch.cat(
                (objective_edge_state, edge_objective_coeffs), dim=-1
            )
        raw_objective_residual = self.objective_energy_residual_head(
            residual_head_input
        ).squeeze(-1)
        # Only differences within an outgoing candidate row affect policy.
        # Center before bounding so a row-constant head output is exactly
        # neutral. The result is a dimensionless correction because the native
        # decoder and PPO replay both divide the raw objective by the same
        # row-centered graph scale before adding this term.
        source = pyg.edge_index[0]
        row_sums = raw_objective_residual.new_zeros(pyg.x.shape[0])
        row_sums.scatter_add_(0, source, raw_objective_residual)
        row_counts = torch.bincount(
            source, minlength=pyg.x.shape[0]
        ).to(raw_objective_residual.dtype)
        centered_objective_residual = raw_objective_residual - (
            row_sums / row_counts.clamp_min(1.0)
        )[source]
        objective_residual = torch.tanh(centered_objective_residual)
        projected_tokens = self.token_projection(tokens)
        edge_active = active[edge_batch] if batched else active[0]
        channel_state = normalized_resources
        channel_events = resource_events
        # Gather the arrival node's descriptor-conditioned resource states and
        # concatenate the corresponding edge states. The ordinary GNN receives
        # only their invariant set summaries; the shared per-resource field
        # retains these equivariant rows until its final channel computation.
        if node_resource_rows is None:
            resource_context_edges = pyg.node_resource[pyg.edge_index[1]]
        else:
            resource_context_edges = torch.cat(
                (
                    pyg.node_resource[pyg.edge_index[1]],
                    node_resource_rows[pyg.edge_index[1]],
                    edge_resource_rows,
                ),
                dim=-1,
            )
        # Monolithic ablation, second half: pooling the tokens alone is NOT
        # enough to remove the factorization. _field_channel also consumes a
        # PER-CHANNEL live-pressure pair (normalized_resources, resource_events),
        # so a token-only collapse leaves each resource its own analytic pathway
        # and the field still differentiates them -- which would let the
        # ablation "survive" for a reason that has nothing to do with learned
        # factorization. Pool that input over the active resources too, so every
        # channel is fed one undifferentiated pressure signal and no per-resource
        # information reaches the field at all.
        if self.monolithic_resource_field:
            edge_weights = edge_active
            edge_denominator = edge_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            channel_state = (
                (normalized_resources * edge_weights).sum(dim=-1, keepdim=True)
                / edge_denominator
            ).expand_as(normalized_resources)
            channel_events = (
                (resource_events * edge_weights).sum(dim=-1, keepdim=True)
                / edge_denominator
            ).expand_as(resource_events)
            # The per-node attribute block is a third per-resource pathway into
            # the field head, so it has to be pooled here as well or the
            # ablation leaves each resource distinguishable after all.
            resource_context_edges = (
                (resource_context_edges * edge_weights[..., None]).sum(
                    dim=1, keepdim=True
                )
                / edge_denominator[..., None]
            ).expand_as(resource_context_edges)
        raw_channels = []
        for channel in range(resource_count):
            token = (
                projected_tokens[:, channel]
                if batched
                else projected_tokens[0, channel]
            )
            resource_edge = self.resource_edge_projection(
                torch.cat(
                    (
                        channel_state[:, channel, None],
                        channel_events[:, channel, None],
                        resource_context_edges[:, channel],
                    ),
                    dim=1,
                )
            )
            if (
                self.emb_net.grad_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                raw = torch.utils.checkpoint.checkpoint(
                    self._field_channel,
                    projected_edges,
                    token,
                    resource_edge,
                    edge_batch,
                    use_reentrant=False,
                )
            else:
                raw = self._field_channel(
                    projected_edges, token, resource_edge, edge_batch
                )
            raw_channels.append(raw)
        # An empty registry stacks to [edges, 0] rather than failing: a problem
        # with no resource rows has no resource energy, which the decoder
        # already accepts as an [edges, 0] field.
        raw_residual = (
            torch.stack(raw_channels, dim=1)
            if raw_channels
            else projected_edges.new_zeros(edge_count, 0)
        )
        # Resource guidance is a signed, zero-neutral learned field. Analytic
        # pressure remains an input feature, but the decoder does not multiply
        # it into energy. Signed terms let PPO reward useful capacity/route-limit
        # edges as well as penalize harmful ones; exact native feasibility
        # remains authoritative.
        residual = torch.tanh(raw_residual) * edge_active

        projected_graph = self.graph_projection(graph_embedding)
        if self.normalize_projections:
            # The multiplier, binding and coupler heads read `state`, whose
            # tanh saturates for exactly the same reason the edge one does.
            projected_graph = F.layer_norm(
                projected_graph, (projected_graph.shape[-1],)
            )
        state = torch.tanh(projected_graph.unsqueeze(1) + tokens)
        graph_state = torch.tanh(projected_graph)
        binding_logits = self.binding_head(state).squeeze(-1)
        multipliers = F.softplus(self.multiplier_head(state).squeeze(-1))
        if self.gate_multipliers_by_binding:
            multipliers = multipliers * torch.sigmoid(binding_logits)
        multipliers = multipliers * active
        coupler_queries = self.coupler_query_head(state)
        coupler_keys = self.coupler_key_head(tokens)
        coupler_weights = torch.einsum(
            "bru,bsu->brs", coupler_queries, coupler_keys
        ) / (coupler_queries.shape[-1] ** 0.5)
        coupler_weights = (
            coupler_weights
            * active.unsqueeze(-1)
            * active.unsqueeze(1)
        )
        coupler_bias = self.coupler_bias_head(state).squeeze(-1) * active
        # State-coupler ablation: emit no live-state modulation so every resource
        # multiplier stays at its per-refresh GNN value in both the Python
        # couple() (2*sigmoid(0)==1) and the exported C++ decoder.
        if not self.couple_state_multipliers:
            coupler_weights = torch.zeros_like(coupler_weights)
            coupler_bias = torch.zeros_like(coupler_bias)
        # Append the always-on objective slot required by the native guidance
        # schema. Keep it exactly one, with zero live-state coupling, for every
        # problem (including constrained ones). A graph-level scalar only
        # changes effective beta; PPO exploited that shortcut on CVRP (about
        # 1.0 -> 3.2) instead of learning useful edge preferences. The signed
        # objective-energy residual is the policy channel for objective-specific
        # ordering; resource multipliers remain learned
        # relative to the fixed objective.
        objective_multiplier = torch.ones(
            graph_state.shape[0],
            dtype=graph_state.dtype,
            device=graph_state.device,
        )
        objective_coupler_weights = torch.zeros(
            (graph_state.shape[0], resource_count),
            dtype=graph_state.dtype,
            device=graph_state.device,
        )
        objective_coupler_bias = torch.zeros_like(objective_multiplier)
        multipliers = torch.cat(
            (multipliers, objective_multiplier.unsqueeze(1)), dim=1
        )
        coupler_weights = torch.cat(
            (coupler_weights, objective_coupler_weights.unsqueeze(1)), dim=1
        )
        coupler_bias = torch.cat(
            (coupler_bias, objective_coupler_bias.unsqueeze(1)), dim=1
        )
        return {
            "objective_residual": objective_residual,
            "residual": residual,

            "multipliers": multipliers,
            "binding_logits": binding_logits,
            "raw_residual": raw_residual,
            "coupler_weights": coupler_weights,
            "coupler_bias": coupler_bias,
            "value_context": graph_state,
            "active_channels": active,
        }

    def couple(self, output, live_state, graph_index=None):
        """Apply the same cheap state modulation evaluated by the C++ decoder."""
        _require_unit_interval("live_state", live_state)
        base = output["multipliers"]
        weights = output["coupler_weights"]
        bias = output["coupler_bias"]
        if weights.ndim != 3:
            raise ValueError("coupler_weights must have shape [B, C, S]")
        if live_state.shape[-1] != weights.shape[-1]:
            raise ValueError(
                "live_state width must match the runtime resource registry"
            )
        graph_count = weights.shape[0]
        if graph_index is None:
            if graph_count != 1:
                raise ValueError(
                    "graph_index is required when coupling multiple graphs"
                )
            graph_index = torch.zeros(
                live_state.shape[0], dtype=torch.long, device=live_state.device
            )
        else:
            graph_index = torch.as_tensor(
                graph_index, dtype=torch.long, device=live_state.device
            )
            if graph_index.shape != live_state.shape[:-1]:
                raise ValueError("graph_index must have one entry per live state")
            if graph_index.numel() and (
                graph_index.min() < 0 or graph_index.max() >= graph_count
            ):
                raise ValueError("graph_index contains an invalid graph")
        selected_weights = weights[graph_index]
        selected_bias = bias[graph_index]
        selected_base = base[graph_index]
        logit = torch.einsum("ncs,ns->nc", selected_weights, live_state)
        logit = logit + selected_bias
        return selected_base * (2.0 * torch.sigmoid(logit))

    def value(self, output, search_progress):
        """Estimate remaining normalized improvement from a refresh state."""
        context = output["value_context"]
        progress = torch.as_tensor(
            search_progress, dtype=context.dtype, device=context.device
        )
        if progress.ndim == 0:
            progress = progress.expand(context.shape[0])
        progress = progress.reshape(-1, 1)
        if progress.shape[0] != context.shape[0]:
            raise ValueError("search_progress must have one value per graph")
        _require_unit_interval("search_progress", progress)
        return self.value_head(torch.cat((context, progress), dim=-1)).squeeze(-1)


def load_constraint_field_state_dict(
    model: ConstraintFieldNet, state_dict: dict
) -> None:
    """Strict schema load. Program pooling intentionally invalidates all old weights."""
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"incompatible ConstraintFieldNet checkpoint for {MODEL_SCHEMA}"
        ) from error
