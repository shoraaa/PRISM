import math

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import prism_decoder


FIELD_CHANNEL_COUNT = prism_decoder.FIELD_CHANNEL_COUNT
LIVE_STATE_FEATURE_COUNT = prism_decoder.LIVE_STATE_FEATURE_COUNT
RESOURCE_DESCRIPTOR_DIM = prism_decoder.RESOURCE_DESCRIPTOR_DIM
RESOURCE_PROGRAM_MAX_TOKENS = prism_decoder.RESOURCE_PROGRAM_MAX_TOKENS
RESOURCE_PROGRAM_CATEGORY_COUNT = prism_decoder.RESOURCE_PROGRAM_CATEGORY_COUNT
RESOURCE_PROGRAM_VALUE_COUNT = prism_decoder.RESOURCE_PROGRAM_VALUE_COUNT
RESOURCE_PROGRAM_EDGE_ROLE_COUNT = prism_decoder.RESOURCE_PROGRAM_EDGE_ROLE_COUNT
RESOURCE_PROGRAM_ROLE_COUNT = prism_decoder.RESOURCE_PROGRAM_ROLE_COUNT
RESOURCE_PROGRAM_OPCODE_COUNT = prism_decoder.RESOURCE_PROGRAM_OPCODE_COUNT
RESOURCE_PROGRAM_INPUT_COUNT = prism_decoder.RESOURCE_PROGRAM_INPUT_COUNT
NODE_FEATURE_COUNT = prism_decoder.NODE_FEATURE_COUNT
EDGE_FEATURE_COUNT = prism_decoder.EDGE_FEATURE_COUNT
# The native feature contract places instance-static node attributes first and
# incumbent replay state afterwards. Edge slots 8/9 mark incumbent/reverse
# incumbent arcs; all other edge slots are instance-static.
STATIC_NODE_FEATURE_COUNT = 12
INCUMBENT_EDGE_FEATURE_START = 8
INCUMBENT_EDGE_FEATURE_END = 10


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
MODEL_SCHEMA = "typed_resource_v7_learned_program_graph"


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
    coefficient values."""
    distance = float(coeffs.get("distance_coeff", 1.0))
    visit = float(coeffs.get("visit_coeff", 0.0))
    miss = float(coeffs.get("miss_coeff", 0.0))
    regularizer = float(coeffs.get("distance_regularizer", 0.0))
    sense = float(coeffs.get("sense", 1.0))
    values = [
        _squash_magnitude(distance), _sign_bit(distance),
        _squash_magnitude(visit), _sign_bit(visit),
        _squash_magnitude(miss), _sign_bit(miss),
        _squash_magnitude(regularizer),
        _sign_bit(sense),
    ]
    return torch.tensor([values], dtype=torch.float32, device=device)
# Canonical constraint order for schema conditioning of the refiner; mirrors the
# native CONSTRAINT_KERNEL registry so a multi-hot indexes constraints stably.
CONSTRAINT_VOCAB = (
    "visit_all", "capacity", "backhaul_order", "pickup_delivery",
    "route_limit", "time_windows", "tour_limit", "prize_quota",
)


def constraint_multihot(constraints, device="cpu"):
    """Multi-hot [len(CONSTRAINT_VOCAB)] over active constraint names."""
    v = torch.zeros(len(CONSTRAINT_VOCAB), dtype=torch.float32, device=device)
    idx = {c: i for i, c in enumerate(CONSTRAINT_VOCAB)}
    for c in constraints:
        if c in idx:
            v[idx[c]] = 1.0
    return v


# Schema descriptor = constraint multi-hot + open-route flag + depot-count scale.
# open/multi-depot are route STRUCTURE, not entries in CONSTRAINT_VOCAB, but the
# refiner must distinguish e.g. cvrp from mdcvrp/ocvrp (same constraint set).
SCHEMA_FEATURE_DIM = len(CONSTRAINT_VOCAB) + 2


def schema_vector(problem, device="cpu"):
    """Full schema descriptor for refiner conditioning [SCHEMA_FEATURE_DIM]."""
    v = constraint_multihot(problem.get("constraints", []), device=device)
    depot_count = float(problem.get("depot_count", 1))
    extra = torch.tensor(
        [float(bool(problem.get("open_route", False))),
         depot_count / (1.0 + depot_count)],  # squashed, matches build_decoder_data
        dtype=torch.float32, device=device,
    )
    return torch.cat((v, extra))


def _require_unit_interval(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if value.numel() and (value.min() < -1e-6 or value.max() > 1.0 + 1e-6):
        raise ValueError(f"{name} must be normalized to [0, 1]")


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
    resource_descriptors = torch.as_tensor(
        decoder.resource_descriptors, dtype=torch.float32, device=device
    ).view(1, resource_count, RESOURCE_DESCRIPTOR_DIM)
    resource_program_categories = torch.as_tensor(
        decoder.resource_program_categories, dtype=torch.long, device=device
    ).view(
        1,
        resource_count,
        RESOURCE_PROGRAM_MAX_TOKENS,
        RESOURCE_PROGRAM_CATEGORY_COUNT,
    )
    resource_program_values = torch.as_tensor(
        decoder.resource_program_values, dtype=torch.float32, device=device
    ).view(
        1,
        resource_count,
        RESOURCE_PROGRAM_MAX_TOKENS,
        RESOURCE_PROGRAM_VALUE_COUNT,
    )
    resource_program_mask = torch.as_tensor(
        decoder.resource_program_mask, dtype=torch.bool, device=device
    ).view(1, resource_count, RESOURCE_PROGRAM_MAX_TOKENS)
    resource_program_edges = torch.as_tensor(
        decoder.resource_program_edges, dtype=torch.float32, device=device
    ).view(
        1,
        resource_count,
        RESOURCE_PROGRAM_EDGE_ROLE_COUNT,
        RESOURCE_PROGRAM_MAX_TOKENS,
        RESOURCE_PROGRAM_MAX_TOKENS,
    )
    resource_program_roots = torch.as_tensor(
        decoder.resource_program_roots, dtype=torch.bool, device=device
    ).view(1, resource_count, RESOURCE_PROGRAM_MAX_TOKENS)
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
    _require_unit_interval("resource_features", resource_features)
    _require_unit_interval("resource_events", resource_events)
    _require_unit_interval("resource_descriptors", resource_descriptors)
    _require_unit_interval("resource_program_values", resource_program_values)
    _require_unit_interval("resource_program_edges", resource_program_edges)
    if resource_program_mask.shape != (
        1,
        resource_count,
        RESOURCE_PROGRAM_MAX_TOKENS,
    ):
        raise ValueError("resource_program_mask has an invalid shape")
    if torch.any(resource_program_roots & ~resource_program_mask):
        raise ValueError("resource program roots must select active clauses")
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
        raw_resource_pressure=raw_resource_pressure,
        resource_features=resource_features,
        resource_events=resource_events,
        resource_descriptors=resource_descriptors,
        resource_program_categories=resource_program_categories,
        resource_program_values=resource_program_values,
        resource_program_mask=resource_program_mask,
        resource_program_edges=resource_program_edges,
        resource_program_roots=resource_program_roots,
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
    learned_candidate_quotas=False,
):
    """Run one model-guided perturbation on an installed incumbent graph."""
    if not decoder.best_solution["feasible"]:
        raise ValueError(
            "decode_iteration requires a feasible installed incumbent; "
            "greedily bootstrap and call decoder.set_incumbent() first"
        )
    graph = build_decoder_data(decoder, device=device)
    output = model(graph)
    if learned_candidate_quotas:
        decoder.set_candidate_resource_quotas(
            output["candidate_quota"][0].detach().cpu().numpy()
        )
        decoder.set_incumbent(decoder.best_solution["route"])
        graph = build_decoder_data(decoder, device=device)
        output = model(graph)
    edge_field = output["residual"].detach().cpu().numpy()
    multipliers = output["multipliers"][0].detach().cpu().numpy()
    solution = decoder.solve(
        1,
        edge_field=edge_field,
        edge_additive=output["additive"].detach().cpu().numpy(),
        multipliers=multipliers,
        coupler_weights=output["coupler_weights"][0].detach().cpu().numpy(),
        coupler_bias=output["coupler_bias"][0].detach().cpu().numpy(),
        objective_residual=output["objective_residual"].detach().cpu().numpy(),
        edge_risk=output["feasibility_risk"].detach().cpu().numpy(),
        risk_penalty=float(risk_penalty),
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
    ):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.grad_checkpointing = grad_checkpointing
        
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)
        
        self.layers = nn.ModuleList([
            GNNLayer(self.units, self.act_fn, self.agg_fn) for _ in range(self.depth)
        ])
        
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


class ResourceProgramEncoder(nn.Module):
    """Encode a canonical resource-program dataflow graph into one token.

    Categories describe executable syntax (clause role, opcode, input kind,
    scope, check phase, and direction), never a resource or variant name. Two
    typed message-passing steps preserve producer/consumer wiring before
    permutation-invariant sum/max/root pooling. The output width matches the
    existing resource-token interface; exact program execution remains native.
    """

    def __init__(self, output_units=32, hidden_units=64, message_depth=2):
        super().__init__()
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.message_depth = message_depth
        self.role_embedding = nn.Embedding(
            RESOURCE_PROGRAM_ROLE_COUNT, hidden_units
        )
        self.opcode_embedding = nn.Embedding(
            RESOURCE_PROGRAM_OPCODE_COUNT, hidden_units
        )
        self.input_embedding = nn.Embedding(
            RESOURCE_PROGRAM_INPUT_COUNT, hidden_units
        )
        self.scope_embedding = nn.Embedding(3, hidden_units)
        self.check_embedding = nn.Embedding(3, hidden_units)
        self.direction_embedding = nn.Embedding(3, hidden_units)
        self.value_projection = nn.Linear(
            RESOURCE_PROGRAM_VALUE_COUNT, hidden_units
        )
        self.edge_projections = nn.ModuleList(
            nn.Linear(hidden_units, hidden_units, bias=False)
            for _ in range(RESOURCE_PROGRAM_EDGE_ROLE_COUNT)
        )
        self.message_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_units, hidden_units),
                nn.SiLU(),
                nn.Linear(hidden_units, hidden_units),
            )
            for _ in range(message_depth)
        )
        self.message_norms = nn.ModuleList(
            nn.LayerNorm(hidden_units) for _ in range(message_depth)
        )
        self.output = nn.Sequential(
            nn.Linear(3 * hidden_units + 1, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, output_units),
        )

    @staticmethod
    def _require_category(name, values, upper):
        if values.dtype != torch.long:
            raise ValueError(f"{name} must use torch.long categorical indices")
        if values.numel() and (values.min() < 0 or values.max() >= upper):
            raise ValueError(f"{name} contains an out-of-vocabulary category")

    def forward(self, categories, values, mask, edges, roots):
        if categories.ndim != 4 or categories.shape[-2:] != (
            RESOURCE_PROGRAM_MAX_TOKENS,
            RESOURCE_PROGRAM_CATEGORY_COUNT,
        ):
            raise ValueError(
                "resource_program_categories must have shape "
                "[batch, resources, tokens, categories]"
            )
        expected_prefix = categories.shape[:3]
        if values.shape != expected_prefix + (RESOURCE_PROGRAM_VALUE_COUNT,):
            raise ValueError("resource_program_values has an invalid shape")
        if mask.shape != expected_prefix or roots.shape != expected_prefix:
            raise ValueError("resource program masks have an invalid shape")
        expected_edges = categories.shape[:2] + (
            RESOURCE_PROGRAM_EDGE_ROLE_COUNT,
            RESOURCE_PROGRAM_MAX_TOKENS,
            RESOURCE_PROGRAM_MAX_TOKENS,
        )
        if edges.shape != expected_edges:
            raise ValueError("resource_program_edges has an invalid shape")
        _require_unit_interval("resource_program_values", values)
        _require_unit_interval("resource_program_edges", edges)
        if torch.any(roots.bool() & ~mask.bool()):
            raise ValueError("resource program roots must select active clauses")

        category_sizes = (
            RESOURCE_PROGRAM_ROLE_COUNT,
            RESOURCE_PROGRAM_OPCODE_COUNT,
            RESOURCE_PROGRAM_INPUT_COUNT,
            3,
            3,
            3,
        )
        for index, upper in enumerate(category_sizes):
            self._require_category(
                f"resource_program_categories[{index}]",
                categories[..., index],
                upper,
            )

        h = (
            self.role_embedding(categories[..., 0])
            + self.opcode_embedding(categories[..., 1])
            + self.input_embedding(categories[..., 2])
            + self.scope_embedding(categories[..., 3])
            + self.check_embedding(categories[..., 4])
            + self.direction_embedding(categories[..., 5])
            + self.value_projection(values)
        )
        clause_mask = mask.bool().unsqueeze(-1)
        h = h * clause_mask
        for update, norm in zip(self.message_updates, self.message_norms):
            incoming = torch.zeros_like(h)
            for role, projection in enumerate(self.edge_projections):
                projected = projection(h)
                incoming = incoming + torch.einsum(
                    "brts,brsu->brtu", edges[:, :, role], projected
                )
            degree = edges.sum(dim=2).sum(dim=-1, keepdim=True).clamp_min(1.0)
            incoming = incoming / degree
            h = norm(h + update(torch.cat((h, incoming), dim=-1)))
            h = h * clause_mask

        count = mask.sum(dim=-1, keepdim=True).to(values.dtype).log1p()
        summed = h.sum(dim=-2)
        maximum = h.masked_fill(~clause_mask, -torch.inf).amax(dim=-2)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
        root_mask = roots.bool().unsqueeze(-1)
        root_sum = (h * root_mask).sum(dim=-2)
        return self.output(torch.cat((summed, maximum, root_sum, count), dim=-1))

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
        program_units=64,
    ):
        super().__init__()
        # Gate resource multipliers by the binding classifier so inactive/slack
        # resources begin damped. The signed edge field still receives policy
        # gradients through this gate; the ungated path remains an ablation.
        self.gate_multipliers_by_binding = gate_multipliers_by_binding
        self.emb_net = EmbNet(
            depth=depth,
            feats=NODE_FEATURE_COUNT,
            edge_feats=EDGE_FEATURE_COUNT,
            units=units,
            act_fn=act_fn,
            agg_fn=agg_fn,
            grad_checkpointing=grad_checkpointing,
        )
        # Learn resource semantics from the canonical executable program graph.
        # The separate context adapter preserves the existing resource-token
        # width and all downstream attention/field/coupler heads.
        self.program_encoder = ResourceProgramEncoder(
            output_units=units, hidden_units=program_units
        )
        context_size = 4 + OBJECTIVE_COEFF_DIM + 1 + 2
        self.resource_context_encoder = nn.Sequential(
            nn.Linear(context_size, units),
            nn.SiLU(),
            nn.Linear(units, units),
        )
        attention_heads = 4 if units % 4 == 0 else 1
        self.resource_attention = nn.MultiheadAttention(
            units, attention_heads, batch_first=True
        )
        self.edge_projection = nn.Linear(units, units)
        self.resource_edge_projection = nn.Linear(2, units)
        self.token_projection = nn.Linear(units, units)
        self.graph_projection = nn.Linear(units, units)
        self.field_head = nn.Linear(units, 1)
        self.additive_head = nn.Linear(units, 1)
        self.feasibility_head = nn.Linear(units, 1)
        self.multiplier_head = nn.Linear(units, 1)
        self.binding_head = nn.Linear(units, 1)
        self.candidate_quota_head = nn.Linear(units, 1)
        self.distance_quota_head = nn.Linear(units, 1)
        self.coupler_query_head = nn.Linear(units, units)
        self.coupler_key_head = nn.Linear(units, units)
        self.coupler_bias_head = nn.Linear(units, 1)
        # Retain the legacy objective-head parameters in the state dict so
        # existing typed-resource checkpoints and optimizer states remain
        # loadable. Forward fixes this slot to one: a scalar objective weight is
        # only a second decoder temperature and cannot learn edge ordering.
        self.objective_multiplier_head = nn.Linear(units, 1)
        self.objective_coupler_query_head = nn.Linear(units, units)
        self.objective_coupler_bias_head = nn.Linear(units, 1)
        self.value_head = nn.Linear(units + 1, 1)
        # Retain the original shared head in the state dict so typed-resource
        # checkpoints keep their parameter/optimizer ordering. It is no longer
        # used directly: one shared signed logit caused severe negative transfer
        # between distance and prize objectives.
        self.edge_logit_head = nn.Linear(units, 1)
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
        self.objective_energy_residual_head = nn.Sequential(
            nn.Linear(units + OBJECTIVE_COEFF_DIM, units),
            nn.SiLU(),
            nn.Linear(units, 1),
        )
        # Schema-conditioned advantage scale g_phi(schema): a detached, learned
        # RELATIVE multiplier on the batch-pooled advantage scale. It reads the
        # same algebra descriptor the field consumes (constraint multi-hot + route
        # structure), so a schema's dispersion relative to the batch is predicted
        # as a smooth function of the composition and generalises to held-out
        # variants -- replacing any hand-grouped per-objective normaliser. It is
        # initialised at 1.0 so it starts as an exact no-op on top of the pooled
        # scale (a stationary global magnitude); predicting absolute dispersion
        # instead would ramp the effective step size by orders of magnitude and
        # diverge. Because the output depends only on the schema (not the option),
        # every option of a schema shares one multiplier, so degenerate
        # low-variance options are not amplified. Trained by a separate log-space
        # regression, never through the policy loss (detached where it scales).
        self.reward_scale_head = nn.Sequential(
            nn.Linear(SCHEMA_FEATURE_DIM, units),
            nn.SiLU(),
            nn.Linear(units, 1),
        )
        nn.init.zeros_(self.field_head.weight)
        nn.init.zeros_(self.field_head.bias)
        nn.init.zeros_(self.additive_head.weight)
        nn.init.zeros_(self.additive_head.bias)
        # Preserve the neutral objective policy exactly at initialization and
        # when upgrading an older typed-resource checkpoint.
        nn.init.zeros_(self.edge_logit_head.weight)
        nn.init.zeros_(self.edge_logit_head.bias)
        # Small non-zero init on the final layer so the hidden (coefficient-
        # conditioning) layer receives gradient from step 0. Bias stays zero so a
        # row-constant output remains neutral after row-centering; the residual
        # starts small and is bounded by row-center + tanh + --objective-residual-l2.
        nn.init.normal_(
            self.objective_energy_residual_head[-1].weight,
            std=OBJECTIVE_RESIDUAL_HEAD_INIT_STD,
        )
        nn.init.zeros_(self.objective_energy_residual_head[-1].bias)
        nn.init.zeros_(self.feasibility_head.weight)
        # A fresh model must reproduce the plain-objective (E = c(e)) neutral
        # search -- the initial policy from which the field is learned, not the
        # flat fields-off baseline. sigmoid(-20) is below ranking tolerance even
        # after a full route is aggregated, while BCE-with-logits still gives
        # positive risk labels a gradient near -1.
        nn.init.constant_(self.feasibility_head.bias, -20.0)
        nn.init.zeros_(self.coupler_query_head.weight)
        nn.init.zeros_(self.coupler_query_head.bias)
        nn.init.zeros_(self.coupler_bias_head.weight)
        nn.init.zeros_(self.coupler_bias_head.bias)
        # These legacy heads are deliberately neutral and unused by forward.
        nn.init.zeros_(self.objective_multiplier_head.weight)
        nn.init.zeros_(self.objective_multiplier_head.bias)
        nn.init.zeros_(self.objective_coupler_query_head.weight)
        nn.init.zeros_(self.objective_coupler_query_head.bias)
        nn.init.zeros_(self.objective_coupler_bias_head.weight)
        nn.init.zeros_(self.objective_coupler_bias_head.bias)
        # A zero value is the neutral bootstrap for old checkpoints and makes
        # enabling temporal credit leave the field policy unchanged initially.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        # Start the schema scale at exactly 1.0 (softplus(shift) == 1) so enabling
        # it leaves the advantage untouched until the regression fits dispersion.
        nn.init.zeros_(self.reward_scale_head[-1].weight)
        nn.init.constant_(
            self.reward_scale_head[-1].bias, math.log(math.expm1(1.0))
        )
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
        return (
            self.field_head(interaction).squeeze(-1),
            self.additive_head(interaction).squeeze(-1),
        )

    def reward_scale(self, schema):
        """Positive per-schema advantage scale g_phi(schema) from the algebra
        descriptor [..., SCHEMA_FEATURE_DIM]. Softplus keeps it positive and, at
        initialization, exactly 1.0 so enabling it is a no-op until fitted."""
        return F.softplus(self.reward_scale_head(schema).squeeze(-1))

    def forward(self, pyg):
        _require_unit_interval("node_features", pyg.x)
        _require_unit_interval("edge_features", pyg.edge_attr)
        active = pyg.active_channels
        if active.ndim == 1:
            active = active.unsqueeze(0)
        _require_unit_interval("active_channels", active)
        resource_count = active.shape[-1]
        if resource_count < FIELD_CHANNEL_COUNT:
            raise ValueError("resource registry must include canonical rows")

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
                torch.zeros_like(pyg.x[:, STATIC_NODE_FEATURE_COUNT:]),
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
        edge_embedding = self.emb_net(pyg.x, pyg.edge_index, pyg.edge_attr)

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
        program_categories = pyg.resource_program_categories
        program_values = pyg.resource_program_values
        program_mask = pyg.resource_program_mask
        program_edges = pyg.resource_program_edges
        program_roots = pyg.resource_program_roots
        if program_categories.shape[:2] != (batch_size, resource_count):
            raise ValueError(
                "resource program tensors must begin with "
                "[num_graphs, resource_count]"
            )

        def _broadcast(column):
            return column.unsqueeze(1).expand(-1, resource_count, -1)

        context = torch.cat(
            (
                active.unsqueeze(-1),
                resource_mean.unsqueeze(-1),
                resource_max.unsqueeze(-1),
                _broadcast(open_route),
                _broadcast(objective_coeffs),
                _broadcast(objective_scale),
                _broadcast(multi_route),
                _broadcast(depot_scale),
            ),
            dim=-1,
        )
        _require_unit_interval("resource_context", context)
        tokens = self.program_encoder(
            program_categories,
            program_values,
            program_mask,
            program_edges,
            program_roots,
        ) + self.resource_context_encoder(context)
        padding_mask = ~active.bool()
        no_resources = ~active.bool().any(dim=1)
        if no_resources.any():
            padding_mask = padding_mask.clone()
            padding_mask[no_resources, 0] = False
        coupled_tokens, _ = self.resource_attention(
            tokens, tokens, tokens, key_padding_mask=padding_mask
        )
        tokens = tokens + coupled_tokens

        projected_edges = self.edge_projection(edge_embedding)
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
        raw_objective_residual = self.objective_energy_residual_head(
            torch.cat((objective_edge_state, edge_objective_coeffs), dim=-1)
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
        raw_channels = []
        additive_channels = []
        for channel in range(resource_count):
            token = (
                projected_tokens[:, channel]
                if batched
                else projected_tokens[0, channel]
            )
            resource_edge = self.resource_edge_projection(
                torch.stack(
                    (
                        normalized_resources[:, channel],
                        resource_events[:, channel],
                    ),
                    dim=1,
                )
            )
            if (
                self.emb_net.grad_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                raw, additive_channel = torch.utils.checkpoint.checkpoint(
                    self._field_channel,
                    projected_edges,
                    token,
                    resource_edge,
                    edge_batch,
                    use_reentrant=False,
                )
            else:
                raw, additive_channel = self._field_channel(
                    projected_edges, token, resource_edge, edge_batch
                )
            raw_channels.append(raw)
            additive_channels.append(additive_channel)
        raw_residual = torch.stack(raw_channels, dim=1)
        # Resource guidance is a signed, zero-neutral learned field. Analytic
        # pressure remains an input feature, but the decoder does not multiply
        # it into energy. Signed terms let PPO reward useful capacity/route-limit
        # edges as well as penalize harmful ones; exact native feasibility
        # remains authoritative.
        residual = torch.tanh(raw_residual)
        additive = torch.tanh(torch.stack(additive_channels, dim=1))
        feasibility_logits = self.feasibility_head(
            torch.tanh(projected_edges)
        ).squeeze(-1)
        feasibility_risk = torch.sigmoid(feasibility_logits)
        edge_active = active[edge_batch] if batched else active[0]
        residual = residual * edge_active
        additive = additive * edge_active

        projected_graph = self.graph_projection(graph_embedding)
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
        resource_quota_logits = self.candidate_quota_head(state).squeeze(-1)
        resource_quota_logits = resource_quota_logits.masked_fill(
            ~active.bool(), torch.finfo(resource_quota_logits.dtype).min
        )
        distance_quota_logit = self.distance_quota_head(graph_state)
        candidate_quota_logits = torch.cat(
            (resource_quota_logits, distance_quota_logit), dim=1
        )
        candidate_quota = torch.softmax(candidate_quota_logits, dim=1)
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
            "additive": additive,
            "feasibility_logits": feasibility_logits,
            "feasibility_risk": feasibility_risk,
            "multipliers": multipliers,
            "binding_logits": binding_logits,
            "raw_residual": raw_residual,
            "coupler_weights": coupler_weights,
            "coupler_bias": coupler_bias,
            "candidate_quota_logits": candidate_quota_logits,
            "candidate_quota": candidate_quota[:, :-1],
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


def _sinusoidal_positions(length, units, device, dtype):
    """Plain sinusoidal PE over route position.

    A first, cheap stand-in for CaR's cyclic positional encoding. Route order is
    what distinguishes an improvement policy from the order-agnostic field, so
    even this simple version is load-bearing; swap in the cyclic variant later.
    """
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, units, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / units)
    )
    pe = torch.zeros(length, units, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


class RefinementDecoder(nn.Module):
    """Learned remove-and-reinsert refinement operator (CaR Path A).

    Replaces the hand-designed C++ perturb / scope_restricted_refine move
    generator with a neural policy: given the current incumbent as an ordered
    sequence, a *ruin* head selects rm_num customers to remove and a *recreate*
    head sequentially chooses a reinsertion gap for each. The C++ decoder stays
    the feasibility+cost oracle (`evaluate`), so the policy only proposes routes;
    it never has to re-derive the resource algebra.

    The encoder is shared with ConstraintFieldNet via `emb_net` (CaR's
    unified_encoder), so construction/field and refinement reuse one node
    representation.
    """

    def __init__(
        self,
        units=32,
        rm_num=3,
        emb_net=None,
        depth=12,
        act_fn="silu",
        agg_fn="mean",
        grad_checkpointing=False,
    ):
        super().__init__()
        self.units = units
        self.rm_num = rm_num
        if emb_net is None:
            emb_net = EmbNet(
                depth=depth,
                feats=NODE_FEATURE_COUNT,
                edge_feats=EDGE_FEATURE_COUNT,
                units=units,
                act_fn=act_fn,
                agg_fn=agg_fn,
                grad_checkpointing=grad_checkpointing,
            )
        self.emb_net = emb_net
        # Live resource state has a graph-dependent width (resource_count). We
        # summarize it to a fixed 3-dim [mean, max, spread] per node so the same
        # weights apply to any registry. TODO: condition on resource_descriptors
        # like ConstraintFieldNet to keep per-resource identity.
        self.state_proj = nn.Linear(3, units)
        self.pos_proj = nn.Linear(units, units)
        self.route_attention = nn.MultiheadAttention(
            units, 4 if units % 4 == 0 else 1, batch_first=True
        )
        self.route_norm = nn.LayerNorm(units)
        # Ruin head: per-route-position removal logit.
        self.remove_head = nn.Sequential(
            nn.Linear(units, units), nn.SiLU(), nn.Linear(units, 1)
        )
        # Recreate head: score inserting node h_node into gap (h_u, h_v).
        self.insert_head = nn.Sequential(
            nn.Linear(3 * units, units), nn.SiLU(), nn.Linear(units, 1)
        )

    def encode_static(self, graph):
        """Route-independent GNN node embeddings [N, units].

        Depends only on static (instance-level) node/edge features, so it can be
        computed ONCE per instance and reused across every rollout and step --
        the key to a batched, oracle-free rollout. Dynamic route state is folded
        in separately by fuse_state().
        """
        _, node_emb = self.emb_net(
            graph.x, graph.edge_index, graph.edge_attr, return_nodes=True
        )
        return node_emb

    def fuse_state(self, node_emb, live_state):
        """Add the dynamic per-node resource state to static embeddings."""
        summary = torch.stack(
            (
                live_state.mean(dim=1),
                live_state.amax(dim=1),
                live_state.amax(dim=1) - live_state.amin(dim=1),
            ),
            dim=1,
        )  # [N, 3]
        return node_emb + self.state_proj(summary)  # [N, units]

    def encode_nodes(self, graph, live_state):
        """Convenience: static encode + fuse (used by the C++-graph path)."""
        return self.fuse_state(self.encode_static(graph), live_state)

    def route_context(self, node_h, route):
        """Order-aware per-position embeddings for one route (cheap: PE + MHA)."""
        seq = node_h[route]  # [L, units]
        pe = _sinusoidal_positions(
            seq.shape[0], self.units, seq.device, seq.dtype
        )
        seq = seq + self.pos_proj(pe)
        attended, _ = self.route_attention(
            seq.unsqueeze(0), seq.unsqueeze(0), seq.unsqueeze(0)
        )
        return self.route_norm(seq + attended.squeeze(0))  # [L, units]

    def forward(
        self, graph, route, live_state, depot_count, greedy=False, adj=None,
        node_emb=None,
    ):
        """Propose one refined route.

        node_emb: optional precomputed static node embeddings [N, units] from
        encode_static(). Pass it to skip the per-step GNN forward (compute once
        per instance) -- the oracle-free batched rollout relies on this. When
        None, the GNN runs on `graph` as before.

        adj: optional [N, N] bool adjacency (e.g. the decoder's candidate-graph
        neighborhood). When given, reinsertion is restricted to gaps adjacent to
        the removed node's neighbors -- this is the difference between a ~0.6%
        and a ~40% improving-move density, since scoring all gaps blindly almost
        never lands near an improving position.

        Returns (new_route: list[int], logp: scalar tensor, entropy: scalar
        tensor). Feasibility/cost are NOT checked here -- pass new_route to
        decoder.evaluate() and reward accordingly.
        """
        device = node_emb.device if node_emb is not None else graph.x.device
        route = torch.as_tensor(route, device=device).long()
        if node_emb is None:
            node_emb = self.encode_static(graph)
        node_h = self.fuse_state(node_emb, live_state)  # [N, units]
        seq = self.route_context(node_h, route)  # [L, units]
        route_list = route.tolist()

        # ---- Ruin: sample rm_num distinct customer positions ----
        is_customer = route >= depot_count  # depots are never removed
        remove_logits = self.remove_head(seq).squeeze(-1)  # [L]
        remove_logits = remove_logits.masked_fill(~is_customer, float("-inf"))
        logp = seq.new_zeros(())
        entropy = seq.new_zeros(())
        removed_positions = []
        n_remove = min(self.rm_num, int(is_customer.sum().item()))
        for _ in range(n_remove):
            # Fresh mask each step: mutating a tensor already captured by an
            # earlier masked_fill breaks autograd (version-counter error).
            taken = torch.zeros_like(is_customer)
            if removed_positions:
                taken[removed_positions] = True
            logits = remove_logits.masked_fill(taken, float("-inf"))
            dist = torch.distributions.Categorical(logits=logits)
            pos = logits.argmax() if greedy else dist.sample()
            logp = logp + dist.log_prob(pos)
            entropy = entropy + dist.entropy()
            removed_positions.append(int(pos.item()))

        removed_nodes = [route_list[p] for p in removed_positions]
        partial = [
            n for i, n in enumerate(route_list) if i not in set(removed_positions)
        ]

        # ---- Recreate: sequentially reinsert each removed node into a gap ----
        # Gap endpoints reuse the cached node embeddings directly, so no GNN or
        # attention pass runs inside this loop. (Trade-off: gap scoring loses
        # full-route context vs re-encoding; add it back if reinsertion quality
        # is the bottleneck.)
        for node in removed_nodes:
            partial_t = torch.as_tensor(partial, device=device).long()
            gap_h = node_h[partial_t]  # [P, units]
            gaps = gap_h.shape[0] - 1
            u = gap_h[:-1]
            v = gap_h[1:]
            node_e = node_h[node].expand(gaps, -1)
            gap_score = self.insert_head(
                torch.cat((node_e, u, v), dim=-1)
            ).squeeze(-1)  # [gaps]
            if adj is not None:
                # Allow a gap only if one of its endpoints neighbors the node.
                near = adj[node, partial_t]  # [P] bool
                allowed = near[:-1] | near[1:]  # [gaps]
                if allowed.any():
                    gap_score = gap_score.masked_fill(~allowed, float("-inf"))
            dist = torch.distributions.Categorical(logits=gap_score)
            gap = gap_score.argmax() if greedy else dist.sample()
            logp = logp + dist.log_prob(gap)
            entropy = entropy + dist.entropy()
            partial.insert(int(gap.item()) + 1, node)

        return partial, logp, entropy


class BatchedRelocate(nn.Module):
    """Fully-batched relocate operator (rm_num=1) over [B, L] routes.

    The proven single-node relocate (52%-improving neighborhood) done for a whole
    batch of rollouts at once -- no Python per-rollout loop, no per-move .item()
    sync. This is what makes GPU actually pay off: one forward proposes B moves.
    Encoder is shared via `emb_net`; static node embeddings are passed in
    precomputed (encode_static), so the GNN runs once per instance, not per step.
    """

    def __init__(self, units=32, depth=12, emb_net=None, grad_checkpointing=False,
                 heads=4, max_seg=1):
        super().__init__()
        self.units = units
        self.max_seg = max_seg  # OR-OPT: relocate contiguous segments of len 1..max_seg
        if emb_net is None:
            emb_net = EmbNet(
                depth=depth, feats=NODE_FEATURE_COUNT, edge_feats=EDGE_FEATURE_COUNT,
                units=units, grad_checkpointing=grad_checkpointing,
            )
        self.emb_net = emb_net
        self.state_proj = nn.Linear(3, units)
        self.pos_proj = nn.Linear(units, units)
        self.route_attention = nn.MultiheadAttention(
            units, 4 if units % 4 == 0 else 1, batch_first=True
        )
        self.route_norm = nn.LayerNorm(units)
        # N2S-style node-pair heads (CaR-constraint models/SINGLEModel.py). The
        # removal score of a node comes from its compatibility with its route
        # PREDECESSOR and SUCCESSOR (how badly it fits between them) -- the
        # learnable signal a flat per-node head lacks, which is why my removal CE
        # plateaued. Reinsertion scores each gap by the removed node's
        # compatibility with the gap's two endpoints.
        self.heads = heads if units % heads == 0 else 1
        self.rm_q = nn.Linear(units, units, bias=False)
        self.rm_k = nn.Linear(units, units, bias=False)
        self.rm_agg = nn.Sequential(
            nn.Linear(self.heads, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.ins_q = nn.Linear(units, units, bias=False)
        self.ins_k = nn.Linear(units, units, bias=False)
        self.ins_agg = nn.Sequential(
            nn.Linear(2 * self.heads, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        # Schema conditioning: a graph-level descriptor (constraint multi-hot +
        # open-route + depot-count scale) projected and added to every node
        # embedding, so ONE refiner behaves per-schema (CaR-constraint's
        # constraint generalization, extended to route structure).
        self.schema_proj = nn.Linear(SCHEMA_FEATURE_DIM, units)
        # OR-OPT heads: segment length (from graph context) and reversal (from the
        # segment's head/tail). Present even at max_seg=1 (unused) so checkpoints
        # are shape-stable across max_seg settings.
        self.len_head = nn.Sequential(
            nn.Linear(units, units), nn.SiLU(), nn.Linear(units, max_seg)
        )
        self.rev_head = nn.Sequential(
            nn.Linear(2 * units, units), nn.SiLU(), nn.Linear(units, 1)
        )

    def encode_static(self, graph):
        """Static node embeddings [N, units] -- compute once per instance."""
        _, node_emb = self.emb_net(
            graph.x, graph.edge_index, graph.edge_attr, return_nodes=True
        )
        return node_emb

    @staticmethod
    def _summary(live):  # [B, N, C] -> [B, N, 3]
        return torch.stack(
            (live.mean(-1), live.amax(-1), live.amax(-1) - live.amin(-1)), dim=-1
        )

    def _encode_route(self, node_emb, live, rt, valid, schema=None):
        """Fuse static node emb with live state (+ schema), gather the route,
        self-attend. Returns (seq [B,L,U], node_h [B,N,U]). schema [K] multi-hot
        of active constraints, broadcast to all nodes."""
        B, L = rt.shape
        U = self.units
        node_h = node_emb.unsqueeze(0) + self.state_proj(self._summary(live))  # [B,N,U]
        if schema is not None:
            node_h = node_h + self.schema_proj(schema).view(1, 1, U)
        safe = rt.clamp(min=0)
        seq = torch.gather(node_h, 1, safe.unsqueeze(-1).expand(B, L, U))
        pe = _sinusoidal_positions(L, U, node_emb.device, seq.dtype).unsqueeze(0)
        seq = seq + self.pos_proj(pe)
        attn, _ = self.route_attention(seq, seq, seq, key_padding_mask=~valid)
        return self.route_norm(seq + attn), node_h

    def _removal_logits(self, seq, rt, valid, depot_count, tabu_node=None):
        """N2S node-pair removal score per route position: compatibility of each
        node with its predecessor and successor (Q_pre.K + Q.K_post - Q_pre.K_post
        per head, aggregated). tabu_node [B] optionally forbids re-removing a
        node (anti-cycling, so greedy hill-climb can't oscillate)."""
        B, L, U = seq.shape
        H, d = self.heads, self.units // self.heads
        q = self.rm_q(seq).view(B, L, H, d)
        k = self.rm_k(seq).view(B, L, H, d)
        zpad = q.new_zeros(B, 1, H, d)
        q_pre = torch.cat((zpad, q[:, :-1]), dim=1)   # predecessor query
        k_post = torch.cat((k[:, 1:], zpad), dim=1)   # successor key
        compat = ((q_pre * k).sum(-1) + (q * k_post).sum(-1)
                  - (q_pre * k_post).sum(-1))          # [B, L, H]
        logit = self.rm_agg(compat).squeeze(-1)        # [B, L]
        is_cust = valid & (rt >= depot_count)
        if tabu_node is not None:
            is_cust = is_cust & (rt != tabu_node.view(B, 1))
        return logit.masked_fill(~is_cust, float("-inf"))

    @staticmethod
    def _partial(rt, valid, pos_r):
        """Route with position pos_r [B] removed, order preserved (pad=-1).
        Returns (partial [B,L], m [B])."""
        B, L = rt.shape
        dev = rt.device
        rows = torch.arange(B, device=dev)
        keep = valid.clone()
        keep[rows, pos_r] = False
        m = valid.sum(1) - 1
        dest = keep.cumsum(1) - 1
        partial = torch.full((B, L), -1, dtype=torch.long, device=dev)
        partial[rows.unsqueeze(1).expand(B, L)[keep], dest[keep]] = rt[keep]
        return partial, m

    def _gap_logits(self, node_h, partial, m, removed, adj, gap_feas_fn):
        """N2S node-pair reinsertion score per gap: the removed node's
        compatibility with the gap's two endpoints (u = partial[g],
        v = partial[g+1]). Masked to the proximity-restricted feasible action
        space. Returns gscore [B,L-1]."""
        B, L = partial.shape
        U = self.units
        H, d = self.heads, self.units // self.heads
        dev = node_h.device
        rows = torch.arange(B, device=dev)
        psafe = partial.clamp(min=0)
        ph = torch.gather(node_h, 1, psafe.unsqueeze(-1).expand(B, L, U))
        qc = self.ins_q(node_h[rows, removed]).view(B, 1, H, d)  # removed-node query
        ku = self.ins_k(ph[:, :-1]).view(B, L - 1, H, d)         # gap left endpoint
        kv = self.ins_k(ph[:, 1:]).view(B, L - 1, H, d)          # gap right endpoint
        compat_u = (qc * ku).sum(-1)  # [B, L-1, H]
        compat_v = (qc * kv).sum(-1)
        gscore = self.ins_agg(torch.cat((compat_u, compat_v), -1)).squeeze(-1)
        gap_valid = torch.arange(L - 1, device=dev).unsqueeze(0) < (m - 1).unsqueeze(1)
        # adj=None disables the proximity restriction (feasibility alone defines
        # the action space). Proximity densified an untrained random policy but
        # can exclude the teacher's best gap, breaking imitation CE, and a trained
        # policy does not need it.
        prox = gap_valid
        if adj is not None:
            near = adj[removed]  # [B, N]
            prox = gap_valid & (
                torch.gather(near, 1, psafe[:, :-1]) | torch.gather(near, 1, psafe[:, 1:])
            )
        if gap_feas_fn is not None:
            feas_gap = gap_feas_fn(partial, m, removed)
            prox_feas = prox & feas_gap
            allowed = torch.where(
                prox_feas.any(1, keepdim=True), prox_feas,
                torch.where(feas_gap.any(1, keepdim=True), feas_gap, gap_valid),
            )
        else:
            allowed = torch.where(prox.any(1, keepdim=True), prox, gap_valid)
        return gscore.masked_fill(~allowed, float("-inf"))

    @staticmethod
    def _apply_insert(partial, m, removed, gap):
        """Insert `removed` after position `gap` in `partial`. Returns
        (new_rt [B,L], new_valid [B,L])."""
        B, L = partial.shape
        dev = partial.device
        ar = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
        at = ar == (gap + 1).unsqueeze(1)
        after = ar > (gap + 1).unsqueeze(1)
        src_idx = torch.where(after, (ar - 1).clamp(min=0), ar)
        gathered = torch.gather(partial, 1, src_idx)
        new_rt = torch.where(at, removed.unsqueeze(1), gathered)
        new_valid = ar < (m + 1).unsqueeze(1)
        new_rt = torch.where(new_valid, new_rt, torch.full_like(new_rt, -1))
        return new_rt, new_valid

    # ---------- OR-OPT (segment relocate) neural heads ----------
    @staticmethod
    def _shift_up(t, n):
        if n == 0:
            return t
        z = t.new_zeros((t.shape[0], n) + tuple(t.shape[2:]))
        return torch.cat((t[:, n:], z), dim=1)

    def _segment_start_logits(self, seq, rt, valid, depot_count, s):
        """Score each position as the START of a length-s segment to relocate,
        via node-pair compatibility on the segment's boundary (predecessor->head,
        tail->successor, minus predecessor->successor). Masked to positions where
        [i..i+s-1] are all customers (segment stays inside one route)."""
        B, L, U = seq.shape
        H, d = self.heads, self.units // self.heads
        q = self.rm_q(seq).view(B, L, H, d)
        k = self.rm_k(seq).view(B, L, H, d)
        z1 = q.new_zeros(B, 1, H, d)
        q_pred = torch.cat((z1, q[:, :-1]), dim=1)       # q[i-1]
        q_tail = self._shift_up(q, s - 1)                 # q[i+s-1]
        k_succ = self._shift_up(k, s)                     # k[i+s]
        compat = ((q_pred * k).sum(-1) + (q_tail * k_succ).sum(-1)
                  - (q_pred * k_succ).sum(-1))            # [B, L, H]
        logit = self.rm_agg(compat).squeeze(-1)          # [B, L]
        is_cust = valid & (rt >= depot_count)
        ok = is_cust.clone()
        for kk in range(1, s):
            shifted = torch.cat(
                (is_cust[:, kk:], torch.zeros(B, kk, dtype=torch.bool, device=seq.device)),
                dim=1,
            )
            ok = ok & shifted
        return logit.masked_fill(~ok, float("-inf"))

    def _block_gap_logits(self, node_h, partial, block, feas_gap):
        """Score each gap for inserting `block` [B,s]: removed-block head compat
        with the gap's left endpoint + block tail compat with the right endpoint.
        feas_gap [B,G] restricts to feasible insertions. Returns [B,G]."""
        B, L = partial.shape
        U = self.units
        H, d = self.heads, self.units // self.heads
        rows = torch.arange(B, device=node_h.device)
        psafe = partial.clamp(min=0)
        ph = torch.gather(node_h, 1, psafe.unsqueeze(-1).expand(B, L, U))  # [B,L,U]
        v = torch.cat((ph[:, 1:], ph[:, -1:]), dim=1)  # right endpoint (last repeats)
        qh = self.ins_q(node_h[rows, block[:, 0]]).view(B, 1, H, d)
        qt = self.ins_q(node_h[rows, block[:, -1]]).view(B, 1, H, d)
        ku = self.ins_k(ph).view(B, L, H, d)
        kv = self.ins_k(v).view(B, L, H, d)
        compat_u = (qh * ku).sum(-1)
        compat_v = (qt * kv).sum(-1)
        gscore = self.ins_agg(torch.cat((compat_u, compat_v), -1)).squeeze(-1)  # [B,L]
        return gscore.masked_fill(~feas_gap, float("-inf"))

    def oropt_imitation_loss(self, node_emb, live, rt, valid, ev, seg_pos, seg_len,
                             gap, rev, depot_count=1, schema=None):
        """Behaviour-clone the best_oropt teacher: CE over segment length, start
        position, insertion gap, and (for s>1) reversal. Uses the evaluator to
        build the per-row partial for the teacher's segment (teacher forcing)."""
        rt = rt.long()
        B, L = rt.shape
        dev = node_emb.device
        rows = torch.arange(B, device=dev)
        seq, node_h = self._encode_route(node_emb, live, rt, valid, schema=schema)

        # (1) length
        gctx = node_h.mean(dim=1)  # [B, U]
        len_logits = self.len_head(gctx)  # [B, max_seg]
        loss = F.cross_entropy(len_logits, (seg_len - 1).clamp(0, self.max_seg - 1))

        # (2) start position, per teacher length
        start_logits = torch.full((B, L), float("-inf"), device=dev)
        for s in range(1, self.max_seg + 1):
            sl = self._segment_start_logits(seq, rt, valid, depot_count, s)
            m = (seg_len == s).unsqueeze(1)
            start_logits = torch.where(m, sl, start_logits)
        loss = loss + F.cross_entropy(start_logits, seg_pos)

        # (3) gap + (4) reversal, per teacher length (build partial via evaluator)
        gap_loss = seq.new_zeros(())
        rev_loss = seq.new_zeros(())
        rev_count = 0
        for s in range(1, self.max_seg + 1):
            sel = seg_len == s
            if not bool(sel.any()):
                continue
            partial, mm, block = ev.segment_partial_rows(rt, valid, seg_pos, s)
            block_o = torch.where(rev.unsqueeze(1), block.flip(1), block)
            _, feas = ev._segment_insertion_eval(partial, mm, block_o)  # [B,G] bool
            glog = self._block_gap_logits(node_h, partial, block_o, feas)
            gl = F.cross_entropy(glog[sel], gap[sel])
            gap_loss = gap_loss + gl * float(sel.sum())
            if s > 1:
                bt = torch.cat((node_h[rows, block[:, 0]], node_h[rows, block[:, -1]]), -1)
                rev_logit = self.rev_head(bt).squeeze(-1)  # [B]
                rev_loss = rev_loss + F.binary_cross_entropy_with_logits(
                    rev_logit[sel], rev[sel].float(), reduction="sum"
                )
                rev_count += int(sel.sum())
        loss = loss + gap_loss / B
        if rev_count:
            loss = loss + rev_loss / rev_count
        return loss

    @staticmethod
    def _safe_logits(logits):
        """Rows that are entirely -inf (no legal action) would make Categorical
        NaN; replace them with uniform so sampling is defined (those rows are
        masked out downstream anyway)."""
        dead = torch.isinf(logits).all(dim=1, keepdim=True)
        return torch.where(dead, torch.zeros_like(logits), logits)

    def forward_oropt(self, ev, node_emb, live, rt, valid, depot_count=1,
                      greedy=False, schema=None):
        """OR-OPT deployment step: sample (length, start, reversal, gap) and apply
        the segment relocate. Uses the evaluator for per-row segment construction
        and feasibility. Returns (new_rt, new_valid, logp, entropy, removed_head)
        where removed_head is the segment's first node (for anti-cycling tabu)."""
        rt = rt.long()
        B, L = rt.shape
        dev = node_emb.device
        rows = torch.arange(B, device=dev)
        seq, node_h = self._encode_route(node_emb, live, rt, valid, schema=schema)

        len_logits = self.len_head(node_h.mean(dim=1))  # [B, max_seg]
        ld = torch.distributions.Categorical(logits=len_logits)
        s_sel = len_logits.argmax(1) if greedy else ld.sample()
        logp = ld.log_prob(s_sel)
        ent = ld.entropy()
        seg_len = s_sel + 1

        new_rt = rt.clone()
        new_valid = valid.clone()
        removed_head = rt[:, 0].clone()
        for s in range(1, self.max_seg + 1):
            rows_s = seg_len == s
            if not bool(rows_s.any()):
                continue
            slog = self._safe_logits(
                self._segment_start_logits(seq, rt, valid, depot_count, s)
            )
            sd = torch.distributions.Categorical(logits=slog)
            p = slog.argmax(1) if greedy else sd.sample()
            logp = logp + torch.where(rows_s, sd.log_prob(p), logp.new_zeros(()))
            ent = ent + torch.where(rows_s, sd.entropy(), ent.new_zeros(()))
            partial, mm, block = ev.segment_partial_rows(rt, valid, p, s)
            if s > 1:
                bt = torch.cat(
                    (node_h[rows, block[:, 0]], node_h[rows, block[:, -1]]), -1
                )
                rlogit = self.rev_head(bt).squeeze(-1)
                rprob = torch.sigmoid(rlogit)
                rev = (rprob > 0.5) if greedy else (torch.rand_like(rprob) < rprob)
                rl = -torch.nn.functional.binary_cross_entropy_with_logits(
                    rlogit, rev.float(), reduction="none"
                )
                logp = logp + torch.where(rows_s, rl, logp.new_zeros(()))
            else:
                rev = torch.zeros(B, dtype=torch.bool, device=dev)
            block_o = torch.where(rev.unsqueeze(1), block.flip(1), block)
            _, feas = ev._segment_insertion_eval(partial, mm, block_o)  # [B,G]
            glog = self._safe_logits(
                self._block_gap_logits(node_h, partial, block_o, feas)
            )
            gd = torch.distributions.Categorical(logits=glog)
            g = glog.argmax(1) if greedy else gd.sample()
            logp = logp + torch.where(rows_s, gd.log_prob(g), logp.new_zeros(()))
            ent = ent + torch.where(rows_s, gd.entropy(), ent.new_zeros(()))
            cand, cand_valid, _ = ev._segment_insertion_candidates(partial, mm, block_o)
            chosen = cand[rows, g]
            chosen_v = cand_valid[rows, g]
            # only commit rows that picked this length and had a feasible gap
            had_feas = feas.any(dim=1)
            commit = rows_s & had_feas
            new_rt = torch.where(commit.unsqueeze(1), chosen, new_rt)
            new_valid = torch.where(commit.unsqueeze(1), chosen_v, new_valid)
            removed_head = torch.where(commit, block[:, 0], removed_head)
        return new_rt, new_valid, logp, ent, removed_head

    def forward(self, node_emb, live, rt, valid, adj, depot_count=1, greedy=False,
                gap_feas_fn=None, tabu_node=None, schema=None):
        """node_emb [N,U] static; live [B,N,C]; rt long [B,L] (pad=-1);
        valid bool [B,L]; adj bool [N,N]. Returns (new_rt [B,L], new_valid [B,L],
        logp [B], entropy [B], removed [B]).

        gap_feas_fn(partial [B,L], m [B], removed [B]) -> bool [B,L-1] optionally
        restricts recreate to feasible insertions, making every proposed move
        feasible-by-construction (the CaR/SRR feasible action space). tabu_node [B]
        forbids re-removing a node (anti-cycling). schema [K] constraint multi-hot
        conditions the policy per problem schema."""
        rt = rt.long()
        B = rt.shape[0]
        rows = torch.arange(B, device=node_emb.device)
        seq, node_h = self._encode_route(node_emb, live, rt, valid, schema=schema)

        # ---- Ruin: sample one customer per row ----
        rlog = self._removal_logits(seq, rt, valid, depot_count, tabu_node=tabu_node)
        rd = torch.distributions.Categorical(logits=rlog)
        pos_r = rlog.argmax(1) if greedy else rd.sample()
        logp = rd.log_prob(pos_r)
        ent = rd.entropy()
        removed = rt[rows, pos_r]  # [B]

        partial, m = self._partial(rt, valid, pos_r)

        # ---- Recreate: sample a feasible gap ----
        gscore = self._gap_logits(node_h, partial, m, removed, adj, gap_feas_fn)
        gd = torch.distributions.Categorical(logits=gscore)
        gap = gscore.argmax(1) if greedy else gd.sample()
        logp = logp + gd.log_prob(gap)
        ent = ent + gd.entropy()

        new_rt, new_valid = self._apply_insert(partial, m, removed, gap)
        return new_rt, new_valid, logp, ent, removed

    def imitation_loss(self, node_emb, live, rt, valid, adj, rm_pos, gap_target,
                       depot_count=1, gap_feas_fn=None, schema=None):
        """Cross-entropy behaviour-cloning of a teacher relocate.

        Teacher forcing: score the removal head, then conditioned on the teacher's
        removed position `rm_pos` [B] score the gap head over the resulting
        partial, and return CE(removal, rm_pos) + CE(gap, gap_target). This warms
        the policy to reproduce best-improvement local search before RL -- the fix
        my own notes flagged (REINFORCE-from-scratch is too sample-inefficient
        here) and the recipe CaR uses (imitation loss alongside RL). schema [K]
        constraint multi-hot conditions the policy per problem schema."""
        rt = rt.long()
        B = rt.shape[0]
        rows = torch.arange(B, device=node_emb.device)
        seq, node_h = self._encode_route(node_emb, live, rt, valid, schema=schema)
        rlog = self._removal_logits(seq, rt, valid, depot_count)
        rm_loss = torch.nn.functional.cross_entropy(rlog, rm_pos)
        removed = rt[rows, rm_pos]
        partial, m = self._partial(rt, valid, rm_pos)
        gscore = self._gap_logits(node_h, partial, m, removed, adj, gap_feas_fn)
        gap_loss = torch.nn.functional.cross_entropy(gscore, gap_target)
        return rm_loss + gap_loss


def load_constraint_field_state_dict(
    model: ConstraintFieldNet, state_dict: dict
) -> bool:
    """Load a typed-resource checkpoint, upgrading compatible neutral heads.

    V1 checkpoints contain the identity ``resource_types`` buffer and fixed
    seven-wide coupler heads. Silently upgrading those weights would undermine
    the descriptor-only claim, so the v1 boundary is intentionally clean. A
    pre-objective-guidance v2 checkpoint is safe to upgrade because the new
    residual head's zero initialization exactly reproduces its policy. A v2
    checkpoint containing learned logits is rejected because those parameters
    are in log-probability units, not energy units.
    """
    if "resource_types" in state_dict:
        raise RuntimeError(
            "incompatible ConstraintFieldNet v1 checkpoint: typed-resource "
            "v5 scale-equivariant-energy model requires retraining"
        )
    upgraded = False
    state_dict = dict(state_dict)
    if any(
        key.startswith("objective_edge_logit_head.") for key in state_dict
    ):
        raise RuntimeError(
            "incompatible ConstraintFieldNet v2 checkpoint: learned edge "
            "logits must be retrained as objective-energy residuals"
        )
    for prefix, head in (
        ("edge_logit_head", model.edge_logit_head),
    ):
        head_keys = {f"{prefix}.weight", f"{prefix}.bias"}
        missing_head = head_keys - state_dict.keys()
        if not missing_head:
            continue
        if missing_head != head_keys:
            raise RuntimeError(
                "incompatible ConstraintFieldNet v2 checkpoint"
            )
        state_dict[f"{prefix}.weight"] = head.weight.detach().clone()
        state_dict[f"{prefix}.bias"] = head.bias.detach().clone()
        upgraded = True
    # The schema-conditioned advantage-scale head is absent from pre-g_phi
    # checkpoints. Inject its neutral initialization (g_phi == 1, an exact no-op)
    # so strict loading succeeds without altering the restored policy.
    model_state = model.state_dict()
    injectable = ("reward_scale_head.", "objective_energy_residual_head.")
    for key in model_state:
        if key.startswith(injectable) and key not in state_dict:
            # The residual head's final layer is zero-initialized, so injecting
            # its model init reproduces the pre-objective neutral policy exactly.
            state_dict[key] = model_state[key].detach().clone()
            upgraded = True
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "incompatible ConstraintFieldNet v2 checkpoint"
        ) from error
    return upgraded
