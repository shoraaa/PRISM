import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import prism_decoder


FIELD_CHANNEL_COUNT = prism_decoder.FIELD_CHANNEL_COUNT
LIVE_STATE_FEATURE_COUNT = prism_decoder.LIVE_STATE_FEATURE_COUNT
NODE_FEATURE_COUNT = prism_decoder.NODE_FEATURE_COUNT
EDGE_FEATURE_COUNT = prism_decoder.EDGE_FEATURE_COUNT


OBJECTIVE_TYPES = ("distance", "prize", "distance_plus_penalty")


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
    ).view(1, FIELD_CHANNEL_COUNT)
    open_route = torch.tensor(
        [[float(bool(decoder.metadata["open_route"]))]],
        dtype=torch.float32,
        device=device,
    )
    objective_type = torch.zeros(1, len(OBJECTIVE_TYPES), device=device)
    objective_type[0, OBJECTIVE_TYPES.index(decoder.metadata["objective"])] = 1.0
    objective_scale = torch.tensor(
        [[float(decoder.metadata["objective_scale"])]],
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
        resource_scales.shape != (FIELD_CHANNEL_COUNT,)
        or not torch.isfinite(resource_scales).all()
        or torch.any(resource_scales <= 0.0)
    ):
        raise ValueError("resource_scales must be finite and strictly positive")
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        active_channels=active_channels,
        open_route=open_route,
        objective_type=objective_type,
        objective_scale=objective_scale,
        multi_route=multi_route,
        depot_scale=depot_scale,
        raw_resource_pressure=raw_resource_pressure,
        objective_edge_costs=objective_edge_costs,
        resource_scales=resource_scales,
        edge_offsets=edge_offsets,
        graph_version=int(decoder.graph_version),
    )


@torch.no_grad()
def decode_iteration(decoder, model, device="cpu", risk_penalty=10.0):
    """Run one model refresh and one decoder iteration on the current graph."""
    graph = build_decoder_data(decoder, device=device)
    output = model(graph)
    scales = graph.resource_scales.unsqueeze(0)
    edge_field = (output["residual"] * scales).detach().cpu().numpy()
    multipliers = output["multipliers"][0].detach().cpu().numpy()
    solution = decoder.solve(
        1,
        edge_field=edge_field,
        edge_additive=(output["additive"] * scales).detach().cpu().numpy(),
        multipliers=multipliers,
        coupler_weights=output["coupler_weights"][0].detach().cpu().numpy(),
        coupler_bias=output["coupler_bias"][0].detach().cpu().numpy(),
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
        
    def forward(self, x, edge_index, edge_attr):
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
        gate_multipliers_by_binding=False,
    ):
        super().__init__()
        # When False (default), the resource multipliers lambda_r are a plain
        # softplus of the multiplier head, so RL gradients reach them directly
        # from step 0. Gating them by the binding classifier makes the policy
        # gradient vanish wherever the (initially untrained) binding head is
        # unsure -- the chicken-and-egg that stalled field learning -- so it is
        # off by default and kept only as an ablation.
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
        # resource-type one-hot + active flag + mean/max pressure + open_route
        # + objective-type one-hot + objective scale + multi_route + depot scale
        descriptor_size = FIELD_CHANNEL_COUNT + 4 + len(OBJECTIVE_TYPES) + 1 + 2
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
        self.token_projection = nn.Linear(units, units)
        self.graph_projection = nn.Linear(units, units)
        self.field_head = nn.Linear(units, 1)
        self.additive_head = nn.Linear(units, 1)
        self.feasibility_head = nn.Linear(units, 1)
        self.multiplier_head = nn.Linear(units, 1)
        self.binding_head = nn.Linear(units, 1)
        self.coupler_head = nn.Linear(units, LIVE_STATE_FEATURE_COUNT)
        self.coupler_bias_head = nn.Linear(units, 1)
        # Objective multiplier slot (w_obj): a learned, state-conditioned weight
        # on the objective edge cost. It lets the objective enter the search
        # energy through a learned coefficient instead of a hard unit term, and
        # is appended as multiplier slot FIELD_CHANNEL_COUNT.
        self.objective_multiplier_head = nn.Linear(units, 1)
        self.objective_coupler_head = nn.Linear(units, LIVE_STATE_FEATURE_COUNT)
        self.objective_coupler_bias_head = nn.Linear(units, 1)
        self.value_head = nn.Linear(units + 1, 1)
        nn.init.zeros_(self.field_head.weight)
        nn.init.zeros_(self.field_head.bias)
        nn.init.zeros_(self.additive_head.weight)
        nn.init.zeros_(self.additive_head.bias)
        nn.init.zeros_(self.feasibility_head.weight)
        nn.init.constant_(self.feasibility_head.bias, -4.0)
        nn.init.zeros_(self.coupler_head.weight)
        nn.init.zeros_(self.coupler_head.bias)
        nn.init.zeros_(self.coupler_bias_head.weight)
        nn.init.zeros_(self.coupler_bias_head.bias)
        # w_obj initializes to 1.0: softplus(unit_softplus_shift) * 2*sigmoid(0),
        # so the objective term reproduces the plain objective edge cost before
        # search reward moves it.
        nn.init.zeros_(self.objective_multiplier_head.weight)
        nn.init.zeros_(self.objective_multiplier_head.bias)
        nn.init.zeros_(self.objective_coupler_head.weight)
        nn.init.zeros_(self.objective_coupler_head.bias)
        nn.init.zeros_(self.objective_coupler_bias_head.weight)
        nn.init.zeros_(self.objective_coupler_bias_head.bias)
        # A zero value is the neutral bootstrap for old checkpoints and makes
        # enabling temporal credit leave the field policy unchanged initially.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.register_buffer(
            "resource_types", torch.eye(FIELD_CHANNEL_COUNT)
        )
        self.register_buffer(
            "unit_softplus_shift",
            torch.log(torch.expm1(torch.ones(()))),
        )
        # The additive correction supplies learned pressure when the analytic
        # pressure is zero. The C++ search energy uses
        # intensity * max(pressure*residual + additive, 0); parameterize the
        # correction through softplus and shift it to initialize near zero with
        # a healthy gradient, alongside the unit-initialized multiplicative
        # correction.
        self.register_buffer(
            "additive_softplus_shift",
            torch.log(torch.expm1(torch.full((), 0.01))),
        )

    def _field_channel(
        self, edge_projection, token_projection, edge_batch
    ):
        token = (
            token_projection
            if token_projection.ndim == 1
            else token_projection[edge_batch]
        )
        interaction = torch.tanh(
            edge_projection + token
        )
        return (
            self.field_head(interaction).squeeze(-1),
            self.additive_head(interaction).squeeze(-1),
        )

    def forward(self, pyg):
        _require_unit_interval("node_features", pyg.x)
        _require_unit_interval("edge_features", pyg.edge_attr)
        active = pyg.active_channels
        if active.ndim == 1:
            active = active.unsqueeze(0)
        _require_unit_interval("active_channels", active)
        if active.shape[-1] != FIELD_CHANNEL_COUNT:
            raise ValueError(
                f"active_channels must end in {FIELD_CHANNEL_COUNT} channels"
            )

        edge_embedding = self.emb_net(
            pyg.x, pyg.edge_index, pyg.edge_attr
        )
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
        objective_type = _per_graph_descriptor(
            pyg,
            "objective_type",
            len(OBJECTIVE_TYPES),
            batch_size,
            active,
            default=[1.0] + [0.0] * (len(OBJECTIVE_TYPES) - 1),
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

        normalized_resources = pyg.edge_attr[:, 1 : 1 + FIELD_CHANNEL_COUNT]
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
        resource_type = self.resource_types.unsqueeze(0).expand(
            graph_embedding.shape[0], -1, -1
        )
        def _broadcast(column):
            return column.unsqueeze(1).expand(-1, FIELD_CHANNEL_COUNT, -1)

        descriptor = torch.cat(
            (
                resource_type,
                active.unsqueeze(-1),
                resource_mean.unsqueeze(-1),
                resource_max.unsqueeze(-1),
                _broadcast(open_route),
                _broadcast(objective_type),
                _broadcast(objective_scale),
                _broadcast(multi_route),
                _broadcast(depot_scale),
            ),
            dim=-1,
        )
        _require_unit_interval("resource_descriptors", descriptor)
        tokens = self.resource_encoder(descriptor)
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
        projected_tokens = self.token_projection(tokens)
        raw_channels = []
        additive_channels = []
        for channel in range(FIELD_CHANNEL_COUNT):
            token = (
                projected_tokens[:, channel]
                if batched
                else projected_tokens[0, channel]
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
                    edge_batch,
                    use_reentrant=False,
                )
            else:
                raw, additive_channel = self._field_channel(
                    projected_edges, token, edge_batch
                )
            raw_channels.append(raw)
            additive_channels.append(additive_channel)
        raw_residual = torch.stack(raw_channels, dim=1)
        # The residual is now the direct per-edge resource field (the analytic
        # pressure gate has been removed), so it initializes near zero like the
        # additive term and learns its magnitude from search reward instead of
        # scaling a hand-supplied pressure.
        residual = F.softplus(raw_residual + self.additive_softplus_shift)
        additive = F.softplus(
            torch.stack(additive_channels, dim=1) + self.additive_softplus_shift
        )
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
        coupler_weights = self.coupler_head(state) * active.unsqueeze(-1)
        coupler_bias = self.coupler_bias_head(state).squeeze(-1) * active
        # Append the always-on objective multiplier slot (w_obj) so the guidance
        # arrays carry MULTIPLIER_COUNT = FIELD_CHANNEL_COUNT + 1 entries. The
        # objective weight is graph-level and is not gated by active channels.
        objective_multiplier = F.softplus(
            self.objective_multiplier_head(graph_state).squeeze(-1)
            + self.unit_softplus_shift
        )
        objective_coupler_weights = self.objective_coupler_head(graph_state)
        objective_coupler_bias = self.objective_coupler_bias_head(
            graph_state
        ).squeeze(-1)
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
            "residual": residual,
            "additive": additive,
            "feasibility_logits": feasibility_logits,
            "feasibility_risk": feasibility_risk,
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
        if live_state.shape[-1] != LIVE_STATE_FEATURE_COUNT:
            raise ValueError(
                f"live_state must end in {LIVE_STATE_FEATURE_COUNT} features"
            )
        base = output["multipliers"]
        weights = output["coupler_weights"]
        bias = output["coupler_bias"]
        if weights.ndim != 3:
            raise ValueError("coupler_weights must have shape [B, C, S]")
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
) -> bool:
    """Load current or pre-value-head checkpoints without hiding other drift.

    Returns ``True`` when a legacy checkpoint was upgraded by retaining the
    value head's zero initialization.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "value_head.weight",
        "value_head.bias",
        "objective_multiplier_head.weight",
        "objective_multiplier_head.bias",
        "objective_coupler_head.weight",
        "objective_coupler_head.bias",
        "objective_coupler_bias_head.weight",
        "objective_coupler_bias_head.bias",
    }
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected or missing - allowed_missing:
        raise RuntimeError(
            "incompatible ConstraintFieldNet checkpoint: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return bool(missing)
