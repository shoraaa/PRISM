#!/usr/bin/env python
"""Validate PRISM against DeepACO on identical instances, both directions.

Bin packing, multidimensional knapsack and sequential ordering are the classes
PRISM declares outside routing (see ``problem_data.NON_ROUTING_VARIANTS``).
TSP, CVRP and PCTSP are routing classes both methods publish a specialist
model for. DeepACO is the natural reference either way: it is a learned-
heatmap method that publishes a specialist model per class, so it fixes both
the instance distribution and a comparable number.

Instances come from DeepACO's own generators rather than a reimplementation of
them, so distributional parity holds by construction. The same tensors are then
handed to PRISM's decoder and to DeepACO's ACO, and both are scored on the
task's true objective -- bins used for BPP, collected prize for MKP, tour/route
cost for TSP/CVRP/PCTSP -- rather than on either method's internal training
surrogate. DeepACO's BPP fitness is a Falkenauer fill-quality score, which is
not the bin count, so its solutions are re-scored here.

Every instance is also run through all four decoder/guidance combinations:
PRISM's decoder guided by its own field or by DeepACO's heuristic, and
DeepACO's ACO guided by its own heuristic or by PRISM's static edge energy.

DeepACO's BPP module imports numba only for two fitness helpers. When numba is
absent, a pass-through shim is installed so the rest of the module loads
unchanged; the helpers are small enough that pure Python is not the bottleneck.

Usage:
    python scripts/deepaco_compare.py --variant bpp --size 120 --instances 10 \\
        --prism-checkpoint pretrained/v15/best.pt
    python scripts/deepaco_compare.py --variant cvrp --size 100 --instances 10 \\
        --prism-checkpoint pretrained/v15/best.pt
"""

from __future__ import annotations

import argparse
import inspect
import statistics
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEEPACO = ROOT / "baselines" / "DeepACO"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import prism_decoder  # noqa: E402
import problem_data  # noqa: E402

BPP_CAPACITY = problem_data.BPP_CAPACITY
# Matches DeepACO's own CVRP capacity constant exactly, so PRISM's generated
# demand (normalized by this) converts losslessly to DeepACO's raw quantities.
CVRP_CAPACITY = 50
TSP_K_SPARSE = 10

ROUTING_VARIANTS = ("tsp", "cvrp", "pctsp")
DEFAULT_SIZE = {"bpp": 120, "mkp": 50, "sop": 50, "tsp": 100, "cvrp": 100, "pctsp": 100}

# Loaded before a DeepACO task temporarily places its own flat ``net.py`` on
# sys.path.  Keeping the callable avoids resolving PRISM's ``train`` module
# while that task-local module shadows PRISM's ``net`` module.
_PRISM_INFER_INSTANCE = None
_PRISM_FIELD_GUIDANCE = None

# Item index i of a DeepACO MKP instance is PRISM node i+1: PRISM prepends a
# depot node representing "stop selecting". BPP, SOP, TSP, CVRP and PCTSP all
# start at a real depot/root node 0 in either representation (TSP has no depot
# but both sides index the same coordinate list from 0), so no shift applies.
_NODE_OFFSET = {"bpp": 0, "mkp": 1, "sop": 0, "tsp": 0, "cvrp": 0, "pctsp": 0}


def install_numba_shim() -> bool:
    """Make ``import numba`` succeed with a pass-through ``njit``."""
    try:
        import numba  # noqa: F401

        return False
    except ModuleNotFoundError:
        shim = types.ModuleType("numba")

        def njit(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda function: function

        shim.njit = njit
        shim.jit = njit
        sys.modules["numba"] = shim
        return True


@contextmanager
def deepaco_module(name: str):
    """Import one DeepACO task package, whose modules use flat imports."""
    directory = DEEPACO / name
    if not directory.is_dir():
        raise SystemExit(f"DeepACO task directory not found: {directory}")
    sys.path.insert(0, str(directory))
    # The task directories all define `aco`/`utils`/`net`, so a previously
    # imported task would otherwise be returned for the next one.
    stale = {key: sys.modules.pop(key) for key in ("aco", "utils", "net") if key in sys.modules}
    try:
        yield
    finally:
        for key in ("aco", "utils", "net"):
            sys.modules.pop(key, None)
        sys.modules.update(stale)
        sys.path.remove(str(directory))


# --------------------------------------------------------------------------
# scoring, shared by both methods so neither is scored on its own surrogate
# --------------------------------------------------------------------------


def bpp_bins(sequence: list[int], demand: np.ndarray) -> tuple[int, float]:
    """Count bins in a depot-delimited item sequence and return the fullest."""
    bins: list[float] = []
    current = 0.0
    for node in sequence:
        if node == 0:
            if current > 0.0:
                bins.append(current)
            current = 0.0
        else:
            current += float(demand[node])
    if current > 0.0:
        bins.append(current)
    return len(bins), (max(bins) if bins else 0.0)


def bpp_check(sequence: list[int], demand: np.ndarray, item_count: int) -> str:
    served = sorted(node for node in sequence if node != 0)
    if served != list(range(1, item_count + 1)):
        return "items missing or repeated"
    _, fullest = bpp_bins(sequence, demand)
    if fullest > 1.0 + 1e-6:
        return f"bin over capacity ({fullest:.6f})"
    return ""


def mkp_check(selected: list[int], weight: np.ndarray, budget: float) -> str:
    if len(selected) != len(set(selected)):
        return "item selected twice"
    if not selected:
        return ""
    used = weight[selected].sum(axis=0)
    if (used > budget + 1e-4).any():
        return f"budget exceeded ({used.max():.6f} > {budget:.6f})"
    return ""


def sop_check(sequence: list[int], predecessors: list[list[int]]) -> str:
    if sorted(sequence) != list(range(len(predecessors))):
        return "nodes missing or repeated"
    if sequence and sequence[0] != 0:
        return "path does not start at node 0"
    position = {node: index for index, node in enumerate(sequence)}
    for node, required in enumerate(predecessors):
        for before in required:
            if position[before] > position[node]:
                return f"precedence violated: {before} after {node}"
    return ""


def sop_cost(sequence: list[int], distance: np.ndarray) -> float:
    """Open Hamiltonian path cost, matching DeepACO's own scoring."""
    return float(
        sum(distance[sequence[i]][sequence[i + 1]] for i in range(len(sequence) - 1))
    )


def tsp_check(sequence: list[int], node_count: int) -> str:
    if sorted(sequence) != list(range(node_count)):
        return "nodes missing or repeated"
    return ""


def pctsp_check(sequence: list[int], prize: np.ndarray, min_prize: float) -> str:
    # A finished ant's padded solution repeats the depot after it returns;
    # only the prefix up to that return is the actual route.
    if sequence and sequence[0] == 0:
        tail = sequence.index(0, 1) if 0 in sequence[1:] else len(sequence)
        sequence = sequence[:tail]
    customers = [node for node in sequence if node != 0]
    if len(customers) != len(set(customers)):
        return "item selected twice"
    collected = float(prize[customers].sum()) if customers else 0.0
    if collected + 1e-6 < min_prize:
        return f"prize quota unmet ({collected:.6f} < {min_prize:.6f})"
    return ""


def sop_dense_from_predecessors(
    padded: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert PRISM's padded predecessor-list format into DeepACO's own.

    ``padded[node]`` lists what ``node`` must wait for (-1 padded), including
    the structural "node 0 precedes everything" entries PRISM's own generator
    adds -- mirroring ``baselines/DeepACO/sop/utils.py:preceding_mat_gen`` /
    ``adjacency_mat_gen``, which both fold from the identical (predecessor,
    node) pairs.
    """
    prec_mat = torch.zeros(size, size)
    adjacency = torch.ones(size, size)
    adjacency.fill_diagonal_(0)
    for node in range(size):
        for value in padded[node].tolist():
            predecessor = int(value)
            if predecessor < 0:
                continue
            prec_mat[node, predecessor] = 1
            adjacency[node, predecessor] = 0
    return prec_mat, adjacency


# --------------------------------------------------------------------------
# PRISM
# --------------------------------------------------------------------------


def load_prism_model(path: str, device: str):
    """Rebuild a PRISM checkpoint with the architecture flags it trained under.

    Same reconstruction test.py performs: the ablation flags are forward-time
    only, so the state dict loads either way and a missing flag would silently
    evaluate the model on inputs it was never trained to read.
    """
    import torch

    global _PRISM_INFER_INSTANCE, _PRISM_FIELD_GUIDANCE

    from net import ConstraintFieldNet, load_constraint_field_state_dict
    from train import infer_instance, _field_guidance

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config") or {}
    stored_kwargs = checkpoint.get("model_kwargs") or {}
    sources = (checkpoint, stored_kwargs, config)
    aliases = {"pool_node_resources": ("pool_node_resources", "resource_pooling")}
    kwargs = {}
    for name in inspect.signature(ConstraintFieldNet.__init__).parameters:
        if name == "self":
            continue
        keys = aliases.get(name, (name,))
        for source in sources:
            match = next((key for key in keys if key in source), None)
            if match is not None:
                kwargs[name] = source[match]
                break
    model = ConstraintFieldNet(**kwargs).to(device)
    load_constraint_field_state_dict(
        model,
        checkpoint["model_state_dict"],
        model_schema=checkpoint.get("model_schema"),
        config=config,
    )
    model.eval()
    _PRISM_INFER_INSTANCE = infer_instance
    _PRISM_FIELD_GUIDANCE = _field_guidance
    return model, config


def _inference_namespace(problem: dict, iterations: int, seed: int, device: str):
    """The decoder configuration a guided run decodes under.

    ``candidates`` is the complete graph for these classes rather than the
    trained K=64: with every pairwise distance equal, a truncated neighbourhood
    keeps an arbitrary index-ordered subset and makes the rest unreachable.
    That is a genuine input shift away from training and is the first thing to
    suspect if guidance underperforms here.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        candidates=problem_data.candidate_limit(problem, 64),
        n_rollouts=1,
        beta=2.0,
        seed=seed,
        search_iterations=iterations,
        feasibility_lookahead_depth=2,
        device=device,
        static_field=False,
        min_changed_edges=8,
        random_escape=False,
        srr_exploration_budget=0,
    )


def prism_energy_matrix(model, variant: str, data: dict, device: str) -> torch.Tensor:
    """The PRISM field's static (zero live-state) per-edge energy, densified.

    Construction's first decision sees every resource channel at its reset
    value, so the coupler's live-state term drops out and each slot's weight
    collapses to a constant: ``multiplier * 2*sigmoid(coupler_bias)``. That is
    exactly the score PRISM's own construction uses to rank the first edge out
    of the depot, so reusing it as a dense heatmap is not an approximation of
    a different quantity -- it is that quantity, exposed for every pair
    instead of just the reachable ones.
    """
    if _PRISM_FIELD_GUIDANCE is None:
        raise RuntimeError("PRISM inference was not initialized with the model")
    problem = problem_data.decoder_problem(variant, data)
    decoder = prism_decoder.Decoder(
        problem,
        candidate_config={"max_candidates": problem_data.candidate_limit(problem, 64)},
    )
    guidance = _PRISM_FIELD_GUIDANCE(model, decoder, device)
    edge_field = np.asarray(guidance["edge_field"], dtype=np.float64)
    multipliers = np.asarray(guidance["multipliers"], dtype=np.float64)
    coupler_bias = np.asarray(guidance["coupler_bias"], dtype=np.float64)
    # The coupler is per edge, so its zero-live-state collapse is a per-edge
    # weight row rather than one row for the whole graph.
    weights = multipliers[None, :] * (2.0 / (1.0 + np.exp(-coupler_bias)))
    objective_cost = np.asarray(decoder.objective_edge_costs, dtype=np.float64)
    scale = float(decoder.objective_energy_scale)
    energy = weights[:, -1] * (objective_cost / scale)
    if edge_field.shape[1]:
        energy += (edge_field * weights[:, :-1]).sum(axis=1)

    edge_index = np.asarray(decoder.edge_index, dtype=np.int64)
    node_count = int(edge_index.max()) + 1 if edge_index.size else 1
    dense = np.full((node_count, node_count), np.nan, dtype=np.float64)
    dense[edge_index[0], edge_index[1]] = energy
    valid = np.isfinite(dense)
    if not valid.any():
        raise RuntimeError("PRISM candidate graph produced no edges")
    dense = np.where(valid, dense, dense[valid].max() + 1.0)

    # exp(-dense) turns "lower energy is better" into a positive DeepACO-style
    # heuristic without an arbitrary offset; z-scoring keeps the exponent in
    # a range that neither overflows nor collapses to a single value.
    heuristic = np.exp(-dense)

    offset = _NODE_OFFSET[variant]
    heuristic = heuristic[offset:, offset:]
    return torch.as_tensor(heuristic, dtype=torch.float32)


def deepaco_guidance_for_prism(decoder, heuristic: torch.Tensor, variant: str) -> dict:
    """Turn a DeepACO heuristic matrix into PRISM ``objective_residual`` guidance.

    Mirrors ``train._distance_guidance``: every resource channel stays neutral
    and the complete candidate energy is overridden to track the (negated,
    normalized) external heuristic, so higher DeepACO desirability becomes
    lower PRISM energy.
    """
    resources = int(decoder.metadata["resource_count"])
    multiplier_slots = int(decoder.metadata["multiplier_count"])
    edge_count = int(decoder.metadata["edge_count"])
    guidance = {
        "edge_field": np.zeros((edge_count, resources), dtype=np.float32),
        "multipliers": np.zeros(multiplier_slots, dtype=np.float32),
        "coupler_weights": np.zeros(
            (edge_count, multiplier_slots, resources), dtype=np.float32
        ),
        "coupler_bias": np.zeros(
            (edge_count, multiplier_slots), dtype=np.float32
        ),
    }
    guidance["multipliers"][resources] = 1.0

    offset = _NODE_OFFSET[variant]
    node_count = int(np.asarray(decoder.edge_index).max()) + 1
    padded = np.zeros((node_count, node_count), dtype=np.float64)
    heuristic_np = heuristic.detach().cpu().numpy().astype(np.float64)
    size = heuristic_np.shape[0]
    padded[offset : offset + size, offset : offset + size] = heuristic_np

    edge_index = np.asarray(decoder.edge_index, dtype=np.int64)
    edge_cost = -padded[edge_index[0], edge_index[1]]
    scale = max(float(edge_cost.std()), 1.0e-9)
    normalized = (edge_cost - edge_cost.mean()) / scale

    objective_scale = float(decoder.metadata["objective_energy_scale"])
    objective_energy = (
        np.asarray(decoder.objective_edge_costs, dtype=np.float64) / objective_scale
    )
    guidance["objective_residual"] = (normalized - objective_energy).astype(np.float32)
    return guidance


def prism_solve(
    variant: str,
    data: dict,
    iterations: int,
    seed: int,
    model=None,
    device: str = "cpu",
    guidance_fn=None,
    baseline: str | None = None,
) -> dict:
    problem = problem_data.decoder_problem(variant, data)
    if guidance_fn is not None:
        decoder = prism_decoder.Decoder(
            problem,
            candidate_config={
                "max_candidates": problem_data.candidate_limit(problem, 64)
            },
        )
        decoder.seed(seed)
        solution = decoder.solve(iterations, **guidance_fn(decoder))
    elif model is None and baseline is None:
        decoder = prism_decoder.Decoder(
            problem,
            candidate_config={
                "max_candidates": problem_data.candidate_limit(problem, 64)
            },
        )
        decoder.seed(seed)
        solution = decoder.solve(iterations)
    else:
        import torch

        if _PRISM_INFER_INSTANCE is None:
            raise RuntimeError("PRISM inference was not initialized with the model")

        with torch.no_grad():
            _, solution, _ = _PRISM_INFER_INSTANCE(
                model,
                problem,
                _inference_namespace(problem, iterations, seed, device),
                **({"baseline": baseline} if baseline is not None else {}),
            )
    route = [int(node) for node in solution["route"]]
    if variant == "bpp":
        demand = problem["demand"]
        bins, fullest = bpp_bins(route, demand)
        return {
            "score": float(bins),
            "error": solution.get("error", "")
            or bpp_check(route, demand, len(demand) - 1),
            "detail": f"fullest bin {fullest:.4f}",
        }
    if variant == "sop":
        distance = problem["distance"]
        predecessors = [
            [int(value) for value in row if int(value) > 0]
            for row in np.asarray(data["predecessors"]).reshape(
                distance.shape[0], -1
            )
        ]
        return {
            "score": sop_cost(route, distance),
            "error": solution.get("error", "") or sop_check(route, predecessors),
            "detail": f"{len(route)} nodes",
        }
    if variant == "mkp":
        selected = [node for node in route if node != 0]
        weight = np.stack(
            [
                problem["node_attributes"][key]
                for key in sorted(problem["node_attributes"])
            ],
            axis=1,
        )
        budget = float(np.asarray(data["budget"]).reshape(-1)[0])
        return {
            "score": float(problem["prize"][selected].sum()),
            "error": solution.get("error", "") or mkp_check(selected, weight, budget),
            "detail": f"{len(selected)} items",
        }
    # tsp / cvrp / pctsp: PRISM's own decoder already validates visit_all,
    # capacity, and prize_quota, so its objective is the metric being
    # compared -- no separate reconstruction is needed.
    return {
        "score": float(solution["objective"]),
        "error": solution.get("error", ""),
        "detail": f"{len(route)} nodes",
    }


# --------------------------------------------------------------------------
# DeepACO
# --------------------------------------------------------------------------


def deepaco_raw_heuristic(variant: str, model, **tensors) -> torch.Tensor | None:
    """The dense (size, size) heatmap DeepACO's own model would feed its ACO.

    Factored out so the same computation can either drive DeepACO's ACO (the
    normal reference run) or be handed to PRISM's decoder as external
    guidance, without duplicating each task's ``gen_pyg_data`` call.
    """
    if model is None:
        return None
    from utils import gen_pyg_data  # type: ignore

    if variant == "bpp":
        demands = tensors["demands"]
        size = demands.size(0)
        heuristic = model(gen_pyg_data(demands, "cpu"))
        return heuristic.reshape((size, size))
    if variant == "mkp":
        prize, weight = tensors["prize"], tensors["weight"]
        size = prize.size(0)
        heuristic = model(gen_pyg_data(prize, weight))
        return heuristic.reshape((size, size))
    if variant == "sop":
        distance, adjacency = tensors["distance"], tensors["adjacency"]
        size = distance.size(0)
        heuristic = model(gen_pyg_data(distance, adjacency, "cpu"))
        dense = torch.zeros((size, size))
        dense[adjacency.bool()] = heuristic.reshape(-1)
        return dense
    if variant == "tsp":
        coordinates, k_sparse = tensors["coordinates"], tensors["k_sparse"]
        pyg_data, _ = gen_pyg_data(coordinates, k_sparse)
        heuristic = model(pyg_data)
        # DeepACO's own Net.reshape: zero-pads the sparse (k_sparse) heuristic
        # vector back out to a dense (size, size) matrix.
        return type(model).reshape(pyg_data, heuristic)
    if variant == "cvrp":
        demands, distances = tensors["demands"], tensors["distances"]
        size = demands.size(0)
        heuristic = model(gen_pyg_data(demands, distances, "cpu"))
        return heuristic.reshape((size, size))
    if variant == "pctsp":
        distances, prizes, penalties = (
            tensors["distances"],
            tensors["prizes"],
            tensors["penalties"],
        )
        size = prizes.size(0)
        heuristic = model(gen_pyg_data(prizes, penalties, distances))
        return heuristic.reshape((size, size))
    raise ValueError(f"unknown variant: {variant}")


def deepaco_bpp(
    demands: torch.Tensor, ants: int, rounds: int, model, heuristic=None
) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if heuristic is None:
        heuristic = deepaco_raw_heuristic("bpp", model, demands=demands)
    if heuristic is not None:
        size = demands.size(0)
        kwargs["heuristic"] = heuristic.reshape((size, size)) + 1e-10
    aco = ACO(demand=demands, n_ants=ants, device="cpu", **kwargs)
    aco.run(rounds)
    if aco.shortest_path is None:
        return {"score": float("nan"), "error": "no solution", "detail": ""}
    sequence = [int(node) for node in aco.shortest_path]
    demand = (demands / BPP_CAPACITY).numpy()
    bins, fullest = bpp_bins(sequence, demand)
    return {
        "score": float(bins),
        "error": bpp_check(sequence, demand, demands.size(0) - 1),
        "detail": f"fullest bin {fullest:.4f}",
    }


def deepaco_mkp(
    prize: torch.Tensor,
    weight: torch.Tensor,
    ants: int,
    rounds: int,
    model,
    heuristic=None,
) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if heuristic is None:
        heuristic = deepaco_raw_heuristic("mkp", model, prize=prize, weight=weight)
    if heuristic is not None:
        size = prize.size(0)
        kwargs["heuristic"] = heuristic.reshape((size, size)) + 1e-10
    aco = ACO(prize=prize, weight=weight, n_ants=ants, device="cpu", **kwargs)
    best, solution = aco.run(rounds)
    # The ACO appends a dummy node at index n to mean "stop"; it is not an item.
    selected = [int(node) for node in solution if int(node) != prize.size(0)]
    budget = float(prize.size(0) // 2)
    return {
        "score": float(best),
        "error": mkp_check(selected, weight.numpy(), budget),
        "detail": f"{len(selected)} items",
    }


def deepaco_sop(
    distance: torch.Tensor,
    adjacency: torch.Tensor,
    preceding: torch.Tensor,
    ants: int,
    rounds: int,
    model,
    heuristic=None,
) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if heuristic is None:
        heuristic = deepaco_raw_heuristic(
            "sop", model, distance=distance, adjacency=adjacency
        )
    if heuristic is not None:
        kwargs["heuristic"] = heuristic + 1e-10
    aco = ACO(
        distances=distance, prec_cons=preceding, n_ants=ants, device="cpu", **kwargs
    )
    aco.run(rounds)
    if aco.shortest_path is None:
        return {"score": float("nan"), "error": "no solution", "detail": ""}
    sequence = [int(node) for node in aco.shortest_path]
    predecessors = [
        [int(before) for before in row.nonzero().flatten().tolist() if before != 0]
        for row in preceding
    ]
    return {
        "score": sop_cost(sequence, distance.numpy()),
        "error": sop_check(sequence, predecessors),
        "detail": f"{len(sequence)} nodes",
    }


def deepaco_tsp(
    coordinates: torch.Tensor, ants: int, rounds: int, model, k_sparse: int, heuristic=None
) -> dict:
    from aco import ACO  # type: ignore
    from utils import gen_distance_matrix  # type: ignore

    distances = gen_distance_matrix(coordinates)
    kwargs = {}
    if heuristic is None:
        heuristic = deepaco_raw_heuristic(
            "tsp", model, coordinates=coordinates, k_sparse=k_sparse
        )
    if heuristic is not None:
        kwargs["heuristic"] = heuristic + 1e-10
    aco = ACO(distances=distances, n_ants=ants, device="cpu", **kwargs)
    aco.run(rounds)
    if aco.shortest_path is None:
        return {"score": float("nan"), "error": "no solution", "detail": ""}
    route = [int(node) for node in aco.shortest_path]
    return {
        "score": float(aco.lowest_cost),
        "error": tsp_check(route, coordinates.size(0)),
        "detail": f"{len(route)} nodes",
    }


def deepaco_cvrp(
    demands: torch.Tensor,
    distances: torch.Tensor,
    ants: int,
    rounds: int,
    model,
    heuristic=None,
) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if heuristic is None:
        heuristic = deepaco_raw_heuristic(
            "cvrp", model, demands=demands, distances=distances
        )
    if heuristic is not None:
        size = demands.size(0)
        kwargs["heuristic"] = heuristic.reshape((size, size)) + 1e-10
    aco = ACO(distances=distances, demand=demands, n_ants=ants, device="cpu", **kwargs)
    aco.run(rounds)
    if aco.shortest_path is None:
        return {"score": float("nan"), "error": "no solution", "detail": ""}
    route = [int(node) for node in aco.shortest_path]
    demand_normalized = (demands / CVRP_CAPACITY).numpy()
    return {
        "score": float(aco.lowest_cost),
        "error": bpp_check(route, demand_normalized, demands.size(0) - 1),
        "detail": f"{len(route)} nodes",
    }


def deepaco_pctsp(
    distances: torch.Tensor,
    prizes: torch.Tensor,
    penalties: torch.Tensor,
    ants: int,
    rounds: int,
    model,
    heuristic=None,
) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if heuristic is None:
        heuristic = deepaco_raw_heuristic(
            "pctsp", model, distances=distances, prizes=prizes, penalties=penalties
        )
    if heuristic is not None:
        size = prizes.size(0)
        kwargs["heuristic"] = heuristic.reshape((size, size)) + 1e-10
    aco = ACO(
        distances=distances, prizes=prizes, penalties=penalties, n_ants=ants,
        device="cpu", **kwargs,
    )
    best, solution = aco.run(rounds)
    route = [int(node) for node in solution]
    return {
        "score": float(best),
        "error": pctsp_check(route, prizes.numpy(), float(aco.min_prizes)),
        "detail": f"{len(route)} nodes",
    }


def load_model(variant: str, size: int, checkpoint: str | None = None):
    path = Path(checkpoint) if checkpoint else DEEPACO / "pretrained" / variant / f"{variant}{size}.pt"
    if not path.exists():
        return None, f"no pretrained {variant} model at {path}"
    from net import Net  # type: ignore

    model = Net()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, ""


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("bpp", "mkp", "sop", "tsp", "cvrp", "pctsp"),
        required=True,
    )
    parser.add_argument("--size", type=int, default=None, help="items per instance")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument(
        "--prism-iterations",
        type=int,
        default=64,
        help="PRISM search iterations per instance",
    )
    parser.add_argument("--aco-rounds", type=int, default=10)
    parser.add_argument("--aco-ants", type=int, default=20)
    parser.add_argument(
        "--dimensions",
        type=int,
        default=problem_data.MKP_DIMENSIONS,
        help="mkp only: number of knapsack constraints",
    )
    parser.add_argument(
        "--k-sparse",
        type=int,
        default=TSP_K_SPARSE,
        help="tsp only: nearest-neighbor sparsification width for DeepACO's heuristic",
    )
    parser.add_argument(
        "--prism-checkpoint",
        required=True,
        help="PRISM checkpoint whose learned field guides construction and "
        "search; required so every one of the four decoder/guidance "
        "combinations below has a PRISM field to draw on",
    )
    parser.add_argument(
        "--deepaco-checkpoint",
        default=None,
        help="path to the pretrained DeepACO model; defaults to "
        "baselines/DeepACO/pretrained/<variant>/<variant><size>.pt, but "
        "some form of it is required so every combination has a DeepACO "
        "heuristic to draw on",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    size = args.size or DEFAULT_SIZE[args.variant]

    prism_model, prism_config = load_prism_model(args.prism_checkpoint, args.device)
    flags = [
        key
        for key in (
            "index_embedded_resources",
            "monolithic_resource_field",
            "program_blind_resources",
        )
        if getattr(prism_model, key)
    ]
    print(
        f"PRISM field: {args.prism_checkpoint}"
        + (f" [{', '.join(flags)}]" if flags else " [semantic descriptor]")
    )

    shimmed = install_numba_shim()
    if shimmed:
        print("numba is absent; using a pass-through njit shim")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    # Six combinations of {decoder} x {guidance field}: which construction
    # algorithm runs (PRISM's native decoder or DeepACO's ACO) is independent
    # of which model's learned edge valuation it is guided by. "heuristic"
    # is each side's own model-free baseline: PRISM's normalized-distance
    # guidance and DeepACO's hand-designed 1/distance heuristic.
    combos = (
        "prism/prism",
        "prism/deepaco",
        "prism/heuristic",
        "deepaco/prism",
        "deepaco/deepaco",
        "deepaco/heuristic",
    )
    rows: dict[str, list[dict]] = {combo: [] for combo in combos}
    with deepaco_module(args.variant):
        model, why = load_model(args.variant, size, args.deepaco_checkpoint)
        if model is None:
            raise SystemExit(f"--deepaco-checkpoint required: {why}")
        print(f"DeepACO model: {args.deepaco_checkpoint or f'{args.variant}{size}.pt (default)'}")

        # Instances come from PRISM's own generators (problem_data.py), not
        # DeepACO's: bpp/mkp/sop have dedicated PRISM-owned reimplementations
        # of DeepACO's distributions, and tsp/cvrp/pctsp use PRISM's generic
        # ``generated_problem``, which is also what training draws from.
        # DeepACO's own tensors are then derived from that single generated
        # instance so both methods solve literally the same problem.
        batch = None
        if args.variant == "bpp":
            batch = problem_data.generate_bpp_data(size, args.instances, seed=args.seed)
        elif args.variant == "mkp":
            batch = problem_data.generate_mkp_data(
                size, args.instances, seed=args.seed, dimensions=args.dimensions
            )
        elif args.variant == "sop":
            batch = problem_data.generate_sop_data(size, args.instances, seed=args.seed)
        else:
            # generated_problem draws one instance at a time from the global
            # torch RNG rather than a seeded batch, so the seed is set once
            # up front (already done above) and instances are drawn in order.
            pass

        for index in range(args.instances):
            if args.variant == "sop":
                distance = batch["distance"][index]
                padded = batch["predecessors"][index]
                data = {
                    "distance": distance.unsqueeze(0),
                    "predecessors": padded.unsqueeze(0),
                }
                prec_mat, adjacency = sop_dense_from_predecessors(padded, size)
                # PRISM's generator zeros the distance diagonal (required by
                # its own decoder); DeepACO's hand-designed 1/distance
                # heuristic would divide by zero there, so DeepACO sees a
                # large-diagonal copy. No valid tour cost differs between the
                # two since no route ever takes a self-loop.
                deepaco_distance = distance.clone()
                deepaco_distance.fill_diagonal_(1e9)
                preceding = prec_mat
                deepaco_tensors = {
                    "distance": deepaco_distance,
                    "adjacency": adjacency,
                }
            elif args.variant == "bpp":
                demand = batch["demand"][index]
                item_size = batch["item_size"][index]
                data = {"demand": demand.unsqueeze(0)}
                deepaco_tensors = {"demands": item_size}
            elif args.variant == "mkp":
                prize = batch["prize"][index]
                weight = batch["weight"][index]
                budget = batch["budget"][index]
                data = {
                    "prize": prize.unsqueeze(0),
                    "weight": weight.unsqueeze(0),
                    "budget": budget.unsqueeze(0),
                }
                # DeepACO's own mkp instances carry no depot row; PRISM's
                # generator prepends one (index 0), so DeepACO sees it dropped.
                deepaco_tensors = {"prize": prize[1:], "weight": weight[1:]}
            elif args.variant == "tsp":
                problem = problem_data.generated_problem("tsp", size)
                coordinates = torch.as_tensor(
                    problem["coordinates"], dtype=torch.float32
                )
                data = {"xy": coordinates.unsqueeze(0)}
                deepaco_tensors = {"coordinates": coordinates, "k_sparse": args.k_sparse}
            elif args.variant == "cvrp":
                problem = problem_data.generated_problem("cvrp", size)
                demand = torch.as_tensor(problem["demand"], dtype=torch.float32)
                distances = torch.as_tensor(problem["distance"], dtype=torch.float32)
                data = {
                    "demand": demand.unsqueeze(0),
                    "distance": distances.unsqueeze(0),
                }
                # PRISM's generator zeros the distance diagonal; DeepACO's own
                # hand-designed 1/distance heuristic would divide by zero
                # there (its own gen_distance_matrix sets 1e-10, never 0, for
                # exactly this reason). No valid route ever takes a self-loop,
                # so this doesn't change any tour cost.
                deepaco_distances = distances.clone()
                deepaco_distances.fill_diagonal_(1e-10)
                # PRISM's demand is capacity-normalized; DeepACO's ACO expects
                # raw quantities against its own (identical, 50) capacity.
                deepaco_tensors = {
                    "demands": demand * CVRP_CAPACITY,
                    "distances": deepaco_distances,
                }
            else:  # pctsp
                # PRISM's own prize scale (its generator targets an expected
                # total prize of ~2 regardless of size) and quota (schema
                # default 1.0, i.e. collect half of that) differ from
                # DeepACO's own hardcoded ACO quota (node_count / 4). DeepACO
                # still runs correctly against PRISM-scaled prizes -- its
                # depot mask has an "all nodes visited" escape even if the
                # quota is never reached -- it just tends to visit more nodes
                # than its own distribution would call for.
                problem = problem_data.generated_problem("pctsp", size)
                distances = torch.as_tensor(problem["distance"], dtype=torch.float32)
                prize = torch.as_tensor(problem["prize"], dtype=torch.float32)
                penalty = torch.as_tensor(problem["penalty"], dtype=torch.float32)
                data = {
                    "distance": distances.unsqueeze(0),
                    "prize": prize.unsqueeze(0),
                    "penalty": penalty.unsqueeze(0),
                }
                deepaco_tensors = {
                    "distances": distances,
                    "prizes": prize,
                    "penalties": penalty,
                }

            prism_heuristic = prism_energy_matrix(
                prism_model, args.variant, data, args.device
            )
            deepaco_heuristic = deepaco_raw_heuristic(
                args.variant, model, **deepaco_tensors
            )

            def run_aco(heuristic, seed_offset, use_model=True):
                torch.manual_seed(args.seed + index + seed_offset)
                active_model = model if use_model else None
                if args.variant == "sop":
                    return deepaco_sop(
                        deepaco_distance,
                        adjacency,
                        preceding,
                        args.aco_ants,
                        args.aco_rounds,
                        active_model,
                        heuristic=heuristic,
                    )
                if args.variant == "bpp":
                    return deepaco_bpp(
                        deepaco_tensors["demands"],
                        args.aco_ants,
                        args.aco_rounds,
                        active_model,
                        heuristic=heuristic,
                    )
                if args.variant == "mkp":
                    return deepaco_mkp(
                        deepaco_tensors["prize"],
                        deepaco_tensors["weight"],
                        args.aco_ants,
                        args.aco_rounds,
                        active_model,
                        heuristic=heuristic,
                    )
                if args.variant == "tsp":
                    return deepaco_tsp(
                        deepaco_tensors["coordinates"],
                        args.aco_ants,
                        args.aco_rounds,
                        active_model,
                        args.k_sparse,
                        heuristic=heuristic,
                    )
                if args.variant == "cvrp":
                    return deepaco_cvrp(
                        deepaco_tensors["demands"],
                        deepaco_tensors["distances"],
                        args.aco_ants,
                        args.aco_rounds,
                        active_model,
                        heuristic=heuristic,
                    )
                return deepaco_pctsp(
                    deepaco_tensors["distances"],
                    deepaco_tensors["prizes"],
                    deepaco_tensors["penalties"],
                    args.aco_ants,
                    args.aco_rounds,
                    active_model,
                    heuristic=heuristic,
                )

            def run_prism(guidance_fn=None, baseline=None):
                return prism_solve(
                    args.variant,
                    data,
                    args.prism_iterations,
                    args.seed + index,
                    model=None if guidance_fn or baseline else prism_model,
                    device=args.device,
                    guidance_fn=guidance_fn,
                    baseline=baseline,
                )

            deepaco_guidance_fn = lambda decoder, h=deepaco_heuristic: (  # noqa: E731
                deepaco_guidance_for_prism(decoder, h, args.variant)
            )
            results = {
                "prism/prism": run_prism(),
                "prism/deepaco": run_prism(deepaco_guidance_fn),
                "prism/heuristic": run_prism(baseline="distance"),
                "deepaco/prism": run_aco(prism_heuristic, 0),
                "deepaco/deepaco": run_aco(deepaco_heuristic, 1),
                "deepaco/heuristic": run_aco(None, 2, use_model=False),
            }
            for combo in combos:
                rows[combo].append(results[combo])
            print(
                f"[{index:>3}] "
                + "  ".join(
                    f"{combo} {results[combo]['score']:>9.4f}"
                    + (f" err={results[combo]['error']}" if results[combo]["error"] else "")
                    for combo in combos
                )
            )

    if not rows["prism/prism"]:
        return 1
    better = "higher" if args.variant == "mkp" else "lower"
    label = {
        "bpp": "bins",
        "mkp": "prize",
        "sop": "path cost",
        "tsp": "tour cost",
        "cvrp": "route cost",
        "pctsp": "tour+penalty cost",
    }[args.variant]
    print()
    print(f"{args.variant} n={size}  {len(rows['prism/prism'])} instances  ({better} {label} is better)")
    for combo in combos:
        scores = [row["score"] for row in rows[combo]]
        invalid = sum(1 for row in rows[combo] if row["error"])
        print(
            f"  {combo:<16} mean {statistics.fmean(scores):8.4f}   infeasible {invalid}"
        )
    baseline = [row["score"] for row in rows["deepaco/deepaco"]]
    for combo in (
        "prism/prism",
        "prism/deepaco",
        "prism/heuristic",
        "deepaco/prism",
        "deepaco/heuristic",
    ):
        scores = [row["score"] for row in rows[combo]]
        gap = statistics.fmean(
            [
                (b - a) / b * 100.0 if args.variant == "mkp" else (a - b) / b * 100.0
                for a, b in zip(scores, baseline)
                if b
            ]
        )
        print(f"  {combo} mean gap to deepaco/deepaco: {gap:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
