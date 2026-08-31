#!/usr/bin/env python
"""Validate PRISM's non-routing classes against DeepACO on identical instances.

Bin packing, multidimensional knapsack and sequential ordering are the classes
PRISM declares outside routing (see ``problem_data.NON_ROUTING_VARIANTS``).
DeepACO is the natural reference: it is a learned-heatmap method that publishes
a specialist model for each, so it fixes both the instance distribution and a
comparable number.

Instances come from DeepACO's own generators rather than a reimplementation of
them, so distributional parity holds by construction. The same tensors are then
handed to PRISM's decoder and to DeepACO's ACO, and both are scored on the
task's true objective -- bins used for BPP, collected prize for MKP -- rather
than on either method's internal training surrogate. DeepACO's BPP fitness is a
Falkenauer fill-quality score, which is not the bin count, so its solutions are
re-scored here.

DeepACO's BPP module imports numba only for two fitness helpers. When numba is
absent, a pass-through shim is installed so the rest of the module loads
unchanged; the helpers are small enough that pure Python is not the bottleneck.

Usage:
    python scripts/deepaco_compare.py --variant bpp --size 120 --instances 10
    python scripts/deepaco_compare.py --variant mkp --size 50 --instances 10
    python scripts/deepaco_compare.py --variant sop --size 50 --instances 10
"""

from __future__ import annotations

import argparse
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

    from net import ConstraintFieldNet, load_constraint_field_state_dict
    from train import MODEL_SCHEMA

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_schema") != MODEL_SCHEMA:
        raise SystemExit(
            f"checkpoint schema {checkpoint.get('model_schema')!r} != {MODEL_SCHEMA!r}"
        )
    config = checkpoint.get("config", {})
    model = ConstraintFieldNet(
        couple_resource_tokens=config.get("couple_resource_tokens", True),
        linear_objective_residual_head=config.get(
            "linear_objective_residual_head", False
        ),
        unconditioned_objective_residual_head=config.get(
            "unconditioned_objective_residual_head", False
        ),
        couple_state_multipliers=config.get("couple_state_multipliers", True),
        gate_multipliers_by_binding=config.get("gate_multipliers_by_binding", True),
        index_embedded_resources=config.get("index_embedded_resources", False),
        monolithic_resource_field=config.get("monolithic_resource_field", False),
        program_blind_resources=config.get("program_blind_resources", False),
    ).to(device)
    load_constraint_field_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
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
        feasibility_risk_penalty=1.0,
        device=device,
        static_field=False,
        min_changed_edges=8,
        random_escape=False,
        srr_exploration_budget=0,
    )


def prism_solve(
    variant: str,
    data: dict,
    iterations: int,
    seed: int,
    model=None,
    device: str = "cpu",
) -> dict:
    problem = problem_data.decoder_problem(variant, data)
    if model is None:
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

        from train import infer_instance

        with torch.no_grad():
            _, solution, _ = infer_instance(
                model, problem, _inference_namespace(problem, iterations, seed, device)
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
    selected = [node for node in route if node != 0]
    weight = np.stack(
        [problem["node_attributes"][key] for key in sorted(problem["node_attributes"])],
        axis=1,
    )
    budget = float(np.asarray(data["budget"]).reshape(-1)[0])
    return {
        "score": float(problem["prize"][selected].sum()),
        "error": solution.get("error", "") or mkp_check(selected, weight, budget),
        "detail": f"{len(selected)} items",
    }


# --------------------------------------------------------------------------
# DeepACO
# --------------------------------------------------------------------------


def deepaco_bpp(demands: torch.Tensor, ants: int, rounds: int, model) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if model is not None:
        from utils import gen_pyg_data  # type: ignore

        heuristic = model(gen_pyg_data(demands, "cpu"))
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
    prize: torch.Tensor, weight: torch.Tensor, ants: int, rounds: int, model
) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if model is not None:
        from utils import gen_pyg_data  # type: ignore

        heuristic = model(gen_pyg_data(prize, weight))
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
) -> dict:
    from aco import ACO  # type: ignore

    kwargs = {}
    if model is not None:
        from utils import gen_pyg_data  # type: ignore

        heuristic = model(gen_pyg_data(distance, adjacency, "cpu"))
        size = distance.size(0)
        dense = torch.zeros((size, size))
        dense[adjacency.bool()] = heuristic.reshape(-1)
        kwargs["heuristic"] = dense + 1e-10
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


def load_model(variant: str, size: int):
    path = DEEPACO / "pretrained" / variant / f"{variant}{size}.pt"
    if not path.exists():
        return None, f"no pretrained {variant} model for size {size}"
    from net import Net  # type: ignore

    model = Net()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, ""


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("bpp", "mkp", "sop"), required=True)
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
        "--no-pretrained",
        action="store_true",
        help="run DeepACO with its hand-designed heuristic instead of its model",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="PRISM checkpoint whose learned field guides the search; "
        "omitted, the decoder searches unguided",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    size = args.size or (120 if args.variant == "bpp" else 50)

    prism_model = None
    if args.checkpoint:
        prism_model, prism_config = load_prism_model(args.checkpoint, args.device)
        flags = [
            key
            for key in (
                "index_embedded_resources",
                "monolithic_resource_field",
                "program_blind_resources",
            )
            if prism_config.get(key)
        ]
        print(
            f"PRISM guidance: {args.checkpoint}"
            + (f" [{', '.join(flags)}]" if flags else " [semantic descriptor]")
        )

    shimmed = install_numba_shim()
    if shimmed:
        print("numba is absent; using a pass-through njit shim")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    rows: list[tuple[dict, dict]] = []
    with deepaco_module(args.variant):
        if args.variant == "sop":
            from utils import training_instance_gen as gen_instance  # type: ignore
        else:
            from utils import gen_instance  # type: ignore

        model, why = (None, "disabled") if args.no_pretrained else load_model(
            args.variant, size
        )
        if model is None:
            print(f"DeepACO model: {why}; using its hand-designed heuristic")

        # Draw every instance before either method runs. DeepACO's ACO samples
        # from the same global torch generator the instance generator draws
        # from, so generating lazily would make the instance stream depend on
        # how much sampling the reference happened to do -- and the two arms
        # would not be solving the same problems.
        if args.variant == "sop":
            draw = lambda: gen_instance(size, "cpu")  # noqa: E731
        elif args.variant == "bpp":
            draw = lambda: gen_instance(size, "cpu")  # noqa: E731
        else:
            draw = lambda: gen_instance(size, args.dimensions, "cpu")  # noqa: E731
        instances = [draw() for _ in range(args.instances)]

        for index, instance in enumerate(instances):
            if args.variant == "sop":
                distance, adjacency, preceding = instance
                reference = deepaco_sop(
                    distance, adjacency, preceding, args.aco_ants, args.aco_rounds, model
                )
                # DeepACO leaves the diagonal populated (it never traverses a
                # self-loop, and its 1/d heuristic would divide by zero); PRISM
                # validates a proper distance matrix, so it sees a zeroed copy.
                # No tour cost differs between the two.
                matrix = distance.clone()
                matrix.fill_diagonal_(0.0)
                width = max(1, int(preceding.sum(dim=1).max().item()))
                padded = torch.full((size, width), -1, dtype=torch.long)
                for node in range(size):
                    required = preceding[node].nonzero().flatten()
                    if required.numel():
                        padded[node, : required.numel()] = required
                data = {
                    "distance": matrix.unsqueeze(0),
                    "predecessors": padded.unsqueeze(0),
                }
            elif args.variant == "bpp":
                demands = instance
                data = {
                    "demand": (demands / BPP_CAPACITY).unsqueeze(0),
                }
                reference = deepaco_bpp(demands, args.aco_ants, args.aco_rounds, model)
            else:
                prize, weight = instance
                zeros = torch.zeros(1)
                data = {
                    "prize": torch.cat((zeros, prize)).unsqueeze(0),
                    "weight": torch.cat(
                        (torch.zeros(1, args.dimensions), weight)
                    ).unsqueeze(0),
                    "budget": torch.full((1,), float(size // 2)),
                }
                reference = deepaco_mkp(
                    prize, weight, args.aco_ants, args.aco_rounds, model
                )
            ours = prism_solve(
                args.variant,
                data,
                args.prism_iterations,
                args.seed + index,
                model=prism_model,
                device=args.device,
            )
            rows.append((ours, reference))
            print(
                f"[{index:>3}] prism {ours['score']:>9.4f} ({ours['detail']})"
                f"  deepaco {reference['score']:>9.4f} ({reference['detail']})"
                + (f"  PRISM: {ours['error']}" if ours["error"] else "")
                + (f"  DeepACO: {reference['error']}" if reference["error"] else "")
            )

    if not rows:
        return 1
    better = "higher" if args.variant == "mkp" else "lower"
    ours = [row[0]["score"] for row in rows]
    theirs = [row[1]["score"] for row in rows]
    invalid_ours = sum(1 for row in rows if row[0]["error"])
    invalid_theirs = sum(1 for row in rows if row[1]["error"])
    label = {"bpp": "bins", "mkp": "prize", "sop": "path cost"}[args.variant]
    print()
    print(f"{args.variant} n={size}  {len(rows)} instances  ({better} {label} is better)")
    print(f"  PRISM   mean {statistics.fmean(ours):8.4f}   infeasible {invalid_ours}")
    print(f"  DeepACO mean {statistics.fmean(theirs):8.4f}   infeasible {invalid_theirs}")
    gap = statistics.fmean(
        [
            (b - a) / b * 100.0 if args.variant == "mkp" else (a - b) / b * 100.0
            for a, b in zip(ours, theirs)
            if b
        ]
    )
    print(f"  PRISM mean gap to DeepACO: {gap:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
