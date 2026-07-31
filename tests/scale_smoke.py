#!/usr/bin/env python3
"""Run one reproducible large-scale field-decoder inference gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from net import ConstraintFieldNet  # noqa: E402
from train import infer_instance, setup_seeds  # noqa: E402
from problem_data import generated_problem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-node", type=int, default=10_000)
    parser.add_argument("--variant", default="cvrp")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--search-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    setup_seeds(cli.seed)
    problem = generated_problem(cli.variant, cli.n_node, 50)
    model = ConstraintFieldNet().to(cli.device)
    if cli.checkpoint:
        payload = torch.load(
            cli.checkpoint, map_location=cli.device, weights_only=False
        )
        model.load_state_dict(payload["model_state_dict"])
    model.eval()
    decoder_args = SimpleNamespace(
        candidates=64,
        n_ants=32,
        beta=2.0,
        seed=cli.seed,
        search_iterations=cli.search_iterations,
        feasibility_lookahead_depth=2,
        feasibility_risk_penalty=10.0,
        device=cli.device,
    )
    started = time.perf_counter()
    average, best, metrics = infer_instance(model, problem, decoder_args)
    result = {
        "variant": cli.variant,
        "n_node": cli.n_node,
        "checkpoint": str(cli.checkpoint) if cli.checkpoint else None,
        "elapsed_seconds": time.perf_counter() - started,
        "average": average,
        "objective": float(best["objective"]),
        "feasible": bool(best["feasible"]),
        "route_size": len(best["route"]),
        **metrics,
    }
    print(json.dumps(result, sort_keys=True))
    if not result["feasible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
