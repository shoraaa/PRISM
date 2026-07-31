"""Generate fixed validation artifacts for variants without benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from problem_data import DEFAULT_DATASET_DIR, generate_vrptw_validation_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize VRPTW validation data from its train generator"
    )
    parser.add_argument("--n-node", type=int, default=100)
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0x54570000)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.dataset_dir / "vrptw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"vrptw{args.n_node}_n{args.count}_seed{args.seed}.pt"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing data: {output}")
    data = generate_vrptw_validation_data(
        args.n_node, args.count, seed=args.seed
    )
    torch.save(data, output)
    print(f"saved {args.count} VRPTW instances to {output}")


if __name__ == "__main__":
    main()
