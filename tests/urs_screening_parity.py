#!/usr/bin/env python3
"""Verify O(1) planned SRR labels against full resource evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines" / "URS"))

import prism_decoder  # noqa: E402
from problem.ProblemSet import ProblemSet  # noqa: E402
from urs_data import SavedURS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ants", type=int, default=1)
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()

    saved = SavedURS(args.size)
    fast = 0
    fallback = 0
    failures = []
    variants = list(ProblemSet.get())
    for index, variant in enumerate(variants):
        try:
            problem, _ = saved.load(variant)
            decoder = prism_decoder.Decoder(
                problem,
                search_config={
                    "use_pheromone": False,
                    "verify_screening_resources": True,
                },
                n_ants=args.ants,
            )
            decoder.seed(20260727 + index)
            bootstrap = decoder.solve(1)
            batch = decoder.sample_traced()
            trace = batch["trace"]
            if not bootstrap["feasible"]:
                raise RuntimeError(bootstrap["error"])
            if trace["screening_verification_failures"]:
                raise RuntimeError(
                    "resource mismatch by channel: "
                    + str(
                        trace[
                            "screening_verification_failures_by_channel"
                        ].tolist()
                    )
                )
            fast += int(trace["screening_fast_evaluations"])
            fallback += int(trace["screening_fallback_evaluations"])
        except Exception as exc:
            failures.append((variant, str(exc)))

    print(
        f"URS_SCREENING_PARITY passed={len(variants) - len(failures)} "
        f"failed={len(failures)} fast={fast} fallback={fallback}"
    )
    for variant, error in failures:
        print(f"FAIL {variant}: {error}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
