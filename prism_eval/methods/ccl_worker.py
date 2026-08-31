#!/usr/bin/env python3
"""Evaluate a CCL-MTLVRP checkpoint. Runs inside CCL's virtual environment.

This file is never imported by PRISM: it is executed by ``ccl.CclMethod`` with
CCL's own interpreter, because rl4co/lightning/hydra live only there. It reads
one .npz written from PRISM's instances, runs CCL's policy over it, and prints
a single ``CCL_RESULT_JSON`` line with one objective per instance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ccl-reld", required=True, type=Path)
    parser.add_argument("--multi-depot", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--augmentations", type=int, default=8)
    parser.add_argument(
        "--starts",
        type=int,
        default=0,
        help="0 uses the environment default, i.e. full POMO multi-start",
    )
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(args.ccl_reld))
    if args.device == "cpu":
        # The released checkpoint builds nested modules on CUDA whenever ROCm is
        # visible, even after load_from_checkpoint(map_location="cpu"). Hiding
        # the accelerators before importing torch keeps this process CPU-bound
        # rather than leaving a CUDA tensor inside its attention path.
        os.environ["ROCR_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import torch
    from rl4co.data.transforms import StateAugmentation
    from rl4co.utils.ops import unbatchify
    from routefinder.data.utils import get_dataloader
    from routefinder.envs import MTDVRPEnv, MTVRPEnv
    from routefinder.models import RouteFinderBase, RouteFinderMoE
    from routefinder.models.baselines.mtpomo import MTPOMO
    from routefinder.models.baselines.mvmoe import MVMoE

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable in {sys.executable}")
    device = torch.device(args.device)

    checkpoint = str(args.checkpoint)
    if "mvmoe" in checkpoint:
        model_class = MVMoE
    elif "mtpomo" in checkpoint:
        model_class = MTPOMO
    elif "moe" in checkpoint:
        model_class = RouteFinderMoE
    else:
        model_class = RouteFinderBase
    model = model_class.load_from_checkpoint(
        checkpoint, map_location="cpu", strict=False
    )
    policy = model.policy.to(device).eval()

    env = MTDVRPEnv() if args.multi_depot else MTVRPEnv()
    td_all = env.load_data(str(args.dataset))
    total = int(td_all.batch_size[0])
    dataloader = get_dataloader(td_all, batch_size=args.batch_size or total)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    rewards = []
    for batch in dataloader:
        td = env.reset(batch).to(device)
        starts = args.starts or env.get_num_starts(td)
        if args.augmentations > 1:
            td = StateAugmentation(
                num_augment=args.augmentations, augment_fn="dihedral8"
            )(td)
        # CCL's stock test.py enables CUDA autocast. At n=500 that produces NaN
        # logits with the released size-100 checkpoint, so this runner keeps
        # float32 inference deliberately.
        with torch.inference_mode():
            output = policy(
                td, env, phase="test", num_starts=starts, return_actions=False
            )
        reward = unbatchify(output["reward"], (args.augmentations, starts))
        if starts > 1:
            reward = reward.max(dim=-1).values
        if args.augmentations > 1:
            reward = reward.max(dim=1).values
        rewards.append(reward.detach().reshape(-1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started

    # CCL maximizes a negative reward; PRISM minimizes the distance itself.
    objectives = [-float(value) for value in torch.cat(rewards)]
    print(
        "CCL_RESULT_JSON "
        + json.dumps(
            {
                "objectives": objectives,
                "direction": "minimize",
                "seconds": seconds,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
