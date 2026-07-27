# Training profile

Date: 2026-07-27

The profile uses the real `train.py` path on the local 16-thread CPU and ROCm
GPU. Each row is one generated variant, 32 ants, 16 search iterations, one
optimizer update, no validation, and synchronized accelerator timing. The
scalable configuration uses one full-graph PPO pass per rollout.

| Nodes | Revision | Total | Decoder | Neural emission | PPO | Rollout other |
|---:|---|---:|---:|---:|---:|---:|
| 500 | Before | 30.07 s | 8.03 s | 2.53 s | 16.95 s | 2.54 s |
| 500 | Final | 14.83 s | 6.39 s | 2.26 s | 3.85 s | 2.30 s |
| 1,000 | Before sparse/GPU fixes | 38.72 s | 21.37 s | 6.47 s | 8.85 s | 2.01 s |
| 1,000 | Final | 23.25 s | 19.02 s | 0.65 s | 1.69 s | 1.86 s |
| 1,000 | Final, RL active | 27.66 s | 23.28 s | 0.62 s | 1.75 s | 2.00 s |

At the target n=1,000 training scale, SRR is the bottleneck at 81.8% of the
training unit. PPO is 7.3%, rollout labels/bookkeeping are 8.0%, and the GNN
emission is 2.8%. Python mapping is not a material hotspot. A representative
n=1,000, 32-ant native step visits essentially all nodes, averages roughly
9,500 to 10,000 exact move evaluations per ant, and keeps the requested
unlimited BFS revisits. Further throughput work therefore belongs in
incremental SRR route-cache maintenance.

With learned guidance and PPO replay active from epoch zero, the conclusion is
unchanged: native decoding is 84.2% of the 27.66-second unit, while PPO is
6.3%. The higher decoder time is caused by the different field-guided search
trajectory, not trace replay in Python.

The original Python `cProfile` at n=100 also identified repeated full-graph PPO
passes before the native decoder became dominant: `ppo_update` used 1.57 s,
25 model forwards used 0.82 s, backward used 0.42 s, and 64 explicit device
synchronizations used 0.32 s. The final trainer defaults to one on-policy PPO
pass, defers metric scalar transfers, and synchronizes individual phases only
under `--profile-timing`.

The final n=1,000 `cProfile` run reports 1.73 seconds in `ppo_update`, 0.10
seconds across all `_step_loss` calls, and 0.09 seconds in graph assembly.
Native search releases the Python GIL, so it is intentionally absent as a
Python call frame; the synchronized phase timer measured that interval at
19.47 seconds in the same run.

Large Euclidean inputs no longer materialize an n-by-n distance matrix. The
10K candidate graph has 650,000 directed edges and builds in 1.02 s. A real
field inference gate at n=10,000 loaded the checkpoint from the measured
n=1,000 RL update and completed in 4.28 s total, including graph construction
and bootstrap; the neural emission used 0.58 s, the greedy decoder step used
0.21 s, and the returned CVRP solution was feasible. This one-update smoke is a
scale/correctness result, not a learned-quality result.

Commands:

```bash
PYTHONPATH=src uv run --no-sync python train.py --n-node 1000 --epochs 1 \
  --steps-per-epoch 1 --grad-accum-variants 1 --skip-validation \
  --no-wandb --profile-timing

PYTHONPATH=src uv run --no-sync python tests/scale_smoke.py --n-node 10000
```
