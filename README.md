# PRISM Routing Decoder

PRISM is a sparse, constraint-aware, field-guided ILS decoder for a composable
routing problem family. Its C++ backend builds a directed `O(nK)` candidate
graph with `K=64`, runs parallel ants with static OpenMP scheduling, and uses
scope-restricted refinement with composable resource summaries. The classical
ACO terms are available only as an explicit benchmark mode.

Traced SRR evaluates planned capacity, route-limit, tour-limit, backhaul, and
prize resource labels from cached route aggregates instead of replaying the
whole solution. Once the planned sequence summaries are available, resource
replacement is constant in the number and length of routes. Time-window and
pickup-delivery labels retain the exact legacy replay:
the current cascading lateness sum and open-pair maximum do not have the same
fixed-size concatenation algebra. Trace output reports fast and fallback
counts. Set `search_config={"verify_screening_resources": true}` to compare
every fast result against the legacy evaluator during a correctness run.

Accepted SRR plans replace only their one or two affected route caches. Edge
membership uses reference counts, resource extrema use versioned heaps, and the
BFS scope is derived from changed local links, so this maintenance scales with
the affected route lengths rather than all nodes. Stable route slots let depot
split, merge, and reassignment moves use the same local update without
renumbering downstream nodes. Solutions report
`srr_incremental_rebuilds`, `srr_full_rebuilds`, and `srr_rebuilt_nodes`.
Set `search_config={"verify_incremental_srr": true}` to reconstruct and compare
the complete cache after every incremental acceptance.

The neural path emits one edge field per constraint resource. All node, edge,
resource-token, and live-state values presented to the GNN or state coupler are
normalized to `[0, 1]`. A field is refreshed when the incumbent improves or a
new candidate graph is otherwise installed. A stagnation cap ends the current
SMDP option without recomputing an identical field; the lightweight state
coupler is replayed at every logged stochastic decision.

The decoder also emits a scalar feasibility risk per candidate edge. During
auxiliary pretraining, C++ labels immediately masked counterfactual edges and
uses a configurable sparse look-ahead for legal edges. Learned field and risk
penalties remain disabled until pretraining completes; afterward risk enters
proposal, perturbation, and SRR sequence energy with policy gradients stopped
at the risk prediction.

PPO is replayed per stochastic decoder decision, with option returns broadcast
from ants and inverse decision-count weights preserving equal ant influence.
Graph normalization uses current-instance statistics in both train and
evaluation modes rather than curriculum-dependent running statistics.
Auxiliary losses use full weight during pretraining and default to `0.1` during
RL fine-tuning; adjust the latter with `--aux-rl-scale`. The detached-output
small-VRAM update is selected automatically for HIP/ROCm devices and can be
overridden with `--no-smallvram`.

Auxiliary-only pretraining is disabled by default (`--pretrain-epochs 0`), so
learned guidance and RL are active from epoch 0. Set a positive value explicitly
to run a warm-up experiment.

## Build and test

```bash
uv sync
uv run python setup.py build_ext --inplace
uv run pytest -q
```

Problem definitions, random generators, dataset discovery, and saved-file
readers are owned by [`problem_data.py`](problem_data.py). PRISM does not import
Python code from a baseline repository. The existing 110-task benchmark files
remain input artifacts; point to a relocated copy with `PRISM_DATASET_DIR` or
`--dataset-dir`.

Run the end-to-end size-100 URS feasibility gates with:

```bash
uv run python tests/urs_one_each.py --iterations 2 --guidance classical --no-pheromone
uv run python tests/urs_one_each.py --iterations 2 --guidance field --no-pheromone
uv run python tests/urs_screening_parity.py
uv run python tests/urs_one_each.py --ants 1 --threads 1 --iterations 2 \
  --no-pheromone --verify-incremental-srr
```

The `field` mode above is a neutral-field interface gate, not checkpoint
inference. Compare a trained checkpoint with the non-neural decoder on the same
saved size-100 instances of every benchmark variant, plus generated VRPTW,
using:

```bash
PYTHONPATH=src uv run --no-sync python tests/compare_decoders.py \
  --checkpoint pretrained/best.pt --variants all --iterations 16 \
  --csv results/checkpoint_vs_non_neural.csv
```

By default this averages the first 8 instances of each variant and refreshes
the neural field whenever the incumbent graph changes. Both sides use the same
per-instance seed, candidate budget, post-bootstrap iteration count, and
available oracle reference. Use `--val-size 1 --variants tsp,cvrp,cvrptw` for a
quick subset, or `--static-field` to reproduce frozen-field inference.

The 12-task training curriculum includes `vrptw`, a closed, multi-route
time-window task with no demand or capacity constraint. It appears in the
one-resource phase; `cvrptw` and `ocvrptw` enter later to train capacity/TW and
open-route/TW interactions. Validation generates deterministic VRPTW instances
and measures their gap against a same-budget classical decoder reference.
Before epoch 0, validation runs the same-budget classical decoder, caches its
per-instance objectives, and stores its macro gap once as the run-level W&B
summary `baseline/gap`. It is not a training epoch. Every learned epoch is
paired against those per-instance objectives.
VRPTW alone has no external benchmark, so its fixed validation batch is
materialized once from the current training distribution with
`uv run python generate_validation_data.py --n-node 100`; later runs load the
saved artifact instead of regenerating it.
W&B reports baseline improvement, reference coverage, and per-variant gaps; the
oracle gap is a macro-average over referenced variants. Best-checkpoint ranking
is feasibility-first, then maximizes the average paired improvement percentage
over the classical baseline, with oracle macro gap as the final tie-breaker. Checkpoints also
save global step and Python, NumPy, Torch, CUDA, and curriculum RNG state for
exact continuation.

The compact W&B `val_summary/` namespace contains `macro_gap`, the
variant-macro `macro_improvement`, and `macro_score` (the instance-average
paired improvement used for checkpoint selection). Detailed changing
per-variant metrics remain under `val/variants/`; static manifest counts and
coverage bookkeeping are not sent to W&B.

Train the resource-token field and event-driven decoder with:

```bash
uv run python train.py --n-node 1000 --no-wandb
```

Run the large-scale inference gate, optionally with a trained checkpoint:

```bash
uv run python tests/scale_smoke.py --n-node 10000
uv run python tests/scale_smoke.py --n-node 10000 --checkpoint pretrained/best.pt
```

Large Euclidean problems use coordinate-backed distances and do not allocate a
dense distance matrix. The GNN uses direct single-graph reductions to avoid a
high-contention scatter on the 10K graph. Training automatically enables
activation checkpointing from n=1000; inference remains checkpoint-free.

The feasibility defaults are a two-step look-ahead and a `10.0` cost-unit soft
penalty. They are configurable with `--feasibility-lookahead-depth` and
`--feasibility-risk-penalty`.

Validation loads eight instances per selected problem by default. Change
`--val-size` to adjust that count; all training variants are included unless
`--val-seen` is set explicitly.
