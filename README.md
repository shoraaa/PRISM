# PRISM Routing Decoder

PRISM is a sparse, constraint-aware, field-guided ILS decoder for the URS
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
Auxiliary losses use full weight during pretraining and default to `0.1` during
RL fine-tuning; adjust the latter with `--aux-rl-scale`. The detached-output
small-VRAM update is selected automatically for HIP/ROCm devices and can be
overridden with `--no-smallvram`.

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

The training registry also includes `vrptw`, a closed multi-route problem with
time windows but no demand or capacity constraint. Generate its fixed validation
artifact with:

```bash
uv run python generate_validation_data.py --n-node 100
```

Later runs load the saved artifact from the configured dataset directory.

Training uses an epoch-local balanced schedule: every eligible variant appears
before any variant receives an extra rollout, and variants within each PPO
accumulation group are distinct. Validation defaults to all 12 training
variants plus a fixed 16-variant held-out manifest stratified across objective,
pickup-delivery, symmetry, depot count, and constraint count. With the default
`--val-size 8`, this is 224 fixed instances. Missing manifest entries are an
error unless `--allow-missing-validation` is explicitly supplied.

Before epoch 0, the same-budget non-neural decoder is evaluated once on the
validation manifest. W&B then reports per-variant objective, feasibility, gap,
and paired baseline improvement, plus `val_summary/macro_gap`,
`val_summary/macro_improvement`, and `val_summary/macro_score`. Checkpoint
selection is lexicographic: worst-variant feasibility, overall feasibility,
paired-baseline coverage, variant-macro improvement, then variant-macro gap.
Raw costs from different objectives are retained only as diagnostic values.

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
first saved size-100 instance of every URS variant using:

```bash
PYTHONPATH=src uv run --no-sync python tests/compare_decoders.py \
  --checkpoint pretrained/best.pt --variants all --iterations 16 \
  --csv results/checkpoint_vs_non_neural.csv
```

The checkpoint path uses dynamic refinement by default: its sparse field is
recomputed whenever an incumbent change installs a new candidate graph. Both
sides use the same seed, candidate budget, post-bootstrap iteration count, and
available oracle reference. Use `--static-field` for the frozen one-field
ablation, or `--variants tsp,cvrp,cvrptw` for a quick subset before the complete
run. The report and CSV include the field mode and neural evaluation count.

Train the resource-token field and event-driven decoder with:

```bash
uv run python train.py --n-node 1000 --pretrain-epochs 3 \
  --pretrain-lr 1e-4 --pretrain-aux-scale 1.0 --no-wandb
```

Set `--pretrain-epochs 0` to start PPO immediately. The equivalent long-form
aliases are `--pretraining-epochs`, `--pretraining-lr`, and
`--pretraining-aux-scale`; all values are saved in the checkpoint and W&B
configuration.

After pretraining, winner-gated Monte Carlo temporal credit is enabled by
default. The complete sampled reward-to-go is assigned only to the ant that
installed the next incumbent. Tune it with `--temporal-credit-weight` (default
`0.1`), or set that weight to `0` for the local-POMO-only ablation. The optional
progress-conditioned GAE critic remains available with, for example,
`--gae-lambda 0.95 --value-loss-weight 0.5`; the defaults are `1.0` and `0.0`.
Checkpoints from before the critic are accepted with its value head initialized
at zero.

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

Validation loads one saved instance per selected problem by default. Increase
`--val-size` to evaluate more instances per problem.
