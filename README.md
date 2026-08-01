# PRISM: Compositional Constraint Interaction Fields for Routing Search

Routing constraints rarely act in isolation. Capacity changes which time-window
transitions remain useful; route limits alter the value of returning to a depot;
pickup-delivery precedence reshapes feasible neighborhoods; and these effects
change again as the incumbent solution evolves. A binary feasibility mask can
reject an illegal move, but it does not express how several active resources
jointly reshape the value of otherwise feasible choices.

PRISM introduces **constraint interaction fields**: typed, continuous edge-level
signals that represent how individual routing resources and their interactions
reshape the value of candidate search moves. The fields combine analytic
resource pressure, learned pressure corrections, live state-dependent
modulation, and continuation-risk potential. A unified native decoder maintains
exact routing semantics and feasibility while the learned fields provide
continuous guidance before hard constraint violations occur.

The C++ backend builds a directed `O(nK)` candidate graph with `K=64`, runs
parallel search ants with static OpenMP scheduling, and improves solutions with
scope-restricted refinement (SRR). The neural model operates directly on this
search graph and is refreshed whenever an accepted incumbent installs a new
graph. Classical ACO guidance remains available as a matched search baseline.

## Constraint interaction fields

For candidate edge `e` in search state `s`, PRISM uses the constraint-aware
search energy

```text
E(e | s) = c(e)
         + sum_r lambda_r(s) * [p_r(e) * rho_r(e) + a_r(e)]_+
         + kappa * q(e),
```

where:

- `c(e)` is the objective-aware edge cost;
- `p_r(e)` is the analytic pressure of resource `r`;
- `rho_r(e)` and `a_r(e)` are learned multiplicative and additive pressure
  corrections;
- `lambda_r(s)` is the learned, live-state-modulated intensity of that resource;
  and
- `q(e)` is a learned continuation-risk potential.

The multiplicative term adapts known physical pressure, while the additive term
represents interactions that remain informative when the analytic pressure is
zero. Resource-token attention lets capacity, time windows, route limits,
backhaul, pickup-delivery, and prize requirements change one another's learned
intensities. A lightweight state coupler then updates those intensities at every
logged decision without rerunning the full GNN.

Together, the resource-specific channels form the constraint interaction field.
Each **resource field** describes one typed constraint over the candidate graph;
resource-token attention makes every channel depend on the complete active
constraint composition, and state modulation adapts it to the current partial
solution.

The field is trained at two resolutions. Exact C++ screening traces teach it to
predict counterfactual resource changes and binding structure; sparse
continuation look-ahead teaches the continuation-risk potential. PPO then
optimizes the resulting search policy from accepted improvements, using
event-driven SMDP options to carry credit across incumbent changes.

## Contributions

- **Constraint interaction fields.** PRISM factorizes neural search guidance
  into typed resource-specific fields whose values depend jointly on candidate
  edges, active constraint composition, and live search state. This structure
  provides continuous guidance before hard constraint violation while retaining
  exact decoder-enforced feasibility.
- **Compositional routing representation.** A shared resource-token model and a
  unified decoder cover capacity, time windows, route and tour limits, backhaul,
  pickup-delivery, prize quota, open routes, multiple depots, optional customers,
  and symmetric or asymmetric costs.
- **Search-native learning.** Neural fields are recomputed after incumbent
  changes, coupled to live decoder state at individual decisions, and trained by
  replaying the exact stochastic decisions made by the native search.
- **Scalable exact resource accounting.** Sparse candidate support, composable
  route summaries, incremental caches, and affected-scope repair make learned
  search practical from mixed size-100 training through size-10,000 inference.

## Search and learning architecture

### Typed constraint fields

The GNN emits one field for each constraint resource. Node, edge,
resource-token, and live-state inputs are normalized to `[0, 1]`. Active
resource tokens attend to one another before producing edge pressure
corrections, global resource intensities, binding predictions, and live-state
coupler parameters. The decoder is the source of truth for graph dimensions,
resource scales, and active channels.

The field is refreshed when the incumbent improves or a new candidate graph is
installed. Stagnation ends the current SMDP option without recomputing an
identical graph, while the state coupler continues to respond to load, time,
route progress, and other live search variables at each stochastic choice.

### Counterfactual feasibility learning

At traced states, the decoder labels candidate edges that are immediately
masked and applies configurable sparse look-ahead to currently legal edges.
The feasibility head learns the resulting continuation risk. After auxiliary
pretraining, this risk guides proposal, perturbation, and SRR sequence energy,
with the prediction detached from policy replay.

### Exact and incremental SRR

SRR evaluates planned capacity, route-limit, tour-limit, backhaul, and prize
changes from cached route summaries. Once a planned sequence is available,
these resource replacements are constant in the number and length of routes.
Time-window cascading lateness and pickup-delivery open-pair maxima use exact
route replay, preserving their full state-dependent semantics. Trace output
reports both summary and replay evaluation counts.

Accepted plans replace only the affected route caches. Edge membership uses
reference counts, resource extrema use versioned heaps, and the repair scope is
derived from changed local links. Stable route slots allow depot split, merge,
and reassignment moves to use the same local update without renumbering the
remaining solution.

For correctness runs, enable:

```python
search_config={
    "verify_screening_resources": True,
    "verify_incremental_srr": True,
}
```

The first option compares every summary-derived resource label with exact
replay. The second reconstructs and compares the complete cache after every
accepted incremental update. Solutions expose `srr_incremental_rebuilds`,
`srr_full_rebuilds`, and `srr_rebuilt_nodes`.

### Decision-level PPO

Training replays one probability ratio per stochastic decoder decision. Option
returns are assigned to ants, and inverse decision-count weighting gives every
ant equal total influence independent of trace length. Auxiliary resource,
binding, and feasibility losses use full weight during pretraining and default
to `0.1` during RL fine-tuning through `--aux-rl-scale`.

Winner-gated Monte Carlo credit connects consecutive incumbent improvements.
The ant that installs the next incumbent receives the sampled continuation
advantage, preserving temporal information after within-option POMO centering.
An optional progress-conditioned GAE critic supplies a learned continuation
value for longer search horizons.

On HIP/ROCm, training automatically selects the detached-output small-VRAM
update. `--no-smallvram` selects the conventional retained-graph update.

## Build and test

```bash
uv sync
uv run python setup.py build_ext --inplace
uv run pytest -q
```

Problem schemas, generators, dataset discovery, and saved-file readers live in
[`problem_data.py`](problem_data.py). Benchmark files can be placed under
`baselines/URS/dataset`, relocated with `PRISM_DATASET_DIR`, or selected with
`--dataset-dir`.

The registry contains 110 benchmark compositions plus `vrptw`, a closed
multi-route time-window problem without demand or capacity. Generate its fixed
validation artifact with:

```bash
uv run python generate_validation_data.py --n-node 100
```

## Training

Train the resource-token interaction fields and event-driven decoder with:

```bash
uv run python train.py --n-node 1000 --pretrain-epochs 3 \
  --pretrain-lr 1e-4 --pretrain-aux-scale 1.0 --no-wandb
```

Set `--pretrain-epochs 0` to begin joint PPO training immediately. The aliases
`--pretraining-epochs`, `--pretraining-lr`, and `--pretraining-aux-scale` are
equivalent, and all settings are stored in checkpoints and W&B configuration.

Training uses an epoch-local balanced schedule. Every eligible routing
composition appears before another receives an extra rollout, and each PPO
accumulation group contains distinct variants. The curriculum first learns
individual resource effects and then introduces increasingly rich interactions.

Temporal credit is controlled by `--temporal-credit-weight` (default `0.1`). Set
it to `0` for the local-POMO ablation. The optional critic can be enabled with,
for example:

```bash
uv run python train.py --gae-lambda 0.95 --value-loss-weight 0.5
```

The default `--gae-lambda 1.0 --value-loss-weight 0.0` uses critic-free sampled
Monte Carlo continuation credit.

## Validation and checkpoint selection

Validation covers all 12 training variants and a fixed 16-variant held-out
manifest stratified by objective, pickup-delivery structure, symmetry, depot
count, and number of interacting constraints. With `--val-size 8`, the complete
manifest contains 224 fixed instances. Missing entries fail validation unless
`--allow-missing-validation` is selected.

Before epoch 0, PRISM evaluates the matched non-neural decoder under the same
search budget. W&B records per-variant objective, feasibility, reference gap,
and paired learned-field improvement, together with:

- `val_summary/macro_gap`
- `val_summary/macro_improvement`
- `val_summary/macro_score`

Checkpoint selection is lexicographic: worst-variant feasibility, overall
feasibility, paired-baseline coverage, variant-macro improvement, and finally
variant-macro reference gap. This ordering favors constraint reliability across
the full compositional family before average objective quality.

## Evaluation

Run the end-to-end size-100 feasibility and resource-parity gates:

```bash
uv run python tests/urs_one_each.py --iterations 2 --guidance classical --no-pheromone
uv run python tests/urs_one_each.py --iterations 2 --guidance field --no-pheromone
uv run python tests/urs_screening_parity.py
uv run python tests/urs_one_each.py --ants 1 --threads 1 --iterations 2 \
  --no-pheromone --verify-incremental-srr
```

`--guidance field` exercises the neutral typed-field interface. Evaluate trained
constraint interaction fields against the matched non-neural decoder on every
benchmark composition with:

```bash
PYTHONPATH=src uv run --no-sync python test.py \
  --checkpoint pretrained/best.pt --variants all --iterations 16 \
  --csv results/checkpoint_vs_non_neural.csv
```

By default, `test.py` evaluates all 110 benchmark variants using the first eight
saved instances of each. It reports overall results plus separate `SEEN` results
for the 15 benchmark variants in the training curriculum and `HELDOUT` results
for the remaining 95. Both decoder paths use the same paired per-instance seed,
candidate budget, ant count, post-bootstrap iteration count, and matched oracle
reference. Dynamic field refinement is enabled by default. For a focused subset,
use `--val-size 1 --variants tsp,cvrp,cvrptw`; use `--static-field` to evaluate
one frozen field for the full solve. The report and optional CSV record the data
split, mean objective, baseline improvement, reference gap, runtime, field mode,
and neural evaluation count for every variant.

## Large-scale inference

Run the size-10,000 inference gate with either native or learned guidance:

```bash
uv run python tests/scale_smoke.py --n-node 10000
uv run python tests/scale_smoke.py --n-node 10000 --checkpoint pretrained/best.pt
```

Large Euclidean instances retain coordinates instead of allocating a dense
distance matrix. Candidate construction combines bounded KD-tree neighborhoods
with resource-indexed pools, and the GNN uses direct single-graph reductions.
Training enables activation checkpointing automatically from `n=1000`, while
inference remains checkpoint-free.

The feasibility look-ahead defaults to two steps. Its learned risk is conditioned
on the decoder's live resource state, and its contribution is bounded by
`--feasibility-risk-penalty` times the instance objective scale (default `1.0`).
The supervised risk classifier remains isolated from PPO; PPO instead controls a
separate sigmoid trust gate that can suppress unreliable risk guidance. Configure
the look-ahead with `--feasibility-lookahead-depth`.
