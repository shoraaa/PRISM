# PRISM: Learned Constraint-Interaction Fields for Routing Search

PRISM is a **learned search policy** for vehicle routing under composed
constraints. A native decoder defines an action space of candidate moves that
are guaranteed feasible, and a graph neural network defines the policy that
ranks those moves. That policy is a **constraint-interaction field**: a typed,
continuous, edge-level scoring function that is trained end to end by
reinforcement learning from the search's own accepted improvements. No
hand-tuned heuristic term appears in the ranking — classical quantities such as
edge distance and analytic resource pressure enter only as network inputs or as
ablatable references, never as fixed terms in the energy.

Routing constraints rarely act in isolation. Capacity changes which time-window
transitions remain useful; route limits alter the value of returning to a depot;
pickup-delivery precedence reshapes feasible neighborhoods; and these effects
change again as the incumbent solution evolves. A binary feasibility mask can
reject an illegal move, but it cannot express how several active resources
jointly reshape the value of the moves that remain legal. PRISM learns exactly
that continuous, composition- and state-dependent value, while the decoder keeps
every constructed solution feasible by construction.

The C++ backend builds a directed `O(nK)` candidate graph with `K=64`, runs
parallel search rollouts with static OpenMP scheduling, and improves solutions
with scope-restricted refinement (SRR). The neural policy operates directly on
this search graph and is refreshed whenever an accepted incumbent installs a new
graph.

## The learned policy

The network defines the search policy: for candidate edge `e` in search state
`s`, PRISM ranks moves by the energy

```text
E(e | s) = w_obj(s) * c(e)
         + sum_r lambda_r(s) * [field_r(e) + a_r(e)]_+
         + kappa * q(e),
```

and samples the next node from `softmax(-beta * E)`. Every term is produced by
the network:

- `c(e)` is the objective edge cost, entered through a learned,
  state-conditioned weight `w_obj(s)` rather than a hard unit coefficient;
- `field_r(e)` and `a_r(e)` are the learned per-edge field and additive term for
  resource `r` (scaled to that resource's units);
- `lambda_r(s)` is the learned, live-state-modulated intensity of resource `r`,
  a Lagrangian-style multiplier shaped by search reward rather than supervision;
- `q(e)` is an optional learned continuation-risk potential.

Nothing in the ranking is hand-tuned. Analytic resource pressure — the classical
signal a heuristic would multiply in here — is **not** a term in the energy; it
is exposed to the GNN only as an input edge feature, so the network must *learn*
where each resource matters instead of scaling a supplied pressure. The plain
objective cost is likewise only recovered as an ablation (`lambda_r = 0,
w_obj = 1`), not assumed.

Resource-token attention lets capacity, time windows, route limits, backhaul,
pickup-delivery, and prize requirements change one another's learned intensities,
so each channel depends on the complete active constraint composition. A
lightweight state coupler then updates those intensities -- and `w_obj` -- at
every logged decision without rerunning the full GNN, adapting the policy to the
live partial solution.

## Contributions

- **A learned policy, not a heuristic with a learned nudge.** The GNN defines
  the entire move-ranking energy end to end. The objective weight and every
  constraint intensity are learned and state-conditioned; distance and analytic
  pressure are inputs or ablations, never load-bearing terms. Ablating the field
  reduces PRISM to distance-only search, isolating exactly what the network adds.
- **Compositional constraint-interaction fields.** Typed per-resource fields
  with resource-token attention make each constraint's learned intensity depend
  on the full active constraint composition and the live search state, targeting
  zero-shot transfer to unseen constraint compositions.
- **Reinforcement learning over a feasibility-guaranteed action space.**
  Decision-level PPO trains the policy from the native search's accepted
  improvements. The exact decoder guarantees feasibility, so learning shapes
  *quality* rather than legality. Multipliers are shaped by RL instead of pinned
  to a supervised target, and policy gradients reach them from the first step.
- **Scalable exact resource accounting.** Sparse candidate support, composable
  route summaries, incremental caches, and affected-scope repair make the learned
  search practical from mixed size-100 training through size-10,000 inference.

## Architecture

### Runtime resource algebra

The decoder accepts a runtime `resources` registry in addition to the seven
canonical compatibility rows. A declarative row names an operator primitive,
its state dimension and direction, an extension input, reset events, scale, and
the phase at which lower or upper bounds are checked. The first supported
primitive is a scalar affine accumulator; it covers resources such as energy,
fuel, emissions, and driver-hour budgets without a Python callback in the hot
loop:

```python
problem["node_attributes"] = {"charger": charger_mask}
problem["resources"] = [{
    "name": "battery",
    "operator": "affine_accumulator",
    "state_dim": 1,
    "direction": "forward",
    "scope": "route",
    "initial": {"scalar": "battery_capacity"},
    "scale": {"scalar": "battery_capacity"},
    "increment": {"edge_attribute": "distance", "coefficient": -1.0},
    "reset": {
        "node_attribute": "charger",
        "value": {"scalar": "battery_capacity"},
    },
    "bounds": [{"lower": 0.0, "check": "transition"}],
}]
```

The binding resolves named arrays once, and native `extend + bound` replay then
drives construction masks, incumbent validation, resource labels, pressure
features, and live state. Each row also produces a 32-dimensional descriptor
from algebraic properties (operator family, bound type, check phase, direction,
scope, reset form, input coupling, scale, and tightness); names and registry
positions are excluded. Resource tensors grow with the registry, while the
canonical seven rows remain at their historical indices so the distance-only
control path retains exact behavior.

Candidate construction is itself schema-driven, so it extends to a new resource
with no per-variant tuning. In the default `schema` mode, native admission ranks
edges by a registry-derived relevance (consumption and reset signals from each
row's `extend + bound`) and fills the remaining budget by distance. The
per-resource allocation is a uniform equal-share prior over the active rows plus
an implicit geometric slot -- a single variant-agnostic rule, not a table of
hand-tuned quotas -- so a freshly declared resource is covered before its quota
head has ever been trained. `--learned-candidate-quotas` lets the typed
multinomial policy (trained with the same winner-gated PPO return) *reweight*
that allocation rather than enable it; the schema neighborhood is present either
way. `--candidate-mode geometric` is an explicit ablation that drops the resource
channels and keeps only the k-d-tree distance neighborhood plus the required
depot overlay. Because the fields-off baseline shares the same candidate mode as
the field, field-on-vs-off stays a clean ablation of the field alone. The more
invasive learned edge-scorer stage remains gated on the documented
known-resource noninferiority criterion rather than silently changing topology.

### Compositional field network

The GNN emits one field for each registry resource plus the objective weight.
Node, edge, resource-token, and live-state inputs are normalized to `[0, 1]`;
the decoder is the source of truth for graph dimensions, resource scales, and
active channels. Active resource tokens attend to one another before producing
per-edge resource fields, global resource intensities, a state-conditioned
objective weight, binding predictions, and live-state coupler parameters. The
decoder first installs one objective-only greedy incumbent, so the network's
first input already contains incumbent route positions, forward/backward resource
state, and incumbent-edge indicators. The neural policy does not participate in
initial construction; it starts with perturbation and SRR from that incumbent.
The field is refreshed when the incumbent improves or a new candidate graph is
installed; stagnation ends the current SMDP option without recomputing an
identical graph, while the state coupler keeps responding to load, time, route
progress, and other live variables at each stochastic choice.

The token encoder consumes the native algebra descriptor rather than a
constraint-identity one-hot. Field, multiplier, quota, and token-to-token state
coupler heads are shared across rows, so appending a resource adds no model
parameter. This is a clean `typed_resource_v2` checkpoint boundary: v1 identity
checkpoints are rejected and must be retrained.

### Guaranteed-feasible action space

A unified native decoder covers capacity, time windows, route and tour limits,
backhaul, pickup-delivery, prize quota, open routes, multiple depots, optional
customers, and symmetric or asymmetric costs. It enforces every hard constraint
during greedy construction and refinement, so the learned policy only chooses
among feasible perturbation/refinement moves and can express continuous
preference *before* a violation would occur.

### Reinforcement learning

Training replays one probability ratio per stochastic decoder decision. Option
returns are assigned to rollouts, and inverse decision-count weighting gives
every rollout equal total influence independent of trace length. The update runs
multiple PPO passes per rollout, so the probability ratio departs from one and
clipping engages as the policy moves. The resource multipliers are left ungated
by the binding classifier by default, so RL gradients reach them from step 0
(`--gate-multipliers-by-binding` restores the gate as an ablation), and the
multiplier-to-binding supervision is off by default (`--price-weight 0`) so the
policy's intensities are learned from search outcome.

Auxiliary resource, binding, and feasibility heads are trained as representation
pretraining and downweighted to `0.1` during RL fine-tuning (`--aux-rl-scale`);
they inform the policy but do not define it. Winner-gated Monte Carlo credit
connects consecutive incumbent improvements: the rollout that installs the next
incumbent receives the sampled continuation advantage, preserving temporal
information after within-option POMO centering. An optional progress-conditioned
GAE critic supplies a learned continuation value for longer search horizons. On
HIP/ROCm, training automatically selects the detached-output small-VRAM update;
`--no-smallvram` selects the conventional retained-graph update.

### Counterfactual feasibility learning

At traced states, the decoder labels candidate edges that are immediately masked
and applies configurable sparse look-ahead to currently legal edges. The
feasibility head learns the resulting continuation risk; after auxiliary
pretraining it can guide proposal, perturbation, and SRR sequence energy, with
the prediction detached from policy replay.

### Exact and incremental SRR

SRR evaluates planned capacity, route-limit, tour-limit, backhaul, and prize
changes from cached route summaries. Once a planned sequence is available, these
resource replacements are constant in the number and length of routes.
Time-window cascading lateness and pickup-delivery open-pair maxima use exact
route replay, preserving their full state-dependent semantics. Trace output
reports both summary and replay evaluation counts.

The default repair policy ports DyNACO's bounded local-search mechanics into
this unified evaluator: it scans at most 32 ranked edges per affected node,
accepts the first exact improving move, and uses don't-look bits until a changed
local link reactivates the node. Its hot loop contains the same compact move
families (relocate, swap, 2-opt-star, and intra-route 2-opt), while optional-node,
depot-structure, and pickup-delivery moves remain enabled when their schema
requires them. These are schema-independent scheduling rules, not a
CVRP-specialized move evaluator. Single-route static distance/capacity/prize
plans are certified by the exact planned-resource algebra before replacement;
multi-route boundaries, stateful rows (time windows, backhaul,
pickup-delivery), route/tour limits, and runtime rows additionally use full
route replay. `srr_candidate_limit`,
`srr_first_improvement`,
`srr_dont_look`, and `srr_extended_operators` expose the policy and permit an
exhaustive legacy ablation.

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

Train the constraint-interaction field policy and event-driven decoder with:

```bash
uv run python train.py --n-node 1000 --pretrain-epochs 3 \
  --pretrain-lr 1e-4 --pretrain-aux-scale 1.0 --no-wandb
```

Set `--pretrain-epochs 0` to begin joint PPO training immediately. The aliases
`--pretraining-epochs`, `--pretraining-lr`, and `--pretraining-aux-scale` are
equivalent, and all settings are stored in checkpoints and W&B configuration.

Training uses an epoch-local balanced schedule over `--variants` (the current
training variants by default). Every selected routing composition appears before
another receives an extra rollout, and each PPO accumulation group contains
distinct variants. Add `--curriculum` to phase those variants by resource count;
without it, every selected variant is eligible from epoch 0.

Temporal credit is controlled by `--temporal-credit-weight` (default `0.1`). Set
it to `0` for the local-POMO ablation. The optional critic can be enabled with,
for example:

```bash
uv run python train.py --gae-lambda 0.95 --value-loss-weight 0.5
```

The default `--gae-lambda 1.0 --value-loss-weight 0.0` uses critic-free sampled
Monte Carlo continuation credit.

## Validation and checkpoint selection

Validation covers exactly the same `--variants` selected for training, with no
seen/held-out split. Missing entries fail validation unless
`--allow-missing-validation` is selected. Validation inherits `--n-rollouts` by
default; use `--val-n-rollouts` to give it a separate rollout budget.

PRISM is scored against an **ablation of itself**: before epoch 0 it evaluates
the identical decoder and search with the field ablated to pure distance
(`E = c(e)`), i.e. *PRISM without fields* (`--baseline fields-off`, the default).
Reported quality uses each aligned saved oracle reference when available. Before
training, instances without a saved oracle are solved once by the hand-tuned
classical-proximity decoder under the matched validation budget, and that result
is cached as their reference. The fields-off ablation remains a separate paired
comparison rather than becoming the fallback reference. W&B records per-variant
objective, feasibility, oracle-or-classical reference gap, and paired field
improvement, together with `val_summary/macro_gap`,
`val_summary/macro_improvement`, and `val_summary/macro_score`.

`best.pt` is selected solely by the lowest variant-macro reference gap: average
the normalized per-instance gaps within each variant, then give every variant
equal weight. Feasibility and paired fields-off improvement remain logged
diagnostics but do not enter checkpoint ranking.

The hand-tuned classical-proximity ranking is used as the cached reference only
where a saved oracle is unavailable. `--baseline classical` can also select it
as the paired comparison instead of the default fields-off ablation.

## Evaluation

Run the end-to-end size-100 feasibility and resource-parity gates:

```bash
uv run python tests/urs_one_each.py --iterations 2 --guidance classical
uv run python tests/urs_one_each.py --iterations 2 --guidance field
uv run python tests/urs_screening_parity.py
uv run python tests/urs_one_each.py --rollouts 1 --threads 1 --iterations 2 \
  --verify-incremental-srr
```

`--guidance field` exercises the typed-field interface with a neutral field.
Evaluate the trained policy against PRISM with its field ablated (distance-only)
on every benchmark composition with:

```bash
PYTHONPATH=src uv run --no-sync python test.py \
  --checkpoint pretrained/best.pt --variants all --iterations 16 \
  --csv results/prism_vs_fields_off.csv
```

By default, `test.py` evaluates all 110 benchmark variants using the first eight
saved instances of each, reporting overall results plus separate `SEEN` and
`HELDOUT` splits. The learned policy and its fields-off ablation share the same
paired per-instance seed, candidate budget, rollout count, post-bootstrap
iteration count, and oracle reference, so the only difference is the learned
field (`--baseline` selects the reference; `fields-off` is the default). Dynamic
field refinement is enabled by default. For a focused subset, use
`--val-size 1 --variants tsp,cvrp,cvrptw`; use `--static-field` to evaluate one
frozen field for the full solve. The report and optional CSV record the data
split, mean objective, field improvement, oracle gap, runtime, field mode, and
neural evaluation count for every variant.

## Large-scale inference

Run the size-10,000 inference gate with either distance-only or learned guidance:

```bash
uv run python tests/scale_smoke.py --n-node 10000
uv run python tests/scale_smoke.py --n-node 10000 --checkpoint pretrained/best.pt
```

Large Euclidean instances retain coordinates instead of allocating a dense
distance matrix. Candidate construction combines bounded KD-tree neighborhoods
with resource-indexed pools, and the GNN uses direct single-graph reductions.
Training enables activation checkpointing automatically from `n=1000`, while
inference remains checkpoint-free.

The feasibility look-ahead defaults to two steps and its search penalty to
`10.0` objective units. Configure them with `--feasibility-lookahead-depth` and
`--feasibility-risk-penalty`.
