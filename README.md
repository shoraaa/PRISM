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
E_tilde(e | s) = c(e) / s_obj + delta_obj(e)
         + sum_r lambda_r(s) * [field_r(e) + a_r(e)]
         + kappa * q(e),
```

and samples the next node from `softmax(-beta * E_tilde)`. The native decoder
supplies the exact objective and scale; the remaining energy terms are learned:

- `c(e)` is the exact canonical objective edge cost, `s_obj` is a row-centered
  RMS objective scale, and `delta_obj(e)` is its signed,
  objective-family-conditioned dimensionless learned residual;
- `field_r(e)` and `a_r(e)` are the learned per-edge field and additive term for
  resource `r`, expressed directly in dimensionless energy units;
- `lambda_r(s)` is the learned, live-state-modulated intensity of resource `r`,
  a Lagrangian-style multiplier shaped by search reward rather than supervision;
- `q(e)` is an optional learned continuation-risk potential.

Analytic resource pressure is **not** automatically charged in the energy; it is
exposed to the GNN only as an input edge feature. Resource fields and the
objective residual initialize exactly at zero, so the initial policy is the
plain objective `E = c(e)` and every deviation from it must be learned. This
initial policy is distinct from the fields-off baseline, which flattens the
field to an identical value on every edge (`E = 1`).

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

The decoder uses one `resources` registry for compiled constraint kernels and
declarative kernels alike. A declarative row names an operator primitive,
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
positions are excluded. Resource tensors grow with the registry, while field
channels retain their model indices so the distance-only control path retains
exact behavior.

Native execution requires an explicit normalized schema. `constraints`,
`objective`, `depot_count`, `multi_route`, and `open_route` must be declared;
`name` is optional metadata and never supplies solver semantics. Incomplete
name-only inputs are rejected. `prism_decoder.normalize_problem_schema(problem)`
exposes the exact normalized dictionary and fills only schema-independent
numeric defaults. `problem_data.py` owns benchmark-name-to-schema conversion.

Constraints are registered as compiled kernels with their schema
name, stable field-channel slot, resource operator, and search capabilities
(route/solution state, ordering, reversal sensitivity, and relations). The
decoder selects its active kernel set from the declared `constraints`; generic
operators consult those capabilities instead of variant-name lists. The kernel
bodies remain specialized native code so capacity, time windows,
pickup-delivery, backhaul, and multi-depot search retain their existing speed
and exact behavior. Algebra-declared resources append generic `extend + bound`
kernels to the same registry after the seven model field rows.

Candidate construction is itself schema-driven, so it extends to a new resource
with no per-variant tuning. In the default `schema` mode, native admission ranks
edges by a registry-derived relevance (consumption and reset signals from each
row's `extend + bound`) and fills the remaining budget by distance. The
per-resource allocation is a uniform equal-share prior over the active rows plus
an implicit geometric slot -- a single variant-agnostic rule, not a table of
hand-tuned quotas -- so a freshly declared resource is covered before its quota
head has ever been trained. `--learned-candidate-quotas`
(**EXPERIMENTAL** — off by default; may be removed or evolved in the future)
lets the typed
multinomial policy (trained with the same winner-gated PPO return) *reweight*
that allocation rather than enable it; the schema neighborhood is present either
way. `--candidate-mode geometric` is an explicit ablation that drops the resource
channels and keeps only the k-d-tree distance neighborhood plus the required
depot overlay. Because the fields-off baseline shares the same candidate mode as
the field, field-on-vs-off stays a clean ablation of the field alone. The more
invasive learned edge-scorer stage (**EXPERIMENTAL**) remains gated on the
documented known-resource noninferiority criterion rather than silently changing
topology.

### Compositional field network

The GNN emits one field for each registry resource plus a signed
objective-energy residual.
Node, edge, resource-token, and live-state inputs are normalized to `[0, 1]`;
the decoder is the source of truth for graph dimensions, resource scales, and
active channels. Active resource tokens attend to one another before producing
per-edge resource fields, global resource intensities, the objective residual,
binding predictions, and live-state coupler parameters. The native decoder adds
the residual to normalized objective edge cost before applying the single
sampling temperature, so native sampling, PPO replay, and SRR share one
dimensionless energy formula. The decoder first installs one objective-only
greedy incumbent, so
the network's
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
parameter. This is a clean `typed_resource_v5_scale_equivariant_energy` checkpoint
boundary: older checkpoints are rejected and must be retrained because the
policy energy units and objective-residual units changed.

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
pretraining. Carrying their loss into RL fine-tuning (`--aux-rl-scale`) is
**EXPERIMENTAL** and off by default (`--aux-rl-scale 0`); when enabled they
inform the policy but do not define it. Winner-gated Monte Carlo credit
connects consecutive incumbent improvements: the rollout that installs the next
incumbent receives the sampled continuation advantage, preserving temporal
information after within-option POMO centering. An optional
(**EXPERIMENTAL** — off by default, `--value-loss-weight 0`)
progress-conditioned
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
this unified evaluator: it energy-ranks each bounded candidate row, scans the
complete row, evaluates the full relocate, Or-opt, exchange, swap, 2-opt-star,
and intra-route 2-opt neighborhood, and accepts the best exact improving move.
Optional-node, depot-structure, and pickup-delivery moves remain enabled when
their schema requires them. These are schema-independent scheduling rules, not
a CVRP-specialized move evaluator. An explicit route-structure check plus exact
planned-resource summaries certify single- and multi-depot capacity,
time-window, route/tour-limit, and prize changes before replacement. A
route-order certificate handles the capacity-to-empty transition before the
backhaul-only suffix. Pickup-delivery plans replay pair identities only over
affected route pieces and combine their maximum open-pair pressure with cached
unaffected routes.
Custom reset, battery, and other runtime algebra rows replay only their own
state over the materialized candidate. Depot structural moves and any candidate
whose load-order certificate cannot be proven still retain full replay.

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
it to `0` for the local-POMO ablation. The optional critic
(**EXPERIMENTAL** — off by default; may be removed or evolved in the future) can
be enabled with, for example:

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
the identical decoder and search with the field flattened to an identical value
on every edge (`E = 1`), i.e. *PRISM without any field guidance*.
Reported gap uses only each aligned saved oracle reference; instances without a
saved oracle remain in validation for cost and feasibility metrics but are
excluded from gap aggregation. No classical fallback reference is cached, so
changing the search budget cannot change the gap target. The fields-off ablation
remains a separate paired comparison rather than becoming a reference.
W&B records per-variant objective, feasibility, saved-oracle reference gap, and
paired field improvement, plus saved/missing reference counts and gap coverage,
together with `val_summary/macro_gap`,
`val_summary/macro_improvement`, and `val_summary/macro_score`.

`best.pt` is selected by the lowest mean canonical best cost across feasible
validation instances. The oracle-only macro gap remains a reporting diagnostic;
feasibility and paired fields-off improvement do not enter checkpoint ranking.

The fields-off decoder is the sole paired comparison baseline.

## Evaluation

Run the end-to-end size-100 feasibility and resource-parity gates:

```bash
uv run python tests/urs_one_each.py --iterations 2
uv run python tests/urs_one_each.py --iterations 2 --guidance field
uv run python tests/urs_screening_parity.py
uv run python tests/urs_one_each.py --rollouts 1 --threads 1 --iterations 2 \
  --verify-incremental-srr
```

`--guidance field` exercises the typed-field interface with a neutral field.
Evaluate the trained policy against PRISM with its field ablated to a flat,
identical value (`E = 1`) on every benchmark composition with:

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
field. Dynamic field refinement is enabled by default. For a focused subset, use
`--val-size 1 --variants tsp,cvrp,cvrptw`; use `--static-field` to evaluate one
frozen field for the full solve. The report and optional CSV record the data
split, mean objective, field improvement, oracle gap, runtime, field mode, and
neural evaluation count for every variant.

TSPTW is available as an explicit external, held-out probe without changing the
default 110-variant evaluation:

```bash
PYTHONPATH=src uv run --no-sync python test.py \
  --checkpoint pretrained/best.pt --variants tsptw
```

This defaults to the first eight hard, size-100 instances and paired LKH costs
from `baselines/CaR-constraint/data/TSPTW`. Select another CaR dataset with
`--tsptw-size` and `--tsptw-hardness`, or use fresh instances from CaR's own
generator with `--tsptw-source generator`. Generated instances have no oracle
reference, so they are compared only against the paired fields-off decoder. For
hard instances, both decoders start from the same feasible tour embedded by
CaR's generator; the LKH tour is reference-only. Custom saved hard datasets must
pass their generation seed through `--tsptw-dataset-seed` (CaR defaults to
2025). CaR's easy and medium generators do not retain a guaranteed witness
tour, so those modes use the evaluator's ordinary field construction and may
serve as a stricter construction-feasibility diagnostic.

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

The feasibility look-ahead defaults to two steps. Its learned risk classifier
is trained and reported, and its detached search penalty defaults to `1.0`.
A measured ablation with
`--feasibility-risk-penalty 0` disables this ranking term while retaining exact
decoder legality and risk-head training. Configure these controls with
`--feasibility-lookahead-depth` and `--feasibility-risk-penalty`.
