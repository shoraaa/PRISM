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
E_tilde(e | s) = c(e) / s_obj + field_obj(e)
         + sum_r lambda_r(s) * field_r(e),
```

and samples the next node from `softmax(-beta * E_tilde)`. The native decoder
supplies the exact objective and its scale; every term added to them is learned:

- `c(e)` is the exact canonical objective edge cost and `s_obj` is a
  row-centered RMS objective scale, so the anchor is dimensionless and
  invariant to a positive rescale of the declared coefficients;
- `field_obj(e)` is the learned **objective channel**: a signed per-edge
  correction to that anchor, produced by the same resource-token encoder,
  attention and field head as every constraint row;
- `field_r(e)` is the learned per-edge field for resource `r`, expressed
  directly in dimensionless energy units;
- `lambda_r(s)` is the learned, live-state-modulated intensity of resource `r`,
  a Lagrangian-style multiplier shaped by search reward rather than supervision.

The objective's *intensity* stays a pinned unit anchor while its *geometry* is
learned. A graph-level scalar on the objective is exactly a search temperature,
and PPO took that shortcut when the slot was free (about 1.0 -> 3.2 on CVRP)
instead of learning edge preferences; on a bare TSP, where the objective is the
only channel, it would be the only thing an intensity head could do.

The objective is a channel and not a bolt-on head for two reasons. It is the
one row every problem declares, so it is the only channel that can carry
transfer between compositions -- capacity's field says nothing about a time
window, and neither says anything about a plain tour. And a problem that
declares no constraint (`tsp`, `atsp`) has an empty resource registry: with an
analytic-only objective its field stacked to `[E, 0]`, its sole multiplier was
the pinned objective constant, every model output was discarded, and `logp` was
constant in theta -- measured gradient norm exactly `0.0`. Those variants
trained nothing and ran as distance-ranked construction plus 2-opt.

Analytic resource pressure is **not** automatically charged in the energy; it is
exposed to the GNN only as an input edge feature. The shared field head is
zero-initialized, so `field_obj` and every `field_r` are exactly zero at step 0:
the initial policy is the plain objective `E = c(e)` and every deviation from it
must be learned. This initial policy is distinct from the fields-off baseline,
which flattens the field to an identical value on every edge (`E = 1`).

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
    "terms": [
        {"edge_attribute": "distance", "coefficient": -1.0,
         "op": "add", "phase": "before_bound"},
        {"value": {"scalar": "battery_capacity"},
         "op": "assign", "phase": "after_bound",
         "when": "reset_arrival", "at_depot": True,
         "at_nodes": {"node_attribute": "charger"}},
    ],
    "bounds": [{
        "lower": 0.0,
        "check": "transition",
        "horizon": "return_construction",
    }],
}]
```

A bound also declares its **horizon** -- how far ahead it is projected before it
is tested. `transition` tests the arrival state only. `return` additionally
projects the cheapest depot return leg and is enforced as a feasibility
condition; it is sound whenever the row cannot be replenished mid-route, and it
is what route and tour limits require. `return_construction` applies the same
projection during construction only, as a stranding guard for rows that *can*
be replenished at an interior node (a charger, a rest stop), where a closed
route may legitimately reach the depot through a reset and so must not be
rejected. Before this was declarable, the projection lived inside the compiled
route- and tour-limit kernels and inside a signature test over spec fields; it
is now a property of the row like any other.

The binding resolves named arrays once, and native `extend + bound` replay then
drives construction masks, incumbent validation, resource labels, pressure
features, and live state. Each row also produces a 20-property row vector and a
variable-size set of 24-property term vectors. The properties describe behavior
(source locality, read point, algebraic operation, phase, trigger, gate,
coefficient, bound, scope, and horizon), not enum identities. Shared term
weights pool each set to a fixed-width resource type, so adding an operation
does not add an input dimension. Names and registry positions are excluded.

### Compiled kernels are fast paths, not a private semantics

A compiled constraint kernel is an execution specialization, not a second way of
defining what a constraint means. Every kernel publishes the declarative
resource row it implements through `Decoder.resource_declarations`, and that row
-- never a switch over the kernel enum -- is what the row/term properties are
derived from. `tests/python/test_resource_algebra_equivalence.py` rebuilds
each declarable constraint from its published row alone, with the compiled
kernel dropped from the schema, and requires the identical solution. Capacity,
route limit, tour limit, and prize quota pass that test, so for those the
network cannot tell a compiled kernel from a declared row, and the generality
claim rests on the algebra rather than on the kernel table.

Every compiled kernel is expressible. Two abstain (`declared: false`) on
instances that leave the domain their declaration covers, and the equivalence
test encodes that rule rather than a fixed list, so a new abstention has to be
justified there before the suite passes:

| kernel | abstains | why the language cannot express it |
| --- | --- | --- |
| `capacity` | a class order supplied only as another declared row | publication cannot infer that row's opening-load convention from the problem flags |
| `time_windows` | a row with optional resets is active | a break costs wall time, so one row's reset must increment another row's state |

Signed demand, including guarded depot opening/reload behavior, is now fully
factored into terms. The remaining wall-time case is genuinely cross-row. One
deviation is known and narrow: the compiled prize-quota
kernel also lets a route close once every customer has been visited, so an
instance whose total prize cannot reach the quota stays feasible for the kernel
while the declared bound rejects it.

### The tropical operator

The `max_plus` semiring expresses resources whose arrival value is raised to a
per-node floor before it is bounded -- the classic time REF. Time windows are
declared by composing `add` and `join` terms:

```python
{"name": "time", "operator": "affine_accumulator", "semiring": "max_plus",
 "terms": [
   {"edge_attribute": "distance", "op": "add", "phase": "before_bound"},
   {"node_attribute": "tw_start", "op": "join", "phase": "before_bound"},
   {"node_attribute": "service_time", "op": "add", "phase": "after_bound"}],
 "bounds": [{"upper": {"node_attribute": "tw_end"},
             "check": "transition", "horizon": "return"}]}
```

The `join` term is the tropical floor applied to the arrival value. The service
term is charged *after* the bound, so it delays the next transition
without being tested against the arriving node's own window. A bound side may now
be a named node attribute instead of a scalar, which is what a window that varies
node by node requires. Resource `direction` accepts only `forward`:
`backward` and `bidirectional` name real REF directions and have property-space
coordinates, but execution extends a route forward only, so declaring them is rejected
rather than silently run under a forward extension.

### The precedence family

Backhaul ordering and pickup--delivery are not resource extension functions:
nothing accumulates, and admissibility is a predicate over what the route has
already served. They are declared through a third operator whose two relation
forms cover both, and whose properties describe behavior rather than kernel
names -- a declared precedence row is byte-identical in its properties to
the compiled kernel it replaces:

```python
{"name": "delivery_order", "operator": "precedence", "relation": "pairwise",
 "scope": "route", "predecessor": {"node_attribute": "required_before"}}

{"name": "linehaul_first", "operator": "precedence", "relation": "class_order",
 "scope": "route", "class": {"node_attribute": "haul_class"}}
```

`pairwise` requires each node's declared predecessor to be served first;
`class_order` requires classes to be served in non-decreasing order. The inverse
of `predecessor` is derived, never declared twice, and a node may take part in
one pairwise relation only -- generic operators move a pair together and assume
a single partner.

`scope` is executed rather than decorative: a `route`-scoped relation must be
resolved before the route closes, which is what confines a pair to one route,
while a `solution`-scoped one only requires the predecessor to come first
somewhere and so may span routes. `tour` is rejected: it is indistinguishable
from `solution` in execution, and accepting it would run solution semantics
under another name. A row is also rejected if it declares a key its operator
cannot act on -- `bounds` on a precedence row, `predecessor` on an accumulator --
since a silently ignored key is how a declaration ends up meaning something
other than it says.

Neither kernel is the general case. Pickup--delivery is a depth-1 pair and
backhaul a two-class order, so an arbitrary precedence chain `a -> b -> c` is a
structurally new instantiation of the same declared relation, reachable with no
new kernel, output head, or variant name;
`test_a_precedence_chain_runs_although_no_kernel_expresses_one` exercises one.

Generic operators consult `relational()` -- the consumer of
`KERNEL_RELATIONAL`, which previously had none -- to decide whether a route may
be cut between two linked nodes, and `relation_partner()` to move a pair
together. Neither names a pickup or a delivery, so a declared relation
constrains the move set exactly as the compiled kernel always has.

The model inputs are generic too. Live state is **one vector indexed by
resource**, in construction and in the model alike. `State` no longer carries
private per-constraint scalars: `load`, `current_time`, `route_distance`,
`collected_prize`, `open_pickups`, and the backhaul latch are accessors onto
`resource_state`, each in the convention its published declaration states, and
the registry always holds every field channel so each slot is a permanent home.
A compiled kernel keeps its specialized arithmetic; what it no longer keeps is a
second copy of the state. The live-state feature is read from the declaration
rather than from a per-channel formula. The per-node flags are roles -- *opens an obligation* and
*requires one* -- rather than *is a pickup* and *is a delivery*, and the prefix
and suffix obligation counters come from the same `open_relation_delta` the piece
cutter uses. A declared pairwise row therefore
produces node features identical to the kernel it replaces, and a chain's middle
node correctly holds both roles at once. A node may take part in one pairwise
relation across the whole registry, compiled kernel included, because
`relation_partner` returns a single partner.

Per-node attributes are indexed by resource rather than named after a
constraint. `node_resource_features` is `[N, resource_count, 5]` -- the node
increment split by sign, the tropical clamp, the finite bound, and the departure
term -- read off each row's published algebra. `net.py` gathers the arrival
node's block onto each edge and feeds it through the same shared
`resource_edge_projection` that already carried per-edge pressure and reset
events, so the width grows with the number of resources and no weight is
per-resource. Demand, window start and end, and service time therefore no longer
have node-feature slots of their own: `NODE_FEATURE_COUNT` is 19, and what
remains there is geometry, depot structure, objective coefficients, relation
roles, and incumbent prefix/suffix state.

This is the one place that reads `published_algebra` rather than
`declared_algebra`, so an abstaining kernel still contributes its node
attributes. That is sound because in all three abstention cases it is the
*extension rule* that escapes the language -- a state-dependent reload, a
cross-row duration, an exempt class -- never the per-node quantities. Signed
demand covers 64 of 110 benchmark variants, so the alternative would have blanked
demand across most of the suite.

Two consequences follow from the operator rather than from a table. Search
capabilities -- route or solution state, order sensitivity, reversal sensitivity,
and the relational flag -- are derived from each row's algebra by
`derived_capabilities`: a tropical clamp, an interior reset, or a mixed-sign
increment each destroy order invariance.

And resource pressure is computed from the published declaration for *every*
kernel that has one, not from a per-channel formula. An accumulator's pressure
includes the depot return leg when it declares a return horizon; a tropical row's
is the wait-plus-warp quantity read off its clamp, bound, and departure fields; a
precedence row's is the share of its relations an edge puts in the wrong order.
The hand-written `analytic_resource_pressure` table now runs only for a kernel
that abstains. `test_pressure_comes_from_the_declaration_not_the_kernel` pins all
seven channels: without it a declared route-limit row reported half the pressure
of the kernel it is otherwise equivalent to, because the generic formula ignored
the declared horizon.

All four resource replay paths -- the construction mask, the SRR trial validator,
incumbent evaluation, and the resource report -- share a single
`extend_declared`, so an operator primitive cannot be wired into one path and
silently missed by the others.

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

The GNN emits one field for each registry resource.
Node, edge, resource-token, and live-state inputs are normalized to `[0, 1]`;
the decoder is the source of truth for graph dimensions, resource scales, and
active channels. Active resource tokens attend to one another before producing
per-edge resource fields, global resource intensities, binding predictions, and
live-state coupler parameters. The native decoder uses normalized exact
objective edge cost before applying the single sampling temperature, so native
sampling, PPO replay, and SRR share one
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

The token encoder consumes pooled row/term property sets rather than a
constraint-identity one-hot. Field, multiplier, quota, and token-to-token state
coupler heads are shared across rows, so appending a resource adds no model
parameter. This is a clean `typed_resource_v13_pooled_terms` checkpoint boundary:
older checkpoints are rejected and must be retrained because the model input
contract changed.

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
[`problem_data.py`](problem_data.py). Benchmark files live under
`datasets/benchmarks/<variant>/`, one directory per variant holding every scale
generated or converted for it. Evaluation consumes the canonical neutral tensor
dictionary named `<variant><scale>_prism.pt`; retained `.pkl`, `.txt`, and
foreign-key `.pt` files are import provenance, not runtime dataset formats.
Materialize or verify the canonical size-100 suite with:

```bash
uv run python scripts/normalize_benchmarks.py --scale 100
```

Persist a fresh full size-1000 suite (16 instances per variant, capacity 200,
seed 1234) with:

```bash
uv run python scripts/generate_benchmark_suite.py --size 1000 --count 16 \
  --capacity 200 --seed 1234 --force
```

Generation keeps the existing PRISM distributions unchanged. Each artifact has
an adjacent JSON sidecar recording provenance, the absence of a paired oracle,
and generator-certifiable infeasibility counts; the command prints those counts
instead of filtering or rescaling samples.

Evaluation CSV rows are synchronized after every completed in-process instance,
so an interrupted run retains its latest measurement. Resume the same command
and checkpoint by adding `--resume` while keeping the same `--csv` path:

```bash
uv run test.py --checkpoint checkpoint.pt --csv results/eval.csv ...
uv run test.py --checkpoint checkpoint.pt --csv results/eval.csv --resume ...
```

Completed variant/method batches are skipped. An interrupted partial batch is
rerun, but already persisted instance rows are not duplicated. `--cached`
remains the separate facility for importing baseline rows from another CSV.

For memory-bounded large-scale evaluation of constructive neural baselines,
pass `--aug 1`. This shared override makes both URS and CCL evaluate one
augmentation instead of their native test-time defaults, and is included in
their CSV configuration and cache identity. Because augmentation changes the
evaluation protocol, do not resume a CSV that was started without the same
`--aug` value; start a new output CSV instead.

Each variant directory may also hold per-instance oracle costs. The root can be relocated with
`PRISM_DATASET_DIR` or selected per run with `--dataset-dir`, and the scale is
chosen with `--dataset-scale`.

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
