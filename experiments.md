# Experiment design for the eight review comments

`train.py` and `test.py` answer one question: *what is the final gap?* Every comment
below asks a different question — about **what the model knows**, **why the design
has the shape it has**, and **where the claim stops holding**. Those need
instrumentation that reports on internal quantities and on counterfactual inputs,
not on end-to-end objective values.

The plan is organised so that **one new harness (`scripts/probe_field.py`) unlocks
six of the experiments**. Build that first.

## Current implementation boundary (v13)

The live schema is `typed_resource_v13_pooled_terms`. Resets, opening values,
arrival floors, departure charges, checkpoints, and restores are executable
terms with independent `(op, phase, when, gate)` coordinates. The neural input
is no longer a flat descriptor: each resource has 20 row properties plus a
variable-size set of 24 term-property rows, mapped with shared weights and
pooled to a fixed 32-dimensional resource type. New term counts therefore do
not create new input slots. `scripts/probe_descriptor.py` now intervenes on
these row/term properties and its census compares whole term sets. All earlier
v6/v11 descriptor results below are retained as historical checkpoint evidence,
not claims about the current input contract.

---

## 0. The probe harness — **built**

`scripts/probe_field.py` (dump) and `scripts/probe_report.py` (E4a/E4c/E4f/E7b
analyses) exist. Because `../epoch235_015.pt` is schema
`typed_resource_v6_objective_coeff_algebra` while the working tree has moved to
`v11` (descriptor width 32 → 36, different node/edge feature layout), the probe
imports model code from `--code-root` rather than from this repository. A v6
checkout with its own compiled extension lives at `../PRISM-v6`
(`git worktree add ../PRISM-v6 1ef28d2d` + `python setup.py build_ext --inplace`);
the main tree and its running training are untouched.

```
.venv/bin/python scripts/probe_field.py \
    --checkpoint ../epoch235_015.pt --code-root ../PRISM-v6 \
    --dataset-dir datasets/benchmarks \
    --variants tsp,op,cvrp,cvrpb,cvrpl,ocvrp,cvrptw,cvrpbl,cvrpbtw,cvrpltw,cvrpbp,pdcvrp,cvrpbltw,acvrptw,mdcvrptw \
    --instances 8 --states 16 --out results/probe/v6_e4.csv
.venv/bin/python scripts/probe_report.py results/probe/v6_e4.csv
```

Two dumps are kept: `v6_e4.csv` (`--incumbent field`, reproduces inference) and
`v6_e4_distinc.csv` (`--incumbent distance`, zero-field control bootstrap).
Any comparison between the energy and the incumbent's own successor is circular
on the first and must use the second.

**Original design note.** Load a checkpoint, walk instances/variants, snapshot
the decoder at *K* sampled decision states per instance, run the net **once** per
state, and dump a tidy table. No search, no oracle, minutes not hours.

One row per (variant, instance, state, candidate edge):

| group | columns | source |
|---|---|---|
| identity | `variant, instance, state_id, step_frac, from, to` | decoder |
| inputs | `distance, objective_cost, resource_pressure[r], live_state[r]` | `decoder.resource_pressure`, `incumbent_live_state` |
| model out | `energy, residual[r], additive[r], multiplier[r], feasibility_logit, binding_logit, objective_residual` | `net.forward` |
| labels | `mask_feasible` (1-step), `lookahead_label` (see E4b), `in_reference_solution` | `decoder.mask(prefix)`, `evaluate`, reference route |
| context | `resource_descriptors[r]`, `active_channels`, `n_active_resources` | `decoder.resource_descriptors` |

Everything the C++ side needs is already exported (`binding.cpp:1474-1514`):
`mask`, `evaluate_resources`, `resource_pressure`, `edge_index`, `edge_offsets`,
`resource_descriptors`, `resource_declarations`. `net.build_decoder_data` +
`ConstraintFieldNet.forward` give the model side. **No C++ changes required** for
E1a, E1c, E2b, E4a–E4d, E7b, and the λ-magnitude answer to reviewer Q1.

Then E4/E1/E7b become pandas queries over one parquet file instead of six eval runs.

---

## Findings so far (v6 checkpoint `epoch235_015.pt`)

Measured on `../PRISM-v6`, 8 instances x 16 states over 15 variants unless noted.
Preliminary sample sizes -- enough to see the shape, not enough for a table.

**The v6 descriptor collapsed to an operator one-hot, and v11 fixes it.**
`scripts/probe_descriptor.py --census` compares the rows two co-active resources
receive. Under v6, 2 of 7 co-active pairs are *byte-identical* (capacity and
route-limit in `cvrpbl`/`cvrpbltw`), the mean pair differs in 3.6 of 32 slots,
and on 3 of 7 pairs the operator one-hot is the **only** separating component.
The cause is that compiled kernels never populated the declared row, so
increment source, increment sign, normalization and the depot reset were written
as constant zero -- structurally dead, not merely unused. Under v11 no pair is
identical, the mean pair differs in 11.7 of 36 slots, and 5-7 components separate
each pair.

This matters for how the paper argues, not just for the code. Reviewer weakness 4
("the algebra might simply be a fixed embedding layer for 10 known constraint
types") was substantially *correct* for the evaluated model, and the Table 5
`Identity` ablation was therefore a much weaker contrast than it reads as: it
replaced a near-one-hot with a learned one-hot. Both the descriptor census and
the `Identity` ablation should be re-measured on v11, where the claim is true and
newly checkable. The census needs no checkpoint, so it can run the moment a v11
decoder builds.

**Descriptor interventions (E1a/E1b) are only informative on v11.** With the v6
layout, zeroing the operator component moves the candidate ranking most
(Kendall tau 0.937, `d_auc` +0.031) and everything else is a smaller
perturbation of a constant input; shuffling is an exact no-op for 9 of 12
components because the co-active rows already agree. The report now prints an
`input_changed` column so a no-op cannot be misread as model indifference.

**The constraint knowledge is in the resource term, not the composite energy
(E4a).** Scored on unvisited non-depot candidates only, the resource energy beats
the distance null in 12 of 14 variants (`cvrp` +0.18, `cvrpl` +0.24, `cvrptw`
+0.20, `acvrptw` +0.40 AUROC). The *total* energy is often below 0.5, because the
objective term is 1.4-7x larger and distance itself anti-correlates with
feasibility under time windows. Stable across both incumbent sources. The
defensible claim is "the resource energies encode constraint semantics", not
"the heatmap is feasibility-aware".

**The energy adds little feasibility information over the features it is handed
(E4c).** Resource-pressure features alone reach AUROC 0.62-0.88; adding the
learned energy moves it by <=0.03 on every multi-resource variant, and only
+0.11 to +0.17 on `op`/`cvrp`/`cvrpl`. Predicting one-step feasibility is not the
field's job, so this is evidence about what the energy encodes rather than a
defect -- but it argues the value/cost-shaping story over the feasibility-
awareness story.

**Intensities are large and nearly state-independent (E4f, reviewer Q1).**
lambda_r(s) sits at 6.9-11.8 with coefficient of variation 1.5-10.5%. Magnitude
alone is not meaningful -- the field absorbs scale, and the resource term is
14-52% of |total energy| -- but the near-constancy is what bears on "value, not
penalty". The redeeming detail: lambda's state variation correlates positively
with live resource state (r = 0.20-0.77), so state-dependent pricing is doing
something real at +/-8% amplitude, consistent with the ablation showing static
pricing costs +0.77 gap points.

**The continuation-risk head is close to inert at inference.** Its logits vary by
~0.04 across candidates and its own feasibility AUROC is 0.37-0.73 with no clear
pattern. Worth checking against E2b before keeping it in v11.

### v11 checkpoint (`pretrained/original/best.pt`, epochs 4 and 7)

The harness runs unchanged on v11 once the CSV is sized for the widest registry:
v11 carries only the resources a variant activates (1 for `cvrp`, 2 for
`cvrptw`) where v6 always held all seven channels, so narrower variants leave
their trailing per-resource columns blank -- blank, not zero, because an absent
channel is not an inactive one. v6 output is byte-identical after the change.

This checkpoint is 7 epochs into a 1000-epoch run at `val_gap` 1.045, against
0.1495 for the v6 model, so a null result here cannot distinguish "component
unused" from "not learned yet". Three things are still worth having:

**The descriptor discriminates far more, confirming the census behaviourally.**
Under v6 a shuffle was a real intervention for only 3 of 12 components; under
v11 it bites for 8 of 13, and zeroing is a real intervention for 12 of 13
against v6's 9 of 12. `reserved` (the slot that held v6's constraint-name bit)
is a verified exact no-op in both, which validates the measurement.

**No component ordering yet.** At epoch 4 and again at epoch 7 every component
sits at `d_energy` 0.004-0.019 and Kendall tau 0.99+, with no movement between
the two checkpoints. Re-run at a converged checkpoint; the numbers are stored per
epoch so the series is the deliverable, not any single row.

**lambda_r(s) is ~0.6 early and ~10 when trained -- the large intensities are
learned.** v11 at epoch 4 prices every resource at 0.56-0.79; the trained v6
model prices them at 6.9-11.8. That inflation over training is a sharper answer
to reviewer Q1 than any single snapshot, and it does not favour the paper's
framing: an intensity that grows by an order of magnitude during training behaves
more like a penalty weight finding its level than like a value on a fixed scale.
Worth plotting lambda against epoch off the saved checkpoints.

**The instrument distinguishes trained from untrained.** On v11-epoch-4 the E4a
`lift_energy` is within +/-0.011 of zero on 13 of 14 variants and `lift_resource`
is negative on 6 -- the learned energy is still essentially the distance ranking.
The trained v6 model shows `lift_resource` positive on 12 of 14, up to +0.40. A
measurement that separates those two is measuring the field, not an artifact of
the probing procedure.

One correction to the earlier read: v11's continuation-risk head is *not* more
informative than v6's. Its higher `auc_risk_head` values sit on an essentially
constant logit (within-state spread 0.016 versus v6's 0.020), so the AUROC there
is noise on a flat signal. Both heads look close to inert at inference.

### Training support: the variable that explains the rest (v6)

Census over all 22 training variants: only **18 of 32 descriptor slots are ever
nonzero during training**. The other 14 -- two-sided bounds (7), check phases
9/10, directions 12/13, scopes 15/16, depot reset (18), event reset (19), node
and edge increment source (23/24), normalization (30), event density (31) -- are
constant zero in every training row, so those input dimensions of the descriptor
MLP receive no gradient and their weights stay at initialization.

Now put the EVRP probe against that. Battery is a *declared* row, not a compiled
kernel, so unlike the training constraints its descriptor is richly populated: 12
active slots, differing from capacity in 8 (lower vs upper bound, event vs no
reset, edge vs node increment, negative vs positive sign, plus normalization and
event density). **4 of those 12 -- slots 19, 24, 30, 31 -- were never nonzero in
training.** Whatever the model computes from them is an untrained projection.

This cuts both ways and both matter:

- The transfer is **not** explained by the model reading battery's novel algebra.
  It cannot be: those directions were never trained.
- It *is* consistent with battery landing inside the trained region on the slots
  that carry it -- accumulator (0), lower bound (5), negative increment (27) --
  values the model did see, via the prize-quota lower bound and backhaul's signed
  demand. Plus the shared per-resource machinery and edge-level pressure features
  u_r(e), which are fully trained and resource-agnostic.

So the defensible mechanism is compositional generalization **over descriptor
field values already seen in training**, carried by shared machinery -- not an
open algebra in which any new declaration works. That is narrower than the paper
claims and stronger than nothing, and unlike the current claim it is falsifiable.

**It also makes a prediction worth testing**, and it is the sharp version of E1c:
rank a new resource by *how many of its active slots fall outside the training
support*, and transfer quality should degrade with that count. Battery scores 4.
A two-sided-bound resource (slot 7), a terminal-check resource (9/10), or a
solution-scoped one (15/16) would score higher and should transfer worse. If they
do not, the descriptor matters even less than this analysis suggests and the
credit belongs almost entirely to u_r(e).

The cheap fix for v11 is the one already noted in the algebra memory:
domain-randomize declarative rows during training so the descriptor's support is
not a 18/32 subspace. Until that is done, "unseen resource" means "unseen
combination of seen descriptor values", and the paper should say so.

---

## 1. Why nine components? How general is the representation?

The 9-tuple is `(o, d, s, x⁰, σ, δ, b, χ, ζ)` (paper Eq. 2), realised as the
36-slot descriptor in `decoder.cpp:build_resource_descriptors`. Three experiments,
in increasing order of persuasiveness.

### E1a — Descriptor-field sensitivity (cheap, no retraining)
Group the 36 slots into their 9 semantic fields. For each field, at **inference
only**, either zero it or randomise it across resources, then measure:
1. mean |Δ energy| per candidate edge, normalised by the energy's own std;
2. Kendall-τ between the perturbed and unperturbed edge rankings;
3. end-to-end gap degradation on a 20-variant subset.

Produces a 9 × resource-family sensitivity heatmap. A field whose ablation is inert
is a field you should delete — reporting that you *checked* is what makes "nine"
a claim rather than an accident of implementation.

### E1b — Counterfactual field-flip (the semantic version of E1a)
Flip **one** descriptor field on a resource while the decoder's actual algebra stays
unchanged: tell the model a capacity row is `deplete` instead of `accumulate`, or
`tour`-scoped instead of `route`-scoped, or reset-free instead of depot-reset. Measure
whether the energy moves *in the semantically correct direction* (e.g. claiming
depot-reset is absent should raise the energy of long multi-route continuations).

This is the difference between "the model uses the bits" (E1a) and "the model uses
the bits *as their semantics dictate*" (E1b). The second is the one that answers
"is this an algebra or a fixed embedding of 10 known types" (reviewer weakness 4).

### E1c — New resources chosen to span the descriptor space
Battery and driving time are already done, but both are `AFFINE_ACCUMULATOR`
+ reset. Declare 4–6 more that exercise *fields the trained set never varied*:

| new resource | field it exercises | descriptor slot |
|---|---|---|
| minimum route load / min duration | lower bound instead of upper | 5 vs 6 |
| peak-leg load, max temperature | `AFFINE_MAX` operator | 1, 21 |
| zone/LIFO ordering | `CLASS_ORDER` precedence | 2 |
| per-customer accumulated limit | per-node bound | 32 |
| cash pickup with bank drops | negative increment + event reset | 19, 27 |
| two-sided duration window | both bounds | 7 |

Then the key plot: **gap reduction vs. descriptor distance to the nearest training
resource** (Hamming over the categorical slots + L1 over the continuous ones). If
transfer degrades *smoothly* with descriptor distance, you have direct evidence that
the descriptor space — not the identity set — is what generalises. That single
scatter plot is the strongest available answer to both comment 1 and reviewer Q3.

### E1d — Expressiveness census (a table, not a run)
Enumerate 25–30 constraint families from the literature in `baseline_papers/`.
For each: *expressible & trained* / *expressible & zero-shot tested* /
*expressible, untested* / **not expressible**. Give the actual declaration for the
expressible ones (`decoder.resource_declarations` prints it). The honest
inexpressible column — stochastic travel times, soft/penalty constraints,
cross-route synchronisation, cumulative objectives — is what converts "we cover
whatever our representation covers" into a falsifiable scope claim. This is exactly
the move `theory.md` §6 recommends.

---

## 2. Why four extra loss terms?

Terms in `train.py:_step_loss`: `dual` (predicted resource delta), `feasibility`
(continuation risk), `binding` (which resources bind), `price` (multiplier ≈
binding, plus the coupled dynamic target), plus the `objective_residual` L2 and
optional critic.

### E2a — Leave-one-term-out retraining
6 runs (full, −dual, −feasibility, −binding, −price, RL-only) at identical budget,
seed, schedule, and validation-based selection. Report final gap **and** the
per-head diagnostic that term supervises. The story you want is not "gap goes up"
but "this capability disappears when you remove its supervision".

If 6 × 1000 epochs is unaffordable, run 300 epochs and report **training curves**
(already in wandb) — the auxiliary terms mostly buy early-training sample
efficiency, and a curve shows that better than a final number does.

### E2b — Head quality decoupled from gap (cheap, from the probe dump)
For each head, on **held-out compositions**:
- `dual` → R² of predicted resource delta vs. true delta from `evaluate_resources`
- `feasibility` → AUROC / average precision vs. ground-truth continuation risk
- `binding` → accuracy vs. `binding_target`
- `price` → calibration of λ_r against realised bindingness

Then correlate head quality with per-variant gap reduction. This reframes "we added
four terms" as "each term supervises a measurable quantity, and the terms whose
heads are accurate are the terms that pay".

### E2c — Shuffled-target control (the decisive one)
Retrain with each auxiliary target **replaced by a shuffled target of the same shape
and scale**. If the shuffled run matches the real one, the term is a regulariser,
not constraint supervision — and you should say so. This is the exact analogue of
the zero-init field-off control you already use for the field, applied to the loss.
It is the cleanest possible answer to "why four?".

### E2d — Weight sensitivity
Single-factor sweep of each weight over {0, ¼×, 1×, 4×}. Flat curves rebut the
"over-engineered and tuned to this benchmark" charge (reviewer weakness 3); sharp
curves tell you which term is actually load-bearing.

---

## 3. Conceptual model / PACE-style story

`theory.md` already has the right skeleton (continuation equivalence → resource
sufficiency → composition closure). Make the theorems **executable property tests**
so the conceptual story is audited rather than asserted. Extend
`tests/python/test_resource_algebra_equivalence.py`:

### E3a — Sufficiency audit (Theorem 1)
Sample prefix pairs `(p, p′)` with **equal resource state but different histories**;
assert `mask(p) == mask(p′)` on shared candidates and that sampled continuations are
feasible after `p` iff after `p′`. Report: "verified on N prefix pairs across 110
compositions, 0 violations". This has already caught real bugs in this codebase
(the backhaul `bp` load-reset divergence), so it pays for itself twice.

### E3b — Composition closure audit (Theorem 2)
Assert that transition admissibility under a conjunction equals the AND of
per-resource admissibility computed independently. **Where it fails you have found a
genuine cross-resource coupling** — and that failure set is the empirical
justification for `Context_θ` existing at all. Report it as content, not as a bug.

### E3c — Necessity probe (the Proposition)
For each resource, construct prefix pairs differing *only* in that resource's state
and count what fraction of the candidate set flips feasibility. Gives a per-resource
"semantic information content" number: direct evidence the retained state is
necessary, not chosen.

---

## 4. Has the heatmap learned constraints, or is it riding the mask? — **highest priority**

The reviewer's sharpest implicit doubt, and the one comment 4 names directly. All of
these come out of the probe dump; only E4e needs a decoder change.

### E4a — Feasibility AUROC on edges the mask would reject
The decoder masks infeasible actions, so the field is *never obliged* to know
feasibility. So score the energy on the **full** candidate set including
mask-rejected edges, and measure AUROC of `−energy` against `mask_feasible`.
- AUROC ≈ 0.5 → the field is feasibility-blind; all feasibility is the mask.
- AUROC high → the field has internalised constraint semantics.

Break out by resource family and by number of active resources. Compare against
distance-only, which is the null model.

### E4b — Dead-end awareness (the deeper version)
One-step feasibility is easy. The real question is whether the field avoids moves
that are feasible *now* but force a dead end or an expensive detour. Label each
candidate by a *k*-step lookahead: take the edge, complete greedily with the field
off, record whether extra routes were forced and the final cost. Then measure rank
correlation between energy and lookahead label, **relative to the same correlation
for `c(e)` alone**. The delta is precisely "what the field knows that distance does
not and the mask cannot tell you".

### E4c — Incremental information over hand-crafted features
Logistic regression predicting one-step feasibility from (a) distance,
(b) distance + the resource-pressure features the net already receives,
(c) + learned energy. Report the incremental AUROC of (c) over (b). Answers "does
the network add anything beyond the features it is handed".

### E4d — Constraint-shift counterfactual (best figure in the paper)
Fix coordinates and demands; sweep **only** the capacity Q (then, separately, TW
width). The distance term `c(e)` is invariant by construction, so the
distance-only ranking has Kendall-τ = 1.0 across the sweep *by definition*. Measure
the learned energy's τ across the same sweep.

Then plot the two heatmaps side by side for one instance at Q-tight vs. Q-loose. If
they are identical, the field is not constraint-aware. If the tight one concentrates
on compact intra-cluster edges and the loose one spreads, that is a picture a
reader believes immediately — and it is unforgeable by any mask or search argument.

### E4e — Feasibility rate under a weakened mask
The most direct answer, and the only item needing C++: add a decoder flag that
**relaxes or disables hard masking** (fall back to penalty + repair). Then report
solution feasibility rate with field-on vs. field-off. If field-on stays feasible
far more often once the safety net is removed, the field has demonstrably learned
the constraints. This one number answers comment 4 and feeds comment 6.

### E4f — λ_r magnitude audit (answers reviewer Q1 for free)
From the same dump: distribution of the learned intensities λ_r(s) across variants
and states. Show they stay bounded and track *bindingness* rather than diverging.
This is the empirical content of the "value, not penalty" argument — and if they
*do* look like growing multipliers, better you find that than a reviewer.

---

## 5. What can we handle, and how far does it generalise?

### E5a — Stop averaging over 110 heterogeneous variants
Reviewer weakness 5 is correct that a single mean over variants with wildly
different reference quality is misleading. Replace with:
- per-variant **paired** reduction with bootstrap CIs (paired is reference-invariant
  — state this explicitly rather than relying on the reader to notice);
- gap vs. #active resources (`scripts/plot_gap_vs_constraints.py` exists);
- **gap vs. constraint tightness** — generate instance families sweeping capacity
  utilisation, TW width, and battery range. This answers "to what extent can we
  handle *hard* constraints" literally. Expect the field's advantage to grow with
  tightness up to a turning point where everything degrades; **showing that turning
  point is a genuine insight**, not a weakness.

### E5b — The generalisation surface, not the diagonal
A matrix, not a list: composition novelty (seen / held-out / unseen-resource) ×
scale (n = 100 / 200 / 500 / 1000). You currently report the diagonal of this.

### E5c — Failure taxonomy
For the worst 10 variants, categorise *why*: route-count fragmentation (already
known for TW at n=1000 — construction emits ~64% too many routes), construction
stranding, weak reference, near-collinear constraints. A named failure mode reads as
understanding; an unexplained outlier reads as luck.

### E5d — Reference-quality audit
Re-run the suspicious variants (URS at 48% on `amdcvrp`) with a longer LKH/HGS/
OR-Tools budget and report how the numbers move. Pre-empts weakness 5 rather than
conceding it.

---

## 6. Run the baselines with the checkpoint; check feasibility rate

### E6a — Feasibility table across every method
`prism_eval.results.Row` already carries a `feasible` column and every method's
route is validated through `Decoder.evaluate`, so this is **aggregation, not new
instrumentation**. Report, per method × variant: % fully-feasible solutions under
*your* independent checker, plus per-constraint violation counts for the failures.
Baselines trained on their own generators often violate on your instances; if they
do not, saying so strengthens your comparison rather than weakening it.

### E6b — Refinement-augmented baselines — **the single most damaging open criticism**
Reviewer weakness 2: PRISM runs SRR, URS is single-pass construction, so the margin
may be "any refinement at all" rather than "the learned field". The control that
closes this:

```
URS raw  →  URS + SRR (field off)  →  URS + SRR (field on)  →  PRISM
```

`set_incumbent` already exists, so this is a small `prism_eval/methods/refined.py`
wrapping any baseline's solution as the SRR incumbent. Do the same for CCL. Report
**equal-wall-clock** curves alongside final numbers (reviewer Q4).

Note this also cuts your way: if URS + SRR(field off) still loses to
URS + SRR(field on), the field's contribution is isolated on a *foreign*
construction, which is stronger evidence than your own paired control.

---

## 7. Compare against other learned-heatmap methods

### E7a — Fixed search, swapped scorer
The only fair protocol: hold the decoder and SRR budget fixed and swap **only** the
edge-scoring function.
1. distance-only (your existing zero-init control)
2. PRISM field
3. **plain GNN edge-cost regressor, no descriptors, no resource tokens** — reviewer
   weakness 3 asks for exactly this "minimal viable innovation" baseline
4. a DIMES/DIFUSCO/UTSP-style heatmap
5. an AM/POMO/URS policy's edge marginals used as a heatmap

Most heatmap work is TSP/CVRP-only; restrict this table to CVRP/CVRPTW and say so.
Item 3 is the one the reviewer will look for.

### E7b — Intrinsic heatmap metrics (no search at all)
Standard in the DIMES/DIFUSCO line, and free from the probe dump: edge-level
precision/recall against the reference solution's edge set, and top-*k* hit rate
("is the reference edge in the top-5 by energy"). Lets you situate against
published numbers without re-running anyone's search.

### E7c — PIP-style risk penalty as an ablation row
Bi et al. (2024)'s proactive infeasibility prevention is named as missing related
work — but you already have the plumbing: `edge_risk` + `risk_penalty` are
arguments on `sample`/`solve`. Run "risk penalty only, no field" as a row. Answers a
missing-related-work criticism with an experiment instead of a citation.

---

## 8. Is the network general beyond this refinement loop?

### E8a — Pure construction mode
Run PRISM with 0 SRR iterations (greedy and sampled) against URS/POMO-class
construction baselines. This is the apples-to-apples construction comparison, and it
decomposes your own number into construction quality vs. refinement quality —
answering weakness 2 from the opposite side to E6b.

### E8b — Field as a foreign solver's candidate rule
Export the energy as an edge cost and feed it to an off-the-shelf solver's edge
ranking — LKH-3's candidate set / α-nearness, or PyVRP's granular neighbourhood
(`prism_eval/methods/oracle.py` already drives PyVRP). If the field improves a
*classical* solver's candidate generation at fixed iteration count, generality is
demonstrated far more convincingly than any within-framework ablation.

### E8c — Another graph COP, cheaply
The credible in-scope option is variants that are genuinely different COPs but
already expressible in the algebra: **orienteering / team orienteering / PCTSP**
(prize collection = a lower-bounded accumulated-reward resource). Those cost a
declaration, not a model. Genuinely distant COPs (scheduling, constrained flow) are
out of scope for this paper — put that in Limitations explicitly rather than leaving
the "other graph-based COPs" claim untested.

---

## Tooling to build

| # | artefact | unlocks | cost |
|---|---|---|---|
| 1 | `scripts/probe_field.py` — static probe dump | E1a, E1b, E2b, E4a–d, E4f, E7b | **build first** |
| 2 | `scripts/descriptor_intervention.py` | E1a, E1b, E1c distance metric | small |
| 3 | `prism_eval/methods/refined.py` — wrap any solution as SRR incumbent | E6b, E8a | small |
| 4 | `prism_eval/methods/heatmap.py` — pluggable external scorer | E7a | medium |
| 5 | decoder `--relax-mask` bypass | E4e | small C++ |
| 6 | property tests in `test_resource_algebra_equivalence.py` | E3a–c | small |
| 7 | tightness-sweep generator in `scripts/generate_benchmark_suite.py` | E5a | small |
| 8 | new resource declarations | E1c | medium |

## Priority

**Tier 1 — cheap, and each closes a specific attack on the paper**
E4a + E4d (heatmap is constraint-aware), E6b (URS + SRR control), E6a (feasibility
table), E1a (descriptor fields earn their place), E4f (λ audit).

**Tier 2 — moderate cost, converts "good" into "convincing"**
E7a items 1–3 (fixed-search scorer swap incl. the plain-GNN baseline), E1c
(descriptor-distance transfer curve), E5a (tightness sweep), E3a–b (executable
theory), E2b (head quality).

**Tier 3 — expensive, retraining or new infrastructure**
E2a + E2c (loss ablations and shuffled-target controls), E4e (relaxed mask),
E8b (foreign solver), E8c (orienteering family), E1d (census — cheap in compute,
expensive in reading).

Tier 1 alone addresses comments 4 and 6 outright and materially moves 1 and 5.
