# paper.tex → v14 migration: outstanding items

Prose and architecture descriptions in `paper.tex` were moved to the v14
implementation (commits `067a72aa`, `b6720d43`). Per decision, **all reported
numbers were left in place and unmarked**. They were produced by pre-v14
checkpoints. This file records what that leaves outstanding.

## Blocking: every number predates the described architecture

`MODEL_SCHEMA = "slack_energy_v14"` (`net.py:85`) loads strictly, so no
pre-v14 checkpoint evaluates under the current code. The paper therefore
describes v14 while reporting v13 results.

| Location | Figure | Regenerate with |
|---|---|---|
| §5.2 `tables/sota_seen_12.tex` | **file is 0 bytes** — `\input` renders nothing | `prism_eval` runner |
| §5.2 `tables/unseen.tex` | 32 OOD variants | `prism_eval` runner |
| §5.3–5.4 | 3.14% → 0.88% headline | `prism_eval` runner |
| §5.5 semantic ablation | 1.50/1.60, 1.04/1.11, 0.76/0.88 | `scripts/ablation_eval.py` |
| §5.5 composition ablation | 2.97 / 3.35 / 3.89 | `scripts/ablation_eval.py` |
| §5.5 + App D.1 probe | 68.5 / 51.1 / 52.2 | see below — script is broken |
| App D.3 per-variant tables | all | `prism_eval` runner |

In-flight runs, both **program-blind** (it is the default now; the directory
name is legacy):

- `pretrained/v14` — `--lr 5e-6 --epoch 1000 --aux-rl-scale 0.1`
- `pretrained/program-blind-resource-v14` — `--lr 5e-5`, otherwise identical

Note `--aux-rl-scale` defaults to `0.0` and `--pretrain-epochs` to `0`, so
`auxiliary_scale` is zero unless passed explicitly. Both runs pass `0.1`,
which matches the scale stated in App C.7. **A run that omits the flag trains
with no semantic term at all.**

## Feasibility probe: RESOLVED (2026-09-01)

`scripts/probe_feasibility.py` is new and runs against the working tree. It
pairs the per-resource field with the executed signed margin and reports
per-resource AUROC against the admissibility label, plus per-state top-1.
`tables/feasibility_probe.tex`, §5.5 and App D.1 are regenerated from it.

`scripts/probe_field.py` and `probe_report.py` still read
`output["feasibility_risk"]` and still fail on v14. Left alone deliberately:
they load a checkpoint's own checkout via `--code-root` and are the only way to
probe the v6 ablation checkpoints.

Measured on `pretrained/program-blind-resource-v14/best.pt` (**epoch 13 of
1000**, val_gap 0.512), 16 instances x 4 formulations, 1,161,224
candidate-constraint pairs over 5,449 informative states — the informative-state
count reproduces the published protocol exactly.

| resource | AUROC(field -> admissible) |
|---|---|
| `backhaul_order` | 0.985 |
| `time_window` | 0.827 |
| `capacity` | 0.784 |
| `route_limit` | n/a (admissible at every probed state) |

Consistent across all four variants (backhaul 0.971-1.000, TW 0.771-0.902,
capacity 0.746-0.840).

**Two findings worth acting on.**

1. *Top-1 selection is at or below chance.* Lowest-energy candidate is
   admissible in 49.1% of informative states vs 50.7% distance and 53.6%
   uniform. This is defensible — the energy ranks quality among already-masked
   candidates and is never asked to screen — but it is a different claim from
   the deleted risk head's 68.5%, and the paper now says so.

2. *The slack target may be signed against the energy.* The field enters the
   energy with a positive multiplier (lower energy preferred) while
   `_slack_loss` regresses it onto a margin that GROWS with remaining slack. So
   supervision prices a comfortably-satisfied row as more expensive. Negating
   the resource term moves top-1 from 49.1% to 52.2%. Still below uniform, so
   the sign is not the whole story, but worth checking against
   `net.py` / `_slack_loss` before the final run.

Re-run at convergence: numbers above are from an epoch-13 checkpoint.

**These probe numbers are now superseded.** They were measured under the
pre-fix slack sign (see below). The AUROC magnitudes should carry over -- the
discriminative content is unchanged and only the field's orientation flips --
but `tables/feasibility_probe.tex`, the Sec. 5.5 paragraph and App D.1 must all
be regenerated from a checkpoint trained with the corrected objective.

## Slack sign: investigated, NOT a bug, reverted (2026-09-01)

I proposed negating `_slack_loss`'s target and it measured worse. Reverted.

**The argument that looked right.** The decoder adds `lambda_r * field` to an
energy it minimizes with `lambda_r` strictly positive, so a larger field is a
less preferred edge, while `SIGNED_MARGIN = bound - post_state` is largest where
a row is comfortably satisfied. That reads as pricing slack as expensive.

**Why it is wrong.** It assumes the field has one global orientation. The
margin's sign relative to consumption is set by each row's own convention:

| row kind | example | corr(post_state, margin) | centered(margin) vs centered(delta) |
|---|---|---|---|
| accumulating (upper bound) | `route_limit` on cvrpl | **-0.88** | opposite |
| depleting (lower bound) | `capacity` on cvrp | **+0.62** | **same** |

So a global negation fixes the accumulating rows and inverts capacity and
battery, and capacity is active in nearly every benchmark variant. The deleted
`_dual_loss` was no more consistent: `screened_resource_delta` is `+d` for
route_limit and `-a_v` for capacity.

Capacity is worse still. Its bound is two-sided `[0, 1]`, so its margin is a
tent function of the state and which side the `min` takes depends on whether
backhauls are active: corr(post_state, margin) is **+0.62 on cvrp** and
**-0.475 on cvrpbpltw** -- same row, opposite orientation, same checkpoint.

**Measured.** Matched configs (the run config is now baked into the defaults, so
the bare-args run reproduces the explicit-args one):

| epoch | ValCost, negated | ValCost, as shipped |
|---|---|---|
| 0 | 10.5332 (BEST) | 10.5485 (BEST) |
| 1 | 10.5365 | 10.5174 (BEST) |
| 2 | 10.5387 | 10.5131 (BEST) |
| 3 | 10.5561 | 10.5681 |
| 4 | 10.5762 | 10.5244 |
| 5 | 10.5555 | 10.5202 |
| 11 | -- | 10.4955 (BEST) |

Negated: one BEST, at epoch 0, then monotone worsening. As shipped: four BESTs
in twelve epochs.

**Kept from the investigation.** `_slack_loss` now raises on a field/margin
shape disagreement instead of returning zero -- that was a genuine second silent
path to training as plain PPO, independent of the sign. Four regression tests
pin the orientation, the row-constant invariance, the raise, and the
empty-registry no-op, with the measurement recorded so nobody re-flips it.

**The real finding, still open.** There is no consistent orientation for the
resource field across rows, and capacity's two-sided margin is non-monotone in
its own state. If this is revisited, orient each row's margin against its own
binding side rather than applying a global sign. That is a design change to
Eq. (30), not a bug fix.

## Probe numbers

A1's numbers (`tables/feasibility_probe.tex`, Sec. 5.5, App D.1) were measured
on `pretrained/program-blind-resource-v14/best.pt` at epoch 13, under the
as-shipped sign, which is the sign that survived. They stand, but still need
regenerating at convergence.
