# Decoder implementation summary

Date: 2026-07-27

All results below use the final `prism_decoder` C++ extension, the first saved
size-100 URS instance of each variant, `K=64`, all available OpenMP threads,
static ant scheduling, two decoder iterations, and pheromone disabled.

## Feasibility gates

| Guidance | Passed | Failed | Improved | SRR active |
|---|---:|---:|---:|---:|
| Classical | 110 | 0 | 110 | 110 |
| Neutral resource field | 110 | 0 | 109 | 109 |

The neutral-field run is an interface and correctness gate, not a learned-model
quality result. Its one unchanged case was `apdtsp`.

## URS reference comparison

References are available for 104 of 110 variants. After construction, the mean
gap was 82.260%. One perturbation plus unlimited-BFS SRR reduced the mean gap to
13.171% and the median gap to 8.610%. SRR improved all 110 solutions relative
to their bootstrap construction. The saved per-variant data is
`decoder_classical_no_pheromone_allx2.csv`.

| Group | References | Bootstrap mean gap | SRR mean gap | SRR median gap |
|---|---:|---:|---:|---:|
| All | 104 | 82.260% | 13.171% | 8.610% |
| Symmetric | 51 | 89.615% | 16.049% | 7.773% |
| Asymmetric | 53 | 75.182% | 10.401% | 8.891% |
| Time windows | 48 | 69.697% | 10.643% | 8.091% |
| Multi-depot | 48 | 91.363% | 13.533% | 9.461% |
| Strict backhaul | 32 | 82.803% | 7.866% | 6.794% |
| Open route | 50 | 69.294% | 12.301% | 7.395% |
| Pickup-delivery | 5 | 61.141% | 10.441% | 6.374% |

No saved reference was available for `op`, `atsp`, `pctsp`, `pdtsp`, `spctsp`,
or `aspctsp`; these variants are included in the feasibility counts but not in
the gap aggregates.

## Training smoke

One small-VRAM mixed update passed all 11 URS training compositions. At size 20,
the feasibility pass produced 2,691 counterfactual/look-ahead edge labels with
1,372 positives (50.98%). At the default training shape, one size-100 CVRPTW
iteration with 32 ants produced 8,768 labels in 0.481 decoder seconds. The
43-test focused Python suite
covers exact C++ trace replay, normalized input/state contracts, shared C++ and
Python dimensions, resource-token masking, deterministic validation,
event-driven options, variable-duration SMDP returns, additive zero-pressure
corrections, auxiliary price/coupler gradients, and normal/small-VRAM PPO
updates.

PPO now replays one probability ratio per stochastic decision and broadcasts
the ant's option return to those decisions. Inverse trace-length weights keep
each ant's total contribution equal without shrinking every policy gradient by
the number of decisions. A PPO-only regression test confirms that policy
parameters move and that post-update KL becomes nonzero with all auxiliary
losses disabled. During RL fine-tuning, auxiliary losses default to a `0.1`
scale; pretraining retains their full scale.

## Correctness hardening

The C++ extension is the sole source of truth for field, live-state, node, and
edge dimensions. Candidate resource features and move-screening targets share
the same exported per-resource physical scales. The learned correction contains
both a multiplicative residual and an additive normalized term, so a resource
channel remains trainable when its analytic edge pressure is zero.

Validation uses deterministic greedy construction and perturbation. Neural
fields are cached by `graph_version`, so a stagnation-truncated option does not
repeat an identical GNN call. Training assigns finite-horizon option returns as
`R + gamma^tau G'`; resource-delta fallback evaluation runs only when SRR did
not emit screening labels.

The field model now has an independently supervised per-edge feasibility-risk
head. C++ labels all stored counterfactual edges at traced states: immediately
illegal edges are positive, while legal edges use sparse k-step continuation
look-ahead. Rollout-batch class-balanced BCE trains feasibility and binding
heads, and
class-balanced smooth-L1 upweights nonzero dual targets. Learned guidance is
disabled during auxiliary pretraining; the risk head is stop-gradient where it
enters policy replay, then contributes a configurable soft penalty to proposal,
perturbation, and composable SRR sequence energy.

Training logs PPO forward, backward, and optimizer time separately, together
with generation, decoder, neural, other-rollout, and unaccounted epoch time.
Validation records the actual average final cost instead of a hard-coded zero.
HIP/ROCm training selects the tested detached-output small-VRAM PPO path by
default; `--no-smallvram` remains available for explicit benchmarking.

These are backend correctness and training-signal results. No trained
feasibility-head quality claim is made yet.

## Scale profile

Generated Euclidean problems above 512 nodes now keep coordinates instead of a
dense distance matrix. Candidate support is built from bounded KD-tree and
resource-index pools; a 10K CVRP graph contains 650,000 directed edges and was
constructed in 1.02 seconds. Direct single-graph reductions avoid the ROCm
atomic-contention failure caused by pooling every edge into graph ID zero.

The final n=1,000 training unit took 23.25 seconds: 19.02 seconds in parallel
native decoding/SRR, 1.69 seconds in PPO, 0.65 seconds in neural emissions, and
1.86 seconds in other rollout work. A real n=10,000 field inference gate loaded
the checkpoint from an n=1,000 RL update and returned a feasible CVRP solution
in 4.28 seconds total. Full commands and the before/after table are in
`train_profile_2026-07-27.md`.

## Exact screening acceleration

SRR screening labels now replace the affected route aggregates in constant
time for capacity, route/tour limits, valid backhaul ordering, and prize quota.
The incumbent cache stores the three largest per-route extrema, global additive
totals, and the incumbent edge set. Planned relocate, swap, Or-opt, 2-opt,
2-opt-star, insert, delete, replace, and exchange moves therefore avoid a full
solution resource replay whenever their active resources have exact compact
summaries. Structurally invalid empty-route plans are discarded without the
previous redundant evaluate-and-replay pair.

The existing time-window target sums cascading lateness at every node, and the
pickup-delivery binding target is a maximum over live open pairs. Preserving
those exact labels requires the legacy route replay; replacing them with the
Vidal minimal-time-warp monoid would change training behavior. The backend uses
an explicit fallback for those cases and exposes fast/fallback counters.

The all-variant verifier recomputed every fast label with the old evaluator on
the first saved size-100 instance of every URS variant: 110/110 variants passed
with zero per-channel mismatches. It observed 5,749 fast planned evaluations
and 14,071 exact fallbacks. A five-seed size-100 timing sample with 16 ants had
median untraced/traced decoder times of 0.0969/0.1188 seconds for CVRP and
0.1636/0.1621 seconds for CVRPTW. These timings are local throughput evidence,
not solver-quality results.

Commands:

```bash
uv run --no-sync pytest -q
uv run --no-sync python tests/urs_one_each.py --iterations 2 --guidance classical --no-pheromone
uv run --no-sync python tests/urs_one_each.py --iterations 2 --guidance field --no-pheromone
uv run --no-sync python tests/urs_srr_report.py --iterations 2 --no-pheromone --csv results/decoder_classical_no_pheromone_allx2.csv
uv run --no-sync python tests/urs_screening_parity.py
```
