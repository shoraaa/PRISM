# Pheromone ablation

Matched first-instance URS evaluation at size 100 using 16 ants, 16 threads,
identical seeds, unlimited SRR, and the same binary. Oracle/reference gaps are
available for 104 of the 110 variants.

| Iterations | Pheromone | Mean gap | Median gap | Runtime | Better / tie / worse |
|---:|:---:|---:|---:|---:|---:|
| 2 | on  | 13.246% | 8.751% | 12.578 s | baseline |
| 2 | off | 13.171% | 8.610% | 12.502 s | 6 / 94 / 4 |
| 5 | on  | 9.524% | 5.946% | 20.230 s | baseline |
| 5 | off | 9.483% | 6.214% | 20.208 s | 31 / 43 / 30 |

Better/tie/worse is paired against pheromone-on over the 104 referenced
variants. At two iterations, disabling pheromone improves mean gap by 0.075
percentage points. At five iterations it improves mean gap by 0.042 percentage
points, while the median is 0.268 points worse. Runtime is effectively equal.

The typed-field plus no-pheromone compatibility run passed all 110 variants,
with every final route replaying as feasible. It used a deterministic objective
channel, not a trained field, so it is an integration check rather than a
learned-quality result.
