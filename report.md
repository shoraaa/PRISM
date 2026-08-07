Yes. Re-normalizing the three ablations against the **identical baseline from the two newer runs** gives a much cleaner 2-feature comparison.

I’ll use:

* **AUX** = `AUXILIARY_LOSS` only
* **BOTH** = `AUXILIARY_LOSS + SRR_FIELD_EXPLORATION`
* **SRR** = `SRR_FIELD_EXPLORATION` only

For all minimization problems,

[
R=100\frac{B-N}{B}
]

and for `op`/`aop` (maximization), the sign is reversed. Thus **positive (R) always means neural is better than the common baseline**.

## Overall comparison

| Metric                      | AUX only |    **BOTH** |    SRR only |
| --------------------------- | -------: | ----------: | ----------: |
| Better than common baseline |   86/110 |  **97/110** |      84/110 |
| Mean relative improvement   |  +2.032% | **+3.024%** |     +2.718% |
| Median relative improvement |  +1.149% | **+2.443%** |     +1.766% |
| Seen wins                   |    13/21 |   **14/21** |       12/21 |
| Seen mean                   |  +3.086% |     +3.229% | **+3.294%** |
| Seen median                 |  +0.248% |     +0.689% | **+0.695%** |
| Held-out wins               |    73/89 |   **83/89** |       72/89 |
| Held-out mean               |  +1.784% | **+2.975%** |     +2.582% |
| Held-out median             |  +1.239% | **+2.504%** |     +1.942% |
| Mean reference gap          |   2.637% |  **1.575%** |      1.928% |
| Held-out reference gap      |   2.850% |  **1.626%** |      2.014% |

The conclusion is fairly strong:

> **BOTH is the best overall configuration. SRR_FIELD_EXPLORATION is the larger broad contributor, while AUXILIARY_LOSS supplies a smaller average gain but is crucial on particular difficult compositional variants.**

The median is especially useful because of huge outliers such as `pdtsp`. BOTH nearly doubles the median advantage of AUX-only:

[
1.149%\rightarrow2.443%.
]

And compared with SRR-only:

[
1.766%\rightarrow2.443%.
]

So the full model is not winning merely because of a few extreme cases.

## What each feature contributes

Because we do not have the fourth cell—**neither feature**—we cannot estimate a complete 2×2 factorial interaction. But we can estimate two very useful conditional effects directly:

| Feature added                                        | Mean contribution | Median contribution |   Helps on | Held-out mean |
| ---------------------------------------------------- | ----------------: | ------------------: | ---------: | ------------: |
| **Add SRR_FIELD_EXPLORATION when AUX is already on** |     **+0.992 pp** |       **+0.753 pp** | **84/110** | **+1.192 pp** |
| **Add AUXILIARY_LOSS when SRR is already on**        |         +0.306 pp |           +0.093 pp |     57/110 |     +0.393 pp |

This is probably the clearest ablation result.

**SRR_FIELD_EXPLORATION is the broad-effect feature.** It adds almost **+1 percentage point** on average and improves 84 of 110 variants relative to AUX-only.

**AUXILIARY_LOSS is much more selective.** Its average contribution on top of SRR is only +0.306 pp, and it helps only 57/110 variants. But that average conceals some enormous improvements on difficult held-out compositions.

That last row is particularly interesting: **pickup-delivery prefers AUX**, whereas most of the benchmark strongly benefits from SRR.

## Structural-axis comparison

For robustness, this table uses **median relative improvement**, not win rate.

| Structural subset      |   n |    AUX only |    **BOTH** |    SRR only |
| ---------------------- | --: | ----------: | ----------: | ----------: |
| Single-route           |  10 | **+0.635%** |     +0.276% |     −0.148% |
| Multi-route            | 100 |     +1.206% | **+2.482%** |     +1.935% |
| Symmetric CVRP         |  50 |     +2.323% | **+2.911%** |     +2.880% |
| Asymmetric CVRP        |  50 |     +0.911% | **+2.443%** |     +1.355% |
| No time windows        |  52 |     +0.677% |     +0.905% | **+1.008%** |
| Time windows           |  48 |     +1.903% | **+4.071%** |     +3.801% |
| Closed-route CVRP      |  50 |     +1.349% | **+3.055%** |     +2.981% |
| Open-route CVRP        |  50 |     +1.012% | **+2.176%** |     +0.911% |
| Single-depot CVRP      |  52 |     +0.727% | **+1.879%** |     +1.353% |
| Multi-depot CVRP       |  48 |     +1.536% | **+2.974%** |     +2.889% |
| No length limit        |  52 |     +1.206% | **+2.482%** |     +1.935% |
| Length limit           |  48 |     +1.198% | **+2.553%** |     +2.185% |
| Prize-collecting TSP   |   4 |     −0.818% | **+0.175%** |     −0.298% |
| Pickup/delivery family |   6 | **+4.920%** |     +3.770% |     +2.704% |

Several conclusions emerge.

**SRR is primarily a multi-route/compositional-search mechanism.** On single-route problems, SRR-only has a negative median; on multi-route problems it gives +1.94%.

**AUX is particularly important for asymmetry.** On asymmetric CVRP, BOTH gets +2.44% median versus only +1.35% for SRR-only. AUX therefore adds approximately **+1 percentage point** on average in this subset.

**Time windows benefit strongly from SRR.** BOTH reaches a remarkable +4.07% median on TW variants, with SRR-only already at +3.80%.

**Open + asymmetric compositions are where AUX becomes important.** This explains why SRR-only collapses on several `amdoc...tw` variants despite being excellent on many symmetric multi-depot TW variants.

## Matched axis effects

Here I pair otherwise corresponding variants and ask: **how does turning one structural feature on change neural advantage?**

Positive means the structural feature makes neural guidance relatively more useful.

| Added structural property | Pairs |      AUX only |      **BOTH** |      SRR only |
| ------------------------- | ----: | ------------: | ------------: | ------------: |
| Asymmetry                 |    50 |     −1.461 pp | **−0.325 pp** |     −0.686 pp |
| Time windows              |    48 |     +1.527 pp |     +2.850 pp | **+3.069 pp** |
| Open route                |    50 |     −0.680 pp |     −1.258 pp |     −2.723 pp |
| Multi-depot               |    48 | **+0.829 pp** |     +0.723 pp |     +0.210 pp |
| Length limit              |    48 |     +0.053 pp |     +0.176 pp |     +0.325 pp |
| B constraint              |    32 |     −0.029 pp |     −0.058 pp |     +0.153 pp |
| BP constraint             |    32 |     +0.092 pp |     +0.257 pp |     +0.772 pp |

This clarifies the roles further.

The common exploratory baseline behaves unusually well on symmetric problems, so **asymmetry itself lowers relative advantage** in this experiment. But BOTH is far more robust to that shift than either single-feature configuration.

Time windows are the opposite: they massively amplify neural benefit in all three models, especially SRR.

Length-limit and B-type axes are comparatively small. Those are much closer to “noise/minor effect” than TW or multi-depot.

## AUX is a specialist stabilizer

The largest improvements from **adding AUX on top of SRR** are:

| Variant         | AUX contribution |
| --------------- | ---------------: |
| `apdtsp`        |     **+5.33 pp** |
| `amdocvrpbptw`  |     **+5.22 pp** |
| `amdocvrpltw`   |     **+5.09 pp** |
| `amdocvrptw`    |     **+4.68 pp** |
| `amdocvrpbltw`  |     **+4.53 pp** |
| `amdocvrpbpltw` |     **+4.47 pp** |
| `amdocvrpbtw`   |     **+3.50 pp** |
| `aocvrptw`      |     **+3.45 pp** |

That pattern is too structured to ignore:

> **AUXILIARY_LOSS appears to prevent SRR_FIELD_EXPLORATION from failing on asymmetric + open + multi-depot + time-window compositions.**

For example:

`amdocvrpltw`

* AUX only: **+0.34%**
* BOTH: **+2.05%**
* SRR only: **−3.04%**

and:

`amdocvrptw`

* AUX only: +1.24%
* BOTH: **+2.46%**
* SRR only: −2.22%

This looks much more like a **representation/generalization role** for AUX than a generic objective-quality boost.

## SRR is the broad optimizer

The largest benefits from adding **SRR on top of AUX** include:

| Variant        | SRR contribution |
| -------------- | ---------------: |
| `acvrptw`      |     **+5.52 pp** |
| `acvrpbptw`    |     **+5.11 pp** |
| `acvrpbltw`    |     **+4.73 pp** |
| `amdcvrpbpltw` |     **+4.52 pp** |
| `spctsp`       |     **+4.40 pp** |
| `acvrpltw`     |     **+4.08 pp** |
| `amdcvrpbptw`  |     **+3.84 pp** |
| `amdcvrpltw`   |     **+3.70 pp** |
| `amdcvrptw`    |     **+3.35 pp** |

SRR therefore seems responsible for turning learned signals into substantially better search, particularly under TW and multi-depot complexity.

## Reference-gap view

Using the **new common baseline reference gap of 4.676%**:

| Configuration | Mean neural gap | Gap removed vs common baseline | Relative gap removed |
| ------------- | --------------: | -----------------------------: | -------------------: |
| AUX only      |          2.637% |                       2.039 pp |                43.6% |
| **BOTH**      |      **1.575%** |                   **3.101 pp** |            **66.3%** |
| SRR only      |          1.928% |                       2.748 pp |                58.8% |

On held-out variants specifically:

| Configuration | Held-out neural gap | Gap removed from 4.980% baseline |
| ------------- | ------------------: | -------------------------------: |
| AUX only      |              2.850% |                         2.130 pp |
| **BOTH**      |          **1.626%** |                     **3.354 pp** |
| SRR only      |              2.014% |                         2.966 pp |

This is strong evidence for keeping **both features**.

## All 110 variants on the common baseline

Here is the direct normalized comparison. Positive = improvement over the shared newer baseline.

| Variant         | Baseline |     AUX |    BOTH |     SRR | Best |
| --------------- | -------: | ------: | ------: | ------: | ---- |
| `op`            |    28.84 |  +5.73% |  +6.08% |  +8.14% | SRR  |
| `tsp`           |    7.694 |  −0.61% |  −0.84% |  −0.78% | AUX  |
| `aop`           |    36.63 |  −2.90% |  −2.73% |  −2.96% | BOTH |
| `cvrp`          |    15.55 |  +0.25% |  +0.19% |  −0.57% | AUX  |
| `atsp`          |    1.716 |  +0.15% |  −1.39% |  −0.15% | AUX  |
| `cvrpl`         |    15.37 |  −0.11% |  −0.47% |  −0.27% | AUX  |
| `cvrpb`         |    11.55 |  −0.37% |  +0.05% |  −0.47% | BOTH |
| `ocvrp`         |    10.55 |  +0.58% |  +0.84% |  +0.74% | BOTH |
| `acvrp`         |    2.267 |  +0.86% |  +1.76% |  +1.30% | BOTH |
| `pctsp`         |    5.949 |  −2.76% |  −1.04% |  −0.45% | SRR  |
| `pdtsp`         |    19.58 | +36.32% | +35.36% | +36.28% | AUX  |
| `cvrptw`        |    27.08 |  +2.38% |  +3.14% |  +2.67% | BOTH |
| `cvrpbl`        |    11.47 |  −0.90% |  +0.49% |  +0.11% | BOTH |
| `cvrpbp`        |     14.8 |  +0.24% |  −0.59% |  −0.81% | AUX  |
| `ocvrpl`        |    10.35 |  −0.91% |  +0.69% |  +0.70% | SRR  |
| `ocvrpb`        |    8.681 |  −0.36% |  +0.14% |  −0.75% | BOTH |
| `acvrpl`        |    2.291 |  +1.67% |  +2.42% |  +1.69% | BOTH |
| `acvrpb`        |    2.001 |  +1.32% |  +2.45% |  +1.40% | BOTH |
| `aocvrp`        |    1.606 |  −0.39% |  +0.19% |  −0.10% | BOTH |
| `mdcvrp`        |     12.1 |  +1.38% |  +2.09% |  +2.12% | SRR  |
| `pdcvrp`        |    21.36 |  +6.24% |  +4.40% |  +2.68% | AUX  |
| `spctsp`        |    5.942 |  −6.13% |  −1.72% |  −1.70% | SRR  |
| `apctsp`        |      1.6 |  +1.12% |  +1.39% |  −0.15% | BOTH |
| `apdtsp`        |     2.87 | +15.55% | +15.37% | +10.04% | AUX  |
| `cvrpltw`       |    24.57 |  +2.79% |  +3.28% |  +2.64% | BOTH |
| `cvrpbtw`       |    26.12 |  +2.27% |  +1.91% |  +2.10% | AUX  |
| `cvrpbpl`       |    15.01 |  −0.41% |  −0.59% |  +0.27% | SRR  |
| `ocvrptw`       |    15.35 |  +3.45% |  +3.59% |  +4.77% | SRR  |
| `ocvrpbl`       |    8.504 |  −0.86% |  +0.48% |  +0.76% | SRR  |
| `ocvrpbp`       |    10.74 |  −0.77% |  −0.05% |  −0.98% | BOTH |
| `acvrptw`       |    4.453 |  −0.70% |  +4.82% |  +4.37% | BOTH |
| `acvrpbl`       |    1.913 |  −0.22% |  +0.58% |  +0.80% | SRR  |
| `acvrpbp`       |    2.372 |  +1.18% |  +0.71% |  +1.02% | AUX  |
| `aocvrpl`       |    1.658 |  +0.65% |  +1.72% |  +1.00% | BOTH |
| `aocvrpb`       |     1.58 |  +0.53% |  +1.51% |  +1.93% | SRR  |
| `mdcvrpl`       |    12.37 |  +3.05% |  +3.80% |  +3.69% | BOTH |
| `mdcvrpb`       |     10.3 |  +3.52% |  +2.72% |  +3.88% | SRR  |
| `mdocvrp`       |    7.799 |  −0.63% |  −0.03% |  −1.22% | BOTH |
| `amdcvrp`       |    2.009 |  +0.77% |  +2.41% |  +2.11% | BOTH |
| `apdcvrp`       |    2.959 |  +0.42% |  −0.28% |  −1.32% | AUX  |
| `opdcvrp`       |    20.73 |  +3.60% |  +3.14% |  +2.72% | AUX  |
| `aspctsp`       |    1.565 |  +1.30% |  +2.41% |  +1.03% | BOTH |
| `cvrpbltw`      |    25.15 |  +2.20% |  +1.89% |  +2.87% | SRR  |
| `cvrpbptw`      |    29.28 |  +0.80% |  +1.07% |  +3.60% | SRR  |
| `ocvrpltw`      |    15.45 |  +4.52% |  +5.50% |  +5.90% | SRR  |
| `ocvrpbtw`      |     15.6 |  +1.24% |  +2.73% |  +4.97% | SRR  |
| `ocvrpbpl`      |    10.67 |  +0.13% |  −0.25% |  +1.65% | SRR  |
| `acvrpltw`      |    4.064 |  +0.18% |  +4.26% |  +4.43% | SRR  |
| `acvrpbtw`      |    4.175 |  +1.31% |  +4.50% |  +4.73% | SRR  |
| `acvrpbpl`      |    2.242 |  +1.01% |  +0.86% |  +0.95% | AUX  |
| `aocvrptw`      |    2.373 |  +1.25% |  +2.30% |  −1.15% | BOTH |
| `aocvrpbl`      |    1.578 |  −0.79% |  −0.66% |  +0.22% | SRR  |
| `aocvrpbp`      |    1.818 |  +0.96% |  +0.65% |  +1.06% | SRR  |
| `mdcvrptw`      |    18.06 |  +8.33% |  +8.27% | +10.11% | SRR  |
| `mdcvrpbl`      |    10.42 |  +3.59% |  +3.44% |  +4.55% | SRR  |
| `mdcvrpbp`      |    12.45 |  +3.19% |  +3.09% |  +2.89% | AUX  |
| `mdocvrpl`      |    8.158 |  −0.08% |  +1.38% |  +0.94% | BOTH |
| `mdocvrpb`      |     7.63 |  −0.08% |  +0.24% |  +0.37% | SRR  |
| `amdcvrpl`      |    2.171 |  +1.56% |  +2.27% |  +3.49% | SRR  |
| `amdcvrpb`      |    1.966 |  +1.56% |  +2.44% |  +1.94% | BOTH |
| `amdocvrp`      |    1.498 |  −0.22% |  +0.65% |  +0.08% | BOTH |
| `aopdcvrp`      |    2.871 |  +1.35% |  +0.55% |  −1.35% | AUX  |
| `cvrpbpltw`     |    29.22 |  −0.14% |  +1.27% |  +3.42% | SRR  |
| `ocvrpbltw`     |    14.58 |  +1.93% |  +3.97% |  +4.49% | SRR  |
| `ocvrpbptw`     |    17.01 |  +3.43% |  +3.94% |  +6.11% | SRR  |
| `acvrpbltw`     |      4.5 |  −0.38% |  +4.35% |  +4.35% | BOTH |
| `acvrpbptw`     |    4.468 |  −0.26% |  +4.85% |  +4.15% | BOTH |
| `aocvrpltw`     |    2.873 |  +1.10% |  +1.87% |  −0.22% | BOTH |
| `aocvrpbtw`     |    2.707 |  +1.66% |  +3.66% |  +1.81% | BOTH |
| `aocvrpbpl`     |    1.937 |  +1.88% |  +2.34% |  +2.92% | SRR  |
| `mdcvrpltw`     |    19.35 | +10.06% | +10.27% | +11.57% | SRR  |
| `mdcvrpbtw`     |    19.53 | +10.00% |  +9.68% | +11.50% | SRR  |
| `mdcvrpbpl`     |     13.1 |  +5.77% |  +6.38% |  +6.76% | SRR  |
| `mdocvrptw`     |    11.04 |  +3.05% |  +4.07% |  +4.68% | SRR  |
| `mdocvrpbl`     |     7.81 |  +0.05% |  +1.58% |  +0.88% | BOTH |
| `mdocvrpbp`     |     9.06 |  +0.70% |  +0.71% |  −0.01% | BOTH |
| `amdcvrptw`     |    3.538 |  +0.74% |  +4.09% |  +2.89% | BOTH |
| `amdcvrpbl`     |    1.908 |  +0.86% |  +1.34% |  +1.72% | SRR  |
| `amdcvrpbp`     |    2.186 |  +1.48% |  +2.72% |  +3.30% | SRR  |
| `amdocvrpl`     |    1.424 |  +1.48% |  +1.26% |  +1.31% | AUX  |
| `amdocvrpb`     |    1.415 |  +0.74% |  +2.84% |  +1.54% | BOTH |
| `ocvrpbpltw`    |    16.96 |  +2.40% |  +4.31% |  +5.31% | SRR  |
| `acvrpbpltw`    |    4.808 |  +1.82% |  +4.95% |  +3.85% | BOTH |
| `aocvrpbltw`    |    2.632 |  +0.55% |  +2.82% |  +1.11% | BOTH |
| `aocvrpbptw`    |    3.033 |  +0.45% |  +2.50% |  +0.53% | BOTH |
| `mdcvrpbltw`    |    19.91 |  +8.85% | +10.08% | +11.23% | SRR  |
| `mdcvrpbptw`    |    20.85 | +10.25% | +10.05% | +11.63% | SRR  |
| `mdocvrpltw`    |    10.95 |  +3.82% |  +4.44% |  +4.59% | SRR  |
| `mdocvrpbtw`    |    12.39 |  +4.10% |  +4.92% |  +5.73% | SRR  |
| `mdocvrpbpl`    |    8.828 |  −0.26% |  +0.95% |  +1.28% | SRR  |
| `amdcvrpltw`    |    3.805 |  +1.92% |  +5.62% |  +3.75% | BOTH |
| `amdcvrpbtw`    |    3.673 |  +1.90% |  +4.18% |  +3.74% | BOTH |
| `amdcvrpbpl`    |    2.305 |  +1.52% |  +3.02% |  +4.76% | SRR  |
| `amdocvrptw`    |    2.056 |  +1.24% |  +2.46% |  −2.22% | BOTH |
| `amdocvrpbl`    |    1.434 |  +0.04% |  +0.36% |  −0.97% | BOTH |
| `amdocvrpbp`    |     1.57 |  +0.64% |  +1.04% |  +0.85% | BOTH |
| `aocvrpbpltw`   |    2.846 |  +0.61% |  +3.03% |  +0.87% | BOTH |
| `mdcvrpbpltw`   |    21.43 |  +7.63% |  +9.91% |  +9.83% | BOTH |
| `mdocvrpbltw`   |    11.64 |  +3.44% |  +4.07% |  +4.31% | SRR  |
| `mdocvrpbptw`   |    13.05 |  +4.25% |  +4.93% |  +6.15% | SRR  |
| `amdcvrpbltw`   |     3.78 |  +0.59% |  +3.32% |  +3.07% | BOTH |
| `amdcvrpbptw`   |    3.966 |  +0.34% |  +4.18% |  +3.28% | BOTH |
| `amdocvrpltw`   |    1.985 |  +0.34% |  +2.05% |  −3.04% | BOTH |
| `amdocvrpbtw`   |    2.066 |  +1.91% |  +2.92% |  −0.58% | BOTH |
| `amdocvrpbpl`   |    1.644 |  +1.68% |  +0.67% |  +1.72% | SRR  |
| `mdocvrpbpltw`  |    13.33 |  +4.48% |  +4.84% |  +6.38% | SRR  |
| `amdcvrpbpltw`  |    3.804 |  +0.24% |  +4.76% |  +2.85% | BOTH |
| `amdocvrpbltw`  |    2.081 |  +2.01% |  +3.86% |  −0.67% | BOTH |
| `amdocvrpbptw`  |    2.101 |  +1.06% |  +2.93% |  −2.29% | BOTH |
| `amdocvrpbpltw` |    2.212 |  +1.30% |  +2.68% |  −1.79% | BOTH |

Among the three neural configurations, **BOTH is individually best on 49 variants, SRR-only on 45, and AUX-only on 16**. That count understates BOTH's value, though: its important advantage is that it avoids the severe failure modes of SRR-only on the complex `amdoc...` family while retaining most of SRR's large gains elsewhere.

The resulting architectural interpretation is fairly clean: **SRR_FIELD_EXPLORATION supplies breadth and search power; AUXILIARY_LOSS supplies robustness/compositional representation; the combination gives the best held-out performance and by far the best reference gap.**


Three-feature ablation configurations

Relative improvement against the common baseline used by the two newer runs. Positive is better. Mean and median across all 110 variants.

median	mean	config
1.149	2.032	AUX only
2.443	3.024	Both
1.766	2.718	SRR only

Conditional contribution of each feature

Mean percentage-point improvement obtained by adding one feature while the other is already enabled. This shows where each mechanism contributes.

group	srr	aux
All variants	0.992	0.306
Held-out	1.192	0.393
Asymmetric CVRP	1.582	1.032
Time-window CVRP	1.717	0.515
Open CVRP	0.882	0.686
Multi-depot CVRP	1.079	0.442
Prize-collecting TSP	1.875	0.576
Pickup-delivery	-0.822	1.582
