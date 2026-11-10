# Data-snooping tests across the whole candidate set

Candidates (11): M1, M2, M2wide, M3derived, M3, M4arch, M4, M5_lb0p1, M5_lb0p5, M5_lb2p0, M5_lb5p0
Pooled weeks: 487. Stationary bootstrap, 5000 replications, mean block max(4, h).

The null is that no candidate beats the benchmark once the full set is accounted for. A small p-value would be evidence that the best candidate's advantage is not an artefact of searching over many of them.

### QLIKE, realized variance

| h | best candidate | its mean advantage | Reality Check p | SPA p |
|---|---|---|---|---|
| 1 | M5_lb0p5 | +14.0% | 0.182 | 0.170 |
| 2 | M5_lb0p5 | +16.6% | 0.212 | 0.253 |
| 4 | M5_lb0p5 | +24.4% | 0.203 | 0.219 |
| 8 | M5_lb5p0 | +34.3% | 0.204 | 0.221 |
| 12 | M5_lb5p0 | +42.5% | 0.179 | 0.216 |

### MSE, realized variance

| h | best candidate | its mean advantage | Reality Check p | SPA p |
|---|---|---|---|---|
| 1 | M5_lb0p5 | +25.5% | 0.132 | 0.129 |
| 2 | M5_lb0p5 | +20.7% | 0.155 | 0.159 |
| 4 | M5_lb0p5 | +18.4% | 0.215 | 0.180 |
| 8 | M5_lb0p1 | +6.5% | 0.530 | 0.301 |
| 12 | M5_lb0p1 | +4.7% | 0.562 | 0.301 |

### QLIKE, volatility (as published)

| h | best candidate | its mean advantage | Reality Check p | SPA p |
|---|---|---|---|---|
| 1 | M5_lb0p5 | +4.5% | 0.336 | 0.442 |
| 2 | M5_lb0p5 | +1.9% | 0.594 | 0.661 |
| 4 | M5_lb0p5 | +6.2% | 0.402 | 0.519 |
| 8 | M5_lb0p5 | +11.0% | 0.337 | 0.432 |
| 12 | M5_lb0p5 | +13.7% | 0.297 | 0.441 |

### MAE, log volatility (as published)

| h | best candidate | its mean advantage | Reality Check p | SPA p |
|---|---|---|---|---|
| 1 | M3derived | +0.9% | 0.614 | 0.351 |
| 2 | M3derived | -0.6% | 0.937 | 0.654 |
| 4 | M3derived | -0.9% | 0.948 | 0.671 |
| 8 | M3derived | -2.3% | 0.986 | 0.695 |
| 12 | M3derived | -2.8% | 0.994 | 0.869 |

### MSE, log volatility (as published)

| h | best candidate | its mean advantage | Reality Check p | SPA p |
|---|---|---|---|---|
| 1 | M3derived | +1.9% | 0.573 | 0.406 |
| 2 | M3derived | -0.5% | 0.850 | 0.770 |
| 4 | M3derived | -2.2% | 0.938 | 0.771 |
| 8 | M3derived | -4.9% | 0.987 | 0.821 |
| 12 | M3derived | -3.4% | 0.970 | 0.833 |

## Reading

On the scales where the point estimates favour the candidates, the tests do not reject: the best of eleven specifications does not beat the benchmark by more than a search over eleven specifications would be expected to produce. On the log scale the point estimates favour the benchmark, so the tests cannot reject by construction. Either way the conclusion is the same as the horizon-wise inference reached, and it now accounts for the size of the search rather than for five horizons alone.
