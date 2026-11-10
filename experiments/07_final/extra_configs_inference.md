# Additional configurations: stopping rule and constraint strength

Pooled weeks: 487 (2017-01-06 to 2026-05-01)

## S1. The early-stopping asymmetry

The ladder's M4 monitors the composite objective for early stopping while M4arch monitors the forecast term alone, so the M4arch-to-M4 contrast mixes the auxiliary gradient with a change in the stopping criterion. M4_fcstop is M4 with the forecast term monitored, which makes the M4arch-to-M4_fcstop contrast the auxiliary gradient alone.

**QLIKE** — mean loss

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.1206 | 0.0939 | 0.0974 | 0.1211 | 0.1418 |
| M4arch | 0.1193 | 0.0957 | 0.1002 | 0.1234 | 0.1409 |
| M4 | 0.1190 | 0.0953 | 0.0999 | 0.1237 | 0.1415 |
| M4_fcstop | 0.1190 | 0.0953 | 0.0999 | 0.1237 | 0.1415 |

**MAE** — mean loss

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.3557 | 0.2981 | 0.2818 | 0.2873 | 0.2881 |
| M4arch | 0.3643 | 0.3186 | 0.3073 | 0.3238 | 0.3179 |
| M4 | 0.3634 | 0.3168 | 0.3059 | 0.3225 | 0.3175 |
| M4_fcstop | 0.3634 | 0.3168 | 0.3059 | 0.3225 | 0.3175 |

**MSE** — mean loss

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.2120 | 0.1561 | 0.1490 | 0.1692 | 0.1862 |
| M4arch | 0.2198 | 0.1704 | 0.1632 | 0.1894 | 0.1996 |
| M4 | 0.2189 | 0.1691 | 0.1622 | 0.1892 | 0.2003 |
| M4_fcstop | 0.2189 | 0.1691 | 0.1622 | 0.1892 | 0.2003 |

### Decomposing the reported auxiliary-task effect (Holm-adjusted p; negative statistic favours the first specification)

**as reported (gradient + stopping rule): M4 vs M4arch**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | -2.23 (0.131) | -1.68 (0.374) | -1.17 (0.729) | 0.76 (0.729) | 1.03 (0.729) |
| MAE | -2.89 (0.016) | -3.25 (0.006) | -1.98 (0.146) | -1.36 (0.350) | -0.31 (0.755) |
| MSE | -3.28 (0.006) | -2.79 (0.022) | -1.91 (0.171) | -0.25 (0.967) | 0.70 (0.967) |

**auxiliary gradient alone: M4_fcstop vs M4arch**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | -2.23 (0.131) | -1.68 (0.374) | -1.17 (0.729) | 0.76 (0.729) | 1.03 (0.729) |
| MAE | -2.89 (0.016) | -3.25 (0.006) | -1.98 (0.146) | -1.36 (0.350) | -0.31 (0.755) |
| MSE | -3.28 (0.006) | -2.79 (0.022) | -1.91 (0.171) | -0.25 (0.967) | 0.70 (0.967) |

**stopping rule alone: M4 vs M4_fcstop**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | nan (0.000) | nan (0.000) | nan (0.000) | nan (0.000) | nan (0.000) |
| MAE | nan (0.000) | nan (0.000) | nan (0.000) | nan (0.000) | nan (0.000) |
| MSE | nan (0.000) | nan (0.000) | nan (0.000) | nan (0.000) | nan (0.000) |

## S2. Constraint strength beyond the reported range

The sweep reported in the paper tops out at a weight of 0.5 and moves the balance-consistency error by 2.3%, which is smaller than the dispersion across seeds; that leaves open whether the constraint was ever binding. Weights of 2 and 5 settle it.

### Balance-consistency error, pooled over the five windows (10³ bbl/week)

| Weight | mean | versus the unpenalised network |
|---|---|---|
| 0 (no penalty) | 5,223 | — |
| 2.0 | 2,467 | -52.8% |
| 5.0 | 2,349 | -55.0% |

**QLIKE** — mean loss across the constraint weights

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.1206 | 0.0939 | 0.0974 | 0.1211 | 0.1418 |
| M4 | 0.1190 | 0.0953 | 0.0999 | 0.1237 | 0.1415 |
| M5_lb0p1 | 0.1184 | 0.0955 | 0.0980 | 0.1208 | 0.1380 |
| M5_lb0p5 | 0.1152 | 0.0921 | 0.0914 | 0.1077 | 0.1224 |
| M5_lb2p0 | 0.1287 | 0.1114 | 0.1144 | 0.1304 | 0.1382 |
| M5_lb5p0 | 0.1343 | 0.1141 | 0.1157 | 0.1289 | 0.1334 |

**MAE** — mean loss across the constraint weights

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.3557 | 0.2981 | 0.2818 | 0.2873 | 0.2881 |
| M4 | 0.3634 | 0.3168 | 0.3059 | 0.3225 | 0.3175 |
| M5_lb0p1 | 0.3641 | 0.3204 | 0.3086 | 0.3260 | 0.3206 |
| M5_lb0p5 | 0.3651 | 0.3221 | 0.3113 | 0.3262 | 0.3243 |
| M5_lb2p0 | 0.3940 | 0.3684 | 0.3681 | 0.3914 | 0.3860 |
| M5_lb5p0 | 0.4014 | 0.3726 | 0.3718 | 0.3926 | 0.3889 |

**MSE** — mean loss across the constraint weights

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.2120 | 0.1561 | 0.1490 | 0.1692 | 0.1862 |
| M4 | 0.2189 | 0.1691 | 0.1622 | 0.1892 | 0.2003 |
| M5_lb0p1 | 0.2202 | 0.1724 | 0.1631 | 0.1901 | 0.2007 |
| M5_lb0p5 | 0.2212 | 0.1728 | 0.1625 | 0.1858 | 0.1967 |
| M5_lb2p0 | 0.2564 | 0.2257 | 0.2277 | 0.2597 | 0.2602 |
| M5_lb5p0 | 0.2637 | 0.2283 | 0.2291 | 0.2598 | 0.2583 |

### Each weight against the unpenalised dual-branch network (M4; Holm-adjusted p, negative favours the penalised specification)

**M5_lb0p1 vs M4**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | -0.26 (1.000) | 0.09 (1.000) | -0.61 (1.000) | -0.68 (1.000) | -0.71 (1.000) |
| MAE | 0.35 (1.000) | 1.35 (0.887) | 0.81 (1.000) | 0.94 (1.000) | 0.79 (1.000) |
| MSE | 0.53 (1.000) | 1.23 (1.000) | 0.27 (1.000) | 0.21 (1.000) | 0.10 (1.000) |

**M5_lb0p5 vs M4**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | -0.63 (1.000) | -0.48 (1.000) | -0.80 (1.000) | -0.92 (1.000) | -0.92 (1.000) |
| MAE | 0.36 (1.000) | 1.01 (1.000) | 0.74 (1.000) | 0.42 (1.000) | 0.70 (1.000) |
| MSE | 0.32 (1.000) | 0.51 (1.000) | 0.03 (1.000) | -0.22 (1.000) | -0.20 (1.000) |

**M5_lb2p0 vs M4**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | 1.14 (1.000) | 1.56 (0.601) | 0.95 (1.000) | 0.27 (1.000) | -0.11 (1.000) |
| MAE | 3.21 (0.004) | 3.55 (0.002) | 3.57 (0.002) | 3.03 (0.005) | 2.62 (0.009) |
| MSE | 2.83 (0.019) | 2.99 (0.015) | 2.71 (0.021) | 2.10 (0.073) | 1.56 (0.120) |

**M5_lb5p0 vs M4**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | 1.63 (0.446) | 1.70 (0.446) | 0.93 (1.000) | 0.18 (1.000) | -0.22 (1.000) |
| MAE | 3.75 (0.001) | 3.88 (0.001) | 3.66 (0.001) | 2.87 (0.008) | 2.59 (0.010) |
| MSE | 3.21 (0.007) | 3.22 (0.007) | 2.77 (0.018) | 1.99 (0.093) | 1.37 (0.171) |

### The same weights excluding 2020

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.1014 | 0.0678 | 0.0541 | 0.0499 | 0.0615 |
| M4 | 0.1034 | 0.0740 | 0.0619 | 0.0607 | 0.0703 |
| M5_lb0p5 | 0.1062 | 0.0776 | 0.0648 | 0.0643 | 0.0714 |
| M5_lb2p0 | 0.1209 | 0.0989 | 0.0913 | 0.0941 | 0.0957 |
| M5_lb5p0 | 0.1258 | 0.1009 | 0.0930 | 0.0960 | 0.0957 |

