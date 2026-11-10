# Unanchored specifications

Pooled weeks: 487 (2017-01-06 to 2026-05-01)

## U1. The cost of removing the anchor

**QLIKE** — mean loss

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.1206 | 0.0939 | 0.0974 | 0.1211 | 0.1418 |
| M1 | 0.1207 | 0.0953 | 0.0998 | 0.1253 | 0.1464 |
| M5_lb0p1 | 0.1184 | 0.0955 | 0.0980 | 0.1208 | 0.1380 |
| M1_noanchor | 0.1298 | 0.1034 | 0.1029 | 0.1210 | 0.1410 |
| M5_lb0p1_noanchor | 0.1263 | 0.0964 | 0.0917 | 0.1027 | 0.1171 |

**MAE** — mean loss

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.3557 | 0.2981 | 0.2818 | 0.2873 | 0.2881 |
| M1 | 0.3578 | 0.3028 | 0.2880 | 0.3031 | 0.3039 |
| M5_lb0p1 | 0.3641 | 0.3204 | 0.3086 | 0.3260 | 0.3206 |
| M1_noanchor | 0.3618 | 0.3068 | 0.2902 | 0.2977 | 0.2937 |
| M5_lb0p1_noanchor | 0.3602 | 0.3064 | 0.2972 | 0.3065 | 0.3047 |

**MSE** — mean loss

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.2120 | 0.1561 | 0.1490 | 0.1692 | 0.1862 |
| M1 | 0.2127 | 0.1594 | 0.1541 | 0.1789 | 0.1937 |
| M5_lb0p1 | 0.2202 | 0.1724 | 0.1631 | 0.1901 | 0.2007 |
| M1_noanchor | 0.2226 | 0.1680 | 0.1564 | 0.1720 | 0.1847 |
| M5_lb0p1_noanchor | 0.2215 | 0.1650 | 0.1520 | 0.1641 | 0.1737 |

Percentage cost of dropping the anchor, relative to the anchored counterpart (positive means the unanchored version is worse).

| Pair and loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| M1_noanchor vs M1, QLIKE | +7.5% | +8.4% | +3.1% | -3.4% | -3.7% |
| M1_noanchor vs M1, MAE | +1.1% | +1.3% | +0.8% | -1.8% | -3.3% |
| M1_noanchor vs M1, MSE | +4.7% | +5.4% | +1.5% | -3.8% | -4.7% |
| M5_lb0p1_noanchor vs M5_lb0p1, QLIKE | +6.7% | +0.9% | -6.4% | -15.0% | -15.1% |
| M5_lb0p1_noanchor vs M5_lb0p1, MAE | -1.1% | -4.3% | -3.7% | -6.0% | -5.0% |
| M5_lb0p1_noanchor vs M5_lb0p1, MSE | +0.6% | -4.3% | -6.8% | -13.7% | -13.5% |

## U2. Does the physical block help when the anchor is absent?

If anchoring removed the channel through which fundamentals act, the physical specification should beat the price-only one once the anchor is gone. Entries are Holm-adjusted p-values; a negative statistic favours the physical specification.

**with the anchor (as in the paper): M5_lb0p1 vs M1**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | -0.75 (1.000) | 0.04 (1.000) | -0.39 (1.000) | -0.62 (1.000) | -0.87 (1.000) |
| MAE | 1.25 (0.426) | 2.17 (0.153) | 1.97 (0.196) | 1.67 (0.284) | 1.12 (0.426) |
| MSE | 1.42 (0.629) | 1.76 (0.395) | 1.07 (0.859) | 0.97 (0.859) | 0.54 (0.859) |

**without the anchor: M5_lb0p1_noanchor vs M1_noanchor**

| Loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| QLIKE | -0.98 (0.686) | -1.49 (0.686) | -1.36 (0.686) | -1.47 (0.686) | -1.32 (0.686) |
| MAE | -0.28 (1.000) | -0.05 (1.000) | 0.72 (1.000) | 0.56 (1.000) | 0.59 (1.000) |
| MSE | -0.19 (1.000) | -0.46 (1.000) | -0.48 (1.000) | -0.58 (1.000) | -0.61 (1.000) |

## U3. The unanchored specifications against the benchmark

| Specification and loss | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| M1_noanchor, QLIKE | +7.6% (1.000) | +10.1% (1.000) | +5.6% (1.000) | -0.1% (1.000) | -0.6% (1.000) |
| M1_noanchor, MAE | +1.7% (1.000) | +2.9% (1.000) | +3.0% (1.000) | +3.6% (1.000) | +2.0% (1.000) |
| M1_noanchor, MSE | +5.0% (1.000) | +7.6% (1.000) | +4.9% (1.000) | +1.7% (1.000) | -0.8% (1.000) |
| M5_lb0p1_noanchor, QLIKE | +4.7% (1.000) | +2.6% (1.000) | -5.8% (1.000) | -15.2% (1.000) | -17.4% (1.000) |
| M5_lb0p1_noanchor, MAE | +1.3% (0.940) | +2.8% (0.940) | +5.5% (0.940) | +6.7% (0.940) | +5.8% (0.940) |
| M5_lb0p1_noanchor, MSE | +4.5% (1.000) | +5.7% (1.000) | +2.0% (1.000) | -3.0% (1.000) | -6.7% (1.000) |

## U4. Excluding 2020

| Specification | h = 1 | h = 2 | h = 4 | h = 8 | h = 12 |
|---|---|---|---|---|---|
| HAR | 0.1014 | 0.0678 | 0.0541 | 0.0499 | 0.0615 |
| M1 | 0.1063 | 0.0741 | 0.0627 | 0.0620 | 0.0733 |
| M5_lb0p1 | 0.1057 | 0.0767 | 0.0640 | 0.0631 | 0.0725 |
| M1_noanchor | 0.1148 | 0.0810 | 0.0683 | 0.0655 | 0.0737 |
| M5_lb0p1_noanchor | 0.1137 | 0.0789 | 0.0655 | 0.0598 | 0.0661 |

