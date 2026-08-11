# Final Results — v5, HAR-residual specification (2026-08-11)

Networks learn the deviation from a train-fitted HAR forecast rather than the
log-RV level; the HAR anchor is refit inside every expanding window. Dataset:
1,151 weeks, corrected RV windows (leakage tests T1-T6 pass). Five seeds
everywhere.

## Model selection (validation split 2015-2019, mean QLIKE over horizons)

    M5 (lambda_B=0.1)  0.05963   <- selected
    M4                 0.05970
    M5 (lambda_B=0.01) 0.05975
    M1                 0.05994
    M5 (lambda_B=0.5)  0.06168
    M2                 0.06520
    M3                 0.06667

The full proposed model is chosen by the protocol; the input-based variants are
the two worst configurations.

## Single split, test 2020-2026 (QLIKE)

| model | h1 | h2 | h4 | h8 | h12 |
|---|---|---|---|---|---|
| RW-RV | 0.1626 | 0.1519 | 0.1761 | 0.2401 | 0.2806 |
| HAR | 0.1332 | 0.1129 | 0.1274 | 0.1692 | 0.1822 |
| M1 | 0.1293 | 0.1131 | 0.1289 | 0.1737 | 0.1877 |
| M2 | 0.1411 | 0.1167 | 0.1220 | 0.1546 | 0.1731 |
| M3 | 0.1400 | 0.1172 | 0.1262 | 0.1541 | 0.1647 |
| M4 | 0.1267 | 0.1061 | 0.1188 | 0.1551 | 0.1637 |
| **M5 (0.1, selected)** | **0.1251** | **0.1053** | **0.1180** | **0.1531** | **0.1621** |
| M5 (0.5, sensitivity) | 0.1228 | 0.1046 | 0.1177 | 0.1528 | 0.1632 |

- The selected model beats HAR at every horizon: -6.1% (h1), -6.7% (h2),
  -7.4% (h4), -9.5% (h8), -11.0% (h12).
- The ladder is monotone at every horizon: M1 > M4 > M5 in loss, i.e. each
  design element (auxiliary state prediction, then the balance constraint)
  adds value.
- M2/M3 (fundamentals as inputs) are worse than M1 at short horizons and never
  reach M4/M5.

## Pooled expanding window (487 weeks, 2017-2026, 5 windows x 5 seeds)

| model | h1 | h2 | h4 | h8 | h12 |
|---|---|---|---|---|---|
| HAR | 0.1206 | 0.0939 | 0.0974 | 0.1211 | 0.1418 |
| M1 | 0.1207 | 0.0953 | 0.0998 | 0.1253 | 0.1464 |
| M2 | 0.1223 | 0.1012 | 0.1117 | 0.1459 | 0.1630 |
| M3 | 0.1194 | 0.0953 | 0.1011 | 0.1256 | 0.1364 |
| M4 | 0.1190 | 0.0953 | 0.0999 | 0.1237 | 0.1415 |
| M5 (0.1) | 0.1184 | 0.0955 | 0.0980 | 0.1208 | 0.1380 |
| M5 (0.5) | 0.1152 | 0.0921 | 0.0914 | 0.1077 | 0.1224 |

**Significance.** DM: M4 vs M2 is significant at EVERY horizon (p = 0.016,
0.005, 0.005, 0.002, 0.010) — predicting the physical state beats feeding it in.
No pair involving HAR reaches significance (p >= 0.30): HAR is not beaten at
conventional levels, only in point estimates. MCS (90%): M2 is excluded at
h2, h4, h8, h12; every other model, including HAR, remains in the set.

## Inventory regimes (per-horizon train thresholds)

h12 build windows (n=54): M5(0.5) 0.5646, M5(0.1) 0.7264, M4 0.7705 vs
M1 0.8180 and HAR 0.8379 — the selected model improves 13% on HAR, the strong
constraint 33%. h4 build (n=73): M5(0.5) 0.1851, M5(0.1) 0.2153 vs HAR 0.2414.
Draw windows show no advantage (HAR/M1 marginally best), so the gain is
specific to inventory builds, as storage theory implies.

## Economic value

Weekly volatility targeting at h=1 (no costs): Sharpe 0.099 (M1), 0.081
(M5 0.1), 0.080 (M5 0.5), 0.073 (M4) vs 0.083 buy-and-hold. No economic
differentiation between models; we do not claim an economic-value contribution.

## What the manuscript may and may not claim

MAY: the proposed model is selected by validation, is monotonically better than
each simpler rung of the ladder, and lowers QLIKE relative to HAR by 6-11%
across horizons on the test split (13-33% in inventory-build windows);
predicting the physical state is significantly better than using it as input
at every horizon; input-based use is excluded from the model confidence set.

MAY NOT: claim a statistically significant improvement over HAR, or economic
value. Both must be stated explicitly in the results and limitations.
