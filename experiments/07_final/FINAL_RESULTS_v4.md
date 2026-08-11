# Final Results after the RV-window fix (v4, 2026-08-11)

All numbers below come from the corrected dataset (1,151 weeks; RV windows of
exactly 7h calendar days, verified by leakage tests T6a/T6b). The earlier v3
numbers are void — see PROJECT_STATUS.md.

Evidence base: single split (train ->2014, val 2015-19, test 2020-26, 5 seeds)
and a five-window expanding evaluation pooling 487 non-overlapping test weeks
(2017-01 -> 2026-05, 5 seeds, 6 neural configs + HAR).

## What is statistically established

**1. Feeding fundamentals as inputs is actively harmful.**
Pooled EW, M2/M3 vs M1 (DM): h1 p=0.001/0.000, h2 p=0.007/0.000, h4
p=0.021/0.004, h8 p=0.076/0.376. M2 and M3 are EXCLUDED from the 90% model
confidence set at h1, h2, h4, h8 (M2 also near-excluded at h12, p=0.071),
while HAR/M1/M4/M5 all remain.

**2. Predicting the physical state repairs that damage, significantly.**
M4 vs M2 (DM): h1 p=0.000, h2 p=0.003, h4 p=0.006, h8 p=0.014, h12 p=0.020 —
significant at EVERY horizon. M4 vs M3: significant at h1-h8.
=> The same information helps or hurts depending on how the model consumes it.
This is the paper's cleanest, fully significant contribution.

## What is directionally supported but NOT significant

**3. Multi-task models give the lowest mean losses at multi-week horizons.**
Pooled EW mean QLIKE at h12: M5(0.5) 0.1110, M4 0.1164, M5(0.1) 0.1171 vs
M1 0.1410 and HAR 0.1418 (-17% to -21%). At h8: M5(0.5) 0.1010, M5(0.1) 0.1027,
M4 0.1033 vs M1 0.1210, HAR 0.1211. But DM p ~ 0.24-0.37 vs M1 and 0.37-0.46
vs HAR; all remain jointly in the MCS. **We cannot claim to beat HAR.**

**4. The balance constraint is neutral on average.**
M5 vs M4 pooled: |DM| <= 1.17, p >= 0.24 at every horizon. Validation selects
lambda_B = 0.1, which lowers mean QLIKE at 4 of 5 horizons on the single split
and 3 of 5 pooled — consistent but not significant. BCE falls with lambda
(6,168 -> 5,767 kb/wk at lambda=0.5).

## Regime findings (descriptive, means)

**5. Gains concentrate in inventory-BUILD windows** (thresholds recomputed per
horizon from train-period h-week average stock changes):
- h12 build (n=54): M5(0.5) 0.4618, M4 0.5998 vs M1 0.7955, HAR 0.8379
  (-42% vs M1, -45% vs HAR)
- h4 build (n=73): M5(0.5) 0.1914, M4 0.2093 vs M1 0.2655, HAR 0.2414
- Draw windows show no such advantage (M1 is marginally best at h4/h12).

**6. Honest counter-evidence on the strong constraint.** In calm weeks
(RV below median) lambda=0.5 is significantly WORSE than M4 at h1 (p=0.027)
and h2 (p=0.021). Excluding COVID-2020, M4 is significantly worse than HAR at
h1 (p=0.026) and h2 (p=0.034). The physical channel does not pay at short
horizons in quiet markets.

## Other checks

- **External baselines** (single split, test QLIKE): LSTM 0.154/0.135/0.141/
  0.165/0.179; XGBoost 0.244/0.191/0.203/0.208/0.216; TCN 0.222/0.222/0.246/
  0.253/0.255; ARIMA worse at all horizons. Off-the-shelf learners given the
  full feature set do not reach the multi-task models at h8/h12.
- **Methodology breaks**: the 2016-07 weekly-exports change predates every
  pooled test week, so it can affect training data only. Rankings are stable
  across the 2022-04 (SPR drawdowns) and 2023-03 (EIA transfers line) cuts.
- **Forecast combination** (HAR + M5, equal weights): significantly WORSE than
  HAR at h1 (p=0.040), better in means but not significantly at h8/h12.
  Combination does not rescue a significant win over HAR.
- **Economic value** (weekly vol targeting, h=1, no costs): Sharpe 0.140 (M1),
  0.135 (M4), 0.140 (M5 lambda=0.1), 0.140 (M5 lambda=0.5) vs 0.083 buy-and-
  hold. Vol targeting beats buy-and-hold; models are indistinguishable.

## Bottom line for the manuscript

Defensible headline: **how a model consumes physical fundamentals decides
whether they help** — as inputs they significantly degrade weekly WTI
volatility forecasts and drop out of the model confidence set; as auxiliary
prediction targets they significantly repair that damage and deliver the lowest
mean losses at multi-week horizons, with gains concentrated in inventory-build
windows. Not defensible: a claim of beating HAR, or of a significant accuracy
gain from the balance constraint.
