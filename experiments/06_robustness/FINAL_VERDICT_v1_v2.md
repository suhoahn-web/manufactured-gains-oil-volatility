# Experimental Verdict after Two Design Iterations (2026-08-11)

## Question tested
Does a soft petroleum-balance consistency loss (M5) improve weekly WTI return
forecasts over an identical unconstrained multi-task model (M4)? (Design doc
section 9: the critical test M5 > M4.)

## Evidence (both iterations, 5 seeds, validation-only selection)

v1 (level aux targets, horizons 1/2/4w) and v2 (change aux targets — harder,
non-trivial auxiliary task — horizons 1/2/4/8/12w):

1. **M5 = M4 statistically everywhere.** DM tests: |DM| <= 0.78, p >= 0.43 at
   every horizon in v2 (p >= 0.63 in v1). No horizon, lambda, inventory regime,
   or COVID-exclusion subsample shows a significant M5 advantage.
2. **Validation always selects the weakest constraint** (lambda_B = 0.01 in
   both iterations; performance degrades monotonically in lambda_B).
3. **Fundamentals add no price-forecast value.** M1 (price-only GRU) is the
   best model at every horizon in both iterations. In v2 the multi-task models
   close to within noise of M1 (DM p >= 0.25) — fundamentals no longer hurt,
   but never help. This replicates Miao et al. (2017; LASSO never selects
   supply-side variables) and Yu et al. (2026; ERT discards production and
   inventory) with a stricter leak-free protocol.
4. **The constraint does what it claims physically.** Balance Consistency
   Error falls monotonically with lambda_B (v2: 6952 -> 6425 kb/wk, -8%), and
   the change-target redesign cut BCE ~16x vs v1. The constraint produces
   internally consistent physical predictions at zero price-accuracy cost —
   but zero benefit too.

## Verdict per design doc sections 25/34
The structural-constraint claim for WEEKLY WTI RETURN forecasting is not
supported and must not be forced. Options: reframe the contribution or pivot
the target variable. See PROJECT_STATUS.md for the decision.

## What survives for any successor design
- Leak-free release-aligned dataset + 13 automated tests (fully reusable)
- Balance audit validating the EIA identity and soft-constraint rationale
- Literature base: 27 papers read, novelty matrix, key-paper structure maps
- Model ladder code (M0-M5) with validation-based selection machinery
- Negative result itself: rigorous evidence that fundamentals-as-inputs and
  balance constraints do not improve weekly WTI return forecasts post-2020
