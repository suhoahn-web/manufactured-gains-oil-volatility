# Balance Definition Decision (design doc steps 3-4)

Date: 2026-08-11. Based on `balance_audit.py` results over 1991-02-08 -> 2026-07-31 (1852 weeks).

## Audit findings

1. Reconstructed net flow `7*(production + imports - refinery_inputs - exports)` tracks the
   observed weekly stock change with r = 0.72 (total stocks incl SPR) / r = 0.66 (commercial ex-SPR).
2. The residual (implied adjustment) has mean ~263 kb/d, std ~545 kb/d (total-stock definition).
   This matches the magnitude EIA itself reports for its official crude-oil adjustment
   ("unaccounted-for crude oil"), including the well-documented large positive adjustments in
   2022-2024 that led EIA to add the "transfers to crude oil supply" line in 2023.
   => The equation reproduces the EIA balance structure correctly; the residual IS the adjustment.
3. Residual variance grows after the mid-2016 weekly-exports methodology change and again after
   2022 (production methodology + transfers). Recorded in `data/metadata/methodology_breaks.csv`.

## Decisions

**D1 — Stock definition for the model: commercial crude stocks ex-SPR (`WCESTUS1`).**
Commercial stocks are the price-relevant inventory measure (as in the design doc) and the series
used in WPSR headlines. SPR is policy-driven, not market-driven.

**D2 — Balance equation used in the loss (explicit SPR term):**

    d_commercial_t + d_SPR_t = 7 * (production_t + imports_t + adjustment_t
                                     - refinery_inputs_t - exports_t)

d_SPR is observed (and near-zero most weeks); at prediction time the model predicts commercial
stocks, and the balance loss uses last-known SPR change = 0 assumption OR predicts total-stock
side consistently. This keeps definitions unmixed (design doc section 3 requirement).

**D3 — Adjustment variable: implied residual (audited).**
Official weekly adjustment history is not available in navigator bulk files; the audit above
validates the implied-residual derivation:

    adjustment_t = (d_commercial_t + d_SPR_t)/7 - production_t - imports_t
                   + refinery_inputs_t + exports_t

By construction the historical balance closes exactly with this variable; the model's balance
loss penalizes PREDICTED states whose implied inconsistency deviates, with adjustment predicted
as one of the auxiliary heads (it is highly autocorrelated, esp. post-2016).

**D4 — Soft constraint confirmed.** EIA components are independently estimated; residual std of
~550 kb/d (≈ 3.8M barrels/week) is far from zero => hard equality constraint would be wrong.
Lambda sweep as designed (0, 0.01, 0.05, 0.1, 0.5, 1.0).

**D5 — Common clean sample: constrained by Cushing stocks (starts 2004-04-09).**
All 10 series complete from 2004-04-09. Design split adapts to:
  - Train: 2004-04-09 -> 2014-12-31 (~560 weeks)
  - Validation: 2015-01-01 -> 2019-12-31 (~261 weeks)
  - Test: 2020-01-01 -> latest (~344 weeks)
