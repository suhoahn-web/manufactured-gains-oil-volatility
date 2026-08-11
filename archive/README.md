# Archive — superseded material

Nothing in this folder is used by the results reported in the manuscript. It is kept
so that the development path is inspectable rather than hidden.

## Why these were superseded

**Return-target experiments.** The study began by forecasting weekly WTI *returns* rather than
realized volatility. No configuration improved on a price-only benchmark at any horizon, and the
target was changed. `run_v2_ladder.py`, `run_lambda_sweep.py` and `run_tuning_sweep.py` belong to
that stage.

**Pre-correction volatility experiments.** `run_v3_vol_ladder.py` and `v3_vol_analysis.py` were run
before an internal audit found that the realized-volatility window spanned `7h − 5` calendar days
instead of `7h`, which made the one-week target a two-day realized volatility and silently dropped
95 weeks. **Their numerical output is invalid and must not be used.** The corrected definition is
enforced by tests `T6a`/`T6b` in `src/data/test_leakage.py`, and every result in the manuscript
comes from the corrected pipeline.

**Superseded drivers.** `rerun_all_v4.py`, `rerun_stage2.py` and `rerun_v6_stage2.py` are stage
drivers from intermediate runs, replaced by `src/training/rerun_v5_har_residual.py` and
`src/training/rerun_v6_controls.py`.

**Superseded analyses.** `combination_test.py` tested a HAR + neural forecast combination that did
not beat HAR and is not reported. `FINAL_VERDICT_v1_v2.md` and `FINAL_RESULTS_v4.md` record the
conclusions of the return-target stage and of the pre-HAR-anchor volatility stage respectively;
both are retained for provenance and are contradicted by the final results.
