# Petroleum fundamentals and crude oil volatility forecasts

Replication materials for **When Do Petroleum Fundamentals Improve Crude Oil Volatility Forecasts? Evidence Across Representations, Evaluation Scales, and Market Periods**, by Suho Ahn (KAIST; ORCID 0009-0008-0232-2317).

This update accompanies the Journal of Forecasting submission and extends the original v1.0.0 archive ("Manufactured gains") with representation controls, nested selection, HAR-X, and balance-consistency results. No specification consistently improves on HAR-RV across evaluation scales and samples. This finding is conditional on the tested WTI setting.

## Environment and data

Run from this folder after installing requirements:
~~~
python -m pip install -r requirements.txt
~~~
The included EIA vintage and processed weekly dataset permit reproduction without downloading revised data. Sources are in data/metadata/eia_series_metadata.json. Inputs are in data/interim and the aligned dataset is data/processed/dataset_weekly.csv. Downloading newer EIA data can change results.

Neural estimates use seeds 0–4. Newly trained predictions can depend on hardware and library versions. Stored per-week losses allow inference without retraining. This update does not certify a clean-environment dependency installation.

## Current manuscript mapping

| Evidence | Script or artifact |
|---|---|
| Table 1: sources and variables | data/metadata/eia_series_metadata.json; src/data/build_dataset.py |
| Table 2: descriptive statistics | experiments/01_data_audit/descriptive_stats.py |
| Table 3: specifications | src/models/ladder.py; src/training/run_ladder.py |
| Tables 4–5: pooled QLIKE and 2020 | experiments/07_final/final_inference.py |
| Table 6: evaluation scales | experiments/07_final/scale_and_stability.py |
| Table 7: directional accuracy | experiments/06_robustness/ew_extra_*.csv |
| Table 8: model contrasts | experiments/07_final/representation_decomposition.py; experiments/07_final/extra_configs_inference.py |
| Table 9: subsamples | experiments/06_robustness/subsample_robustness.py |
| Supplement S10: HAR-X | experiments/07_final/harx_inference.py |
| Supplement S11: multiplicity | experiments/07_final/ladder_multiplicity.py |
| Supplement S12: nested selection | experiments/07_final/nested_inference.py; experiments/06_robustness/nested_selection.csv |
| Balance consistency / Figure 6 | experiments/06_robustness/ew_bce_ladder.csv; experiments/06_robustness/ew_bce_extra.csv |
| Final figures | submission_figures/Fig1.pdf through Fig6.pdf |

To check the evidence without training:
~~~
python src/data/test_leakage.py
python experiments/07_final/final_inference.py
python experiments/07_final/scale_and_stability.py
python experiments/07_final/representation_decomposition.py
python experiments/07_final/nested_inference.py
python experiments/07_final/harx_inference.py
python experiments/07_final/ladder_multiplicity.py
~~~

## Interpretation and aggregation

The leakage check has 26 assertions. Its chronological split check concerns the single split, not selection independence of the fixed-setting pooled evaluation. The common 2015–2019 validation interval overlaps part of the pooled test period. The nested check covers M2, M3derived, and M5 only: seed 0 selects settings within each window, then five seeds are refitted.

Balance-consistency error is the equal-weight mean of five window means, not the week-weighted mean over 487 weeks. Relative to about 5,223 thousand barrels, weights 0.1 and 0.5 give about 3,505 and 2,864 (32.9% and 45.2% reductions). It conditions on observed future SPR changes and is an ex-post diagnostic; those observations are not forecast inputs. Lower balance error does not establish better volatility forecasting.

Raw and derived inputs differ in dimension and composition. Their contrast is not a pure causal effect of representation. Older generated reports may use prior table numbering and labels such as "published"; the current manuscript and mapping above govern interpretation.

## Retraining and figures

experiments/06_robustness contains expanding_window.py (base ladder), expanding_window_representation.py (M3derived/M2wide), expanding_window_nested.py (three-model nested selection), expanding_window_extra.py and ew_bce_backfill.py (penalty checks), and harx_robustness.py (HAR-X).

No model training was repeated for this packaging update. Final vector figures, including layout corrections, are supplied in submission_figures. The legacy src/visualization/make_figures.py does not reproduce every final layout adjustment. results and archive retain older outputs for provenance. Pre-correction RV-window outputs and return-target experiments are not evidence for the current manuscript.

## Archive and license

Original public archive: https://doi.org/10.5281/zenodo.21892583 (v1.0.0).
This is an update package. The original version does not contain these added files; a new archive version is needed for full public correspondence with the submitted manuscript.

Code: MIT. EIA data remain subject to the provider's terms.
