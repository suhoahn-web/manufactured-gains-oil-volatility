# Manufactured gains — replication package

Replication code and data for the manuscript

> **Manufactured gains: how loss choice and sample period create forecasting improvements that a matched ladder removes**

The study asks whether weekly U.S. petroleum fundamentals improve forecasts of weekly WTI
realized volatility beyond a HAR benchmark. The answer is negative, and the package is built so
that a reviewer can verify both the negative result and the two checks that establish it: the
apparent quasi-likelihood advantage of the richest specification comes entirely from 52 weeks of
2020, and it reverses under mean absolute error.

Everything below runs on CPU. A complete reproduction takes roughly **6 hours on 8 cores**; the
individual stages are listed with their own runtimes so that a reviewer can verify any single
claim without rerunning the whole pipeline.

---

## 1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/data/download_eia.py        # ~2 min   (or skip: the vintage is in data/raw/eia)
python src/data/build_dataset.py       # ~3 min
python src/data/test_leakage.py        # ~2 min   MUST print "ALL LEAKAGE TESTS PASSED"
```

`test_leakage.py` is the first thing to run. It is the guard on every result in the paper: it
verifies release alignment, that each horizon's realized-volatility window spans exactly `7h`
calendar days, that every target lies strictly after its forecast origin, and that the splits are
chronological. If it fails, nothing downstream is trustworthy.

---

## 2. Reproducing each claim in the paper

| Paper claim | Command | Runtime | Output |
|---|---|---|---|
| Table 1, Table 2 (data description) | `python experiments/01_data_audit/descriptive_stats.py` | 1 min | `experiments/01_data_audit/descriptive_stats.md` |
| Balance identity audit (Section 2.3) | `python experiments/01_data_audit/balance_audit.py` | 2 min | `balance_audit.csv`, `balance_audit_summary.md` |
| Single-split ladder, five seeds | `python src/training/rerun_v5_har_residual.py` | ~3 h | `results/tables/ladder_results_v5*.csv` |
| Architecture control (M4arch) | `python src/training/run_ladder.py --models M4 --lambda-f 0 --lambda-b 0 --target vol --har-residual --seeds 5 --tag _v6arch` | ~25 min | `ladder_results_v6arch.csv` |
| External baselines (ARIMA, XGBoost, LSTM, TCN) | `python experiments/02_baselines/external_baselines_vol.py` | ~50 min | `ladder_results_v5ext.csv` |
| **Table 3, Table 5 — pooled expanding window** | `python experiments/06_robustness/expanding_window.py` | ~3 h | `ew_losses_*.csv`, `ew_extra_*.csv` |
| Model confidence set | `python experiments/06_robustness/mcs_test.py` | 3 min | `mcs_results.csv` |
| **Table 4 — the 2020 decomposition, and all DM tests** | `python experiments/07_final/final_inference.py` | 2 min | `experiments/07_final/final_inference.md` |
| Sub-period robustness | `python experiments/06_robustness/subsample_robustness.py` | 1 min | `subsample_robustness.md` |
| Inventory regimes, economic value | `python experiments/07_final/stress_and_econ_value.py` | 2 min | `stress_econ_results.md` |
| Figures 1–6 | `python src/visualization/make_figures.py` | 2 min | `results/figures/*.pdf` |

**To check the headline claims without training anything**, the per-week loss files are shipped in
`experiments/06_robustness/` (`ew_losses_*.csv`, `ew_extra_*.csv`). Running
`experiments/07_final/final_inference.py` on them reproduces every Diebold–Mariano statistic,
every Holm-adjusted p-value, and the 2020 decomposition in about two minutes.

---

## 3. What the specifications are

All specifications share the release-aligned data, a 12-week lookback, GRU encoders, the same
optimizer and early-stopping rule, five seeds (0–4), and the same HAR anchor. They differ only in
the rung being tested.

| Name | Physical variables | Encoder | Auxiliary task | Balance penalty |
|---|---|---|---|---|
| `HAR` | — | OLS benchmark | — | — |
| `M1` | none | single | — | — |
| `M2` | raw levels | single, concatenated | — | — |
| `M3` | raw + balance-derived | single, concatenated | — | — |
| `M4arch` | raw + balance-derived | **dedicated branch** | — | — |
| `M4` | raw + balance-derived | dedicated branch | yes (λ_F > 0) | — |
| `M5` | raw + balance-derived | dedicated branch | yes | yes (λ_B > 0) |

`M4arch` is the architecture control. It has **exactly the same information set as M3** and differs
only in where that information enters the network. Without it, comparing `M4arch` with `M2`
returns significance at every horizon for an effect that is in fact null — this is the confound the
paper is built to expose.

---

## 4. Repository layout

```
src/
  data/          download_eia.py, build_dataset.py, test_leakage.py
  models/        ladder.py            model definitions, scalers, balance loss
  training/      run_ladder.py        the single entry point for all specifications
                 rerun_v5_har_residual.py, rerun_v6_controls.py   stage drivers
  visualization/ make_figures.py
experiments/
  01_data_audit/     balance audit, descriptive statistics
  02_baselines/      ARIMA, XGBoost, LSTM, TCN
  05_inventory_stress/  shared helpers (DM test, seed-ensemble loader)
  06_robustness/     expanding window, MCS, sub-period analysis, per-week losses
  07_final/          final inference, inventory regimes, economic value
data/
  raw/eia/           the EIA vintage as downloaded (31 July 2026), unmodified
  eia_series_metadata.json, methodology_breaks.csv, release_alignment.md
results/
  tables/            per-seed results for every specification
  figures/           Figures 1–6 as vector PDF and 600 dpi PNG
```

---

## 5. Data

All inputs are public series published by the U.S. Energy Information Administration. The exact
vintage used in the paper (downloaded 31 July 2026) is included under `data/raw/eia/` so that the
results reproduce even if the EIA revises the series; `src/data/download_eia.py` re-downloads the
current vintage if you prefer. Series identifiers, units, and coverage are in
`data/eia_series_metadata.json`.

Two data facts that matter for interpretation and are handled explicitly in the code:

- **The crude-oil adjustment is not published** in the weekly navigator series. It is derived as the
  residual of the balance identity (`build_dataset.py`), so the identity closes exactly in sample.
- **The WTI spot price was −36.98 USD/bbl on 20 April 2020.** The two daily log returns spanning
  that print are undefined and are excluded from realized-volatility sums, which understates
  volatility for one `h = 1` week and up to twelve consecutive `h = 12` origins.

The series are **revised vintages, not real-time vintages**. This is a stated limitation of the
paper: a null obtained on revised data is weaker than one obtained on vintage data.

---

## 6. Inference

- **Diebold–Mariano** with the Harvey small-sample correction and a Newey–West long-run variance,
  bandwidth `max(h-1, floor(4(n/100)^(2/9)))`. The customary truncation at `h-1` is *not* used
  because it applies no autocovariance correction at all at `h = 1`; under that truncation several
  contrasts reported as null here would appear significant.
- **Holm–Bonferroni** correction within each family of five horizon-wise tests. Raw and adjusted
  p-values are both reported.
- **Model confidence set** (Hansen, Lunde & Nason, 2011), `T_max`, stationary bootstrap with 5,000
  replications and mean block `max(4, h)`. Elimination runs to a single survivor so every model
  receives its own p-value.
- **Minimum detectable improvement** at 80% power is reported in the paper so that readers can see
  what the design can and cannot rule out; it is computed in `final_inference.py`.

---

## 7. Environment

```
python 3.13
numpy 2.2.5   pandas 2.2.3   scipy 1.15.3   torch 2.12.1
scikit-learn 1.7.0   statsmodels 0.14.6   xgboost 3.4.0
matplotlib 3.10.9   requests 2.34.2   xlrd 2.0.2   openpyxl 3.1.5
```

Seeds 0–4 are fixed in every training script. Neural results reported in the paper are means over
seeds on the single split and seed-ensemble forecasts in the pooled expanding-window evaluation;
the two estimands are not interchangeable and the paper's inference uses the latter throughout.

---

## 8. Known limitations of the code

These are documented rather than hidden, because two of them shaped the paper's conclusions.

1. An earlier version of `rv_window()` in `build_dataset.py` spanned `7h − 5` calendar days instead
   of `7h`, making the `h = 1` target a two-day rather than a one-week realized volatility. It was
   found by an internal audit and corrected; tests `T6a`/`T6b` in `test_leakage.py` were added to
   make the error impossible to reintroduce, and all results in the paper use the corrected
   definition.
2. The balance-consistency error uses the **realized** future SPR change rather than a forecast of
   it, so it is not a pure property of the forecast. In addition, the adjustment term enters the
   identity linearly and is itself defined as the identity's residual, so a network can reduce the
   penalty through that channel alone.
3. Early stopping monitors the composite objective for `M4`/`M5` but the forecast loss alone for
   `M1`–`M3`/`M4arch`. The `M4arch` → `M4` contrast therefore mixes the auxiliary training signal
   with the stopping rule.
4. Hyperparameters shared by all specifications were selected on the 2015–2019 validation span, and
   156 of the 487 pooled test weeks fall inside that span.

## 9. License

Code is released under the MIT License (see `LICENSE`). The EIA source data are U.S. Government
works and are in the public domain.
