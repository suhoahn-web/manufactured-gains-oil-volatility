"""Pending experiments for the volatility paper.

1. Inventory-regime analysis (RQ3 supplement): pooled expanding-window QLIKE
   split by the realized inventory regime of the TARGET window (average weekly
   commercial-stock change over (origin, origin+h], thresholds = train q10/q90
   of weekly changes). Reported for h1 and h12.
2. Economic value: volatility-targeting exercise at h=1 (Bollerslev-style):
   weekly position w_t = min(sigma_target / forecast_vol, 3), long WTI, weekly
   rebalancing over the pooled EW test weeks; report annualized return, vol,
   Sharpe per model. Transaction costs ignored (noted in text).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EW = PROJECT_ROOT / "experiments" / "06_robustness"
MODELS = ["HAR", "M1", "M2", "M3", "M4arch", "M4", "M5_lb0p1", "M5_lb0p5"]
HORIZONS = {"h1": 1, "h4": 4, "h12": 12}

ds = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_weekly.csv",
                 parse_dates=["date", "origin_date"])

# stocks series on the weekly Friday grid for target-window regime computation
stocks = ds.set_index("date")["stocks_commercial"]


def avg_weekly_change(dates: pd.Series, h: int) -> pd.Series:
    """Average weekly stock change over the h-week window after the origin."""
    origin = dates + pd.Timedelta(days=7)
    end = origin + pd.Timedelta(days=7 * h)
    s0 = stocks.reindex(origin).to_numpy()
    s1 = stocks.reindex(end).to_numpy()
    return pd.Series((s1 - s0) / h, index=dates.index)


def horizon_thresholds(h: int) -> tuple:
    """Train-period q10/q90 of the SAME statistic used for classification.

    Thresholds must be recomputed per horizon: h-week averages of weekly
    changes are far less dispersed than single-week changes, so applying
    one-week quantiles to multi-week averages yields degenerate regimes
    (e.g. zero draw windows at h=12).
    """
    tr_dates = ds.loc[ds.split == "train", "date"].reset_index(drop=True)
    stat = avg_weekly_change(tr_dates, h).dropna()
    return stat.quantile(0.10), stat.quantile(0.90)


def target_regime(dates: pd.Series, h: int) -> pd.Series:
    return avg_weekly_change(dates, h)


losses = {m: pd.read_csv(EW / f"ew_losses_{m}.csv", parse_dates=["date"])
          for m in MODELS}

lines = ["# Inventory-Regime + Economic-Value Results (pooled EW weeks)", ""]

for h, hn in HORIZONS.items():
    q10, q90 = horizon_thresholds(hn)
    lines += [f"## Mean QLIKE by target-window inventory regime, {h}",
              f"(train q10/q90 of the {hn}-week average weekly stock change: "
              f"{q10:,.0f} / {q90:,.0f} kb)", "", "```"]
    rows = []
    for m in MODELS:
        t = losses[m].copy()
        avg = target_regime(t["date"], hn)
        build, draw = avg > q90, avg < q10
        normal = ~(build | draw) & avg.notna()
        rows.append({
            "model": m,
            "normal": t.loc[normal, h].mean(),
            "build": t.loc[build, h].mean(),
            "draw": t.loc[draw, h].mean(),
            "n_build": int(build.sum()), "n_draw": int(draw.sum()),
        })
    lines += [pd.DataFrame(rows).set_index("model").round(4).to_string(),
              "```", ""]

# ---- economic value: volatility targeting at h=1 -------------------------
# realized next-week return and RV from dataset targets
ret1 = ds.set_index("date")["y_h1"]
rv1_target = np.exp(ds.set_index("date")["yv_h1"])  # realized next-week vol

# forecasts: reconstruct from saved EW loss files? Losses don't store preds.
# Recompute forecast vol from stored per-week losses is not possible ->
# use the saved single-split ensemble preds for the 2020+ segment AND note
# scope; better: economic value on the v3 single-split test (305 weeks).
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "05_inventory_stress"))
import stress_analysis  # noqa: E402
stress_analysis.HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
from stress_analysis import ensemble_preds  # noqa: E402

SIGMA_TARGET = None  # set to train-period median weekly RV below
sigma_target = float(np.exp(ds.loc[ds.split == "train", "yv_h1"]).median())

lines += ["## Economic value: volatility targeting, h=1, single-split test 2020+",
          "",
          f"(sigma_target = train median weekly RV = {sigma_target:.4f}; "
          "position w = min(sigma_target/forecast, 3); no transaction costs)",
          "", "```"]
rows = []
for m, tag in [("M1", "M1_v5"), ("M4", "M4_v5"), ("M5_lb0p1", "M5_v5lb0p1"),
               ("M5_lb0p5", "M5_v5lb0p5")]:
    p = ensemble_preds(tag)
    f_vol = np.exp(p["yhat_h1"].to_numpy())
    r_next = ret1.reindex(p["date"]).to_numpy()
    w = np.clip(sigma_target / f_vol, 0, 3)
    strat = w * r_next
    ann_ret = strat.mean() * 52
    ann_vol = strat.std() * np.sqrt(52)
    rows.append({"model": m, "ann_ret": ann_ret, "ann_vol": ann_vol,
                 "sharpe": ann_ret / ann_vol})
# buy-and-hold reference
r_all = ret1.reindex(ensemble_preds("M1_v5")["date"]).to_numpy()
rows.append({"model": "buy&hold", "ann_ret": r_all.mean() * 52,
             "ann_vol": r_all.std() * np.sqrt(52),
             "sharpe": r_all.mean() * 52 / (r_all.std() * np.sqrt(52))})
lines += [pd.DataFrame(rows).set_index("model").round(4).to_string(), "```", ""]

out = Path(__file__).resolve().parent / "stress_econ_results.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
