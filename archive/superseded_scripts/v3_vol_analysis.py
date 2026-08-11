"""v3 volatility verdict: QLIKE DM tests, per-protocol and sensitivity.

Per-protocol comparisons (lambda selected on validation = 0.1):
  M4 vs M1, M4 vs HAR, M5(0.1) vs M4, M5(0.1) vs HAR
Sensitivity (NOT selectable on validation; report as regime-dependence):
  M5(0.5) vs M4
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "05_inventory_stress"))
import stress_analysis  # noqa: E402
from stress_analysis import dm_test, ensemble_preds  # noqa: E402

HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
stress_analysis.HORIZONS = HORIZONS

ds = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_weekly.csv",
                 parse_dates=["date"])
splits = {s: ds.index[ds.split == s].to_numpy() for s in ("train", "val", "test")}


def har_preds() -> pd.DataFrame:
    """Recompute HAR-RV forecasts (train-fit OLS) for the test rows."""
    out = ds.loc[splits["test"], ["date"]].copy()
    X = np.column_stack([np.ones(len(ds)), np.log(ds["px_rv_1w"]),
                         np.log(ds["px_rv_4w"]), np.log(ds["px_rv_12w"])])
    for h in HORIZONS:
        y = ds[f"yv_{h}"].to_numpy()
        beta, *_ = np.linalg.lstsq(X[splits["train"]], y[splits["train"]],
                                   rcond=None)
        out[f"y_{h}"] = y[splits["test"]]
        out[f"yhat_{h}"] = X[splits["test"]] @ beta
    return out


def qlike_losses(pred: pd.DataFrame, h: str) -> np.ndarray:
    ratio = np.exp(pred[f"y_{h}"].to_numpy() - pred[f"yhat_{h}"].to_numpy())
    return ratio - np.log(ratio) - 1.0


models = {
    "M1": ensemble_preds("M1_v3"),
    "M4": ensemble_preds("M4_v3"),
    "M5s": ensemble_preds("M5_v3lb0p1"),   # per-protocol selected lambda
    "M5x": ensemble_preds("M5_v3lb0p5"),   # sensitivity only
    "HAR": har_preds(),
}
# align on common dates (NN preds start after lookback warmup)
common = set.intersection(*[set(m["date"]) for m in models.values()])
models = {k: m[m["date"].isin(common)].sort_values("date").reset_index(drop=True)
          for k, m in models.items()}

lines = ["# v3 Volatility Verdict (QLIKE, seed-ensemble preds)", "",
         f"Common test weeks: {len(common)}", ""]

lines += ["## Mean QLIKE by model/horizon", "", "```"]
rows = []
for name, p in models.items():
    rows.append({"model": name, **{h: np.mean(qlike_losses(p, h))
                                   for h in HORIZONS}})
lines += [pd.DataFrame(rows).set_index("model").round(4).to_string(), "```", ""]

PAIRS = [("M4", "M1"), ("M4", "HAR"), ("M5s", "M4"), ("M5s", "HAR"),
         ("M5x", "M4"), ("M5x", "HAR")]
lines += ["## DM tests on QLIKE differentials (negative DM = first better)", "", "```"]
for h, hn in HORIZONS.items():
    parts = []
    for a, b in PAIRS:
        la, lb = qlike_losses(models[a], h), qlike_losses(models[b], h)
        d = la - lb
        n = len(d); dbar = d.mean()
        gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(hn)]
        var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
        if var_d <= 0:
            var_d = gamma[0] / n
        from scipy import stats
        dm = dbar / np.sqrt(var_d)
        harvey = np.sqrt((n + 1 - 2 * hn + hn * (hn - 1) / n) / n)
        dmh = dm * harvey
        pv = 2 * (1 - stats.t.cdf(abs(dmh), df=n - 1))
        parts.append(f"{a}vs{b} {dmh:+.2f}(p={pv:.3f})")
    lines.append(f"{h:>4}: " + " | ".join(parts))
lines += ["```", ""]

out = Path(__file__).resolve().parent / "v3_vol_verdict.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
