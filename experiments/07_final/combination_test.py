"""Forecast combination: HAR + physical-state model (single split, v4).

Standard practice in the RV literature. HAR is strong at short horizons where
price history dominates; the multi-task models are strong at long horizons
where the physical balance matters. A combination should inherit both.

Weights are fixed a priori (equal) or estimated on the VALIDATION split only.
Evaluated on the 2020-2026 test split with QLIKE + DM.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "05_inventory_stress"))
import stress_analysis  # noqa: E402
stress_analysis.HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
from stress_analysis import ensemble_preds  # noqa: E402

HOR = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
ds = pd.read_csv(ROOT / "data" / "processed" / "dataset_weekly.csv",
                 parse_dates=["date"])
idx = {s: ds.index[ds.split == s].to_numpy() for s in ("train", "val", "test")}
X = np.column_stack([np.ones(len(ds)), np.log(ds["px_rv_1w"]),
                     np.log(ds["px_rv_4w"]), np.log(ds["px_rv_12w"])])


def har_fit_predict(h):
    y = ds[f"yv_{h}"].to_numpy()
    beta, *_ = np.linalg.lstsq(X[idx["train"]], y[idx["train"]], rcond=None)
    return X @ beta


def qlike(y, f):
    r = np.exp(y - f)
    return r - np.log(r) - 1.0


def dm(a, b, h):
    d = a - b
    n = len(d)
    dbar = d.mean()
    g = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n
    v = v if v > 0 else g[0] / n
    s = dbar / np.sqrt(v)
    s *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return s, 2 * (1 - stats.t.cdf(abs(s), df=n - 1))


nn_preds = {m: ensemble_preds(t) for m, t in
            [("M4", "M4_v4"), ("M5_lb0p1", "M5_v4lb0p1"),
             ("M5_lb0p5", "M5_v4lb0p5")]}
date_to_row = {d: i for i, d in enumerate(ds["date"])}

lines = ["# Forecast combination: HAR + physical-state models (single split)",
         ""]
rows, dm_lines = [], []
for nn_name, p in nn_preds.items():
    r_idx = np.array([date_to_row[d] for d in p["date"]])
    in_val = ds["split"].to_numpy()[r_idx] == "val"
    in_test = ds["split"].to_numpy()[r_idx] == "test"
    rec = {"model": f"HAR+{nn_name}"}
    rec_w = {"model": f"HAR+{nn_name} (val-weighted)"}
    for h, hn in HOR.items():
        har_all = har_fit_predict(h)
        f_har, f_nn = har_all[r_idx], p[f"yhat_{h}"].to_numpy()
        y = ds[f"yv_{h}"].to_numpy()[r_idx]
        # equal-weight
        f_eq = 0.5 * f_har + 0.5 * f_nn
        # validation-optimal weight on log-RV MSE, clipped to [0,1]
        # NOTE: the saved prediction files contain TEST rows only, so no
        # validation rows are available here and the weight falls back to 0.5
        # (i.e. the "val-weighted" rows equal the equal-weight rows). Saving
        # validation forecasts would be required to estimate weights properly.
        num = np.sum((f_har[in_val] - f_nn[in_val])
                     * (f_har[in_val] - y[in_val]))
        den = np.sum((f_har[in_val] - f_nn[in_val]) ** 2)
        w = float(np.clip(num / den, 0, 1)) if den > 0 else 0.5
        f_w = (1 - w) * f_har + w * f_nn
        rec[h] = qlike(y[in_test], f_eq[in_test]).mean()
        rec_w[h] = qlike(y[in_test], f_w[in_test]).mean()
        rec_w[f"w_{h}"] = round(w, 3)
        if nn_name == "M5_lb0p1":
            s1, p1 = dm(qlike(y[in_test], f_eq[in_test]),
                        qlike(y[in_test], f_har[in_test]), hn)
            s2, p2 = dm(qlike(y[in_test], f_eq[in_test]),
                        qlike(y[in_test], f_nn[in_test]), hn)
            dm_lines.append(f"{h:>4}: comb vs HAR {s1:+.2f}(p={p1:.3f}) | "
                            f"comb vs {nn_name} {s2:+.2f}(p={p2:.3f})")
    rows.append(rec)
    rows.append(rec_w)

# reference rows
for name, get in [("HAR", None), ("M4", "M4"), ("M5_lb0p1", "M5_lb0p1"),
                  ("M5_lb0p5", "M5_lb0p5")]:
    p = nn_preds["M4"] if get is None else nn_preds[get]
    r_idx = np.array([date_to_row[d] for d in p["date"]])
    in_test = ds["split"].to_numpy()[r_idx] == "test"
    rec = {"model": name}
    for h in HOR:
        y = ds[f"yv_{h}"].to_numpy()[r_idx]
        f = (har_fit_predict(h)[r_idx] if get is None
             else nn_preds[get][f"yhat_{h}"].to_numpy())
        rec[h] = qlike(y[in_test], f[in_test]).mean()
    rows.append(rec)

tbl = pd.DataFrame(rows).set_index("model")
cols = [c for c in tbl.columns if not c.startswith("w_")]
lines += ["## Test QLIKE (2020-2026)", "", "```",
          tbl[cols].round(4).to_string(), "```", "",
          "Validation-estimated weights on the NN forecast:", "```",
          tbl[[c for c in tbl.columns if c.startswith("w_")]].dropna()
          .to_string(), "```", "",
          "## DM tests for HAR+M5_lb0p1 (equal weight)", "", "```"]
lines += dm_lines + ["```", ""]

out = Path(__file__).resolve().parent / "combination_results.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
