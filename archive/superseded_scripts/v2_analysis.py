"""v2 final verdict analysis: DM tests, ex-COVID subsample, stress split."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "05_inventory_stress"))
import stress_analysis  # noqa: E402
from stress_analysis import dm_test, ensemble_preds  # noqa: E402

HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
stress_analysis.HORIZONS = HORIZONS  # ensemble_preds reads module-level dict
MODELS = ["M1_v2", "M2_v2", "M3_v2", "M4_v2", "M5_v2"]

ds = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_weekly.csv",
                 parse_dates=["date"])
tr = ds[ds.split == "train"]
lo, hi = tr["d_stocks_commercial"].quantile(0.10), tr["d_stocks_commercial"].quantile(0.90)
d_stock_next = (ds.set_index("date")["aux_d_stocks_commercial_next"]
                .rename("d_stock_target"))

preds = {}
for m in MODELS:
    p = ensemble_preds(m).merge(ds[["date", "px_wti"]], on="date", how="left")
    p = p.merge(d_stock_next.reset_index(), on="date", how="left")
    preds[m] = p

lines = ["# v2 Final Analysis", ""]

# --- DM tests -------------------------------------------------------------
lines += ["## DM tests (squared error, seed-ensemble; negative DM = first model better)", "", "```"]
for h, hn in HORIZONS.items():
    e = {m: preds[m][f"yhat_{h}"].to_numpy() - preds[m][f"y_{h}"].to_numpy()
         for m in ("M1_v2", "M4_v2", "M5_v2")}
    dm54, p54 = dm_test(e["M5_v2"], e["M4_v2"], hn)
    dm41, p41 = dm_test(e["M4_v2"], e["M1_v2"], hn)
    lines.append(f"{h:>4}: M5vsM4 DM={dm54:+.3f} p={p54:.3f} | M4vsM1 DM={dm41:+.3f} p={p41:.3f}")
lines += ["```", ""]

# --- ex-COVID (test from 2021-01) ----------------------------------------
lines += ["## Return MAE, test subsample 2021-01 onward (ex-COVID)", "", "```"]
rows = []
for m in MODELS:
    p = preds[m]
    mask = p["date"] >= "2021-01-01"
    rows.append({"model": m, **{
        h: np.mean(np.abs(p.loc[mask, f"yhat_{h}"] - p.loc[mask, f"y_{h}"]))
        for h in HORIZONS}})
lines += [pd.DataFrame(rows).set_index("model").round(4).to_string(), "```", ""]

# --- stress split, h1 and h12 --------------------------------------------
for h in ("h1", "h12"):
    lines += [f"## Return MAE by inventory regime, {h}", "", "```"]
    rows = []
    for m in MODELS:
        p = preds[m]
        build = p["d_stock_target"] > hi
        draw = p["d_stock_target"] < lo
        normal = ~(build | draw)
        rows.append({"model": m, **{
            name: np.mean(np.abs(p.loc[msk, f"yhat_{h}"] - p.loc[msk, f"y_{h}"]))
            for name, msk in (("normal", normal), ("build", build), ("draw", draw))}})
    lines += [pd.DataFrame(rows).set_index("model").round(4).to_string(), "```", ""]

out = Path(__file__).resolve().parent / "v2_analysis.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
