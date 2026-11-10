"""Table 2: descriptive statistics of the modelling sample (design doc 29.2)."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ds = pd.read_csv(ROOT / "data" / "processed" / "dataset_weekly.csv",
                 parse_dates=["date"])

VARS = [
    ("Realized volatility, 1w (%)", lambda d: d["px_rv_1w"] * 100),
    ("log RV, 1w", lambda d: np.log(d["px_rv_1w"])),
    ("WTI spot ($/bbl)", lambda d: d["px_wti"]),
    ("Production (kb/d)", lambda d: d["production"]),
    ("Imports (kb/d)", lambda d: d["imports"]),
    ("Exports (kb/d)", lambda d: d["exports"]),
    ("Refinery inputs (kb/d)", lambda d: d["refinery_inputs"]),
    ("Commercial stocks (Mbbl)", lambda d: d["stocks_commercial"] / 1000),
    ("Total stocks including SPR (Mbbl)",
     lambda d: (d["stocks_commercial"] + d["stocks_spr"]) / 1000),
    ("Strategic Petroleum Reserve stocks (Mbbl)",
     lambda d: d["stocks_spr"] / 1000),
    ("Cushing stocks (Mbbl)", lambda d: d["stocks_cushing"] / 1000),
    ("Utilization (%)", lambda d: d["utilization"]),
    ("Adjustment (kb/d)", lambda d: d["adjustment"]),
    ("d Commercial stocks (Mbbl/wk)",
     lambda d: d["d_stocks_commercial"] / 1000),
]

rows = []
for name, fn in VARS:
    x = fn(ds).replace([np.inf, -np.inf], np.nan).dropna()
    jb, jb_p = stats.jarque_bera(x)
    rows.append({
        "Variable": name, "N": len(x), "Mean": x.mean(), "SD": x.std(),
        "Min": x.min(), "Median": x.median(), "Max": x.max(),
        "Skew": stats.skew(x), "Kurt": stats.kurtosis(x, fisher=False),
        "JB p": jb_p,
    })
tbl = pd.DataFrame(rows).set_index("Variable")

split_rows = []
for s in ("train", "val", "test"):
    d = ds[ds.split == s]
    rv = np.log(d["px_rv_1w"].replace(0, np.nan)).dropna()
    split_rows.append({"Split": s, "Weeks": len(d),
                       "Start": d["date"].min().date(),
                       "End": d["date"].max().date(),
                       "Mean log RV": rv.mean(), "SD log RV": rv.std()})
splits = pd.DataFrame(split_rows).set_index("Split")

out = Path(__file__).resolve().parent / "descriptive_stats.md"
txt = ["# Table 2. Descriptive statistics", "",
       tbl.round(3).to_markdown(), "",
       "## Sample splits", "", splits.round(3).to_markdown(), ""]
out.write_text("\n".join(txt), encoding="utf-8")
tbl.round(4).to_csv(Path(__file__).resolve().parent / "descriptive_stats.csv")
print("\n".join(txt))


