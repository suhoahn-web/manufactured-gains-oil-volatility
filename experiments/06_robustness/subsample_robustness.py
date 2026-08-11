"""Sub-period robustness (design doc section 24 + field convention).

Uses the pooled expanding-window per-week QLIKE losses, so no retraining is
needed. Three cuts:

  1. EIA methodology breaks inside the evaluation period.
     - The 2016-07 weekly-exports change predates every pooled test week
       (evaluation starts 2017-01), so it can only affect training data.
     - The 2023-03 introduction of "transfers to crude oil supply" (which
       changed the crude adjustment) DOES fall inside the test period:
       compare model rankings before/after.
     - The 2022-04 onset of large SPR policy drawdowns likewise.
  2. COVID (2020) vs the rest, the standard sub-period check in this
     literature.
  3. Calm vs turbulent weeks, split at the median realized volatility of the
     pooled test weeks.

Outputs subsample_robustness.md with mean QLIKE per model per subsample and
DM tests for the two decisive pairs (M4 vs M1, M5 vs M4).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
MODELS = ["HAR", "M1", "M2", "M3", "M4arch", "M4", "M5_lb0p1", "M5_lb0p5"]

tables = {}
for m in MODELS:
    p = HERE / f"ew_losses_{m}.csv"
    if p.exists():
        tables[m] = pd.read_csv(p, parse_dates=["date"])
avail = list(tables)

common = tables[avail[0]][["date"]]
for m in avail[1:]:
    common = common.merge(tables[m][["date"]], on="date")
dates = common["date"].reset_index(drop=True)
L = {m: tables[m].set_index("date").loc[dates, list(HORIZONS)].reset_index(drop=True)
     for m in avail}

ds = pd.read_csv(ROOT / "data" / "processed" / "dataset_weekly.csv",
                 parse_dates=["date"])
rv = ds.set_index("date")["px_rv_1w"].reindex(dates).to_numpy()
med = np.nanmedian(rv)

CUTS = {
    "pre-2023-03 (before EIA transfers line)": dates < "2023-03-01",
    "post-2023-03": dates >= "2023-03-01",
    "pre-2022-04 (before SPR drawdowns)": dates < "2022-04-01",
    "post-2022-04": dates >= "2022-04-01",
    "COVID 2020": (dates >= "2020-01-01") & (dates < "2021-01-01"),
    "ex-COVID": ~((dates >= "2020-01-01") & (dates < "2021-01-01")),
    "calm weeks (RV below median)": rv <= med,
    "turbulent weeks (RV above median)": rv > med,
}


def dm(a, b, h):
    d = a - b
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = gamma[0] / n
    stat = dbar / np.sqrt(var_d)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    s = stat * harvey
    return s, 2 * (1 - stats.t.cdf(abs(s), df=n - 1))


lines = ["# Sub-period robustness (pooled expanding-window QLIKE)", "",
         f"Pooled test weeks: {len(dates)} "
         f"({dates.min().date()} to {dates.max().date()})", "",
         "Note: the 2016-07 weekly-exports methodology change predates every",
         "pooled test week, so it cannot contaminate the out-of-sample results;",
         "it is relevant only to training data.", ""]

for name, mask in CUTS.items():
    mask = np.asarray(mask)
    lines += [f"## {name}  (n = {int(mask.sum())})", "", "```"]
    rows = [{"model": m, **{h: L[m].loc[mask, h].mean() for h in HORIZONS}}
            for m in avail]
    lines += [pd.DataFrame(rows).set_index("model").round(4).to_string(), "```", ""]
    pairs = [("M4", "M1"), ("M5_lb0p1", "M4"), ("M5_lb0p5", "M4"),
             ("M4", "HAR")]
    parts = []
    for a, b in pairs:
        if a not in L or b not in L:
            continue
        seg = []
        for h, hn in HORIZONS.items():
            s, p = dm(L[a].loc[mask, h].to_numpy(), L[b].loc[mask, h].to_numpy(), hn)
            seg.append(f"{h}:{s:+.2f}(p={p:.3f})" if np.isfinite(s) else f"{h}:n/a")
        parts.append(f"{a} vs {b}  " + "  ".join(seg))
    lines += ["DM (negative = first model better):", "```"] + parts + ["```", ""]

out = HERE / "subsample_robustness.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
