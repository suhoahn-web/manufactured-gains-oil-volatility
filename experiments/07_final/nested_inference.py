"""Do the conclusions survive selecting hyperparameters inside each window?

`expanding_window_nested.py` removes the selection reuse rather than bounding
it: within each window the grid is fitted on that window's training rows, scored
on its validation rows, and only the winner reaches its test block. The pooled
487 weeks are then clean with respect to model selection.

Three questions:
  1. does the benchmark still hold against the constrained specification?
  2. does the representation contrast survive?
  3. how much does per-window selection move the losses at all?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from forecast_recovery import (  # noqa: E402
    HORIZONS, PUBLISHED, ROBUST, SCALE_LABELS, losses_on_all_scales,
    pooled_index,
)

POOL = HERE.parent / "06_robustness"
GLOBAL = {"M2": "M2", "M3derived": "M3derived", "M5_lb0p1": "M5"}
NESTED = {"M2_nested": "M2", "M3derived_nested": "M3derived",
          "M5_nested": "M5"}
ALL = ["HAR"] + list(GLOBAL) + list(NESTED)
L = {s: losses_on_all_scales(s) for s in ALL}
IDX = pooled_index()

out = []


def w(s=""):
    out.append(s)
    print(s)


def nw(d, q):
    n = len(d); d = d - d.mean(); v = (d @ d) / n
    for k in range(1, q + 1):
        v += 2 * (1 - k / (q + 1)) * ((d[k:] @ d[:-k]) / n)
    return v


def dm(a, b, h):
    d = a - b; n = len(d)
    st = d.mean() / np.sqrt(nw(d, max(h - 1, int(4 * (n / 100) ** (2 / 9)))) / n)
    st *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return st, 2 * (1 - stats.t.cdf(abs(st), df=n - 1))


def holm(ps):
    o = np.argsort(ps); a = np.empty(len(ps)); r = 0.0
    for i, ix in enumerate(o):
        r = max(r, (len(ps) - i) * ps[ix]); a[ix] = min(r, 1.0)
    return a


w("# Nested hyperparameter selection")
w()
sel = pd.read_csv(POOL / "nested_selection.csv")
w("## What each window selected")
w()
w("| Window | Specification | hidden | λ_B | validation MSE |")
w("|---|---|---|---|---|")
for _, r in sel.iterrows():
    lam = "—" if pd.isna(r.get("lam_b", np.nan)) else f"{r['lam_b']:g}"
    w(f"| {int(r['window'])} | {r['spec']} | {int(r['hidden'])} | {lam} | "
      f"{r['val_mse']:.5f} |")
w()
w("Selection used one seed on that window's validation rows; the winner was "
  "then re-estimated with five seeds and ensembled, as elsewhere in the paper.")
w()

w("## Against the benchmark, globally selected versus nested")
w()
w("Percentage change against HAR, Holm-adjusted p in parentheses. Negative "
  "favours the specification.")
w()
for key in ROBUST + PUBLISHED:
    w(f"**{SCALE_LABELS[key]}**")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for s in list(GLOBAL) + list(NESTED):
        pct, ps = [], []
        for h in HORIZONS:
            a, b = L[s][h][key], L["HAR"][h][key]
            _, p = dm(a, b, h); ps.append(p)
            pct.append(100 * (a.mean() - b.mean()) / b.mean())
        adj = holm(np.array(ps))
        w(f"| {s} | " + " | ".join(f"{v:+.1f}% ({p:.3f})"
                                   for v, p in zip(pct, adj)) + " |")
    w()

w("## The representation contrast under nested selection")
w()
w("M3derived against M2, which is the contrast the paper identifies with "
  "representation. A negative statistic favours the derived form.")
w()
for lab, a, b in [("globally selected", "M3derived", "M2"),
                  ("nested", "M3derived_nested", "M2_nested")]:
    w(f"**{lab}**")
    w()
    w("| Loss | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for key in ROBUST + PUBLISHED:
        sts, ps = zip(*[dm(L[a][h][key], L[b][h][key], h) for h in HORIZONS])
        adj = holm(np.array(ps))
        w(f"| {SCALE_LABELS[key]} | "
          + " | ".join(f"{s:.2f} ({p:.3f})" for s, p in zip(sts, adj)) + " |")
    w()

w("## How much does per-window selection move the forecasts?")
w()
w("Percentage change of the nested version against its globally selected "
  "counterpart, quasi-likelihood on volatility.")
w()
w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
w("|---|" + "---|" * len(HORIZONS))
for n, g in [("M2_nested", "M2"), ("M3derived_nested", "M3derived"),
             ("M5_nested", "M5_lb0p1")]:
    cells = []
    for h in HORIZONS:
        a, b = L[n][h]["QLIKE_vol"].mean(), L[g][h]["QLIKE_vol"].mean()
        cells.append(f"{100 * (a - b) / b:+.1f}%")
    w(f"| {n} vs {g} | " + " | ".join(cells) + " |")
w()

path = HERE / "nested_inference.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
