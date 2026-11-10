"""Does the representation result hold outside the neural estimator?

Compares least-squares and ridge HAR-X models against their own HAR comparator
on the same pooled weeks and the same scales as the ladder.
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
)

SPECS = ["HARX_none", "HARX_raw", "HARX_derived", "HARX_both",
         "HARX_raw_ridge", "HARX_derived_ridge", "HARX_both_ridge"]
BASE = "HARX_none"
L = {s: losses_on_all_scales(s) for s in SPECS}

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


w("# HAR-X: the representation question without a neural network")
w()
w("Least-squares and ridge regressions adding the same physical blocks to the "
  "benchmark. No hidden layer, no optimiser, no seed. `HARX_none` is the "
  "benchmark re-estimated on the same rows, so it is the correct comparator.")
w()

for key in ROBUST + PUBLISHED:
    w(f"### {SCALE_LABELS[key]}")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    best = {h: min(SPECS, key=lambda s: L[s][h][key].mean()) for h in HORIZONS}
    for s in SPECS:
        cells = []
        for h in HORIZONS:
            v = f"{L[s][h][key].mean():.4g}"
            cells.append(f"**{v}**" if best[h] == s else v)
        w(f"| {s} | " + " | ".join(cells) + " |")
    w()

w("## Against the benchmark, percentage change and Holm-adjusted p")
w()
w("Negative favours the HAR-X model.")
w()
for key in ROBUST + PUBLISHED:
    w(f"**{SCALE_LABELS[key]}**")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for s in SPECS[1:]:
        pct, ps = [], []
        for h in HORIZONS:
            a, b = L[s][h][key], L[BASE][h][key]
            _, p = dm(a, b, h); ps.append(p)
            pct.append(100 * (a.mean() - b.mean()) / b.mean())
        adj = holm(np.array(ps))
        w(f"| {s} | " + " | ".join(f"{v:+.1f}% ({p:.3f})"
                                   for v, p in zip(pct, adj)) + " |")
    w()

w("## The representation contrast, estimator by estimator")
w()
w("Derived-only against raw-only, which is the contrast the ladder identifies "
  "with M3derived against M2. A negative statistic favours the derived form.")
w()
for tag in ["", "_ridge"]:
    a, b = f"HARX_derived{tag}", f"HARX_raw{tag}"
    w(f"**{a} vs {b}**")
    w()
    w("| Loss | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for key in ROBUST + PUBLISHED:
        sts, ps = zip(*[dm(L[a][h][key], L[b][h][key], h) for h in HORIZONS])
        adj = holm(np.array(ps))
        w(f"| {SCALE_LABELS[key]} | "
          + " | ".join(f"{s:.2f} ({p:.3f})" for s, p in zip(sts, adj)) + " |")
    w()

path = HERE / "harx_inference.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
