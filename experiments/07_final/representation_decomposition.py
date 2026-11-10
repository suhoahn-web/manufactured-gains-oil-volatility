"""Does the M2-to-M3 gap come from representation, information, or capacity?

M3 adds the six balance-derived quantities to M2's nine raw series, so the
published contrast moves all three at once. Two further arms separate them:

  M3derived  price block plus ONLY the six derived quantities (18 inputs).
             Against M2 this is a genuine replacement of raw levels by
             stock-flow quantities.
  M2wide     price block plus ONLY the nine raw series, hidden state widened to
             match M3's parameter count. Against M3 this holds capacity fixed.

Read together:
  M3derived vs M2  — representation, holding information volume comparable
  M3 vs M2wide     — information, holding capacity fixed
  M2wide vs M2     — capacity alone
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

SPECS = ["HAR", "M1", "M2", "M2wide", "M3derived", "M3"]
IDX = pooled_index()
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


w("# Representation, information, or capacity?")
w()
w(f"Pooled weeks: {len(IDX)}. Input dimensions: M1 12, M2 21, M2wide 21 "
  f"(hidden 66), M3derived 18, M3 27. Parameter counts: M2 17,029, "
  f"M2wide 17,957, M3derived 16,453, M3 18,181.")
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

CONTRASTS = [
    ("M3derived", "M2", "representation: derived instead of raw"),
    ("M3", "M2wide", "information: derived added, capacity held fixed"),
    ("M2wide", "M2", "capacity alone"),
    ("M3", "M2", "as published: representation, information and capacity together"),
    ("M3", "M3derived", "keeping the raw series alongside the derived ones"),
]

w("## Contrasts")
w()
w("Holm-adjusted p-values across the five horizons; a negative statistic "
  "favours the first specification.")
w()
for a, b, label in CONTRASTS:
    w(f"**{a} vs {b}** — {label}")
    w()
    w("| Loss | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for key in ROBUST + PUBLISHED:
        sts, ps = zip(*[dm(L[a][h][key], L[b][h][key], h) for h in HORIZONS])
        adj = holm(np.array(ps))
        w(f"| {SCALE_LABELS[key]} | "
          + " | ".join(f"{s:.2f} ({p:.3f})" for s, p in zip(sts, adj)) + " |")
    w()

path = HERE / "representation_decomposition.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
