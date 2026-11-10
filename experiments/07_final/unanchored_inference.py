"""Two questions the unanchored runs answer.

  1. Section 4.2 justifies anchoring by asserting that unanchored networks lose
     a substantial margin at short horizons but reports no number. This gives
     the number.

  2. The benchmark absorbs the persistence of realized volatility. If the
     storage mechanism acts through persistence, anchoring removes the channel
     through which fundamentals could help, and the paper's null would be an
     artefact of the anchor. Comparing the physical specification with the
     price-only one WITHOUT the anchor settles it: if the physical block adds
     nothing there either, the reading is unavailable.

Run after `../06_robustness/expanding_window_unanchored.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
POOL = HERE.parent / "06_robustness"
HS = [1, 2, 4, 8, 12]
SPECS = ["HAR", "M1", "M5_lb0p1", "M1_noanchor", "M5_lb0p1_noanchor"]

ql = {s: pd.read_csv(POOL / f"ew_losses_{s}.csv", parse_dates=["date"])
      .set_index("date") for s in SPECS}
ex = {s: pd.read_csv(POOL / f"ew_extra_{s}.csv", parse_dates=["date"])
      .set_index("date") for s in SPECS}
DATES = ql["HAR"].index


def loss(s, kind, h, m=None):
    src = ql[s][f"h{h}"] if kind == "QLIKE" else \
        ex[s][f"{'ae' if kind == 'MAE' else 'se'}_h{h}"]
    return (src[m] if m is not None else src).values


def nw_var(d, q):
    n = len(d)
    d = d - d.mean()
    v = (d @ d) / n
    for k in range(1, q + 1):
        v += 2 * (1 - k / (q + 1)) * ((d[k:] @ d[:-k]) / n)
    return v


def dm(a, b, h):
    d = a - b
    n = len(d)
    q = max(h - 1, int(4 * (n / 100) ** (2 / 9)))
    st = d.mean() / np.sqrt(nw_var(d, q) / n)
    st *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return st, 2 * (1 - stats.t.cdf(abs(st), df=n - 1))


def holm(ps):
    o = np.argsort(ps)
    a = np.empty(len(ps))
    r = 0.0
    for i, ix in enumerate(o):
        r = max(r, (len(ps) - i) * ps[ix])
        a[ix] = min(r, 1.0)
    return a


out = []


def w(s=""):
    out.append(s)
    print(s)


w("# Unanchored specifications")
w()
w(f"Pooled weeks: {len(DATES)} ({DATES.min().date()} to {DATES.max().date()})")
w()

w("## U1. The cost of removing the anchor")
w()
for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"**{kind}** — mean loss")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for s in SPECS:
        w(f"| {s} | " + " | ".join(f"{loss(s, kind, h).mean():.4f}"
                                   for h in HS) + " |")
    w()

w("Percentage cost of dropping the anchor, relative to the anchored "
  "counterpart (positive means the unanchored version is worse).")
w()
w("| Pair and loss | " + " | ".join(f"h = {h}" for h in HS) + " |")
w("|---|" + "---|" * len(HS))
for base, na in [("M1", "M1_noanchor"), ("M5_lb0p1", "M5_lb0p1_noanchor")]:
    for kind in ["QLIKE", "MAE", "MSE"]:
        cells = []
        for h in HS:
            a, b = loss(na, kind, h).mean(), loss(base, kind, h).mean()
            cells.append(f"{100 * (a - b) / b:+.1f}%")
        w(f"| {na} vs {base}, {kind} | " + " | ".join(cells) + " |")
w()

w("## U2. Does the physical block help when the anchor is absent?")
w()
w("If anchoring removed the channel through which fundamentals act, the "
  "physical specification should beat the price-only one once the anchor is "
  "gone. Entries are Holm-adjusted p-values; a negative statistic favours the "
  "physical specification.")
w()
for label, a, b in [
        ("with the anchor (as in the paper)", "M5_lb0p1", "M1"),
        ("without the anchor", "M5_lb0p1_noanchor", "M1_noanchor")]:
    w(f"**{label}: {a} vs {b}**")
    w()
    w("| Loss | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for kind in ["QLIKE", "MAE", "MSE"]:
        sts, ps = zip(*[dm(loss(a, kind, h), loss(b, kind, h), h) for h in HS])
        adj = holm(np.array(ps))
        w(f"| {kind} | " + " | ".join(f"{v:.2f} ({p:.3f})"
                                      for v, p in zip(sts, adj)) + " |")
    w()

w("## U3. The unanchored specifications against the benchmark")
w()
w("| Specification and loss | " + " | ".join(f"h = {h}" for h in HS) + " |")
w("|---|" + "---|" * len(HS))
for s in ["M1_noanchor", "M5_lb0p1_noanchor"]:
    for kind in ["QLIKE", "MAE", "MSE"]:
        sts, ps = zip(*[dm(loss(s, kind, h), loss("HAR", kind, h), h)
                        for h in HS])
        adj = holm(np.array(ps))
        pct = [100 * (loss(s, kind, h).mean() - loss("HAR", kind, h).mean())
               / loss("HAR", kind, h).mean() for h in HS]
        w(f"| {s}, {kind} | " + " | ".join(f"{v:+.1f}% ({p:.3f})"
                                           for v, p in zip(pct, adj)) + " |")
w()

w("## U4. Excluding 2020")
w()
no20 = DATES.year != 2020
w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
w("|---|" + "---|" * len(HS))
for s in SPECS:
    w(f"| {s} | " + " | ".join(f"{loss(s, 'QLIKE', h, no20).mean():.4f}"
                               for h in HS) + " |")
w()

path = HERE / "unanchored_inference.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
