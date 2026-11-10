"""Inference on the additional expanding-window configurations.

  M4_fcstop   isolates the auxiliary gradient from the early-stopping rule
  M5_lb2p0    balance penalty at four and ten times the strongest weight
  M5_lb5p0    reported in the paper, with the balance-consistency error

Run after `experiments/06_robustness/expanding_window_extra.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
POOL = HERE.parent / "06_robustness"
HS = [1, 2, 4, 8, 12]
BASE = ["HAR", "M4arch", "M4", "M5_lb0p1", "M5_lb0p5"]
NEW = ["M4_fcstop", "M5_lb2p0", "M5_lb5p0"]
ALL = BASE + NEW

ql = {s: pd.read_csv(POOL / f"ew_losses_{s}.csv", parse_dates=["date"])
      .set_index("date") for s in ALL}
ex = {s: pd.read_csv(POOL / f"ew_extra_{s}.csv", parse_dates=["date"])
      .set_index("date") for s in ALL}
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


w("# Additional configurations: stopping rule and constraint strength")
w()
w(f"Pooled weeks: {len(DATES)} ({DATES.min().date()} to {DATES.max().date()})")
w()

# ---------------------------------------------------------------- stopping
w("## S1. The early-stopping asymmetry")
w()
w("The ladder's M4 monitors the composite objective for early stopping while "
  "M4arch monitors the forecast term alone, so the M4arch-to-M4 contrast "
  "mixes the auxiliary gradient with a change in the stopping criterion. "
  "M4_fcstop is M4 with the forecast term monitored, which makes the "
  "M4arch-to-M4_fcstop contrast the auxiliary gradient alone.")
w()
for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"**{kind}** — mean loss")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for s in ["HAR", "M4arch", "M4", "M4_fcstop"]:
        w(f"| {s} | " + " | ".join(f"{loss(s, kind, h).mean():.4f}"
                                   for h in HS) + " |")
    w()

w("### Decomposing the reported auxiliary-task effect (Holm-adjusted p; "
  "negative statistic favours the first specification)")
w()
for label, a, b in [("as reported (gradient + stopping rule)", "M4", "M4arch"),
                    ("auxiliary gradient alone", "M4_fcstop", "M4arch"),
                    ("stopping rule alone", "M4", "M4_fcstop")]:
    w(f"**{label}: {a} vs {b}**")
    w()
    w("| Loss | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for kind in ["QLIKE", "MAE", "MSE"]:
        sts, ps = zip(*[dm(loss(a, kind, h), loss(b, kind, h), h) for h in HS])
        adj = holm(np.array(ps))
        w(f"| {kind} | " + " | ".join(f"{s:.2f} ({p:.3f})"
                                      for s, p in zip(sts, adj)) + " |")
    w()

# ---------------------------------------------------------------- lambda
w("## S2. Constraint strength beyond the reported range")
w()
w("The sweep reported in the paper tops out at a weight of 0.5 and moves the "
  "balance-consistency error by 2.3%, which is smaller than the dispersion "
  "across seeds; that leaves open whether the constraint was ever binding. "
  "Weights of 2 and 5 settle it.")
w()

bce = pd.read_csv(POOL / "ew_bce_extra.csv")
w("### Balance-consistency error, pooled over the five windows (10³ bbl/week)")
w()
w("| Weight | mean | versus the unpenalised network |")
w("|---|---|---|")
base_bce = bce.loc[bce.config == "M4_fcstop", "bce_kb_week_mean"].mean()
w(f"| 0 (no penalty) | {base_bce:,.0f} | — |")
for cfg, lam in [("M5_lb2p0", "2.0"), ("M5_lb5p0", "5.0")]:
    v = bce.loc[bce.config == cfg, "bce_kb_week_mean"].mean()
    w(f"| {lam} | {v:,.0f} | {100 * (v - base_bce) / base_bce:+.1f}% |")
w()

for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"**{kind}** — mean loss across the constraint weights")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for s in ["HAR", "M4", "M5_lb0p1", "M5_lb0p5", "M5_lb2p0", "M5_lb5p0"]:
        w(f"| {s} | " + " | ".join(f"{loss(s, kind, h).mean():.4f}"
                                   for h in HS) + " |")
    w()

w("### Each weight against the unpenalised dual-branch network "
  "(M4; Holm-adjusted p, negative favours the penalised specification)")
w()
for s in ["M5_lb0p1", "M5_lb0p5", "M5_lb2p0", "M5_lb5p0"]:
    w(f"**{s} vs M4**")
    w()
    w("| Loss | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for kind in ["QLIKE", "MAE", "MSE"]:
        sts, ps = zip(*[dm(loss(s, kind, h), loss("M4", kind, h), h)
                        for h in HS])
        adj = holm(np.array(ps))
        w(f"| {kind} | " + " | ".join(f"{v:.2f} ({p:.3f})"
                                      for v, p in zip(sts, adj)) + " |")
    w()

w("### The same weights excluding 2020")
w()
no20 = DATES.year != 2020
w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
w("|---|" + "---|" * len(HS))
for s in ["HAR", "M4", "M5_lb0p5", "M5_lb2p0", "M5_lb5p0"]:
    w(f"| {s} | " + " | ".join(f"{loss(s, 'QLIKE', h, no20).mean():.4f}"
                               for h in HS) + " |")
w()

path = HERE / "extra_configs_inference.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
