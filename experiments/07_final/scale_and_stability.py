"""The three tests the referee memo asks for, on recovered forecasts.

  A. Evaluation scale. Every specification scored on realized variance, on
     volatility, and on log volatility. Patton (2011) covers the first only.

  B. Forecast stability. Whether the advantage is concentrated in 2020 is
     currently argued from a sample split chosen after seeing the result. The
     Giacomini-Rossi (2010) fluctuation test, year-by-year differentials and a
     leave-one-year-out sweep put it on formal footing.

  C. Nested comparison. Every neural forecast is the benchmark plus a learned
     correction, so the benchmark is nested inside its rivals. Diebold-Mariano
     is not the right test for that case; Clark-West (2007) is added for the
     squared-error comparisons.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from forecast_recovery import (  # noqa: E402
    HORIZONS, PUBLISHED, ROBUST, SCALE_LABELS, load_forecasts,
    losses_on_all_scales, pooled_index,
)

LADDER = ["HAR", "M1", "M2", "M2wide", "M3derived", "M3", "M4arch", "M4",
          "M5_lb0p1", "M5_lb0p5"]
IDX = pooled_index()
L = {s: losses_on_all_scales(s) for s in LADDER}

out = []


def w(s=""):
    out.append(s)
    print(s)


# ------------------------------------------------------------------ inference
def nw_var(d, q):
    n = len(d)
    d = d - d.mean()
    v = (d @ d) / n
    for k in range(1, q + 1):
        v += 2 * (1 - k / (q + 1)) * ((d[k:] @ d[:-k]) / n)
    return v


def band(n, h):
    return max(h - 1, int(4 * (n / 100) ** (2 / 9)))


def dm(a, b, h):
    d = a - b
    n = len(d)
    st = d.mean() / np.sqrt(nw_var(d, band(n, h)) / n)
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


def clark_west(y, f_small, f_large, h):
    """One-sided test that the larger nested model has lower MSPE.

    f_adj_t = (y-f_s)^2 - [(y-f_l)^2 - (f_s-f_l)^2]; a positive mean favours
    the larger model. The correction removes the noise the extra parameters
    introduce under the null, which is what makes DM undersized here.
    """
    fa = (y - f_small) ** 2 - ((y - f_large) ** 2 - (f_small - f_large) ** 2)
    n = len(fa)
    st = fa.mean() / np.sqrt(nw_var(fa, band(n, h)) / n)
    return st, 1 - stats.norm.cdf(st)          # one-sided


# ---------------------------------------------------------------- A. scale
w("# Evaluation scale, forecast stability, and nested comparison")
w()
w(f"Pooled weeks: {len(IDX)} ({IDX.min().date()} to {IDX.max().date()})")
w()
w("## A. The evaluation scale decides the ranking")
w()
w("Patton (2011) shows that the ranking of variance forecasts is unaffected by "
  "noise in the volatility proxy for QLIKE and squared error evaluated on "
  "variance. The published evaluation uses neither: QLIKE is applied to "
  "volatility and squared error to log volatility. Both scales are reported "
  "here beside the one the result covers.")
w()
for key in ROBUST + PUBLISHED:
    w(f"### {SCALE_LABELS[key]}")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    best = {h: min(LADDER, key=lambda s: L[s][h][key].mean()) for h in HORIZONS}
    for s in LADDER:
        cells = []
        for h in HORIZONS:
            v = L[s][h][key].mean()
            txt = f"{v:.4g}"
            cells.append(f"**{txt}**" if best[h] == s else txt)
        w(f"| {s} | " + " | ".join(cells) + " |")
    w()

w("### Which specification wins, by scale")
w()
w("| Loss and scale | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
w("|---|" + "---|" * len(HORIZONS))
for key in ROBUST + PUBLISHED:
    w(f"| {SCALE_LABELS[key]} | "
      + " | ".join(min(LADDER, key=lambda s: L[s][h][key].mean())
                   for h in HORIZONS) + " |")
w()

w("### Against the benchmark: percentage change and Holm-adjusted p")
w()
w("Negative percentages favour the specification.")
w()
for key in ROBUST + PUBLISHED:
    w(f"**{SCALE_LABELS[key]}**")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for s in LADDER[1:]:
        pct, ps = [], []
        for h in HORIZONS:
            a, b = L[s][h][key], L["HAR"][h][key]
            _, p = dm(a, b, h)
            ps.append(p)
            pct.append(100 * (a.mean() - b.mean()) / b.mean())
        adj = holm(np.array(ps))
        w(f"| {s} | " + " | ".join(f"{v:+.1f}% ({p:.3f})"
                                   for v, p in zip(pct, adj)) + " |")
    w()

w("### Mean forecast bias on the log scale")
w()
w("Positive means the forecast sits above the realization. QLIKE rewards this; "
  "a symmetric loss on the same scale penalizes it.")
w()
w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
w("|---|" + "---|" * len(HORIZONS))
for s in LADDER:
    w(f"| {s} | " + " | ".join(f"{L[s][h]['bias_log'].mean():+.4f}"
                               for h in HORIZONS) + " |")
w()

# ------------------------------------------------------------- B. stability
w("## B. Is the advantage concentrated in 2020, and is that a choice?")
w()
SPEC = "M5_lb0p1"
w(f"All results below are for {SPEC}, the specification validation selects, "
  f"against the benchmark.")
w()

w("### Year-by-year mean loss difference")
w()
years = sorted(set(IDX.year))
for key in ["QLIKE_var", "MSE_var", "QLIKE_vol"]:
    w(f"**{SCALE_LABELS[key]}** — negative favours the specification")
    w()
    w("| Year | n | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|---|" + "---|" * len(HORIZONS))
    for y in years:
        m = IDX.year == y
        cells = [f"{(L[SPEC][h][key][m] - L['HAR'][h][key][m]).mean():+.4g}"
                 for h in HORIZONS]
        w(f"| {y} | {int(m.sum())} | " + " | ".join(cells) + " |")
    w()

w("### Leave-one-year-out")
w()
w("Mean difference over all pooled weeks except the year named. If one year "
  "carries the result, removing it should change the sign.")
w()
for key in ["QLIKE_var", "QLIKE_vol"]:
    w(f"**{SCALE_LABELS[key]}**")
    w()
    w("| Year removed | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for y in ["none"] + years:
        m = np.ones(len(IDX), bool) if y == "none" else (IDX.year != y)
        cells = [f"{(L[SPEC][h][key][m] - L['HAR'][h][key][m]).mean():+.4g}"
                 for h in HORIZONS]
        w(f"| {y} | " + " | ".join(cells) + " |")
    w()

w("### Giacomini-Rossi fluctuation test")
w()
w("The loss differential is standardized over a rolling window and the "
  "largest absolute value compared with the critical value for the "
  "corresponding window fraction. Rejection means relative performance is not "
  "constant over the sample, which is a different and stronger statement than "
  "a split chosen after the fact.")
w()
# critical values, two-sided, 5%, from Giacomini & Rossi (2010) Table 1
GR_CV = {0.1: 3.393, 0.2: 3.170, 0.3: 3.012, 0.4: 2.890, 0.5: 2.779,
         0.6: 2.664, 0.7: 2.552, 0.8: 2.428, 0.9: 2.287}


def fluctuation(d, h, mu=0.3):
    n = len(d)
    m = int(round(mu * n))
    sig = np.sqrt(nw_var(d, band(n, h)))
    stats_ = np.array([d[i:i + m].sum() / (sig * np.sqrt(m))
                       for i in range(n - m + 1)])
    return stats_, GR_CV[round(mu, 1)]


for key in ["QLIKE_var", "QLIKE_vol"]:
    w(f"**{SCALE_LABELS[key]}**, window fraction 0.3 "
      f"({int(0.3 * len(IDX))} weeks), 5% critical value {GR_CV[0.3]}")
    w()
    w("| h | max abs statistic | reject constancy | date of maximum |")
    w("|---|---|---|---|")
    for h in HORIZONS:
        d = L[SPEC][h][key] - L["HAR"][h][key]
        st, cv = fluctuation(d, h)
        j = int(np.argmax(np.abs(st)))
        m = int(round(0.3 * len(IDX)))
        mid = IDX[j + m // 2]
        w(f"| {h} | {np.abs(st).max():.2f} | "
          f"{'yes' if np.abs(st).max() > cv else 'no'} | {mid.date()} |")
    w()

# ---------------------------------------------------------------- C. nested
w("## C. The benchmark is nested inside every rival")
w()
w("Each neural forecast is the fitted benchmark plus a learned correction, so "
  "under the null that the correction is worthless the rival equals the "
  "benchmark plus estimation noise. Diebold-Mariano is undersized in that "
  "case. Clark-West adjusts for it. The test is one-sided: a large positive "
  "statistic favours the larger model.")
w()
for key, scale in [("MSE_var", "variance"), ("MSE_log", "log volatility")]:
    w(f"**Squared error on {scale}** — statistic (one-sided p)")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HORIZONS) + " |")
    w("|---|" + "---|" * len(HORIZONS))
    for s in LADDER[1:]:
        cells = []
        for h in HORIZONS:
            y, fl = load_forecasts(s)[h]
            _, fs = load_forecasts("HAR")[h]
            if scale == "variance":
                y_, fs_, fl_ = np.exp(2 * y), np.exp(2 * fs), np.exp(2 * fl)
            else:
                y_, fs_, fl_ = y, fs, fl
            st, p = clark_west(y_, fs_, fl_, h)
            cells.append(f"{st:.2f} ({p:.3f})")
        w(f"| {s} | " + " | ".join(cells) + " |")
    w()

path = HERE / "scale_and_stability.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")

# machine-readable copy for the figures, so the plot and the tables cannot
# drift apart
rows = []
for s in LADDER[1:]:
    for key in ROBUST + PUBLISHED:
        for h in HORIZONS:
            a, b = L[s][h][key], L["HAR"][h][key]
            _, p = dm(a, b, h)
            rows.append({"spec": s, "loss": key, "h": h,
                         "pct_vs_har": 100 * (a.mean() - b.mean()) / b.mean(),
                         "p_raw": p, "mean_loss": a.mean(),
                         "har_loss": b.mean(),
                         "bias_log": L[s][h]["bias_log"].mean()})
csv = HERE / "scale_comparison.csv"
pd.DataFrame(rows).to_csv(csv, index=False)
print(f"Wrote {csv}")
