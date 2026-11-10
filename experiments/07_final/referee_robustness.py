"""Robustness checks answering four referee objections, computed on the
per-week pooled losses so that nothing needs retraining.

  R1  MSE alongside MAE. Patton (2011) shows MSE and QLIKE are robust to
      noise in the volatility proxy while MAE is not, so a reversal shown
      only under MAE can be dismissed as a proxy-noise artifact.

  R2  The 2020 decomposition under all three losses, not only QLIKE.

  R3  Dropping every origin whose target window spans 20 April 2020. The two
      daily log returns around the negative print are undefined and excluded,
      which understates realized volatility for exactly the weeks that
      Section 5.2 shows to be decisive. If the 2020 finding survives their
      removal it is not a measurement artifact.

  R4  Dropping 2017-2019, the weeks that fall inside the span used to select
      the shared hyperparameters. Sample reuse can only flatter the
      specifications being tested.

  R5  The sample size that would be needed to detect a 5% improvement, which
      is the order of gain this literature reports.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
POOL = HERE.parent / "06_robustness"
SPECS = ["HAR", "M1", "M2", "M3", "M4arch", "M4", "M5_lb0p1", "M5_lb0p5"]
HS = [1, 2, 4, 8, 12]
NEG_PRICE_DATE = pd.Timestamp("2020-04-20")


# ---------------------------------------------------------------- inference
def nw_var(d, q):
    n = len(d)
    d = d - d.mean()
    v = (d @ d) / n
    for k in range(1, q + 1):
        v += 2 * (1 - k / (q + 1)) * ((d[k:] @ d[:-k]) / n)
    return v


def bandwidth(n, h):
    return max(h - 1, int(4 * (n / 100) ** (2 / 9)))


def dm(a, b, h):
    """Harvey-corrected Diebold-Mariano. Positive statistic: A worse than B."""
    d = a - b
    n = len(d)
    v = nw_var(d, bandwidth(n, h))
    st = d.mean() / np.sqrt(v / n)
    st *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return st, 2 * (1 - stats.t.cdf(abs(st), df=n - 1))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(ps) - i) * ps[idx])
        adj[idx] = min(run, 1.0)
    return adj


# ---------------------------------------------------------------- data
def load():
    ql, ex = {}, {}
    for s in SPECS:
        q = pd.read_csv(POOL / f"ew_losses_{s}.csv", parse_dates=["date"])
        e = pd.read_csv(POOL / f"ew_extra_{s}.csv", parse_dates=["date"])
        ql[s] = q.set_index("date")
        ex[s] = e.set_index("date")
    return ql, ex


ql, ex = load()
DATES = ql["HAR"].index


def loss(spec, kind, h, mask=None):
    src = ql[spec][f"h{h}"] if kind == "QLIKE" else \
        ex[spec][f"{'ae' if kind == 'MAE' else 'se'}_h{h}"]
    return (src[mask] if mask is not None else src).values


out = []


def w(line=""):
    out.append(line)
    print(line)


w("# Referee robustness checks")
w()
w(f"Pooled weeks: {len(DATES)}  "
  f"({DATES.min().date()} to {DATES.max().date()})")
w()

# ------------------------------------------------------------------ R1, R2
w("## R1. Pooled loss under all three loss functions")
w()
for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"### {kind}")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    best = {h: min(SPECS, key=lambda s: loss(s, kind, h).mean()) for h in HS}
    for s in SPECS:
        cells = []
        for h in HS:
            v = f"{loss(s, kind, h).mean():.4f}"
            cells.append(f"**{v}**" if best[h] == s else v)
        w(f"| {s} | " + " | ".join(cells) + " |")
    w()
    w(f"Best specification by horizon: "
      + ", ".join(f"h={h}: {best[h]}" for h in HS))
    w()

w("### Benchmark versus each specification (Harvey-corrected DM,")
w("positive statistic = specification worse than HAR; Holm-adjusted p)")
w()
for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"**{kind}**")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for s in SPECS[1:]:
        sts, ps = zip(*[dm(loss(s, kind, h), loss("HAR", kind, h), h)
                        for h in HS])
        adj = holm(np.array(ps))
        w(f"| {s} | " + " | ".join(f"{st:.2f} ({a:.3f})"
                                   for st, a in zip(sts, adj)) + " |")
    w()

# ------------------------------------------------------------------ R2
w("## R2. The 2020 decomposition under all three losses")
w()
in2020 = DATES.year == 2020
w(f"2020 weeks: {in2020.sum()};  other weeks: {(~in2020).sum()}")
w()
for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"### {kind}: mean loss difference against the benchmark "
      f"(negative favours the specification)")
    w()
    w("| Specification and period | " + " | ".join(f"h = {h}" for h in HS)
      + " |")
    w("|---|" + "---|" * len(HS))
    for s in ["M4", "M5_lb0p1", "M5_lb0p5"]:
        for label, m in [("2020", in2020), ("excluding 2020", ~in2020)]:
            diffs = [(loss(s, kind, h, m) - loss("HAR", kind, h, m)).mean()
                     for h in HS]
            w(f"| {s}, {label} | "
              + " | ".join(f"{d:+.4f}" for d in diffs) + " |")
    w()

# ------------------------------------------------------------------ R3
w("## R3. Excluding origins contaminated by the negative price print")
w()
w("An origin at week t with horizon h has a target window covering the h "
  "weeks after t+1. The two undefined daily returns around 20 April 2020 "
  "enter every window containing that date, so the affected origins differ "
  "by horizon. Each horizon is therefore re-tested on its own reduced sample.")
w()
w("| h | contaminated origins | remaining | HAR | M5 (0.1) | M5 (0.5) | "
  "DM vs HAR, M5 (0.1) | DM vs HAR, M5 (0.5) |")
w("|---|---|---|---|---|---|---|---|")
for h in HS:
    # target window for origin t spans (t+1, t+1+h] in weeks
    start = DATES + pd.Timedelta(days=7)
    end = start + pd.Timedelta(days=7 * h)
    hit = (start < NEG_PRICE_DATE) & (NEG_PRICE_DATE <= end)
    keep = ~hit
    cells = [f"{loss(s, 'QLIKE', h, keep).mean():.4f}"
             for s in ["HAR", "M5_lb0p1", "M5_lb0p5"]]
    tests = []
    for s in ["M5_lb0p1", "M5_lb0p5"]:
        st, p = dm(loss(s, "QLIKE", h, keep), loss("HAR", "QLIKE", h, keep), h)
        tests.append(f"{st:.2f} (p = {p:.3f})")
    w(f"| {h} | {int(hit.sum())} | {int(keep.sum())} | "
      + " | ".join(cells) + " | " + " | ".join(tests) + " |")
w()
w("### The 2020 decomposition with the contaminated origins removed (QLIKE)")
w()
w("| Specification and period | " + " | ".join(f"h = {h}" for h in HS) + " |")
w("|---|" + "---|" * len(HS))
for s in ["M5_lb0p1", "M5_lb0p5"]:
    for label, want2020 in [("2020", True), ("excluding 2020", False)]:
        cells = []
        for h in HS:
            start = DATES + pd.Timedelta(days=7)
            end = start + pd.Timedelta(days=7 * h)
            keep = ~((start < NEG_PRICE_DATE) & (NEG_PRICE_DATE <= end))
            m = keep & ((DATES.year == 2020) == want2020)
            cells.append(f"{(loss(s, 'QLIKE', h, m) - loss('HAR', 'QLIKE', h, m)).mean():+.4f}")
        w(f"| {s}, {label} | " + " | ".join(cells) + " |")
w()

# ------------------------------------------------------------------ R4
w("## R4. Excluding 2017-2019, the hyperparameter-selection overlap")
w()
post = DATES.year >= 2020
w(f"Weeks removed: {int((~post).sum())};  remaining: {int(post.sum())}")
w()
for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"### {kind}, 2020 onward only")
    w()
    w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    best = {h: min(SPECS, key=lambda s: loss(s, kind, h, post).mean())
            for h in HS}
    for s in SPECS:
        cells = []
        for h in HS:
            v = f"{loss(s, kind, h, post).mean():.4f}"
            cells.append(f"**{v}**" if best[h] == s else v)
        w(f"| {s} | " + " | ".join(cells) + " |")
    w()

w("### Ladder contrasts on the reduced sample (Holm-adjusted p)")
w()
CONTRASTS = [("M2", "M1"), ("M3", "M2"), ("M4arch", "M3"),
             ("M4", "M4arch"), ("M5_lb0p1", "M4"), ("M4arch", "M2")]
for kind in ["QLIKE", "MAE", "MSE"]:
    w(f"**{kind}**")
    w()
    w("| Contrast | " + " | ".join(f"h = {h}" for h in HS) + " |")
    w("|---|" + "---|" * len(HS))
    for a, b in CONTRASTS:
        sts, ps = zip(*[dm(loss(a, kind, h, post), loss(b, kind, h, post), h)
                        for h in HS])
        adj = holm(np.array(ps))
        w(f"| {a} vs {b} | " + " | ".join(f"{s:.2f} ({p:.3f})"
                                          for s, p in zip(sts, adj)) + " |")
    w()

# ------------------------------------------------------------------ R4b
w("## R4b. Separating the two subsample effects")
w()
w("Removing 2017-2019 removes the hyperparameter-selection overlap, but it "
  "also raises the share of 2020 weeks from 10.7% to 15.7%, and Section 5.2 "
  "shows that those weeks carry the whole apparent advantage. The two effects "
  "must therefore be separated before the reduced sample can be read. "
  "Crossing the two criteria gives four samples; the last is clean of both.")
w()
w("Entries are the percentage change in mean loss relative to the benchmark "
  "with the Holm-adjusted p-value in parentheses. A positive percentage means "
  "the specification is worse than the benchmark.")
w()
SUBSAMPLES = [
    ("Full pooled sample", np.ones(len(DATES), bool)),
    ("Excluding 2017-2019 (selection overlap)", DATES.year >= 2020),
    ("Excluding 2020 only", DATES.year != 2020),
    ("Excluding both (clean of overlap and of 2020)",
     (DATES.year >= 2020) & (DATES.year != 2020)),
]
for label, m in SUBSAMPLES:
    share = (DATES[m].year == 2020).sum() / m.sum()
    w(f"### {label}  — n = {int(m.sum())}, 2020 share {share:.1%}")
    w()
    for kind in ["QLIKE", "MAE", "MSE"]:
        w(f"**{kind}**")
        w()
        w("| Specification | " + " | ".join(f"h = {h}" for h in HS) + " |")
        w("|---|" + "---|" * len(HS))
        for s in ["M4", "M5_lb0p1", "M5_lb0p5"]:
            pct, ps = [], []
            for h in HS:
                a, b = loss(s, kind, h, m), loss("HAR", kind, h, m)
                _, p = dm(a, b, h)
                ps.append(p)
                pct.append(100 * (a.mean() - b.mean()) / b.mean())
            adj = holm(np.array(ps))
            w(f"| {s} | " + " | ".join(f"{v:+.1f}% ({p:.3f})"
                                       for v, p in zip(pct, adj)) + " |")
        w()

# ------------------------------------------------------------------ R5
w("## R5. Sample size required to detect a 5% improvement")
w()
w("The minimum detectable improvement scales as n^(-1/2), so the sample "
  "needed to detect a target improvement follows from the observed long-run "
  "variance. Weeks are converted to years at 52 per year.")
w()
w("| h | benchmark mean loss | MDI at n = 487 | weeks for 5% | years |")
w("|---|---|---|---|---|")
n = len(DATES)
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
for h in HS:
    d = loss("M5_lb0p1", "QLIKE", h) - loss("HAR", "QLIKE", h)
    v = nw_var(d, bandwidth(n, h))
    base = loss("HAR", "QLIKE", h).mean()
    mdi = z * np.sqrt(v / n)
    target = 0.05 * base
    n_req = v * (z / target) ** 2
    w(f"| {h} | {base:.4f} | {100 * mdi / base:.1f}% | "
      f"{n_req:,.0f} | {n_req / 52:,.0f} |")
w()

path = HERE / "referee_robustness.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
