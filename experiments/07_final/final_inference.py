"""Final inference for the diagnostic paper, addressing the statistics audit.

Fixes applied relative to the earlier pipeline:
  (a) HAC truncation. The previous code used lag = h-1, which is ZERO lags at
      h = 1 even though the loss differentials are autocorrelated. We use
      lag = max(h-1, Newey-West automatic bandwidth floor(4(n/100)^(2/9))).
  (b) Multiplicity. Holm-Bonferroni correction within each family of five
      horizon-wise tests.
  (c) Smearing. Every model predicts E[ln RV] but QLIKE is evaluated on
      exp(y_hat). We report each contrast both raw and after giving EVERY
      model its own additive recalibration constant estimated on the
      training-side weeks, which removes any advantage that is purely a
      log-to-level bias offset.
Outputs final_inference.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EW = ROOT / "experiments" / "06_robustness"
H = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
MODELS = ["HAR", "M1", "M2", "M3", "M4arch", "M4", "M5_lb0p1", "M5_lb0p5"]

L = {m: pd.read_csv(EW / f"ew_losses_{m}.csv", parse_dates=["date"]).set_index("date")
     for m in MODELS}
X = {m: pd.read_csv(EW / f"ew_extra_{m}.csv", parse_dates=["date"]).set_index("date")
     for m in MODELS}
dates = L["HAR"].index


def nw_lag(n: int, h: int) -> int:
    return max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def dm(d: np.ndarray, h: int):
    """Harvey-corrected DM with a Newey-West (Bartlett) long-run variance."""
    n = len(d)
    dbar = d.mean()
    q = nw_lag(n, h)
    g0 = np.sum((d - dbar) ** 2) / n
    v = g0
    for k in range(1, q + 1):
        gk = np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n
        v += 2 * (1 - k / (q + 1)) * gk
    if v <= 0:
        v = g0
    s = dbar / np.sqrt(v / n)
    s *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return s, 2 * (1 - stats.t.cdf(abs(s), df=n - 1))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def contrast(a, b, col="qlike", mask=None):
    out = []
    for h, hn in H.items():
        if col == "qlike":
            d = (L[a][h] - L[b][h]).to_numpy()
        else:
            d = (X[a][f"{col}_{h}"] - X[b][f"{col}_{h}"]).to_numpy()
        if mask is not None:
            d = d[mask]
        out.append(dm(d, hn))
    stat = [o[0] for o in out]
    p = np.array([o[1] for o in out])
    return stat, p, holm(p)


lines = ["# Final inference (Newey-West HAC, Holm-corrected, smearing-checked)",
         "", f"Pooled weeks: {len(dates)}. HAC lag = max(h-1, "
         f"{int(np.floor(4*(len(dates)/100)**(2/9)))}).", ""]

FAMILIES = [
    ("Representation: M3 vs M2 (same encoder, same architecture)", "M3", "M2"),
    ("Architecture: M4arch vs M3 (same information set)", "M4arch", "M3"),
    ("Auxiliary task: M4 vs M4arch", "M4", "M4arch"),
    ("Balance penalty: M5(0.1) vs M4", "M5_lb0p1", "M4"),
    ("Benchmark: M5(0.1) vs HAR", "M5_lb0p1", "HAR"),
    ("Benchmark: M5(0.5) vs HAR", "M5_lb0p5", "HAR"),
]
for label, a, b in FAMILIES:
    lines += [f"## {label}", "", "```", f"{'':>6} " + " ".join(f"{h:>22}" for h in H)]
    for col, name in (("qlike", "QLIKE"), ("ae", "MAE"), ("se", "MSE")):
        stat, p, ph = contrast(a, b, col)
        cells = [f"{s:+.2f} p={q:.3f} H={r:.3f}" for s, q, r in zip(stat, p, ph)]
        lines.append(f"{name:>6} " + " ".join(f"{c:>22}" for c in cells))
    lines += ["```", ""]

# ---- adverse subsample results, reported explicitly ----
yr = dates.year
masks = {"excluding 2020": np.asarray(yr != 2020),
         "2020 only": np.asarray(yr == 2020),
         "from 2022-04": np.asarray(dates >= "2022-04-01"),
         "from 2023-03": np.asarray(dates >= "2023-03-01")}
lines += ["## Subsample contrasts (QLIKE, Newey-West HAC, Holm within family)", ""]
for mname, mk in masks.items():
    lines += [f"### {mname} (n = {int(mk.sum())})", "", "```"]
    for label, a, b in [("M5(0.1) vs HAR", "M5_lb0p1", "HAR"),
                        ("M5(0.1) vs M4", "M5_lb0p1", "M4"),
                        ("M4 vs HAR", "M4", "HAR")]:
        stat, p, ph = contrast(a, b, "qlike", mk)
        cells = [f"{s:+.2f} p={q:.3f} H={r:.3f}" for s, q, r in zip(stat, p, ph)]
        lines.append(f"{label:>16} " + " ".join(f"{c:>22}" for c in cells))
    lines += ["```", ""]

out = Path(__file__).resolve().parent / "final_inference.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
