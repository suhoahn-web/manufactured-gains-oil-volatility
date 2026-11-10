"""Do the ladder findings survive multiplicity control across the whole family?

Holm within a family of five horizons is the correction the paper applies to
each contrast separately. The ladder as a whole tests six prespecified contrasts
at five horizons, so thirty hypotheses per loss. This applies the Romano-Wolf
(2005) stepdown, which controls the familywise error rate across all thirty
jointly while exploiting the dependence between them, and reports it beside the
per-contrast Holm values the manuscript already gives.
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

RNG = np.random.default_rng(0)
B = 5000
CONTRASTS = [
    ("M2", "M1", "supply fundamentals as raw levels"),
    ("M3derived", "M2", "represent them as balance quantities"),
    ("M2wide", "M2", "add capacity without information"),
    ("M3", "M2wide", "add information at fixed capacity"),
    ("M4arch", "M3", "encode in a dedicated branch"),
    ("M4", "M4arch", "predict the future physical state"),
    ("M5_lb0p1", "M4", "reconcile it through the identity"),
]
SPECS = sorted({s for a, b, _ in CONTRASTS for s in (a, b)})
IDX = pooled_index()
L = {s: losses_on_all_scales(s) for s in SPECS}


def nw(d, q):
    n = len(d); d = d - d.mean(); v = (d @ d) / n
    for k in range(1, q + 1):
        v += 2 * (1 - k / (q + 1)) * ((d[k:] @ d[:-k]) / n)
    return v


def band(n, h):
    return max(h - 1, int(4 * (n / 100) ** (2 / 9)))


def tstat(d, h):
    n = len(d)
    return d.mean() / np.sqrt(nw(d, band(n, h)) / n)


def holm(ps):
    o = np.argsort(ps); a = np.empty(len(ps)); r = 0.0
    for i, ix in enumerate(o):
        r = max(r, (len(ps) - i) * ps[ix]); a[ix] = min(r, 1.0)
    return a


def blocks(n, block, reps):
    p = 1.0 / block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = RNG.integers(0, n, reps)
    new = RNG.random((reps, n)) < p
    starts = RNG.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def romano_wolf(D, hs, block):
    """D: (n, k) loss differentials. Returns stepdown-adjusted p-values."""
    n, k = D.shape
    t_obs = np.array([tstat(D[:, j], hs[j]) for j in range(k)])
    idx = blocks(n, block, B)
    # centred bootstrap t-statistics, studentised by the same long-run variance
    se = np.array([np.sqrt(nw(D[:, j], band(n, hs[j])) / n) for j in range(k)])
    boot_means = np.stack([D[idx, j].mean(axis=1) for j in range(k)], axis=1)
    T = np.abs((boot_means - D.mean(axis=0)) / se)
    order = np.argsort(-np.abs(t_obs))
    adj = np.empty(k)
    running = 0.0
    for step, j in enumerate(order):
        rest = order[step:]
        maxT = T[:, rest].max(axis=1)
        p = float((maxT >= abs(t_obs[j])).mean())
        running = max(running, p)
        adj[j] = min(running, 1.0)
    return t_obs, adj


out = []


def w(s=""):
    out.append(s)
    print(s)


w("# Multiplicity across the whole ladder, not one contrast at a time")
w()
w(f"Seven prespecified contrasts at five horizons: {7 * len(HORIZONS)} "
  f"hypotheses per loss. Romano-Wolf stepdown with a stationary bootstrap, "
  f"{B} replications, mean block max(4, h). Holm-within-contrast is the "
  f"correction the manuscript reports; both are shown.")
w()

for key in ROBUST + PUBLISHED:
    w(f"### {SCALE_LABELS[key]}")
    w()
    cols, hs, labels = [], [], []
    for a, b, lab in CONTRASTS:
        for h in HORIZONS:
            cols.append(L[a][h][key] - L[b][h][key])
            hs.append(h)
            labels.append((f"{a} vs {b}", h, lab))
    D = np.column_stack(cols)
    t_obs, rw = romano_wolf(D, hs, max(4, max(HORIZONS)))
    raw = np.array([2 * (1 - stats.t.cdf(abs(t), df=len(IDX) - 1))
                    for t in t_obs])
    # Holm within each contrast, as the manuscript reports it
    hw = np.empty_like(raw)
    for i in range(0, len(raw), len(HORIZONS)):
        hw[i:i + len(HORIZONS)] = holm(raw[i:i + len(HORIZONS)])

    w("| Contrast | h | t | Holm within contrast | Romano-Wolf across ladder |")
    w("|---|---|---|---|---|")
    for i, (name, h, lab) in enumerate(labels):
        if hw[i] < 0.10 or rw[i] < 0.10:
            w(f"| {name} | {h} | {t_obs[i]:.2f} | {hw[i]:.3f} | {rw[i]:.3f} |")
    surv = sum((hw < 0.05) & (rw < 0.05))
    w(f"\nRows shown are those significant at 10% under either correction. "
      f"{int((hw < 0.05).sum())} of {len(raw)} hypotheses clear 5% under "
      f"Holm-within-contrast, {int((rw < 0.05).sum())} under Romano-Wolf, "
      f"{surv} under both.")
    w()

path = HERE / "ladder_multiplicity.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
