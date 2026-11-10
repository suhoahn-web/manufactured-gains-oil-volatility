"""Multiplicity across the whole search, not only across horizons.

Holm correction within a family of five horizons handles one dimension of the
search. The study also spans nine specifications, five constraint weights, three
evaluation scales and two anchoring schemes, and a paper about researcher
degrees of freedom should account for that. White's (2000) reality check and
Hansen's (2005) superior predictive ability test ask whether the *best* member of
a set beats the benchmark once the whole set is taken into account.

Both are run on the recovered forecasts with a stationary bootstrap whose mean
block length matches the horizon, as in the model confidence set elsewhere in
the paper.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from forecast_recovery import (  # noqa: E402
    HORIZONS, PUBLISHED, ROBUST, SCALE_LABELS, losses_on_all_scales,
    pooled_index,
)

RNG = np.random.default_rng(0)
B = 5000

# every specification a reader could see as a candidate, plus the constraint
# sweep, which is itself a search
CANDIDATES = ["M1", "M2", "M2wide", "M3derived", "M3", "M4arch", "M4",
              "M5_lb0p1", "M5_lb0p5", "M5_lb2p0", "M5_lb5p0"]
IDX = pooled_index()
L = {s: losses_on_all_scales(s) for s in ["HAR"] + CANDIDATES}


def stationary_bootstrap_indices(n, block, reps):
    """Politis-Romano: geometric block lengths, wrapped."""
    p = 1.0 / block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = RNG.integers(0, n, reps)
    newblock = RNG.random((reps, n)) < p
    starts = RNG.integers(0, n, (reps, n))
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(newblock[:, t], starts[:, t], cont)
    return idx


def spa_and_rc(d_mat, block):
    """d_mat: (n, k) loss differentials, benchmark minus candidate.

    A positive column mean means the candidate beats the benchmark. Returns
    (reality-check p, SPA consistent p).
    """
    n, k = d_mat.shape
    dbar = d_mat.mean(axis=0)
    idx = stationary_bootstrap_indices(n, block, B)
    boot = d_mat[idx].mean(axis=1)                       # (B, k)

    # variance of each column's mean, from the same bootstrap
    omega = boot.var(axis=0, ddof=1) * n
    omega = np.maximum(omega, 1e-12)

    # White's reality check: unstudentised, recentred
    rc_stat = np.sqrt(n) * dbar.max()
    rc_boot = (np.sqrt(n) * (boot - dbar)).max(axis=1)
    rc_p = float((rc_boot > rc_stat).mean())

    # Hansen's SPA: studentised, with the consistent recentring that drops
    # candidates too far below the benchmark to matter
    t_stat = np.sqrt(n) * dbar / np.sqrt(omega)
    spa_stat = max(t_stat.max(), 0.0)
    thresh = -np.sqrt(omega / n) * np.sqrt(2 * np.log(np.log(n)))
    g = np.where(dbar >= thresh, dbar, 0.0)
    spa_boot = (np.sqrt(n) * (boot - g) / np.sqrt(omega)).max(axis=1)
    spa_boot = np.maximum(spa_boot, 0.0)
    spa_p = float((spa_boot > spa_stat).mean())
    return rc_p, spa_p


out = []


def w(s=""):
    out.append(s)
    print(s)


w("# Data-snooping tests across the whole candidate set")
w()
w(f"Candidates ({len(CANDIDATES)}): {', '.join(CANDIDATES)}")
w(f"Pooled weeks: {len(IDX)}. Stationary bootstrap, {B} replications, "
  f"mean block max(4, h).")
w()
w("The null is that no candidate beats the benchmark once the full set is "
  "accounted for. A small p-value would be evidence that the best candidate's "
  "advantage is not an artefact of searching over many of them.")
w()

for key in ROBUST + PUBLISHED:
    w(f"### {SCALE_LABELS[key]}")
    w()
    w("| h | best candidate | its mean advantage | Reality Check p | SPA p |")
    w("|---|---|---|---|---|")
    for h in HORIZONS:
        d = np.column_stack([L["HAR"][h][key] - L[s][h][key]
                             for s in CANDIDATES])
        rc, spa = spa_and_rc(d, max(4, h))
        j = int(np.argmax(d.mean(axis=0)))
        adv = 100 * d.mean(axis=0)[j] / L["HAR"][h][key].mean()
        w(f"| {h} | {CANDIDATES[j]} | {adv:+.1f}% | {rc:.3f} | {spa:.3f} |")
    w()

w("## Reading")
w()
w("On the scales where the point estimates favour the candidates, the tests "
  "do not reject: the best of eleven specifications does not beat the "
  "benchmark by more than a search over eleven specifications would be "
  "expected to produce. On the log scale the point estimates favour the "
  "benchmark, so the tests cannot reject by construction. Either way the "
  "conclusion is the same as the horizon-wise inference reached, and it now "
  "accounts for the size of the search rather than for five horizons alone.")

path = HERE / "snooping_tests.md"
path.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nWrote {path}")
