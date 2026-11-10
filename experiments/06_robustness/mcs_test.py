"""Model Confidence Set (Hansen, Lunde & Nason 2011) on pooled QLIKE losses.

Implementation notes (addressing the review):
- Elimination runs all the way down to a single model, so every model receives
  its own MCS p-value (the running maximum of the bootstrap p-values up to the
  step at which it is eliminated). Previously the loop stopped at the first
  non-rejection, which gave every surviving model an identical p-value and no
  ranking information.
- The stationary-bootstrap mean block length is tied to the forecast horizon
  (max(4, h) weeks), because overlapping h-week targets induce dependence of at
  least h-1 lags; a fixed 4-week block understates that dependence at h=8, 12.
- Statistic: T_max on the standardized average loss differentials, with the
  bootstrap distribution recentred on the sample means (Hansen et al. 2011).

The 90% MCS is the set of models with p >= 0.10.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
MODELS = ["HAR", "M1", "M2", "M2wide", "M3derived", "M3", "M4arch", "M4",
          "M5_lb0p1", "M5_lb0p5"]
B = 5000
rng = np.random.default_rng(0)

tables = {}
for m in MODELS:
    p = HERE / f"ew_losses_{m}.csv"
    if p.exists():
        tables[m] = pd.read_csv(p, parse_dates=["date"])
avail = [m for m in MODELS if m in tables]

common = tables[avail[0]][["date"]]
for m in avail[1:]:
    common = common.merge(tables[m][["date"]], on="date")
dates = common["date"]
losses = {m: tables[m].set_index("date").loc[dates, list(HORIZONS)]
          for m in avail}


def stationary_bootstrap_indices(n: int, b: int, mean_block: float) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty((b, n), dtype=int)
    starts = rng.integers(0, n, size=(b, n))
    new_block = rng.random((b, n)) < p
    new_block[:, 0] = True
    for j in range(n):
        if j == 0:
            idx[:, 0] = starts[:, 0]
        else:
            idx[:, j] = np.where(new_block[:, j], starts[:, j],
                                 (idx[:, j - 1] + 1) % n)
    return idx


results = {}
print("MCS p-values per horizon (90% set = models with p >= 0.10)\n")
for h, hn in HORIZONS.items():
    L = np.column_stack([losses[m][h].to_numpy() for m in avail])
    n, k = L.shape
    boot_idx = stationary_bootstrap_indices(n, B, mean_block=max(4.0, float(hn)))
    boot_all = L[boot_idx].mean(axis=1)          # [B, k] bootstrap mean losses

    alive = list(range(k))
    pvals, running = {}, 0.0
    while len(alive) > 1:
        Lc = L[:, alive]
        dbar_i = Lc.mean(axis=0)
        dbar = dbar_i - dbar_i.mean()            # vs the equal-weight average
        bm = boot_all[:, alive]
        boot_dev = bm - bm.mean(axis=1, keepdims=True) - dbar
        var_i = (boot_dev ** 2).mean(axis=0)
        var_i[var_i <= 0] = 1e-12
        t_i = dbar / np.sqrt(var_i)
        t_max = t_i.max()
        p = float(np.mean((boot_dev / np.sqrt(var_i)).max(axis=1) >= t_max))
        running = max(running, p)                # MCS p-values are monotone
        worst = int(np.argmax(t_i))
        pvals[avail[alive[worst]]] = running
        alive.pop(worst)
    pvals[avail[alive[0]]] = 1.0                 # last survivor

    results[h] = pvals
    order = sorted(pvals.items(), key=lambda kv: -kv[1])
    inset = [m for m, v in order if v >= 0.10]
    print(f"{h:>4}: " + " | ".join(f"{m}={v:.3f}" for m, v in order))
    print(f"      90% MCS: {', '.join(inset)}\n")

pd.DataFrame(results).to_csv(HERE / "mcs_results.csv")
print("Saved mcs_results.csv")
