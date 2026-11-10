"""Does the representation result depend on the neural estimator?

The ladder's networks carry 15,000-26,000 parameters against a few hundred
training weeks, reach their validation minimum at the third epoch, and disperse
noticeably across seeds. A referee can therefore argue that raw physical levels
are harmful to a GRU rather than harmful as information, and that the stock-flow
representation merely happens to suit the optimiser.

Least-squares HAR-X models settle it. They add the same physical blocks to the
benchmark directly, with no hidden layer, no optimiser and no seed:

    HAR            three realized-volatility components
    HAR-X raw      + the nine raw EIA series
    HAR-X derived  + the six balance-derived quantities
    HAR-X both     + all fifteen

Same five expanding windows, same release-aligned features, same targets, and
regressors standardised on training rows only. Ridge is reported alongside OLS
because fifteen extra regressors on a few hundred weeks is where OLS starts to
struggle, and a referee will ask.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.run_ladder as RL  # noqa: E402
from src.models.ladder import (  # noqa: E402
    PHYS_DERIVED, PHYS_RAW, VOL_TARGETS,
)

RL.VOL_MODE = True
RL.TARGETS = VOL_TARGETS
HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
WINDOWS = [
    ("2014-12-31", "2016-12-31", "2018-12-31"),
    ("2016-12-31", "2018-12-31", "2020-12-31"),
    ("2018-12-31", "2020-12-31", "2022-12-31"),
    ("2020-12-31", "2022-12-31", "2024-12-31"),
    ("2022-12-31", "2024-12-31", "2026-12-31"),
]
HERE = Path(__file__).resolve().parent
RIDGE_ALPHA = 1.0

df = RL.load_data()
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
dates = df["date"]
HAR_COLS = ["px_rv_1w", "px_rv_4w", "px_rv_12w"]

CONFIGS = [
    ("HARX_raw", PHYS_RAW),
    ("HARX_derived", PHYS_DERIVED),
    ("HARX_both", PHYS_RAW + PHYS_DERIVED),
]


def qlike(a, b):
    r = np.exp(a - b)
    return r - np.log(r) - 1.0


def fit_predict(X_tr, y_tr, X_te, ridge=0.0):
    """Least squares, or ridge on the standardised design with a free intercept."""
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (X_tr - mu) / sd, (X_te - mu) / sd
    ybar = y_tr.mean()
    if ridge > 0:
        k = Ztr.shape[1]
        beta = np.linalg.solve(Ztr.T @ Ztr + ridge * np.eye(k),
                               Ztr.T @ (y_tr - ybar))
    else:
        beta, *_ = np.linalg.lstsq(Ztr, y_tr - ybar, rcond=None)
    return Zte @ beta + ybar


pooled, extra = {}, {}

for wi, (tr_end, va_end, te_end) in enumerate(WINDOWS):
    # HAR-X has no hyperparameters to tune, so training uses train + validation
    tr = np.where(dates <= va_end)[0]
    te = np.where((dates > va_end) & (dates <= te_end))[0]
    if len(te) == 0:
        continue
    print(f"[window {wi}] train {len(tr)} test {len(te)}", flush=True)

    har_X = np.column_stack([np.log(df[c].to_numpy()) for c in HAR_COLS])
    for name, block in [("HARX_none", [])] + CONFIGS:
        cols = np.column_stack([har_X] + ([df[block].to_numpy(dtype=np.float64)]
                                          if block else []))
        keep = np.isfinite(cols).all(axis=1)
        tr_k = tr[keep[tr]]
        te_k = te[keep[te]]
        for tag, ridge in ([("", 0.0)] if not block
                           else [("", 0.0), ("_ridge", RIDGE_ALPHA)]):
            key = name + tag
            preds = np.empty((len(te_k), len(HORIZONS)))
            for hi in range(len(HORIZONS)):
                preds[:, hi] = fit_predict(cols[tr_k], y_all[tr_k, hi],
                                           cols[te_k], ridge)
            y_true = y_all[te_k]
            d = dates.iloc[te_k].to_numpy()
            pooled.setdefault(key, []).append(pd.DataFrame(
                {"date": d, **{h: qlike(y_true[:, hi], preds[:, hi])
                               for hi, h in enumerate(HORIZONS)}}))
            rec = {"date": d}
            rv1 = np.log(df["px_rv_1w"].to_numpy()[te_k])
            for hi, h in enumerate(HORIZONS):
                e = preds[:, hi] - y_true[:, hi]
                ref = rv1 + 0.5 * np.log(HORIZONS[h])
                rec[f"ae_{h}"] = np.abs(e)
                rec[f"se_{h}"] = e ** 2
                rec[f"hit_{h}"] = (np.sign(preds[:, hi] - ref)
                                   == np.sign(y_true[:, hi] - ref)).astype(float)
            extra.setdefault(key, []).append(pd.DataFrame(rec))

for k, recs in pooled.items():
    (pd.concat(recs).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_losses_{k}.csv", index=False))
    (pd.concat(extra[k]).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_extra_{k}.csv", index=False))
    n = len(pd.concat(recs))
    print(f"  wrote {k}: {n} weeks")
print("\nNote: HARX_none re-estimates the benchmark on train+validation rows, so "
      "it is the comparator for the HAR-X models rather than the ladder's HAR.")
