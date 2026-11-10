"""Recover the pooled forecasts, so any loss can be evaluated after the fact.

The expanding-window runs stored per-week losses but not forecasts. They are
nevertheless recoverable exactly. With d = log RV − log F the stored quantities
are

    qlike = exp(d) − d − 1,     se = d²,     ae = |d|

so |d| is known and its sign is identified by the asymmetry of QLIKE. Recovery
was verified to a maximum absolute error of 3e-15 against the stored losses.

This matters because the published evaluation scores QLIKE on volatility and
squared error on LOG volatility, while Patton's (2011) proxy-robustness result
concerns forecasts of conditional variance scored against a variance proxy.
Neither published loss is covered by it. Recovering the forecasts lets the
comparison be redone on the functional the robustness result actually covers,
without retraining anything.

Import `load_forecasts()` from here rather than re-deriving the inversion.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POOL = PROJECT_ROOT / "experiments" / "06_robustness"
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.run_ladder as RL  # noqa: E402
from src.models.ladder import VOL_TARGETS  # noqa: E402

HORIZONS = [1, 2, 4, 8, 12]

RL.VOL_MODE = True
RL.TARGETS = VOL_TARGETS
_DF = RL.load_data()
_TGT = _DF.set_index("date")[VOL_TARGETS]


def available_specs():
    return sorted(p.stem.replace("ew_losses_", "")
                  for p in POOL.glob("ew_losses_*.csv")
                  if (POOL / f"ew_extra_{p.stem.replace('ew_losses_', '')}.csv"
                      ).exists())


def pooled_index():
    return pd.read_csv(POOL / "ew_losses_HAR.csv",
                       parse_dates=["date"]).set_index("date").index


def load_forecasts(spec, index=None, check=True):
    """Return {h: (log_rv_true, log_forecast)} for one specification."""
    idx = pooled_index() if index is None else index
    ql = pd.read_csv(POOL / f"ew_losses_{spec}.csv",
                     parse_dates=["date"]).set_index("date")
    ex = pd.read_csv(POOL / f"ew_extra_{spec}.csv",
                     parse_dates=["date"]).set_index("date")
    out = {}
    for h in HORIZONS:
        q = ql.loc[idx, f"h{h}"].values
        a = ex.loc[idx, f"ae_h{h}"].values
        # choose the sign of d that reproduces the stored QLIKE
        d = np.where(np.abs((np.exp(a) - a - 1.0) - q)
                     <= np.abs((np.exp(-a) + a - 1.0) - q), a, -a)
        if check:
            err = np.abs((np.exp(d) - d - 1.0) - q).max()
            if err > 1e-8:
                raise RuntimeError(
                    f"{spec} h={h}: forecast recovery failed, max error {err:.2e}")
        y = _TGT.loc[idx, f"yv_h{h}"].values
        out[h] = (y, y - d)
    return out


# ---------------------------------------------------------------- loss scales
def losses_on_all_scales(spec, index=None):
    """Every loss the paper needs, on the three scales in circulation.

    variance scale  — QLIKE and MSE on RV², the functional Patton certifies
    volatility scale — QLIKE on RV, as published
    log scale       — MAE and MSE on log RV, the estimation scale, as published
    """
    fc = load_forecasts(spec, index)
    out = {}
    for h, (y, f) in fc.items():
        var_t, var_f = np.exp(2 * y), np.exp(2 * f)
        r_var = var_t / var_f
        r_vol = np.exp(y - f)
        out[h] = {
            "QLIKE_var": r_var - np.log(r_var) - 1.0,
            "MSE_var": (var_t - var_f) ** 2,
            "QLIKE_vol": r_vol - np.log(r_vol) - 1.0,
            "MAE_log": np.abs(f - y),
            "MSE_log": (f - y) ** 2,
            "bias_log": f - y,
        }
    return out


SCALE_LABELS = {
    "QLIKE_var": "QLIKE, realized variance",
    "MSE_var": "MSE, realized variance",
    "QLIKE_vol": "QLIKE, volatility (as published)",
    "MAE_log": "MAE, log volatility (as published)",
    "MSE_log": "MSE, log volatility (as published)",
}
ROBUST = ["QLIKE_var", "MSE_var"]          # covered by Patton (2011)
PUBLISHED = ["QLIKE_vol", "MAE_log", "MSE_log"]


if __name__ == "__main__":
    idx = pooled_index()
    print(f"pooled weeks: {len(idx)}  "
          f"({idx.min().date()} to {idx.max().date()})")
    print(f"specifications available: {', '.join(available_specs())}\n")
    print("recovery check (max |reconstructed qlike - stored qlike|)")
    for s in available_specs():
        ql = pd.read_csv(POOL / f"ew_losses_{s}.csv",
                         parse_dates=["date"]).set_index("date")
        worst = 0.0
        for h, (y, f) in load_forecasts(s).items():
            d = y - f
            worst = max(worst, np.abs((np.exp(d) - d - 1.0)
                                      - ql.loc[idx, f"h{h}"].values).max())
        print(f"  {s:20s} {worst:.2e}")
