"""Unanchored counterparts of M1 and M5, for two referee questions.

Section 4.2 justifies anchoring by reporting that unanchored networks cost a
substantial margin at short horizons, but gives no number. This supplies one.

It also answers a sharper objection. The benchmark absorbs the persistence of
realized volatility, and if the storage mechanism acts through persistence
then anchoring removes the channel through which fundamentals could help. If
the physical block adds nothing to an unanchored network either, that reading
is unavailable: the null is then not an artefact of the anchor.

Same five windows, five seeds and pooling as `expanding_window.py`; the only
difference is that the targets are log realized volatilities themselves rather
than deviations from the window's fitted benchmark.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.run_ladder as RL  # noqa: E402
from src.models.ladder import (  # noqa: E402
    AUX_TARGETS, FEATURE_SETS, M45_PHYS, M45_PRICE, RV_FEATURES, VOL_TARGETS,
    DualBranchForecaster, GRUForecaster, WindowDataset,
)

RL.VOL_MODE = True
RL.TARGETS = VOL_TARGETS
RL.HORIZON_LABELS = {i: t.replace("yv_", "") for i, t in enumerate(VOL_TARGETS)}
HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}

WINDOWS = [
    ("2014-12-31", "2016-12-31", "2018-12-31"),
    ("2016-12-31", "2018-12-31", "2020-12-31"),
    ("2018-12-31", "2020-12-31", "2022-12-31"),
    ("2020-12-31", "2022-12-31", "2024-12-31"),
    ("2022-12-31", "2024-12-31", "2026-12-31"),
]
CFG = {"lr": 1e-3, "epochs": 200, "patience": 20}
SEEDS = 5
LOOKBACK = 12
LAMBDA_F = 0.1
HERE = Path(__file__).resolve().parent

# name, dual branch, lambda_B
CONFIGS = [("M1_noanchor", False, 0.0), ("M5_lb0p1_noanchor", True, 0.1)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = RL.load_data()
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
dates = df["date"]

pooled, extra_metrics = {}, {}


def qlike(y_true_log, y_pred_log):
    ratio = np.exp(y_true_log - y_pred_log)
    return ratio - np.log(ratio) - 1.0


for wi, (tr_end, va_end, te_end) in enumerate(WINDOWS):
    tr = np.where(dates <= tr_end)[0]
    va = np.where((dates > tr_end) & (dates <= va_end))[0]
    te = np.where((dates > va_end) & (dates <= te_end))[0]
    if len(te) == 0:
        continue
    splits = {"train": tr, "val": va, "test": te}
    print(f"[window {wi}] train {len(tr)} val {len(va)} test {len(te)}",
          flush=True)

    extra = {
        "flow_t": df[["production", "imports", "exports",
                      "refinery_inputs"]].to_numpy(dtype=np.float32),
        "d_spr_next": df["bal_d_spr_next"].to_numpy(dtype=np.float32),
    }
    bal_scale = float(np.std(np.diff(df["stocks_commercial"].to_numpy()[tr])))

    for name, is_dual, lam_b in CONFIGS:
        if is_dual:
            groups = {"price": M45_PRICE + RV_FEATURES, "phys": M45_PHYS}
        else:
            groups = {"x": FEATURE_SETS["M1"] + RV_FEATURES}
        # no anchor: df itself carries the log-RV targets
        feat_arrays, _, y_std, y_sc, aux_std, aux_sc = RL.prepare_arrays(
            df, splits, groups)
        loaders = {}
        for sname, idx in splits.items():
            dset = WindowDataset(df, idx, LOOKBACK, feat_arrays, y_std,
                                 aux_std, extra)
            loaders[sname] = DataLoader(dset, batch_size=64,
                                        shuffle=(sname == "train"))
        preds = []
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            if is_dual:
                model = DualBranchForecaster(
                    len(groups["price"]), len(M45_PHYS), hidden=64,
                    phys_hidden=32, dropout=0.1,
                    n_horizons=len(VOL_TARGETS),
                    n_aux=len(AUX_TARGETS)).to(device)
            else:
                model = GRUForecaster(len(groups["x"]), hidden=64, dropout=0.1,
                                      n_horizons=len(VOL_TARGETS)).to(device)
            model = RL.train_model(model, loaders, device, CFG, is_dual,
                                   aux_sc=aux_sc, lambda_f=LAMBDA_F if is_dual else 0.0,
                                   lambda_b=lam_b, bal_scale=bal_scale)
            _, y_p_std, rows, _ = RL.predict(model, loaders["test"], device,
                                             is_dual)
            preds.append((rows, y_sc.inverse(y_p_std)))

        rows = preds[0][0]
        y_pred = np.mean([p[1] for p in preds], axis=0)   # no anchor added
        y_true = y_all[rows]
        date_arr = dates.iloc[rows].to_numpy()

        losses = {h: qlike(y_true[:, hi], y_pred[:, hi])
                  for hi, h in enumerate(HORIZONS)}
        pooled.setdefault(name, []).append(
            pd.DataFrame({"date": date_arr, **losses}))

        rv1 = np.log(df["px_rv_1w"].to_numpy()[rows])
        rec = {"date": date_arr}
        for hi, h in enumerate(HORIZONS):
            e = y_pred[:, hi] - y_true[:, hi]
            ref = rv1 + 0.5 * np.log(HORIZONS[h])
            rec[f"ae_{h}"] = np.abs(e)
            rec[f"se_{h}"] = e ** 2
            rec[f"hit_{h}"] = (np.sign(y_pred[:, hi] - ref)
                               == np.sign(y_true[:, hi] - ref)).astype(float)
        extra_metrics.setdefault(name, []).append(pd.DataFrame(rec))

        print(f"  {name}: QLIKE "
              + " ".join(f"{h}={losses[h].mean():.4f}" for h in HORIZONS),
              flush=True)

for name, recs in pooled.items():
    (pd.concat(recs).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_losses_{name}.csv", index=False))
    (pd.concat(extra_metrics[name]).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_extra_{name}.csv", index=False))
print("\nwrote " + ", ".join(pooled))
