"""Additional expanding-window configurations answering two referee objections.

Both are run through the same five windows, five seeds, and pooling as
`expanding_window.py`, so the per-week loss files drop straight into the
existing inference code.

  M4_fcstop   dual branch, auxiliary task on, early stopping on the FORECAST
              term alone. The ladder's M4 monitors the composite objective,
              so the M4arch-to-M4 contrast confounds the auxiliary gradient
              with the stopping rule. This separates them.

  M5_lb2p0    the balance penalty at weights an order of magnitude above the
  M5_lb5p0    range reported in the paper. The reported sweep tops out at 0.5
              and moves the balance-consistency error by 2.3%, which is
              smaller than the seed dispersion — leaving open whether the
              constraint was ever binding. If accuracy is still unaffected at
              a weight where the constraint clearly binds, the null is about
              the constraint rather than about its strength.

The balance-consistency error is recorded per window so that the sweep can be
read as a trade-off rather than only as an accuracy result.
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
    AUX_TARGETS, M45_PHYS, M45_PRICE, RV_FEATURES, VOL_TARGETS,
    DualBranchForecaster, Scaler, WindowDataset, balance_terms,
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

# name, lambda_B, stop_on_forecast
CONFIGS = [
    ("M4_fcstop", 0.0, True),
    ("M5_lb2p0", 2.0, False),
    ("M5_lb5p0", 5.0, False),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = RL.load_data()
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
dates = df["date"]

pooled, extra_metrics, bce_rows = {}, {}, []


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
    print(f"[window {wi}] train {len(tr)} val {len(va)} test {len(te)} "
          f"({dates.iloc[te].min().date()} -> {dates.iloc[te].max().date()})",
          flush=True)

    # this window's HAR anchor, fitted on training rows only
    X = np.column_stack([np.ones(len(df)), np.log(df["px_rv_1w"]),
                         np.log(df["px_rv_4w"]), np.log(df["px_rv_12w"])])
    har_mat = np.empty((len(df), len(HORIZONS)))
    for hi in range(len(HORIZONS)):
        beta, *_ = np.linalg.lstsq(X[tr], y_all[tr, hi], rcond=None)
        har_mat[:, hi] = X @ beta

    df_fit = df.copy()
    for hi, tgt in enumerate(VOL_TARGETS):
        df_fit[tgt] = y_all[:, hi] - har_mat[:, hi]

    groups = {"price": M45_PRICE + RV_FEATURES, "phys": M45_PHYS}
    feat_arrays, _, y_std, y_sc, aux_std, aux_sc = RL.prepare_arrays(
        df_fit, splits, groups)
    extra = {
        "flow_t": df[["production", "imports", "exports",
                      "refinery_inputs"]].to_numpy(dtype=np.float32),
        "d_spr_next": df["bal_d_spr_next"].to_numpy(dtype=np.float32),
    }
    bal_scale = float(np.std(np.diff(df["stocks_commercial"].to_numpy()[tr])))
    loaders = {}
    for sname, idx in splits.items():
        dset = WindowDataset(df, idx, LOOKBACK, feat_arrays, y_std, aux_std,
                             extra)
        loaders[sname] = DataLoader(dset, batch_size=64,
                                    shuffle=(sname == "train"))

    for name, lam_b, fcstop in CONFIGS:
        preds, bces = [], []
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = DualBranchForecaster(
                len(groups["price"]), len(M45_PHYS), hidden=64,
                phys_hidden=32, dropout=0.1, n_horizons=len(VOL_TARGETS),
                n_aux=len(AUX_TARGETS)).to(device)
            model = RL.train_model(model, loaders, device, CFG, True,
                                   aux_sc=aux_sc, lambda_f=LAMBDA_F,
                                   lambda_b=lam_b, bal_scale=bal_scale,
                                   stop_on_forecast=fcstop)
            _, y_p_std, rows, aux_p = RL.predict(model, loaders["test"],
                                                 device, True)
            preds.append((rows, y_sc.inverse(y_p_std)))

            # balance-consistency error on this window's test weeks
            with torch.no_grad():
                d_stock, net_acc = balance_terms(
                    torch.tensor(aux_p, dtype=torch.float32),
                    aux_sc,
                    torch.tensor(extra["flow_t"][rows], dtype=torch.float32),
                    torch.tensor(extra["d_spr_next"][rows],
                                 dtype=torch.float32))
                bces.append(float(torch.mean(torch.abs(d_stock - net_acc))))

        rows = preds[0][0]
        y_pred = np.mean([p[1] for p in preds], axis=0) + har_mat[rows]
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

        bce_rows.append({"window": wi, "config": name,
                         "bce_kb_week_mean": float(np.mean(bces)),
                         "bce_kb_week_sd": float(np.std(bces))})
        print(f"  {name}: QLIKE "
              + " ".join(f"{h}={losses[h].mean():.4f}" for h in HORIZONS)
              + f" | BCE {np.mean(bces):.0f} kb/wk", flush=True)

for name, recs in pooled.items():
    (pd.concat(recs).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_losses_{name}.csv", index=False))
    (pd.concat(extra_metrics[name]).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_extra_{name}.csv", index=False))

pd.DataFrame(bce_rows).to_csv(HERE / "ew_bce_extra.csv", index=False)
print("\nwrote ew_losses_/ew_extra_ for "
      + ", ".join(pooled) + " and ew_bce_extra.csv")
