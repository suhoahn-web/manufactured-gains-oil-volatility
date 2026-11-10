"""Break the confound in the M2-to-M3 contrast.

M3 does not replace the raw physical series with stock-flow quantities; it adds
the six derived quantities on top of the nine raw ones, taking the input from 21
to 27 and the parameter count from 17,029 to 18,181. The contrast therefore
varies representation, information and capacity together, and cannot identify a
representation effect on its own.

Two arms separate them, both estimated through the same five expanding windows,
five seeds, optimiser, stopping rule and HAR anchor as the published ladder:

  M3derived   price block plus ONLY the six balance-derived quantities. Against
              M2 this is a genuine replacement of raw levels by stock-flow
              quantities on a comparable input dimension.

  M2wide      price block plus ONLY the nine raw series, with the hidden state
              widened so that the parameter count matches M3's. Against M3 this
              holds capacity fixed while varying information, so any remaining
              M2-to-M3 gap cannot be attributed to model size.
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
    PHYS_DERIVED, PHYS_RAW, PRICE_FEATURES, RV_FEATURES, VOL_TARGETS,
    GRUForecaster, WindowDataset,
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
SEEDS, LOOKBACK = 5, 12
HERE = Path(__file__).resolve().parent

# name, feature list, hidden size
CONFIGS = [
    ("M3derived", PRICE_FEATURES + PHYS_DERIVED + RV_FEATURES, 64),
    ("M2wide", PRICE_FEATURES + PHYS_RAW + RV_FEATURES, 66),
]


def n_params(n_in, hidden, n_out=5):
    """GRU (3 gates, two bias vectors) plus the linear head."""
    return 3 * hidden * (n_in + hidden) + 6 * hidden + hidden * n_out + n_out


print("parameter counts")
for ref_name, ref_in, ref_h in (("M2", 21, 64), ("M3", 27, 64)):
    print(f"  {ref_name:10s} in={ref_in:3d} hidden={ref_h:3d} "
          f"params={n_params(ref_in, ref_h):,}")
for name, feats, hid in CONFIGS:
    print(f"  {name:10s} in={len(feats):3d} hidden={hid:3d} "
          f"params={n_params(len(feats), hid):,}")
print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = RL.load_data()
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
dates = df["date"]
pooled, extra_metrics = {}, {}


def qlike(a, b):
    r = np.exp(a - b)
    return r - np.log(r) - 1.0


for wi, (tr_end, va_end, te_end) in enumerate(WINDOWS):
    tr = np.where(dates <= tr_end)[0]
    va = np.where((dates > tr_end) & (dates <= va_end))[0]
    te = np.where((dates > va_end) & (dates <= te_end))[0]
    if len(te) == 0:
        continue
    splits = {"train": tr, "val": va, "test": te}
    print(f"[window {wi}] train {len(tr)} val {len(va)} test {len(te)}",
          flush=True)

    X = np.column_stack([np.ones(len(df)), np.log(df["px_rv_1w"]),
                         np.log(df["px_rv_4w"]), np.log(df["px_rv_12w"])])
    har = np.empty((len(df), len(HORIZONS)))
    for hi in range(len(HORIZONS)):
        beta, *_ = np.linalg.lstsq(X[tr], y_all[tr, hi], rcond=None)
        har[:, hi] = X @ beta
    df_fit = df.copy()
    for hi, tgt in enumerate(VOL_TARGETS):
        df_fit[tgt] = y_all[:, hi] - har[:, hi]

    extra = {"flow_t": df[["production", "imports", "exports",
                           "refinery_inputs"]].to_numpy(dtype=np.float32),
             "d_spr_next": df["bal_d_spr_next"].to_numpy(dtype=np.float32)}

    for name, feats, hid in CONFIGS:
        groups = {"x": feats}
        feat_arrays, _, y_std, y_sc, aux_std, _ = RL.prepare_arrays(
            df_fit, splits, groups)
        loaders = {s: DataLoader(WindowDataset(df, i, LOOKBACK, feat_arrays,
                                               y_std, aux_std, extra),
                                 batch_size=64, shuffle=(s == "train"))
                   for s, i in splits.items()}
        preds = []
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GRUForecaster(len(feats), hidden=hid, dropout=0.1,
                                  n_horizons=len(VOL_TARGETS)).to(device)
            model = RL.train_model(model, loaders, device, CFG, False)
            _, y_p_std, rows, _ = RL.predict(model, loaders["test"], device,
                                             False)
            preds.append((rows, y_sc.inverse(y_p_std)))

        rows = preds[0][0]
        y_pred = np.mean([p[1] for p in preds], axis=0) + har[rows]
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
