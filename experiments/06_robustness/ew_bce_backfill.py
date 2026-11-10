"""Record the balance-consistency error for the two constraint weights used
in the ladder, which `expanding_window.py` did not capture.

Figure 4 plots forecast loss and balance-consistency error against constraint
strength. The loss arm is available for every weight from the pooled files;
the consistency arm exists only for the weights added later. Backfilling 0.1
and 0.5 lets the whole figure be drawn on the pooled expanding-window design
rather than mixing it with the single split.

The run also re-derives each weight's per-week losses under a temporary name
so they can be checked against the files already used in the paper: the
training is deterministic given the seed, so any difference would indicate the
pipeline had changed underneath the reported results.
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
    DualBranchForecaster, WindowDataset, balance_terms,
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
CFG = {"lr": 1e-3, "epochs": 200, "patience": 20}
SEEDS, LOOKBACK, LAMBDA_F = 5, 12, 0.1
HERE = Path(__file__).resolve().parent
CONFIGS = [("M5_lb0p1", 0.1), ("M5_lb0p5", 0.5)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = RL.load_data()
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
dates = df["date"]
bce_rows, check = [], {}


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
    print(f"[window {wi}]", flush=True)

    X = np.column_stack([np.ones(len(df)), np.log(df["px_rv_1w"]),
                         np.log(df["px_rv_4w"]), np.log(df["px_rv_12w"])])
    har = np.empty((len(df), len(HORIZONS)))
    for hi in range(len(HORIZONS)):
        beta, *_ = np.linalg.lstsq(X[tr], y_all[tr, hi], rcond=None)
        har[:, hi] = X @ beta
    df_fit = df.copy()
    for hi, t in enumerate(VOL_TARGETS):
        df_fit[t] = y_all[:, hi] - har[:, hi]

    groups = {"price": M45_PRICE + RV_FEATURES, "phys": M45_PHYS}
    feat, _, y_std, y_sc, aux_std, aux_sc = RL.prepare_arrays(
        df_fit, splits, groups)
    extra = {"flow_t": df[["production", "imports", "exports",
                           "refinery_inputs"]].to_numpy(dtype=np.float32),
             "d_spr_next": df["bal_d_spr_next"].to_numpy(dtype=np.float32)}
    bal_scale = float(np.std(np.diff(df["stocks_commercial"].to_numpy()[tr])))
    loaders = {s: DataLoader(WindowDataset(df, i, LOOKBACK, feat, y_std,
                                           aux_std, extra),
                             batch_size=64, shuffle=(s == "train"))
               for s, i in splits.items()}

    for name, lam_b in CONFIGS:
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
                                   lambda_b=lam_b, bal_scale=bal_scale)
            _, y_p_std, rows, aux_p = RL.predict(model, loaders["test"],
                                                 device, True)
            preds.append((rows, y_sc.inverse(y_p_std)))
            with torch.no_grad():
                d_stock, net_acc = balance_terms(
                    torch.tensor(aux_p, dtype=torch.float32), aux_sc,
                    torch.tensor(extra["flow_t"][rows], dtype=torch.float32),
                    torch.tensor(extra["d_spr_next"][rows],
                                 dtype=torch.float32))
                bces.append(float(torch.mean(torch.abs(d_stock - net_acc))))

        rows = preds[0][0]
        y_pred = np.mean([p[1] for p in preds], axis=0) + har[rows]
        check.setdefault(name, []).append(pd.DataFrame({
            "date": dates.iloc[rows].to_numpy(),
            **{h: qlike(y_all[rows][:, hi], y_pred[:, hi])
               for hi, h in enumerate(HORIZONS)}}))
        bce_rows.append({"window": wi, "config": name,
                         "bce_kb_week_mean": float(np.mean(bces)),
                         "bce_kb_week_sd": float(np.std(bces))})
        print(f"  {name}: BCE {np.mean(bces):.0f} kb/wk", flush=True)

pd.DataFrame(bce_rows).to_csv(HERE / "ew_bce_ladder.csv", index=False)

print("\nreproduction check against the files used in the paper:")
for name, recs in check.items():
    new = pd.concat(recs).sort_values("date").reset_index(drop=True)
    old = pd.read_csv(HERE / f"ew_losses_{name}.csv")
    diff = np.abs(new.iloc[:, 1:].values - old.iloc[:, 1:].values).max()
    print(f"  {name}: max absolute difference {diff:.2e}")
