"""External baselines for the volatility target (design doc section 19).

Modest, standard set, all evaluated on the SAME split, target, and metrics as
the model ladder:
  ARIMA(p,d,q)  - univariate on log RV, order by AIC on training data
  XGBoost       - gradient boosting on the full M3 feature set (flattened
                  12-week window), one model per horizon
  LSTM / TCN    - architecture swaps for M1's encoder (same features, size,
                  training protocol), 5 seeds each

Results append to outputs/tables/ladder_results_v5ext.csv in the same schema
as run_ladder.py so the manuscript tables can concatenate them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.run_ladder as RL  # noqa: E402
from src.models.ladder import (  # noqa: E402
    FEATURE_SETS, RV_FEATURES, VOL_TARGETS, Scaler, WindowDataset,
)

RL.VOL_MODE = True
RL.TARGETS = VOL_TARGETS
RL.HORIZON_LABELS = {i: t.replace("yv_", "") for i, t in enumerate(VOL_TARGETS)}
HL = RL.HORIZON_LABELS
LOOKBACK = 12
SEEDS = 5
CFG = {"lr": 1e-3, "epochs": 200, "patience": 20}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = RL.load_data()
splits = RL.make_splits(df)
ref = RL.make_ref(df)
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
# ML/DL baselines are HAR-anchored exactly like the model ladder, so the
# comparison isolates the learner rather than the presence of the anchor.
HAR = RL.har_forecast_matrix(df, splits["train"])
df_fit = df.copy()
for _hi, _t in enumerate(VOL_TARGETS):
    df_fit[_t] = y_all[:, _hi] - HAR[:, _hi]
y_res = df_fit[VOL_TARGETS].to_numpy(dtype=np.float64)
results = []


def record(model, hi, y_true, y_pred, rows, seed=0):
    m = RL.evaluate(y_true, y_pred, ref(hi)[rows])
    results.append({"model": model, "horizon": HL[hi], "seed": seed, **m})


# ---------------------------------------------------------------- ARIMA ----
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

tr, te = splits["train"], splits["test"]
for hi, tgt in enumerate(VOL_TARGETS):
    y = y_all[:, hi]
    best = (np.inf, None)
    for p in range(3):
        for q in range(3):
            if p == q == 0:
                continue
            try:
                fit = ARIMA(y[tr], order=(p, 0, q)).fit()
                if fit.aic < best[0]:
                    best = (fit.aic, (p, 0, q))
            except Exception:
                continue
    order = best[1] or (1, 0, 0)
    hn = int(HL[hi].replace("h", ""))
    # y_{t,h} is only observed h weeks after origin t, so at origin i the
    # information set ends at index i-h. Forecast h steps ahead from there.
    params = ARIMA(y[tr], order=order).fit().params
    preds = []
    for i in te:
        end = i - hn + 1                      # exclusive: last observable + 1
        f = ARIMA(y[:end], order=order).filter(params)
        preds.append(float(f.forecast(hn)[-1]))
    record(f"ARIMA{order}", hi, y[te], np.array(preds), te)
    print(f"ARIMA {HL[hi]} order={order} done", flush=True)

# -------------------------------------------------------------- XGBoost ----
import xgboost as xgb  # noqa: E402

feat_cols = FEATURE_SETS["M3"] + RV_FEATURES
X_raw = df[feat_cols].to_numpy(dtype=np.float64)
sc = Scaler().fit(X_raw[tr])
X = sc.transform(X_raw)


def flatten_window(idx):
    keep = idx[idx >= LOOKBACK - 1]
    mat = np.stack([X[i - LOOKBACK + 1: i + 1].ravel() for i in keep])
    return keep, mat


tr_k, Xtr = flatten_window(tr)
va_k, Xva = flatten_window(splits["val"])
te_k, Xte = flatten_window(te)
for hi in range(len(VOL_TARGETS)):
    model = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
        early_stopping_rounds=50, eval_metric="rmse", verbosity=0)
    model.fit(Xtr, y_res[tr_k, hi], eval_set=[(Xva, y_res[va_k, hi])],
              verbose=False)
    record("XGBoost", hi, y_all[te_k, hi],
           model.predict(Xte) + HAR[te_k, hi], te_k)
print("XGBoost done", flush=True)


# ------------------------------------------------------- LSTM / TCN swaps --
class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.1, n_horizons=5):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


class TCNForecaster(nn.Module):
    """Small dilated causal CNN (2 residual blocks, dilations 1 and 2)."""

    def __init__(self, n_features, ch=64, dropout=0.1, n_horizons=5):
        super().__init__()
        def block(cin, dil):
            return nn.Sequential(
                nn.Conv1d(cin, ch, kernel_size=3, padding=2 * dil,
                          dilation=dil),
                nn.ReLU(), nn.Dropout(dropout))
        self.b1, self.b2 = block(n_features, 1), block(ch, 2)
        self.proj = nn.Conv1d(n_features, ch, 1)
        self.head = nn.Linear(ch, n_horizons)

    def forward(self, x):
        z = x.transpose(1, 2)
        h = self.b1(z)[:, :, : z.size(2)] + self.proj(z)
        h = self.b2(h)[:, :, : z.size(2)] + h
        return self.head(h[:, :, -1])


groups = {"x": FEATURE_SETS["M1"] + RV_FEATURES}
feat_arrays, _, y_std, y_sc, aux_std, aux_sc = RL.prepare_arrays(
    df_fit, splits, groups)
loaders = {}
for sname, idx in splits.items():
    dset = WindowDataset(df, idx, LOOKBACK, feat_arrays, y_std, aux_std, {})
    loaders[sname] = DataLoader(dset, batch_size=64,
                                shuffle=(sname == "train"))

for name, cls in (("LSTM", LSTMForecaster), ("TCN", TCNForecaster)):
    for seed in range(SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = cls(len(groups["x"]), n_horizons=len(VOL_TARGETS)).to(device)
        model = RL.train_model(model, loaders, device, CFG, False)
        y_t, y_p, rows, _ = RL.predict(model, loaders["test"], device, False)
        y_true = y_sc.inverse(y_t) + HAR[rows]
        y_pred = y_sc.inverse(y_p) + HAR[rows]
        for hi in range(len(VOL_TARGETS)):
            record(name, hi, y_true[:, hi], y_pred[:, hi], rows, seed)
        print(f"{name} seed {seed} done", flush=True)

res = pd.DataFrame(results)
out = PROJECT_ROOT / "outputs" / "tables" / "ladder_results_v5ext.csv"
res.to_csv(out, index=False)
print("\n=== External baselines, test QLIKE (mean over seeds) ===")
print(res.pivot_table(index="model", columns="horizon",
                      values="qlike")[["h1", "h2", "h4", "h8", "h12"]]
      .round(4).to_string())
