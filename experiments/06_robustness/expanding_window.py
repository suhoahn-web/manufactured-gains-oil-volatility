"""Expanding-window robustness for the volatility ladder (design section 18).

Five chronological windows; per window: fit scalers/models on train, select
nothing (fixed hyperparams from the main protocol: lambda_F=0.1, lookback 12,
hidden 64), evaluate on that window's test years. Pooling the non-overlapping
test weeks across windows gives ~2x the test sample of the single split.

Models per window: HAR (OLS), M1, M4, M5 lambda_B=0.1 (per-protocol),
M5 lambda_B=0.5 (sensitivity). 3 seeds per NN config (ensemble-averaged).

Output: expanding_window_results.md + pooled per-week QLIKE losses CSV.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.run_ladder as RL  # noqa: E402
from src.models.ladder import (  # noqa: E402
    AUX_TARGETS, FEATURE_SETS, M45_PHYS, M45_PRICE, RV_FEATURES, VOL_TARGETS,
    DualBranchForecaster, GRUForecaster, Scaler, WindowDataset,
)

# configure run_ladder helpers for vol mode
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = RL.load_data()
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
dates = df["date"]

pooled = {}  # model -> list of per-week loss DataFrames


def qlike(y_true_log, y_pred_log):
    ratio = np.exp(y_true_log - y_pred_log)
    return ratio - np.log(ratio) - 1.0


extra_metrics = {}


def add_extra(name, date_arr, y_true, y_pred, rows):
    """Per-week absolute/squared log-RV errors and RV-change hit indicator."""
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


def add_pooled(name, date_arr, losses_by_h):
    rec = pd.DataFrame({"date": date_arr,
                        **{h: losses_by_h[h] for h in HORIZONS}})
    pooled.setdefault(name, []).append(rec)


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

    # HAR per window (also the anchor for HAR-residual learning)
    X = np.column_stack([np.ones(len(df)), np.log(df["px_rv_1w"]),
                         np.log(df["px_rv_4w"]), np.log(df["px_rv_12w"])])
    har_mat = np.empty((len(df), len(HORIZONS)))
    losses = {}
    for hi, h in enumerate(HORIZONS):
        beta, *_ = np.linalg.lstsq(X[tr], y_all[tr, hi], rcond=None)
        har_mat[:, hi] = X @ beta
        losses[h] = qlike(y_all[te, hi], har_mat[te, hi])
    add_pooled("HAR", dates.iloc[te].to_numpy(), losses)
    add_extra("HAR", dates.iloc[te].to_numpy(), y_all[te], har_mat[te], te)

    # networks learn the deviation from this window's HAR forecast
    df_fit = df.copy()
    for hi, tgt in enumerate(VOL_TARGETS):
        df_fit[tgt] = y_all[:, hi] - har_mat[:, hi]

    # ("M4arch", ...) is the architecture control: identical dual-branch
    # network, lambda_F = 0, i.e. no auxiliary physical-state task. It
    # separates the effect of the auxiliary task from that of capacity.
    configs = [
        ("M1", False, 0.0, 0.0),
        ("M2", False, 0.0, 0.0),
        ("M3", False, 0.0, 0.0),
        ("M4arch", True, 0.0, 0.0),
        ("M4", True, 0.0, LAMBDA_F),
        ("M5_lb0p1", True, 0.1, LAMBDA_F),
        ("M5_lb0p5", True, 0.5, LAMBDA_F),
    ]
    for name, is_dual, lam_b, lam_f in configs:
        if is_dual:
            groups = {"price": M45_PRICE + RV_FEATURES, "phys": M45_PHYS}
        else:
            groups = {"x": FEATURE_SETS[name] + RV_FEATURES}
        feat_arrays, _, y_std, y_sc, aux_std, aux_sc = RL.prepare_arrays(
            df_fit, splits, groups)
        extra = {
            "flow_t": df[["production", "imports", "exports",
                          "refinery_inputs"]].to_numpy(dtype=np.float32),
            "d_spr_next": df["bal_d_spr_next"].to_numpy(dtype=np.float32),
        }
        bal_scale = float(np.std(np.diff(
            df["stocks_commercial"].to_numpy()[tr])))
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
                    len(groups["price"]), len(M45_PHYS),
                    hidden=64, phys_hidden=32, dropout=0.1,
                    n_horizons=len(VOL_TARGETS),
                    n_aux=len(AUX_TARGETS)).to(device)
            else:
                model = GRUForecaster(len(groups["x"]), hidden=64,
                                      dropout=0.1,
                                      n_horizons=len(VOL_TARGETS)).to(device)
            model = RL.train_model(model, loaders, device, CFG, is_dual,
                                   aux_sc=aux_sc, lambda_f=lam_f,
                                   lambda_b=lam_b, bal_scale=bal_scale)
            y_t_std, y_p_std, rows, _ = RL.predict(model, loaders["test"],
                                                   device, is_dual)
            preds.append((rows, y_sc.inverse(y_p_std)))
        rows = preds[0][0]
        # residual forecasts -> log-RV levels by adding the HAR anchor back
        y_pred = np.mean([p[1] for p in preds], axis=0) + har_mat[rows]
        y_true = y_all[rows]
        losses = {h: qlike(y_true[:, hi], y_pred[:, hi])
                  for hi, h in enumerate(HORIZONS)}
        add_pooled(name, dates.iloc[rows].to_numpy(), losses)
        add_extra(name, dates.iloc[rows].to_numpy(), y_true, y_pred, rows)
        print(f"  {name}: mean QLIKE "
              + " ".join(f"{h}={losses[h].mean():.4f}" for h in HORIZONS),
              flush=True)

# ---- pooled evaluation ----------------------------------------------------
lines = ["# Expanding-Window Robustness (pooled QLIKE, vol ladder)", ""]
tables = {}
for name, recs in pooled.items():
    tables[name] = pd.concat(recs).sort_values("date").reset_index(drop=True)
    tables[name].to_csv(Path(__file__).resolve().parent
                        / f"ew_losses_{name}.csv", index=False)
    if name in extra_metrics:
        (pd.concat(extra_metrics[name]).sort_values("date")
         .reset_index(drop=True)
         .to_csv(Path(__file__).resolve().parent / f"ew_extra_{name}.csv",
                 index=False))

lines += [f"Pooled test weeks: {len(tables['M1'])}", "", "## Mean QLIKE", "", "```"]
rows_tbl = [{"model": n, **{h: t[h].mean() for h in HORIZONS}}
            for n, t in tables.items()]
lines += [pd.DataFrame(rows_tbl).set_index("model").round(4).to_string(),
          "```", ""]

PAIRS = [("M2", "M1"), ("M3", "M1"), ("M4", "M2"), ("M4", "M3"),
         ("M4", "M1"), ("M4", "HAR"), ("M5_lb0p1", "M4"), ("M5_lb0p1", "HAR"),
         ("M5_lb0p1", "M1"), ("M5_lb0p5", "M4"), ("M5_lb0p5", "HAR")]
lines += ["## DM tests on pooled QLIKE differentials (negative = first better)",
          "", "```"]
for h, hn in HORIZONS.items():
    parts = []
    for a, b in PAIRS:
        merged = tables[a][["date", h]].merge(
            tables[b][["date", h]], on="date", suffixes=("_a", "_b"))
        d = merged[f"{h}_a"].to_numpy() - merged[f"{h}_b"].to_numpy()
        n = len(d); dbar = d.mean()
        gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n
                 for k in range(hn)]
        var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
        if var_d <= 0:
            var_d = gamma[0] / n
        dm = dbar / np.sqrt(var_d)
        harvey = np.sqrt((n + 1 - 2 * hn + hn * (hn - 1) / n) / n)
        dmh = dm * harvey
        pv = 2 * (1 - stats.t.cdf(abs(dmh), df=n - 1))
        parts.append(f"{a}vs{b} {dmh:+.2f}(p={pv:.3f})")
    lines.append(f"{h:>4}: " + " | ".join(parts))
lines += ["```", ""]

out = Path(__file__).resolve().parent / "expanding_window_results.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
