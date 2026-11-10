"""Select hyperparameters inside each window instead of once, globally.

The pooled evaluation shares one set of hyperparameters, chosen on the 2015-2019
validation span, and 156 of the 487 pooled weeks fall inside that span. Removing
those weeks afterwards does not undo the fact that they informed the choice.
This design removes the reuse instead of bounding it: within each expanding
window the grid is fitted on that window's training rows, scored on that
window's validation rows, and only the winner is carried to that window's test
block. No later window and no test observation touches the choice.

Grid, kept small so the run is affordable:

    single-encoder rungs   hidden in {32, 64}
    dual-branch rung       hidden in {32, 64} x lambda_B in {0, 0.1, 0.5}

Selection uses one seed; the selected configuration is then re-estimated with
the usual five and ensembled, matching the main protocol.
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
    AUX_TARGETS, FEATURE_SETS, M45_PHYS, M45_PRICE, PHYS_DERIVED,
    PRICE_FEATURES, RV_FEATURES, VOL_TARGETS,
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
SEEDS, LOOKBACK, LAMBDA_F = 5, 12, 0.1
HERE = Path(__file__).resolve().parent

SPECS = {
    "M2_nested": dict(dual=False, feats=FEATURE_SETS["M2"] + RV_FEATURES,
                      grid=[dict(hidden=h) for h in (32, 64)]),
    "M3derived_nested": dict(dual=False,
                             feats=PRICE_FEATURES + PHYS_DERIVED + RV_FEATURES,
                             grid=[dict(hidden=h) for h in (32, 64)]),
    "M5_nested": dict(dual=True, feats=None,
                      grid=[dict(hidden=h, lam_b=b)
                            for h in (32, 64) for b in (0.0, 0.1, 0.5)]),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = RL.load_data()
y_all = df[VOL_TARGETS].to_numpy(dtype=np.float64)
dates = df["date"]
pooled, extra, chosen = {}, {}, []


def qlike(a, b):
    r = np.exp(a - b)
    return r - np.log(r) - 1.0


def build(spec, cfg, groups, feat_arrays, y_std, aux_std, extra_arrays, splits):
    loaders = {s: DataLoader(WindowDataset(df, i, LOOKBACK, feat_arrays, y_std,
                                           aux_std, extra_arrays),
                             batch_size=64, shuffle=(s == "train"))
               for s, i in splits.items()}
    return loaders


for wi, (tr_end, va_end, te_end) in enumerate(WINDOWS):
    tr = np.where(dates <= tr_end)[0]
    va = np.where((dates > tr_end) & (dates <= va_end))[0]
    te = np.where((dates > va_end) & (dates <= te_end))[0]
    if len(te) == 0:
        continue
    splits = {"train": tr, "val": va, "test": te}
    print(f"\n[window {wi}] train {len(tr)} val {len(va)} test {len(te)}",
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

    extra_arrays = {
        "flow_t": df[["production", "imports", "exports",
                      "refinery_inputs"]].to_numpy(dtype=np.float32),
        "d_spr_next": df["bal_d_spr_next"].to_numpy(dtype=np.float32)}
    bal_scale = float(np.std(np.diff(df["stocks_commercial"].to_numpy()[tr])))

    for name, spec in SPECS.items():
        groups = ({"price": M45_PRICE + RV_FEATURES, "phys": M45_PHYS}
                  if spec["dual"] else {"x": spec["feats"]})
        feat_arrays, _, y_std, y_sc, aux_std, aux_sc = RL.prepare_arrays(
            df_fit, splits, groups)
        loaders = build(name, CFG, groups, feat_arrays, y_std, aux_std,
                        extra_arrays, splits)

        def make(hidden):
            if spec["dual"]:
                return DualBranchForecaster(
                    len(groups["price"]), len(M45_PHYS), hidden=hidden,
                    phys_hidden=max(16, hidden // 2), dropout=0.1,
                    n_horizons=len(VOL_TARGETS), n_aux=len(AUX_TARGETS)).to(device)
            return GRUForecaster(len(groups["x"]), hidden=hidden, dropout=0.1,
                                 n_horizons=len(VOL_TARGETS)).to(device)

        # --- selection: one seed, scored on this window's validation rows only
        best, best_cfg = np.inf, None
        for g in spec["grid"]:
            torch.manual_seed(0)
            np.random.seed(0)
            model = make(g["hidden"])
            model = RL.train_model(model, loaders, device, CFG, spec["dual"],
                                   aux_sc=aux_sc,
                                   lambda_f=LAMBDA_F if spec["dual"] else 0.0,
                                   lambda_b=g.get("lam_b", 0.0),
                                   bal_scale=bal_scale)
            y_t, y_p, rows, _ = RL.predict(model, loaders["val"], device,
                                           spec["dual"])
            v = float(np.mean((y_sc.inverse(y_p) - y_sc.inverse(y_t)) ** 2))
            if v < best:
                best, best_cfg = v, g
        chosen.append({"window": wi, "spec": name, **best_cfg,
                       "val_mse": best})
        print(f"  {name}: selected {best_cfg} (val {best:.5f})", flush=True)

        # --- estimation: the selected configuration, five seeds, ensembled
        preds = []
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = make(best_cfg["hidden"])
            model = RL.train_model(model, loaders, device, CFG, spec["dual"],
                                   aux_sc=aux_sc,
                                   lambda_f=LAMBDA_F if spec["dual"] else 0.0,
                                   lambda_b=best_cfg.get("lam_b", 0.0),
                                   bal_scale=bal_scale)
            _, y_p, rows, _ = RL.predict(model, loaders["test"], device,
                                         spec["dual"])
            preds.append((rows, y_sc.inverse(y_p)))

        rows = preds[0][0]
        y_pred = np.mean([p[1] for p in preds], axis=0) + har[rows]
        y_true = y_all[rows]
        d = dates.iloc[rows].to_numpy()
        losses = {h: qlike(y_true[:, hi], y_pred[:, hi])
                  for hi, h in enumerate(HORIZONS)}
        pooled.setdefault(name, []).append(pd.DataFrame({"date": d, **losses}))
        rv1 = np.log(df["px_rv_1w"].to_numpy()[rows])
        rec = {"date": d}
        for hi, h in enumerate(HORIZONS):
            e = y_pred[:, hi] - y_true[:, hi]
            ref = rv1 + 0.5 * np.log(HORIZONS[h])
            rec[f"ae_{h}"] = np.abs(e)
            rec[f"se_{h}"] = e ** 2
            rec[f"hit_{h}"] = (np.sign(y_pred[:, hi] - ref)
                               == np.sign(y_true[:, hi] - ref)).astype(float)
        extra.setdefault(name, []).append(pd.DataFrame(rec))
        print(f"    QLIKE " + " ".join(f"{h}={losses[h].mean():.4f}"
                                       for h in HORIZONS), flush=True)

for k, recs in pooled.items():
    (pd.concat(recs).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_losses_{k}.csv", index=False))
    (pd.concat(extra[k]).sort_values("date").reset_index(drop=True)
     .to_csv(HERE / f"ew_extra_{k}.csv", index=False))
pd.DataFrame(chosen).to_csv(HERE / "nested_selection.csv", index=False)
print("\nwrote " + ", ".join(pooled) + " and nested_selection.csv")
