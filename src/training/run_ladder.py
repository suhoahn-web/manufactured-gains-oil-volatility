"""Train and evaluate the model ladder M0-M5.

Usage:
  python run_ladder.py --models M0 M1 M2 M3            # baselines
  python run_ladder.py --models M4 M5 --lambda-b 0.1   # multi-task / constrained

Outputs (outputs/):
  tables/ladder_results.csv       one row per model x horizon x seed (+aggregates)
  logs/preds_<model>_seed<k>.csv  per-week test predictions
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ladder import (  # noqa: E402
    AUX_TARGETS, FEATURE_SETS, M45_PHYS, M45_PRICE, RV_FEATURES, TARGETS,
    VOL_TARGETS, DualBranchForecaster, GRUForecaster, Scaler, WindowDataset,
    balance_loss, balance_terms,
)

VOL_MODE = False  # set in main() when --target vol

OUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUT_LOGS = PROJECT_ROOT / "outputs" / "logs"
HORIZON_LABELS = {i: t.replace("y_", "") for i, t in enumerate(TARGETS)}


def load_data():
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_weekly.csv",
                     parse_dates=["date", "origin_date"])
    # d_SPR_{t+1} for the balance equation, from interim series (label-time qty)
    spr = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "stocks_spr.csv",
                      parse_dates=["date"])
    spr["date"] = spr["date"] + pd.to_timedelta(4 - spr["date"].dt.weekday, unit="D")
    spr = spr.drop_duplicates("date", keep="last").set_index("date")["value"]
    spr_next = spr.reindex(df["date"] + pd.Timedelta(days=7)).to_numpy()
    spr_now = spr.reindex(df["date"]).to_numpy()
    df["bal_d_spr_next"] = spr_next - spr_now
    df["bal_d_spr_next"] = df["bal_d_spr_next"].fillna(0.0)
    return df


def make_splits(df):
    return {s: np.where(df["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}


def evaluate(y_true, y_pred, ref):
    """Metrics for one horizon.

    Return mode: `ref` is the origin price level; reports return- and
    price-space errors and sign accuracy.
    Vol mode: targets are log realized volatility; `ref` is the log of the
    random-walk RV benchmark (backward window scaled by sqrt(h)); reports
    log-RV errors, QLIKE on RV levels, and direction-of-change accuracy
    relative to the RW reference.
    """
    err = y_pred - y_true
    out = {
        "ret_mae": float(np.mean(np.abs(err))),
        "ret_rmse": float(np.sqrt(np.mean(err ** 2))),
    }
    if VOL_MODE:
        out["dir_acc"] = float(np.mean(
            np.sign(y_pred - ref) == np.sign(y_true - ref)))
        ratio = np.exp(y_true - y_pred)  # RV_true / RV_pred (levels)
        out["qlike"] = float(np.mean(ratio - np.log(ratio) - 1.0))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        out["dir_acc"] = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        p_true = ref * np.exp(y_true)
        p_pred = ref * np.exp(y_pred)
        out["price_mae"] = float(np.mean(np.abs(p_pred - p_true)))
        out["price_rmse"] = float(np.sqrt(np.mean((p_pred - p_true) ** 2)))
        out["price_smape"] = float(np.mean(
            2 * np.abs(p_pred - p_true) / (np.abs(p_pred) + np.abs(p_true))))
    return out


def horizon_weeks(label: str) -> int:
    return int(label.replace("h", ""))


def har_design(df):
    """HAR-RV design matrix from origin-time realized-volatility components."""
    return np.column_stack([
        np.ones(len(df)), np.log(df["px_rv_1w"].to_numpy()),
        np.log(df["px_rv_4w"].to_numpy()), np.log(df["px_rv_12w"].to_numpy())])


def har_forecast_matrix(df, train_idx):
    """[N, H] HAR forecasts of log RV, coefficients fit on TRAIN rows only."""
    X = har_design(df)
    out = np.empty((len(df), len(TARGETS)))
    for hi, tgt in enumerate(TARGETS):
        y = df[tgt].to_numpy()
        beta, *_ = np.linalg.lstsq(X[train_idx], y[train_idx], rcond=None)
        out[:, hi] = X @ beta
    return out


def make_ref(df):
    """Per-horizon reference array factory for evaluate()."""
    if VOL_MODE:
        rv1 = df["px_rv_1w"].to_numpy()

        def ref(hi):
            return np.log(rv1 * np.sqrt(horizon_weeks(HORIZON_LABELS[hi])))
    else:
        px = df["px_wti"].to_numpy()

        def ref(hi):
            return px
    return ref


def run_naive(df, splits, results):
    tr, te = splits["train"], splits["test"]
    y = df[TARGETS].to_numpy()
    ref = make_ref(df)
    for hi, tgt in enumerate(TARGETS):
        r = ref(hi)
        if VOL_MODE:
            # M0a random-walk RV: forecast = backward window scaled sqrt(h)
            m = evaluate(y[te, hi], r[te], r[te])
            results.append({"model": "M0_rw", "horizon": HORIZON_LABELS[hi],
                            "seed": 0, **m})
            # M0b HAR-RV: OLS of log target RV on log(rv_1w/4w/12w), train fit
            X = np.column_stack([
                np.ones(len(df)), np.log(df["px_rv_1w"]),
                np.log(df["px_rv_4w"]), np.log(df["px_rv_12w"])])
            beta, *_ = np.linalg.lstsq(X[tr], y[tr, hi], rcond=None)
            m = evaluate(y[te, hi], X[te] @ beta, r[te])
            results.append({"model": "M0_har", "horizon": HORIZON_LABELS[hi],
                            "seed": 0, **m})
        else:
            m = evaluate(y[te, hi], np.zeros(len(te)), r[te])
            results.append({"model": "M0_rw", "horizon": HORIZON_LABELS[hi],
                            "seed": 0, **m})
            mu = y[tr, hi].mean()
            m = evaluate(y[te, hi], np.full(len(te), mu), r[te])
            results.append({"model": "M0_mean", "horizon": HORIZON_LABELS[hi],
                            "seed": 0, **m})


def prepare_arrays(df, splits, feature_groups):
    """Scale each feature group on train rows; scale targets and aux."""
    tr = splits["train"]
    feat_arrays, scalers = {}, {}
    for name, cols in feature_groups.items():
        x = df[cols].to_numpy(dtype=np.float64)
        sc = Scaler().fit(x[tr])
        feat_arrays[name] = sc.transform(x).astype(np.float32)
        scalers[name] = sc
    y = df[TARGETS].to_numpy(dtype=np.float64)
    y_sc = Scaler().fit(y[tr])
    aux = df[AUX_TARGETS].to_numpy(dtype=np.float64)
    aux_sc = Scaler().fit(aux[tr])
    return feat_arrays, scalers, y_sc.transform(y), y_sc, aux_sc.transform(aux), aux_sc


def train_model(model, loaders, device, cfg, is_dual, aux_sc=None,
                lambda_f=0.0, lambda_b=0.0, bal_scale=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    huber = nn.HuberLoss(delta=1.0)
    best_val, best_state, patience = np.inf, None, 0
    for epoch in range(cfg["epochs"]):
        for phase in ("train", "val"):
            model.train(phase == "train")
            total, n = 0.0, 0
            with torch.set_grad_enabled(phase == "train"):
                for batch in loaders[phase]:
                    if is_dual:
                        yhat, auxhat = model(batch["price"].to(device),
                                             batch["phys"].to(device))
                        loss = huber(yhat, batch["y"].to(device))
                        if lambda_f > 0:
                            loss = loss + lambda_f * huber(
                                auxhat, batch["aux"].to(device))
                        if lambda_b > 0:
                            loss = loss + lambda_b * balance_loss(
                                auxhat, aux_sc,
                                batch["flow_t"].to(device),
                                batch["d_spr_next"].to(device),
                                scale=bal_scale)
                    else:
                        yhat = model(batch["x"].to(device))
                        loss = huber(yhat, batch["y"].to(device))
                    if phase == "train":
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    total += loss.item() * len(batch["y"])
                    n += len(batch["y"])
            if phase == "val":
                val = total / n
                if val < best_val - 1e-5:
                    best_val, patience = val, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    patience += 1
        if patience >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model


def predict(model, loader, device, is_dual):
    model.eval()
    ys, yhats, rows, auxhats = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if is_dual:
                yhat, auxhat = model(batch["price"].to(device),
                                     batch["phys"].to(device))
                auxhats.append(auxhat.cpu().numpy())
            else:
                yhat = model(batch["x"].to(device))
            yhats.append(yhat.cpu().numpy())
            ys.append(batch["y"].numpy())
            rows.append(batch["row"].numpy())
    return (np.concatenate(ys), np.concatenate(yhats), np.concatenate(rows),
            np.concatenate(auxhats) if auxhats else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["M0", "M1", "M2", "M3"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--phys-hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lambda-f", type=float, default=0.3)
    ap.add_argument("--lambda-b", type=float, default=0.1)
    ap.add_argument("--target", choices=["ret", "vol"], default="ret")
    ap.add_argument("--har-residual", action="store_true",
                    help="vol mode: learn the deviation from a train-fitted "
                         "HAR forecast instead of the log-RV level")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    cfg = vars(args)

    global VOL_MODE, TARGETS, HORIZON_LABELS
    feature_sets = dict(FEATURE_SETS)
    m45_price = list(M45_PRICE)
    if args.target == "vol":
        VOL_MODE = True
        TARGETS = VOL_TARGETS
        HORIZON_LABELS = {i: t.replace("yv_", "") for i, t in enumerate(TARGETS)}
        feature_sets = {k: v + RV_FEATURES for k, v in feature_sets.items()}
        m45_price = m45_price + RV_FEATURES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    splits = make_splits(df)
    ref = make_ref(df)
    results = []

    # HAR-residual mode: the networks learn y - HAR(y); forecasts are
    # reconstructed by adding the HAR component back before evaluation.
    har_mat = None
    df_fit = df
    if VOL_MODE and args.har_residual:
        har_mat = har_forecast_matrix(df, splits["train"])
        df_fit = df.copy()
        for hi, tgt in enumerate(TARGETS):
            df_fit[tgt] = df[tgt].to_numpy() - har_mat[:, hi]

    if "M0" in args.models:
        run_naive(df, splits, results)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    for model_name in [m for m in args.models if m in ("M1", "M2", "M3", "M4", "M5")]:
        is_dual = model_name in ("M4", "M5")
        if is_dual:
            groups = {"price": m45_price, "phys": M45_PHYS}
        else:
            groups = {"x": feature_sets[model_name]}
        feat_arrays, _, y_std, y_sc, aux_std, aux_sc = prepare_arrays(
            df_fit, splits, groups)

        extra = {
            "flow_t": df[["production", "imports", "exports",
                          "refinery_inputs"]].to_numpy(dtype=np.float32),
            "d_spr_next": df["bal_d_spr_next"].to_numpy(dtype=np.float32),
        }
        bal_scale = float(np.std(np.diff(
            df["stocks_commercial"].to_numpy()[splits["train"]])))

        loaders = {}
        for split_name in ("train", "val", "test"):
            ds = WindowDataset(df, splits[split_name], args.lookback,
                               feat_arrays, y_std, aux_std, extra)
            loaders[split_name] = DataLoader(
                ds, batch_size=args.batch,
                shuffle=(split_name == "train"), drop_last=False)

        lam_b = args.lambda_b if model_name == "M5" else 0.0
        lam_f = args.lambda_f if is_dual else 0.0

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            if is_dual:
                model = DualBranchForecaster(
                    len(m45_price), len(M45_PHYS), hidden=args.hidden,
                    phys_hidden=args.phys_hidden, dropout=args.dropout,
                    n_horizons=len(TARGETS), n_aux=len(AUX_TARGETS)).to(device)
            else:
                model = GRUForecaster(
                    len(groups["x"]), hidden=args.hidden,
                    dropout=args.dropout, n_horizons=len(TARGETS)).to(device)

            model = train_model(model, loaders, device, cfg, is_dual,
                                aux_sc=aux_sc, lambda_f=lam_f,
                                lambda_b=lam_b, bal_scale=bal_scale)

            y_true_std, y_pred_std, rows, aux_pred = predict(
                model, loaders["test"], device, is_dual)
            y_true = y_sc.inverse(y_true_std)
            y_pred = y_sc.inverse(y_pred_std)

            # validation metrics (for lambda/hyperparameter selection only)
            v_true_std, v_pred_std, v_rows, _ = predict(
                model, loaders["val"], device, is_dual)
            v_true = y_sc.inverse(v_true_std)
            v_pred = y_sc.inverse(v_pred_std)

            if har_mat is not None:  # back to log-RV levels
                y_true = y_true + har_mat[rows]
                y_pred = y_pred + har_mat[rows]
                v_true = v_true + har_mat[v_rows]
                v_pred = v_pred + har_mat[v_rows]
            val_metrics = {}
            for hi in range(len(TARGETS)):
                vm = evaluate(v_true[:, hi], v_pred[:, hi], ref(hi)[v_rows])
                val_metrics[HORIZON_LABELS[hi]] = {
                    f"val_{k}": v for k, v in vm.items()}

            tag = f"{model_name}{args.tag}"
            nh = len(TARGETS)
            pred_df = pd.DataFrame({
                "date": df["date"].to_numpy()[rows],
                **{f"y_{HORIZON_LABELS[h]}": y_true[:, h] for h in range(nh)},
                **{f"yhat_{HORIZON_LABELS[h]}": y_pred[:, h] for h in range(nh)},
            })

            row_metrics = {}
            if is_dual and aux_pred is not None:
                d_stock, net_acc = balance_terms(
                    torch.as_tensor(aux_pred), aux_sc,
                    torch.as_tensor(extra["flow_t"][rows]),
                    torch.as_tensor(extra["d_spr_next"][rows]))
                bce = float(torch.mean(torch.abs(d_stock - net_acc)))
                row_metrics["bce_kb_week"] = bce
                pred_df["bal_d_stock_hat"] = d_stock.numpy()
                pred_df["bal_net_acc_hat"] = net_acc.numpy()

            pred_df.to_csv(OUT_LOGS / f"preds_{tag}_seed{seed}.csv", index=False)

            for hi in range(len(TARGETS)):
                m = evaluate(y_true[:, hi], y_pred[:, hi], ref(hi)[rows])
                results.append({"model": tag, "horizon": HORIZON_LABELS[hi],
                                "seed": seed, **m,
                                **val_metrics[HORIZON_LABELS[hi]],
                                **row_metrics})
            print(f"{tag} seed {seed} done"
                  + (f" (BCE {row_metrics.get('bce_kb_week', float('nan')):.0f} kb/wk)"
                     if row_metrics else ""))

    res = pd.DataFrame(results)
    out_path = OUT_TABLES / f"ladder_results{args.tag}.csv"
    if out_path.exists():
        old = pd.read_csv(out_path)
        res = pd.concat([old[~old["model"].isin(res["model"].unique())], res])
    res.to_csv(out_path, index=False)

    metric_cols = [c for c in ("ret_mae", "ret_rmse", "dir_acc", "qlike",
                               "r2", "price_mae", "price_rmse")
                   if c in res.columns]
    agg = (res.groupby(["model", "horizon"])[metric_cols]
           .agg(["mean", "std"]).round(4))
    print("\n=== Test-set results (mean over seeds) ===")
    print(agg.to_string())


if __name__ == "__main__":
    main()
