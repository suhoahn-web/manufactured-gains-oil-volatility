"""Lambda selection (validation-only) + inventory stress test + DM tests.

Steps:
1. Aggregate all outputs/tables/ladder_results*.csv; report validation metrics
   per lambda_B and pick lambda* by validation mean return-MAE (avg horizons).
2. Inventory stress split (design section 16): thresholds from TRAIN
   d_stocks_commercial quantiles (10/90 primary; 5/95 and 15/85 robustness).
3. Evaluate M1/M2/M3/M4/M5(lambda*) on test subsets: normal / extreme build /
   extreme draw, using seed-ensemble mean predictions.
4. Diebold-Mariano tests (Harvey small-sample correction, HAC variance with
   lag h-1) for M5 vs M4 and M5 vs M1 per horizon on the full test set.

Outputs: experiments/05_inventory_stress/stress_results.md (+ csv tables)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES = PROJECT_ROOT / "outputs" / "tables"
LOGS = PROJECT_ROOT / "outputs" / "logs"
OUT = Path(__file__).resolve().parent
HORIZONS = {"h1": 1, "h2": 2, "h4": 4}


def load_results() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in TABLES.glob("ladder_results*.csv")]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["model", "horizon", "seed"], keep="last")


def pick_lambda(res: pd.DataFrame) -> tuple:
    sweep = res[res["model"].str.startswith("M5_lb")]
    if "val_ret_mae" not in sweep.columns or sweep["val_ret_mae"].isna().all():
        return None, None
    agg = (sweep.groupby("model")["val_ret_mae"].mean().sort_values())
    best = agg.index[0]
    lam = best.replace("M5_lb", "").replace("p", ".")
    return best, (agg, lam)


def ensemble_preds(tag: str) -> pd.DataFrame:
    """Seed-ensemble mean of predictions for one model tag."""
    files = sorted(LOGS.glob(f"preds_{tag}_seed*.csv"))
    if not files:
        raise FileNotFoundError(tag)
    dfs = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    base = dfs[0][["date"] + [c for c in dfs[0].columns if c.startswith("y_")]].copy()
    for h in HORIZONS:
        base[f"yhat_{h}"] = np.mean([d[f"yhat_{h}"].to_numpy() for d in dfs], axis=0)
    return base


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int) -> tuple:
    """DM test on squared-error loss differential, HAC lag h-1, Harvey corr."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = gamma[0] / n
    dm = dbar / np.sqrt(var_d)
    harvey = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_h = dm * harvey
    p = 2 * (1 - stats.t.cdf(abs(dm_h), df=n - 1))
    return dm_h, p


def subset_metrics(pred: pd.DataFrame, mask: pd.Series, origin_price: pd.Series):
    out = {}
    for h in HORIZONS:
        yt = pred.loc[mask, f"y_{h}"].to_numpy()
        yp = pred.loc[mask, f"yhat_{h}"].to_numpy()
        po = origin_price[mask].to_numpy()
        err = yp - yt
        p_t, p_p = po * np.exp(yt), po * np.exp(yp)
        out[h] = {
            "n": int(mask.sum()),
            "ret_mae": np.mean(np.abs(err)),
            "price_mae": np.mean(np.abs(p_p - p_t)),
            "dir_acc": np.mean(np.sign(yp) == np.sign(yt)),
        }
    return out


def main() -> None:
    res = load_results()
    ds = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_weekly.csv",
                     parse_dates=["date"])

    lines = ["# Lambda selection + Inventory Stress Test", ""]

    # ---- 1. lambda selection on validation --------------------------------
    best_tag, info = pick_lambda(res)
    if best_tag is None:
        print("No validation metrics found in sweep results; aborting.")
        sys.exit(1)
    agg, lam_star = info
    lines += ["## Validation mean return-MAE by lambda_B (lower = better)", "",
              "```", agg.to_string(), "```", "",
              f"**Selected lambda_B* = {lam_star}** (validation only; test untouched)", ""]

    # ---- 2. stress thresholds from TRAIN only ----------------------------
    tr = ds[ds.split == "train"]
    te = ds[ds.split == "test"].set_index("date")
    d_stock_next = ds.set_index("date")["aux_stocks_commercial_next"] - \
        ds.set_index("date")["stocks_commercial"]
    # stress defined on the REALIZED stock change of the target week (t+1),
    # thresholds from train distribution of weekly changes
    thr = {}
    for lo, hi, name in ((0.10, 0.90, "q10_90"), (0.05, 0.95, "q05_95"),
                         (0.15, 0.85, "q15_85")):
        d_tr = tr["d_stocks_commercial"]
        thr[name] = (d_tr.quantile(lo), d_tr.quantile(hi))
    lines += ["## Stress thresholds (train quantiles of weekly commercial stock change, kb)",
              "", "```"]
    for k, (lo, hi) in thr.items():
        lines.append(f"{k}: draw < {lo:,.0f}, build > {hi:,.0f}")
    lines += ["```", ""]

    # ---- 3. subset evaluation --------------------------------------------
    models = ["M1", "M2", "M3", "M4", best_tag]
    lo, hi = thr["q10_90"]
    rows = []
    for tag in models:
        pred = ensemble_preds(tag)
        pred = pred.merge(ds[["date", "px_wti"]], on="date", how="left")
        pred = pred.merge(
            d_stock_next.rename("d_stock_target").reset_index(), on="date", how="left")
        build = pred["d_stock_target"] > hi
        draw = pred["d_stock_target"] < lo
        normal = ~(build | draw)
        for name, mask in (("normal", normal), ("build", build), ("draw", draw),
                           ("all", pd.Series(True, index=pred.index))):
            m = subset_metrics(pred, mask, pred["px_wti"])
            for h in HORIZONS:
                rows.append({"model": tag, "period": name, "horizon": h, **m[h]})
    stress = pd.DataFrame(rows)
    stress.to_csv(OUT / "stress_results.csv", index=False)

    for h in HORIZONS:
        piv = stress[stress.horizon == h].pivot_table(
            index="model", columns="period", values="ret_mae")
        piv = piv[["all", "normal", "build", "draw"]].round(4)
        lines += [f"## Return MAE by period, horizon {h}", "", "```",
                  piv.to_string(), "```", ""]
        pivd = stress[stress.horizon == h].pivot_table(
            index="model", columns="period", values="dir_acc")
        pivd = pivd[["all", "normal", "build", "draw"]].round(4)
        lines += [f"### Directional accuracy by period, horizon {h}", "", "```",
                  pivd.to_string(), "```", ""]

    # ---- 4. DM tests ------------------------------------------------------
    lines += ["## Diebold-Mariano tests (squared-error loss, seed-ensemble preds)",
              "", "```"]
    pm4 = ensemble_preds("M4")
    pm1 = ensemble_preds("M1")
    pm5 = ensemble_preds(best_tag)
    for h, hn in HORIZONS.items():
        e5 = pm5[f"yhat_{h}"].to_numpy() - pm5[f"y_{h}"].to_numpy()
        e4 = pm4[f"yhat_{h}"].to_numpy() - pm4[f"y_{h}"].to_numpy()
        e1 = pm1[f"yhat_{h}"].to_numpy() - pm1[f"y_{h}"].to_numpy()
        dm54, p54 = dm_test(e5, e4, hn)
        dm51, p51 = dm_test(e5, e1, hn)
        lines.append(f"{h}: M5 vs M4  DM={dm54:+.3f} p={p54:.3f} "
                     f"(negative = M5 better) | M5 vs M1  DM={dm51:+.3f} p={p51:.3f}")
    lines += ["```", ""]

    (OUT / "stress_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
