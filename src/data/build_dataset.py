"""Build the release-aligned, leakage-free weekly forecasting dataset.

See data/metadata/release_alignment.md for the timing convention.

Row index: physical week-ending Friday date t (EIA WPSR week).
  - physical features:  known through week t (released Wed of week t+1)
  - price features:     known through Friday of week t+1 (the forecast origin)
  - targets:            cumulative log returns from origin Friday t+1 to t+1+h,
                        for h in {1, 2, 4} weeks (i.e., calendar weeks t+2, t+3, t+5)

Output: data/processed/dataset_weekly.csv (+ split column per design section 18)
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"

PHYSICAL = {
    "production": "production.csv",
    "imports": "imports.csv",
    "exports": "exports.csv",
    "refinery_inputs": "refinery_inputs.csv",
    "stocks_commercial": "stocks_commercial_ex_spr.csv",
    "stocks_spr": "stocks_spr.csv",
    "stocks_cushing": "stocks_cushing.csv",
    "utilization": "refinery_utilization.csv",
}

TRAIN_END = "2014-12-31"
VAL_END = "2019-12-31"
HORIZONS = (1, 2, 4, 8, 12)


def load_series(fname: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(INTERIM / fname, parse_dates=["date"])
    return df.rename(columns={"value": name})


def build() -> pd.DataFrame:
    # ---- weekly Friday grid ----------------------------------------------
    # EIA week-ending dates are Fridays except holiday-shifted weeks; snap all
    # dates to the Friday of their ISO week so series merge on a common grid.
    def to_friday_grid(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["date"] = out["date"] + pd.to_timedelta(
            4 - out["date"].dt.weekday, unit="D"
        )
        # holiday shifts can create duplicates within a week -> keep last
        out = out.drop_duplicates(subset="date", keep="last")
        return out

    phys = None
    for name, fname in PHYSICAL.items():
        s = to_friday_grid(load_series(fname, name))
        phys = s if phys is None else phys.merge(s, on="date", how="outer")
    phys = phys.sort_values("date").reset_index(drop=True)

    price = to_friday_grid(load_series("wti_spot.csv", "wti")).sort_values("date")

    # ---- physical derived features (all use info through week t only) ----
    phys["d_stocks_commercial"] = phys["stocks_commercial"].diff()
    phys["d_stocks_spr"] = phys["stocks_spr"].diff()
    phys["d_stocks_cushing"] = phys["stocks_cushing"].diff()
    phys["net_flow_no_adj"] = 7.0 * (
        phys["production"] + phys["imports"]
        - phys["refinery_inputs"] - phys["exports"]
    )
    # implied adjustment per balance_definition_decision.md (D2/D3), kb/d
    phys["adjustment"] = (
        (phys["d_stocks_commercial"] + phys["d_stocks_spr"]) / 7.0
        - phys["production"] - phys["imports"]
        + phys["refinery_inputs"] + phys["exports"]
    )
    phys["balance_residual"] = (
        phys["d_stocks_commercial"] + phys["d_stocks_spr"] - phys["net_flow_no_adj"]
    )  # equals 7*adjustment by construction; kept for M3 feature parity
    phys["days_supply"] = phys["stocks_commercial"] / phys["refinery_inputs"]

    # ---- price features (indexed by their own Friday date) ---------------
    pr = price.copy()
    pr["log_wti"] = np.log(pr["wti"])
    for k in (1, 2, 4):
        pr[f"ret_{k}w"] = pr["log_wti"].diff(k)
    for k in (4, 8, 12):
        pr[f"vol_{k}w"] = pr["ret_1w"].rolling(k).std()
    pr["ma12_dist"] = pr["log_wti"] - pr["log_wti"].rolling(12).mean()

    # ---- realized volatility from daily WTI (volatility-target pivot) ----
    daily = pd.read_csv(INTERIM / "wti_spot_daily.csv", parse_dates=["date"])
    daily = daily.sort_values("date")
    daily["r2"] = np.log(daily["value"]).diff() ** 2

    def rv_window(end_dates: pd.Series, weeks: int) -> np.ndarray:
        """sqrt of summed squared daily log returns over (end - 7*weeks, end].

        The window spans exactly 7*weeks calendar days ending on `end`
        (inclusive), i.e. start = end - 7*weeks + 1 day. Windows containing a
        non-positive price (2020-04-20) lose those daily returns, which are
        undefined in logs; such windows are flagged so they can be reported
        rather than silently understating volatility.
        """
        out = np.empty(len(end_dates))
        n_obs = np.empty(len(end_dates))
        d = daily.set_index("date")["r2"]
        for i, end in enumerate(end_dates):
            win = d.loc[end - pd.Timedelta(days=7 * weeks - 1): end]
            out[i] = np.sqrt(win.sum()) if win.notna().any() else np.nan
            n_obs[i] = win.notna().sum()
        rv_window.last_n_obs = n_obs
        return out

    # ---- release alignment ----------------------------------------------
    # Row t: physical week t; origin Friday = t + 7 days; price features from
    # origin Friday; targets from origin+7h days.
    df = phys.copy()
    df["origin_date"] = df["date"] + pd.Timedelta(days=7)

    price_idx = pr.set_index("date")

    def price_at(dates: pd.Series, col: str) -> np.ndarray:
        return price_idx[col].reindex(dates).to_numpy()

    price_cols = ["wti", "log_wti", "ret_1w", "ret_2w", "ret_4w",
                  "vol_4w", "vol_8w", "vol_12w", "ma12_dist"]
    for col in price_cols:
        df[f"px_{col}"] = price_at(df["origin_date"], col)

    for h in HORIZONS:
        target_date = df["origin_date"] + pd.Timedelta(days=7 * h)
        df[f"y_h{h}"] = (
            price_idx["log_wti"].reindex(target_date).to_numpy()
            - df["px_log_wti"].to_numpy()
        )
        df[f"y_h{h}_target_date"] = target_date

    # HAR-style RV features (known at origin) and log-RV targets over the
    # h-week window strictly after the origin Friday
    df["px_rv_1w"] = rv_window(df["origin_date"], 1)
    rv1_series = pd.Series(df["px_rv_1w"].to_numpy(), index=df["origin_date"])
    df["px_rv_4w"] = rv1_series.rolling(4).mean().to_numpy()
    df["px_rv_12w"] = rv1_series.rolling(12).mean().to_numpy()
    for h in HORIZONS:
        target_date = df["origin_date"] + pd.Timedelta(days=7 * h)
        df[f"yv_h{h}"] = np.log(rv_window(target_date, h))
    # zero-RV weeks (empty daily windows) produce -inf logs -> drop as NaN
    rv_cols = (["px_rv_1w", "px_rv_4w", "px_rv_12w"]
               + [f"yv_h{h}" for h in HORIZONS])
    df[rv_cols] = df[rv_cols].replace([np.inf, -np.inf], np.nan)
    df.loc[df["px_rv_1w"] <= 0, "px_rv_1w"] = np.nan

    # ---- next-week physical targets for auxiliary heads (M4/M5) ---------
    # v2 design: predict CHANGES for physical quantities (persistence makes
    # level prediction trivial); adjustment is predicted in level form.
    aux_vars = ["production", "imports", "exports", "refinery_inputs",
                "stocks_commercial", "adjustment"]
    phys_idx = phys.set_index("date")
    next_week = df["date"] + pd.Timedelta(days=7)
    for v in aux_vars:
        df[f"aux_{v}_next"] = phys_idx[v].reindex(next_week).to_numpy()
        if v != "adjustment":
            df[f"aux_d_{v}_next"] = df[f"aux_{v}_next"] - df[v]
    # Level columns at t are needed at loss time to reconstruct predicted levels.

    # ---- clean sample and split -----------------------------------------
    feature_cols = (
        list(PHYSICAL.keys())
        + ["d_stocks_commercial", "d_stocks_spr", "d_stocks_cushing",
           "net_flow_no_adj", "adjustment", "balance_residual", "days_supply"]
        + [f"px_{c}" for c in price_cols]
        + ["px_rv_1w", "px_rv_4w", "px_rv_12w"]
    )
    target_cols = ([f"y_h{h}" for h in HORIZONS]
                   + [f"yv_h{h}" for h in HORIZONS])
    aux_cols = ([f"aux_{v}_next" for v in aux_vars]
                + [f"aux_d_{v}_next" for v in aux_vars if v != "adjustment"])

    df = df.dropna(subset=feature_cols + target_cols + aux_cols).reset_index(drop=True)

    df["split"] = np.where(
        df["date"] <= TRAIN_END, "train",
        np.where(df["date"] <= VAL_END, "val", "test"),
    )

    keep = (["date", "origin_date", "split"] + feature_cols + aux_cols
            + target_cols + [f"y_h{h}_target_date" for h in HORIZONS])
    out = df[keep]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "dataset_weekly.csv", index=False)

    print(f"Dataset: {len(out)} rows, {out['date'].min().date()} -> {out['date'].max().date()}")
    print(out["split"].value_counts().to_string())
    print(f"Features: {len(feature_cols)} | aux targets: {len(aux_cols)} | horizons: {HORIZONS}")
    return out


if __name__ == "__main__":
    build()
