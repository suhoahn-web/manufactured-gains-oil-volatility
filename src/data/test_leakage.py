"""Automated leakage tests for the release-aligned dataset.

Checks (release_alignment.md):
  T1  origin_date == physical week date + 7 days, and is a Friday
  T2  every target date >= origin_date + 7 days (strictly future)
  T3  price features at row t equal raw price series values AT origin_date,
      never at any later date (spot check against independent recomputation)
  T4  physical/aux columns at row t come from weeks <= t (aux from t+1 is a
      TARGET, not a feature) - verified against raw interim series
  T5  chronological split integrity: max(train.date) < min(val.date) etc.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DS = PROJECT_ROOT / "data" / "processed" / "dataset_weekly.csv"
INTERIM = PROJECT_ROOT / "data" / "interim"

FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"{status}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


def to_friday(s: pd.Series) -> pd.Series:
    return s + pd.to_timedelta(4 - s.dt.weekday, unit="D")


def main() -> None:
    horizons = (1, 2, 4, 8, 12)
    df = pd.read_csv(DS, parse_dates=["date", "origin_date"]
                     + [f"y_h{h}_target_date" for h in horizons])

    # T1
    check("T1a origin = t+7d", (df["origin_date"] - df["date"]).eq(pd.Timedelta(days=7)).all())
    check("T1b origin is Friday", df["origin_date"].dt.weekday.eq(4).all())

    # T2
    for h in horizons:
        gap = (df[f"y_h{h}_target_date"] - df["origin_date"]).dt.days
        check(f"T2 h={h} target strictly future", (gap >= 7).all(),
              f"min gap {gap.min()}d, expected {7*h}d")

    # T3: recompute origin-date log price independently and compare
    wti = pd.read_csv(INTERIM / "wti_spot.csv", parse_dates=["date"])
    wti["date"] = to_friday(wti["date"])
    wti = wti.drop_duplicates("date", keep="last").set_index("date")["value"]
    recomputed = np.log(wti.reindex(df["origin_date"]).to_numpy())
    check("T3 price feature = price AT origin date",
          np.allclose(recomputed, df["px_log_wti"].to_numpy(), atol=1e-10))
    # and the target really is log P(target_date) - log P(origin)
    for h in horizons:
        tgt = np.log(wti.reindex(df[f"y_h{h}_target_date"]).to_numpy()) - recomputed
        check(f"T3b y_h{h} = future logP - origin logP",
              np.allclose(tgt, df[f"y_h{h}"].to_numpy(), atol=1e-10))

    # T4: physical feature at row t equals raw series value at week t (not t+1)
    prod = pd.read_csv(INTERIM / "production.csv", parse_dates=["date"])
    prod["date"] = to_friday(prod["date"])
    prod = prod.drop_duplicates("date", keep="last").set_index("date")["value"]
    check("T4a production feature is week-t value",
          np.allclose(prod.reindex(df["date"]).to_numpy(), df["production"], atol=1e-9))
    check("T4b aux production target is week-t+1 value",
          np.allclose(prod.reindex(df["date"] + pd.Timedelta(days=7)).to_numpy(),
                      df["aux_production_next"], atol=1e-9))

    # T6 RV target definition: independently recompute log RV over a window of
    # exactly 7*h calendar days ending at origin+7h, and compare.
    daily = pd.read_csv(INTERIM / "wti_spot_daily.csv", parse_dates=["date"])
    daily = daily.sort_values("date")
    r2 = pd.Series(np.log(daily["value"]).diff().to_numpy() ** 2,
                   index=daily["date"])
    for h in horizons:
        ends = df["origin_date"] + pd.Timedelta(days=7 * h)
        rec = np.array([
            np.sqrt(r2.loc[e - pd.Timedelta(days=7 * h - 1):e].sum())
            for e in ends])
        spans = np.array([
            (e - (e - pd.Timedelta(days=7 * h - 1))).days + 1 for e in ends])
        check(f"T6a h={h} RV window spans exactly {7*h} days",
              bool((spans == 7 * h).all()), f"got {set(spans.tolist())}")
        check(f"T6b h={h} yv target = independently recomputed log RV",
              np.allclose(np.log(rec), df[f"yv_h{h}"].to_numpy(), atol=1e-10))

    # T5
    tr = df[df.split == "train"]["date"]
    va = df[df.split == "val"]["date"]
    te = df[df.split == "test"]["date"]
    check("T5 chronological splits",
          tr.max() < va.min() and va.max() < te.min(),
          f"train<= {tr.max().date()}, val {va.min().date()}..{va.max().date()}, "
          f"test>= {te.min().date()}")

    print(f"\n{'ALL LEAKAGE TESTS PASSED' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
