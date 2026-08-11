"""Petroleum balance audit (CLAUDE.md section 23).

Reconstructs the weekly crude-oil stock-flow balance from EIA series and
compares observed stock changes with reconstructed net flows, BEFORE any
machine learning is attempted.

Balance concept (flows in kb/d, stocks in kb):
    delta_stock_t ~= 7 * (production + imports + adjustment
                          - refinery_inputs - exports)

The official EIA adjustment is not in the navigator history files, so we
compute the IMPLIED adjustment as the residual and study its distribution.
We do this for two stock definitions:
  A) commercial stocks ex-SPR (WCESTUS1) -> residual absorbs SPR flows too
  B) total stocks incl SPR   (WCRSTUS1) -> residual closer to EIA official
     "crude oil adjustment" (unaccounted-for crude)

Outputs (experiments/01_data_audit/):
  balance_audit.csv        full audit table
  balance_audit_summary.md summary statistics and verdict
  fig_balance_residual.png residual time series + distribution
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM = PROJECT_ROOT / "data" / "interim"
OUT = Path(__file__).resolve().parent

SERIES_FILES = {
    "production": "production.csv",
    "imports": "imports.csv",
    "exports": "exports.csv",
    "refinery_inputs": "refinery_inputs.csv",
    "stocks_commercial": "stocks_commercial_ex_spr.csv",
    "stocks_total": "stocks_total_incl_spr.csv",
    "stocks_spr": "stocks_spr.csv",
    "stocks_cushing": "stocks_cushing.csv",
    "utilization": "refinery_utilization.csv",
    "wti_spot": "wti_spot.csv",
}


def load_merged() -> pd.DataFrame:
    df = None
    for name, fname in SERIES_FILES.items():
        s = pd.read_csv(INTERIM / fname, parse_dates=["date"])
        s = s.rename(columns={"value": name})
        df = s if df is None else df.merge(s, on="date", how="outer")
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    df = load_merged()

    # stock changes (kb per week)
    for col in ("stocks_commercial", "stocks_total", "stocks_spr"):
        df[f"d_{col}"] = df[col].diff()

    # net flow WITHOUT adjustment, converted to kb per week
    df["net_flow_no_adj"] = 7.0 * (
        df["production"] + df["imports"] - df["refinery_inputs"] - df["exports"]
    )

    # implied adjustment under each stock definition (kb/d)
    df["adj_implied_commercial"] = (df["d_stocks_commercial"] - df["net_flow_no_adj"]) / 7.0
    df["adj_implied_total"] = (df["d_stocks_total"] - df["net_flow_no_adj"]) / 7.0

    # balance residuals in kb/week (what a model's balance loss would see)
    df["resid_commercial"] = df["d_stocks_commercial"] - df["net_flow_no_adj"]
    df["resid_total"] = df["d_stocks_total"] - df["net_flow_no_adj"]

    audit_cols = [
        "date", "wti_spot", "production", "imports", "exports",
        "refinery_inputs", "utilization",
        "stocks_commercial", "stocks_total", "stocks_spr", "stocks_cushing",
        "d_stocks_commercial", "d_stocks_total", "d_stocks_spr",
        "net_flow_no_adj", "resid_commercial", "resid_total",
        "adj_implied_commercial", "adj_implied_total",
    ]
    audit = df[audit_cols]
    audit.to_csv(OUT / "balance_audit.csv", index=False)

    # analysis on the common complete-balance sample
    bal = audit.dropna(subset=["d_stocks_total", "net_flow_no_adj"]).copy()
    bal_start, bal_end = bal["date"].min().date(), bal["date"].max().date()

    def stats(s: pd.Series) -> dict:
        return {
            "mean": s.mean(), "std": s.std(), "median": s.median(),
            "p05": s.quantile(0.05), "p95": s.quantile(0.95),
            "mean_abs": s.abs().mean(),
        }

    st_tot = stats(bal["adj_implied_total"])
    st_com = stats(bal["adj_implied_commercial"])

    # scale reference: typical weekly stock change magnitude
    typ_dstock = bal["d_stocks_total"].abs().mean()
    corr = bal[["d_stocks_total", "net_flow_no_adj"]].corr().iloc[0, 1]
    corr_c = bal[["d_stocks_commercial", "net_flow_no_adj"]].corr().iloc[0, 1]

    # era comparison: weekly exports were monthly-based estimates before mid-2016
    pre = bal[bal["date"] < "2016-07-01"]["adj_implied_total"]
    post = bal[bal["date"] >= "2016-07-01"]["adj_implied_total"]

    lines = [
        "# Balance Audit Summary",
        "",
        f"Common balance sample: {bal_start} -> {bal_end} ({len(bal)} weeks)",
        "",
        "## Correlation: observed stock change vs reconstructed net flow (no adjustment)",
        f"- total stocks (incl SPR):      r = {corr:.4f}",
        f"- commercial stocks (ex-SPR):   r = {corr_c:.4f}",
        f"- typical |weekly stock change| (total): {typ_dstock:,.0f} kb/week",
        "",
        "## Implied adjustment (kb/d), TOTAL-stock definition",
        f"- mean {st_tot['mean']:.1f}, std {st_tot['std']:.1f}, median {st_tot['median']:.1f}",
        f"- p05 {st_tot['p05']:.1f}, p95 {st_tot['p95']:.1f}, mean|.| {st_tot['mean_abs']:.1f}",
        "",
        "## Implied adjustment (kb/d), COMMERCIAL-stock definition",
        f"- mean {st_com['mean']:.1f}, std {st_com['std']:.1f}, median {st_com['median']:.1f}",
        f"- p05 {st_com['p05']:.1f}, p95 {st_com['p95']:.1f}, mean|.| {st_com['mean_abs']:.1f}",
        "",
        "## Methodology-era comparison (total-stock implied adjustment, kb/d)",
        f"- pre 2016-07 (exports monthly-based):  mean {pre.mean():.1f}, std {pre.std():.1f}, n={len(pre)}",
        f"- post 2016-07 (exports weekly Census): mean {post.mean():.1f}, std {post.std():.1f}, n={len(post)}",
        "",
        "## Notes",
        "- Commercial-definition residual absorbs SPR flows; total-definition residual",
        "  corresponds to EIA's official 'crude oil adjustment' (unaccounted-for crude).",
        "- EIA states components are independently estimated, so a nonzero adjustment",
        "  is expected -> justifies SOFT balance constraint.",
    ]
    (OUT / "balance_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # figures
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    ax = axes[0]
    ax.plot(bal["date"], bal["d_stocks_total"], lw=0.6, label="observed Δstock (total)")
    ax.plot(bal["date"], bal["net_flow_no_adj"], lw=0.6, alpha=0.7,
            label="reconstructed net flow (no adj)")
    ax.set_title("Observed weekly stock change vs reconstructed net flow")
    ax.set_ylabel("kb/week")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(bal["date"], bal["adj_implied_total"], lw=0.6, color="tab:red",
            label="implied adjustment (total stocks)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(pd.Timestamp("2016-07-01"), color="gray", ls="--", lw=0.8,
               label="weekly exports methodology change (~2016-07)")
    ax.set_title("Implied crude-oil adjustment over time")
    ax.set_ylabel("kb/d")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.hist(bal["adj_implied_total"].dropna(), bins=80, color="tab:red", alpha=0.6,
            label="total-stock definition")
    ax.hist(bal["adj_implied_commercial"].dropna(), bins=80, color="tab:blue", alpha=0.4,
            label="commercial-stock definition")
    ax.set_title("Distribution of implied adjustment (kb/d)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_balance_residual.png", dpi=150)
    print(f"\nWrote {OUT / 'balance_audit.csv'}, summary, and figure.")


if __name__ == "__main__":
    main()
