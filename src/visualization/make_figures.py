"""Manuscript figures F1-F6 (journal style, Okabe-Ito CVD-safe palette).

Fixed entity->color mapping across ALL figures:
  HAR gray #666666 | M1 blue #0072B2 | M4 orange #E69F00
  M5(lambda=0.1) green #009E73 | M5(lambda=0.5) vermillion #D55E00
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG = PROJECT_ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C = {"HAR": "#666666", "M1": "#0072B2", "M4": "#E69F00",
     "M5a": "#009E73", "M5b": "#D55E00", "ink": "#222222"}
# secondary encoding so series are never distinguished by colour alone
STYLE = {"HAR": ("-", "o"), "M1": ("--", "s"), "M4": ("-.", "^"),
         "M5a": (":", "D"), "M5b": ((0, (3, 1, 1, 1)), "v")}

# Elsevier: double-column width 190 mm; all type >= 7 pt at final size;
# vector output (PDF) plus a PNG preview.
COL2 = 190 / 25.4  # inches
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    """Vector PDF for submission + PNG preview, both at final size."""
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)

ds = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_weekly.csv",
                 parse_dates=["date", "origin_date"])


def box(ax, xy, w, h, text, fc="#F2F2F2", ec="#444444", fs=8.5):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=1.0)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C["ink"])


def arrow(ax, p0, p1, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                 color=kw.get("color", "#444444"),
                                 lw=kw.get("lw", 1.1),
                                 linestyle=kw.get("ls", "-")))


# ---- F1: stock-flow conceptual diagram -----------------------------------
fig, ax = plt.subplots(figsize=(COL2, 3.3))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
box(ax, (0.2, 3.4), 1.9, 0.9, "Field production\n(kb/d)")
box(ax, (0.2, 2.0), 1.9, 0.9, "Imports\n(kb/d)")
box(ax, (0.2, 0.6), 1.9, 0.9, "Adjustment\n(kb/d)")
box(ax, (4.0, 1.9), 2.2, 1.3,
    "Commercial crude\nstocks ex-SPR\n(kb)", fc="#E8F1F8")
box(ax, (7.8, 3.4), 1.9, 0.9, "Refinery net\ninputs (kb/d)")
box(ax, (7.8, 2.0), 1.9, 0.9, "Exports\n(kb/d)")
box(ax, (7.8, 0.6), 1.9, 0.9, "SPR transfers\n(kb/wk)")
for y in (3.85, 2.45, 1.05):
    arrow(ax, (2.1, y), (4.0, 2.55 if y != 2.45 else 2.45))
for y in (3.85, 2.45, 1.05):
    arrow(ax, (6.2, 2.55 if y != 2.45 else 2.45), (7.8, y))
ax.text(5.1, 0.25, r"$\Delta S_t + \Delta SPR_t = 7\,(Prod_t + Imp_t + Adj_t - Ref_t - Exp_t)$",
        ha="center", fontsize=9.5, color=C["ink"])
fig.tight_layout(); save(fig, "F1_stockflow")

# ---- F2: PBCF-Net architecture -------------------------------------------
fig, ax = plt.subplots(figsize=(COL2, 3.7))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
box(ax, (0.2, 4.2), 2.4, 1.3,
    "Price features (12 wk)\nreturns, vols,\nRV 1/4/12w", fc="#E8F1F8")
box(ax, (0.2, 0.6), 2.4, 1.3,
    "Physical features (12 wk)\nEIA raw + engineered\nbalance features", fc="#FCF3E3")
box(ax, (3.3, 4.3), 1.7, 1.0, "GRU\n(64)")
box(ax, (3.3, 0.7), 1.7, 1.0, "GRU\n(32)")
box(ax, (5.7, 2.4), 1.9, 1.2, "Fusion\nMLP head")
box(ax, (8.2, 2.5), 1.6, 1.0, "log RV\nh=1..12", fc="#E8F1F8")
box(ax, (5.7, 0.4), 1.9, 1.2,
    "Aux head:\n$\\Delta$state + Adj\n(next week)", fc="#FCF3E3")
box(ax, (8.2, 0.5), 1.6, 1.0, "Balance\nloss $\\lambda_B$", fc="#F6E4DE",
    ec=C["M5b"])
arrow(ax, (2.6, 4.85), (3.3, 4.8))
arrow(ax, (2.6, 1.25), (3.3, 1.2))
arrow(ax, (5.0, 4.8), (5.9, 3.6))
arrow(ax, (5.0, 1.2), (5.9, 2.6))
arrow(ax, (5.0, 1.2), (5.7, 1.0))
arrow(ax, (7.6, 3.0), (8.2, 3.0))
arrow(ax, (7.6, 1.0), (8.2, 1.0))
fig.tight_layout(); save(fig, "F2_architecture")

# ---- F3: RV and stock-change history -------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(COL2, 4.4), sharex=True)
ax = axes[0]
ax.plot(ds["origin_date"], ds["px_rv_1w"] * 100, lw=0.8, color=C["M1"])
ax.set_ylabel("Weekly realized vol (%)")
ax.set_yscale("log")
ax.set_title("(a) WTI weekly realized volatility (log scale)", loc="left")
ax = axes[1]
ax.plot(ds["date"], ds["d_stocks_commercial"] / 1000, lw=0.7, color=C["M4"])
q10 = ds[ds.split == "train"]["d_stocks_commercial"].quantile(0.10) / 1000
q90 = ds[ds.split == "train"]["d_stocks_commercial"].quantile(0.90) / 1000
ax.axhline(q10, color="#999999", lw=0.8, ls="--")
ax.axhline(q90, color="#999999", lw=0.8, ls="--")
ax.text(ds["date"].iloc[5], q90 + 1, "train q90", fontsize=7, color="#666666")
ax.text(ds["date"].iloc[5], q10 - 3, "train q10", fontsize=7, color="#666666")
ax.set_ylabel("$\\Delta$ commercial stocks\n(million bbl/wk)")
ax.set_title("(b) Weekly commercial crude stock change", loc="left")
ax.xaxis.set_major_locator(mdates.YearLocator(4))
fig.tight_layout(); save(fig, "F3_history")

# ---- F4: lambda trade-off (v5, HAR-residual, single split) ---------------
lam = [0.0, 0.01, 0.1, 0.5]
q_h1 = [0.1267, 0.1258, 0.1251, 0.1228]
q_h12 = [0.1637, 0.1624, 0.1621, 0.1632]
bce = [6210, 6161, 6154, 6065]
fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0))
ax = axes[0]
ax.plot(range(4), q_h1, color=C["M1"], ls="-", marker="o", lw=1.5,
        ms=4, label="h = 1")
ax.plot(range(4), q_h12, color=C["M4"], ls="--", marker="s", lw=1.5,
        ms=4, label="h = 12")
ax.set_xticks(range(4)); ax.set_xticklabels(["0\n(M4)", "0.01", "0.1", "0.5"])
ax.set_xlabel("$\\lambda_B$"); ax.set_ylabel("Test QLIKE")
ax.set_title("(a) Forecast loss vs constraint strength", loc="left")
ax.legend()
ax = axes[1]
ax.plot(range(4), [b / 1000 for b in bce], "-o", color=C["M5b"], lw=1.5, ms=4)
ax.set_xticks(range(4)); ax.set_xticklabels(["0\n(M4)", "0.01", "0.1", "0.5"])
ax.set_xlabel("$\\lambda_B$")
ax.set_ylabel("BCE (million bbl/wk)")
ax.set_title("(b) Balance-consistency error", loc="left")
fig.tight_layout(); save(fig, "F4_lambda_tradeoff")

# ---- F5: horizon gains + build-window bars -------------------------------
h = [1, 2, 4, 8, 12]
ew = {  # pooled expanding-window QLIKE, v5 (HAR-residual)
    "HAR": [0.1206, 0.0939, 0.0974, 0.1211, 0.1418],
    "M1": [0.1207, 0.0953, 0.0998, 0.1253, 0.1464],
    "M4": [0.1190, 0.0953, 0.0999, 0.1237, 0.1415],
    "M5a": [0.1184, 0.0955, 0.0980, 0.1208, 0.1380],
}
build12 = {"HAR": 0.8379, "M1": 0.8180, "M4": 0.7705,
           "M5a": 0.7264, "M5b": 0.5646}
labels = {"HAR": "HAR", "M1": "M1 (price-only)", "M4": "M4 (multi-task)",
          "M5a": "M5 ($\\lambda_B$=0.1)", "M5b": "M5 ($\\lambda_B$=0.5)"}
fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.1),
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for m in ("HAR", "M1", "M4", "M5a"):
    ls, mk = STYLE[m]
    ax.plot(h, ew[m], color=C[m], ls=ls, marker=mk, lw=1.5, ms=4.5,
            label=labels[m])
ax.set_xscale("log"); ax.set_xticks(h); ax.set_xticklabels(h)
ax.minorticks_off()
ax.set_xlabel("Forecast horizon (weeks)"); ax.set_ylabel("Pooled QLIKE")
ax.set_title("(a) Expanding-window QLIKE by horizon", loc="left")
ax.legend(fontsize=7.5)
ax = axes[1]
ms = ["HAR", "M1", "M4", "M5a", "M5b"]
ax.bar(range(5), [build12[m] for m in ms],
       color=[C[m] for m in ms], width=0.62)
ax.set_xticks(range(5))
ax.set_xticklabels(["HAR", "M1", "M4", "M5\n0.1", "M5\n0.5"], fontsize=7.5)
for i, m in enumerate(ms):
    ax.text(i, build12[m] + 0.02, f"{build12[m]:.2f}", ha="center", fontsize=7)
ax.set_ylabel("QLIKE, h = 12")
ax.set_title("(b) Inventory-build windows", loc="left")
ax.grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "F5_horizon_gains")

# ---- F6: 2020 case study -------------------------------------------------
import sys
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "05_inventory_stress"))
import stress_analysis
stress_analysis.HORIZONS = {"h1": 1, "h2": 2, "h4": 4, "h8": 8, "h12": 12}
from stress_analysis import ensemble_preds

p1 = ensemble_preds("M1_v5"); p4 = ensemble_preds("M4_v5")
p5 = ensemble_preds("M5_v5lb0p1")
mask = (p1["date"] >= "2020-01-01") & (p1["date"] <= "2020-12-31")
fig, axes = plt.subplots(2, 1, figsize=(COL2, 4.6), sharex=True)
ax = axes[0]
d = p1.loc[mask, "date"]
ax.plot(d, np.exp(p1.loc[mask, "y_h1"]) * 100, lw=1.6, color="#222222",
        label="Realized")
ax.plot(d, np.exp(p1.loc[mask, "yhat_h1"]) * 100, lw=1.2, color=C["M1"],
        ls=STYLE["M1"][0], label=labels["M1"])
ax.plot(d, np.exp(p4.loc[mask, "yhat_h1"]) * 100, lw=1.2, color=C["M4"],
        ls=STYLE["M4"][0], label=labels["M4"])
ax.plot(d, np.exp(p5.loc[mask, "yhat_h1"]) * 100, lw=1.2, color=C["M5a"],
        ls=STYLE["M5a"][0], label=labels["M5a"])
ax.set_ylabel("Next-week RV (%)")
ax.set_yscale("log")
ax.set_title("(a) One-week-ahead realized volatility, 2020", loc="left")
ax.legend(fontsize=7.5, ncol=2)
ax = axes[1]
m20 = (ds["date"] >= "2020-01-01") & (ds["date"] <= "2020-12-31")
ax.plot(ds.loc[m20, "date"], ds.loc[m20, "stocks_commercial"] / 1000,
        lw=1.4, color=C["M4"])
ax.set_ylabel("Commercial stocks\n(million bbl)")
ax.set_title("(b) U.S. commercial crude stocks", loc="left")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.tight_layout(); save(fig, "F6_case2020")

print("Figures written to", FIG)
