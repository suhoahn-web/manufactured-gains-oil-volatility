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
fig.tight_layout(); save(fig, "F1_history")

# ---- F4: constraint strength, pooled expanding window --------------------
# Both arms are read from the pooled per-week files rather than hard-coded, so
# the figure and the tables cannot drift apart. The weights used in the ladder
# stop at 0.5; 2 and 5 are where the constraint actually binds.
POOL = PROJECT_ROOT / "experiments" / "06_robustness"
LAM = [("M4", 0.0), ("M5_lb0p1", 0.1), ("M5_lb0p5", 0.5),
       ("M5_lb2p0", 2.0), ("M5_lb5p0", 5.0)]

q_h1, q_h12 = [], []
for spec, _ in LAM:
    d = pd.read_csv(POOL / f"ew_losses_{spec}.csv")
    q_h1.append(d["h1"].mean())
    q_h12.append(d["h12"].mean())

bce_src = pd.concat([pd.read_csv(POOL / "ew_bce_ladder.csv"),
                     pd.read_csv(POOL / "ew_bce_extra.csv")])
# the unpenalised network is recorded under the stopping-rule control, which
# is bit-identical to M4
bce_key = {0.0: "M4_fcstop", 0.1: "M5_lb0p1", 0.5: "M5_lb0p5",
           2.0: "M5_lb2p0", 5.0: "M5_lb5p0"}
bce = [bce_src.loc[bce_src.config == bce_key[lam], "bce_kb_week_mean"].mean()
       for _, lam in LAM]

har_h1, har_h12 = [pd.read_csv(POOL / "ew_losses_HAR.csv")[c].mean()
                   for c in ("h1", "h12")]
xs = range(len(LAM))
ticks = ["0\n(M4)", "0.1", "0.5", "2", "5"]

fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.0))
ax = axes[0]
ax.plot(xs, q_h1, color=C["M1"], ls="-", marker="o", lw=1.5, ms=4,
        label="h = 1")
ax.plot(xs, q_h12, color=C["M4"], ls="--", marker="s", lw=1.5, ms=4,
        label="h = 12")
ax.axhline(har_h1, color=C["M1"], lw=0.8, ls=":", alpha=0.7)
ax.axhline(har_h12, color=C["M4"], lw=0.8, ls=":", alpha=0.7)
ax.text(0.05, har_h1, "HAR, h = 1", fontsize=6.5,
        color=C["M1"], va="bottom", ha="left")
ax.text(0.05, har_h12, "HAR, h = 12", fontsize=6.5,
        color=C["M4"], va="bottom", ha="left")
# the constraint binds from the smallest weight on; what changes past 0.5 is
# that the accuracy cost becomes statistically significant
ax.axvspan(3, len(LAM) - 1, color="#F0F0F0", zorder=0)
lo, hi = min(q_h1 + q_h12), max(q_h1 + q_h12)
ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.22 * (hi - lo))
ax.text(3.5, ax.get_ylim()[1], "accuracy cost significant", fontsize=6.5,
        color="#666666", ha="center", va="top")
ax.set_xticks(xs); ax.set_xticklabels(ticks)
ax.set_xlabel("$\\lambda_B$"); ax.set_ylabel("Pooled QLIKE")
ax.set_title("(a) Forecast loss vs constraint strength", loc="center")
ax.legend(loc="lower right")

ax = axes[1]
ax.plot(xs, [b / 1000 for b in bce], "-o", color=C["M5b"], lw=1.5, ms=4)
ax.axvspan(3, len(LAM) - 1, color="#F0F0F0", zorder=0)
ax.annotate(f"{100 * (bce[0] - bce[2]) / bce[0]:.0f}% lower at $\\lambda_B$ = 0.5",
            xy=(2, bce[2] / 1000),
            xytext=(3.7, bce[2] / 1000 + 0.22),
            fontsize=6.5, color="#666666", ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7,
                            shrinkA=3, shrinkB=2))
ax.set_xticks(xs); ax.set_xticklabels(ticks)
ax.set_xlabel("$\\lambda_B$")
ax.set_ylabel("BCE (million bbl/wk)")
ax.set_title("(b) Balance-consistency error", loc="center")
fig.tight_layout(); save(fig, "F6_lambda_tradeoff")

# ---- F5: the same forecasts scored on three scales -----------------------
# The paper's central result. Read from the analysis CSV so the figure and the
# tables cannot disagree.
SC = PROJECT_ROOT / "experiments" / "07_final" / "scale_comparison.csv"
sc = pd.read_csv(SC)
h = [1, 2, 4, 8, 12]
SPECS5 = ["M1", "M2", "M3derived", "M3", "M4", "M5_lb0p1", "M5_lb0p5"]
NICE = {"M1": "M1", "M2": "M2", "M3derived": "M3derived", "M3": "M3",
        "M4": "M4", "M5_lb0p1": r"M5 ($\lambda_B$=0.1)",
        "M5_lb0p5": r"M5 ($\lambda_B$=0.5)"}

fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.2),
                         gridspec_kw={"width_ratios": [1.15, 1]})

# (a) one specification, three scales, across horizons
ax = axes[0]
lines = [("QLIKE_var", "QLIKE on variance", C["M5b"], "-", "o"),
         ("MSE_var", "MSE on variance", C["M4"], "-.", "^"),
         ("QLIKE_vol", "QLIKE on volatility", C["M1"], "--", "s"),
         ("MSE_log", "MSE on log volatility", C["ink"], ":", "D")]
sel = "M5_lb0p5"
start_labels = []
for key, lab, col, ls, mk in lines:
    y = [sc[(sc.spec == sel) & (sc.loss == key) & (sc.h == hh)]
         ["pct_vs_har"].iloc[0] for hh in h]
    ax.plot(h, y, color=col, ls=ls, marker=mk, lw=1.5, ms=4, label=lab)
    start_labels.append((lab, col, y[0]))
ax.axhline(0, color="#444444", lw=1.0)
ax.set_xscale("log"); ax.set_xticks(h); ax.set_xticklabels(h)
ax.set_xlim(0.55, 13.2)
ax.minorticks_off()
ax.set_xlabel("Forecast horizon (weeks)")
ax.set_ylabel("Change vs benchmark (%)")
for lab, col, y0 in start_labels:
    ax.text(0.92, y0, lab, color=col, fontsize=6.5,
            ha="right", va="center")
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.10 * (hi - lo))
ax.text(12, ax.get_ylim()[1], "worse than benchmark", fontsize=6.5,
        color="#666666", va="top", ha="right")
ax.text(12, lo + 0.02 * (hi - lo), "better than benchmark", fontsize=6.5,
        color="#666666", va="bottom", ha="right")

# (b) every specification, two scales, at twelve weeks
ax = axes[1]
ypos = np.arange(len(SPECS5))
for key, lab, col, mk in [("QLIKE_var", "on variance", C["M5b"], "o"),
                          ("MSE_log", "on log volatility", C["ink"], "D")]:
    x = [sc[(sc.spec == s) & (sc.loss == key) & (sc.h == 12)]
         ["pct_vs_har"].iloc[0] for s in SPECS5]
    ax.scatter(x, ypos, color=col, marker=mk, s=26, label=lab, zorder=3)
ax.axvline(0, color="#444444", lw=1.0)
ax.set_yticks(ypos)
ax.set_yticklabels([NICE[s] for s in SPECS5], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("Change vs benchmark at h = 12 (%)")
ax.legend(fontsize=7, loc="lower right")
ax.grid(axis="y", visible=False)
axes[0].text(0.0, -0.25, "(a) One specification evaluated under four scoring conventions",
             transform=axes[0].transAxes, ha="left", va="top", fontsize=8)
axes[1].text(0.0, -0.25, "(b) All specifications evaluated on two scales at h = 12",
             transform=axes[1].transAxes, ha="left", va="top", fontsize=8)
fig.tight_layout(rect=(0, 0.13, 1, 1)); save(fig, "F5_scale_sensitivity")

# ---- F6: 2020 case study -------------------------------------------------
labels = {"HAR": "HAR", "M1": "M1 (price-only)", "M4": "M4 (multi-task)",
          "M5a": "M5 ($\\lambda_B$=0.1)", "M5b": "M5 ($\\lambda_B$=0.5)"}
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
ax.set_title("(a) One-week-ahead realized volatility, 2020", loc="center")
ax.legend(fontsize=7.5, ncol=2)
ax = axes[1]
m20 = (ds["date"] >= "2020-01-01") & (ds["date"] <= "2020-12-31")
ax.plot(ds.loc[m20, "date"], ds.loc[m20, "stocks_commercial"] / 1000,
        lw=1.4, color=C["M4"])
ax.set_ylabel("Commercial stocks\n(million bbl)")
ax.set_title("(b) U.S. commercial crude stocks", loc="center")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.tight_layout(); save(fig, "F4_case2020")

print("Figures written to", FIG)
