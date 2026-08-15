# -*- coding: utf-8 -*-
"""
Publication-quality comparative training-progress figure (600 dpi)
Approach 1 (InSAR only) vs Approach 2 (InSAR + rainfall)
Panels: (a) total loss, (b) value-function loss, (c) entropy loss
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator
import os

F1 = r"C:\Users\hp\Desktop\approach1_insar_only progress.csv"
F2 = r"C:\Users\hp\Desktop\approach2_with_rainfall progress.csv"
OUT = r"C:\Users\hp\Desktop\training_progress_analysis"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C1, C2 = "#0072B2", "#D55E00"          # Okabe-Ito blue / vermillion (colour-blind safe)
LS1, LS2 = "-", "--"                   # distinguishable in greyscale print

a1 = pd.read_csv(F1).dropna(subset=["train/loss"]).reset_index(drop=True)
a2 = pd.read_csv(F2).dropna(subset=["train/loss"]).reset_index(drop=True)
x = a1["time/total_timesteps"].values / 1e3          # 10^3 environment steps
PLAT = np.arange(32, 48)                             # final 16 logged iterations
xs, xe = x[PLAT[0]] - 1.024, x[-1] + 1.024


def sm(v, w=5):
    return pd.Series(v).rolling(w, center=True, min_periods=1).mean().values


fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.3), sharex=True,
                         gridspec_kw=dict(hspace=0.13, left=0.115, right=0.895,
                                          top=0.872, bottom=0.072))
LBOX = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.80)
ax_a, ax_b, ax_c = axes

PANELS = [
    (ax_a, "train/loss", "Total loss,  $\\mathcal{L}_{\\mathrm{PPO}}$", True),
    (ax_b, "train/value_loss", "Value loss,  $\\mathcal{L}_{V}$", True),
    (ax_c, "train/entropy_loss", "Entropy loss,  $-\\mathcal{H}(\\pi)$  [nats]", False),
]

for ax, col, ylab, logy in PANELS:
    ax.axvspan(xs, xe, color="0.90", alpha=0.55, lw=0, zorder=0)
    for d, c, ls in [(a1, C1, LS1), (a2, C2, LS2)]:
        y = d[col].values
        ax.plot(x, y, color=c, lw=0.8, alpha=0.30, zorder=2, solid_capstyle="round")
        ax.plot(x, sm(y), color=c, ls=ls, lw=1.9, zorder=3, solid_capstyle="round",
                dash_capstyle="round")
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 109)
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="y", color="0.94", lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------------------------------------------------------- panel (a)
ax_a.set_ylim(33, 460)
ax_a.set_yticks([40, 60, 100, 200, 400])
ax_a.set_yticklabels(["40", "60", "100", "200", "400"])
p1 = a1["train/loss"].values[PLAT]
p2 = a2["train/loss"].values[PLAT]
ax_a.hlines(p1.mean(), xs, xe, color=C1, lw=1.0, ls=(0, (1, 1.6)), zorder=4)
ax_a.hlines(p2.mean(), xs, xe, color=C2, lw=1.0, ls=(0, (1, 1.6)), zorder=4)
ax_a.annotate(f"{p1.mean():.1f} $\\pm$ {p1.std(ddof=1):.1f}", xy=(70.5, p1.mean()),
              xytext=(0, 5), textcoords="offset points", color=C1, fontsize=8, ha="left",
              bbox=LBOX)
ax_a.annotate(f"{p2.mean():.1f} $\\pm$ {p2.std(ddof=1):.1f}", xy=(70.5, p2.mean()),
              xytext=(0, -13), textcoords="offset points", color=C2, fontsize=8, ha="left",
              bbox=LBOX)
ax_a.annotate("", xy=(103.5, p1.mean()), xytext=(103.5, p2.mean()),
              arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.9, shrinkA=0, shrinkB=0))
ax_a.text(106.4, np.sqrt(p1.mean() * p2.mean()), "$-$41.3 %", rotation=90, va="center",
          ha="center", fontsize=8, color="0.20")
ax_a.text(85.0, 415, "plateau window\n(final 16 iterations)", ha="center", va="top",
          fontsize=7.8, color="0.35", linespacing=1.25)

sec = ax_a.secondary_xaxis("top", functions=(lambda v: v * 1e3 / 2048, lambda v: v * 2048 / 1e3))
sec.set_xlabel("PPO policy-update iteration", labelpad=5)
sec.set_xticks([0, 10, 20, 30, 40, 49])
sec.spines["top"].set_linewidth(0.8)

# ---------------------------------------------------------------- panel (b)
ax_b.set_ylim(70, 1000)
ax_b.set_yticks([80, 100, 200, 400, 800])
ax_b.set_yticklabels(["80", "100", "200", "400", "800"])
q1 = a1["train/value_loss"].values[PLAT]
q2 = a2["train/value_loss"].values[PLAT]
ax_b.hlines(q1.mean(), xs, xe, color=C1, lw=1.0, ls=(0, (1, 1.6)), zorder=4)
ax_b.hlines(q2.mean(), xs, xe, color=C2, lw=1.0, ls=(0, (1, 1.6)), zorder=4)
ax_b.annotate(f"{q1.mean():.1f} $\\pm$ {q1.std(ddof=1):.1f}", xy=(70.5, q1.mean()),
              xytext=(0, 5), textcoords="offset points", color=C1, fontsize=8, ha="left",
              bbox=LBOX)
ax_b.annotate(f"{q2.mean():.1f} $\\pm$ {q2.std(ddof=1):.1f}", xy=(70.5, q2.mean()),
              xytext=(0, -13), textcoords="offset points", color=C2, fontsize=8, ha="left",
              bbox=LBOX)
ax_b.annotate("", xy=(103.5, q1.mean()), xytext=(103.5, q2.mean()),
              arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.9, shrinkA=0, shrinkB=0))
ax_b.text(106.4, np.sqrt(q1.mean() * q2.mean()), "$-$40.1 %", rotation=90, va="center",
          ha="center", fontsize=8, color="0.20")
ax_b.text(3, 900, "plateau difference: $p = 3.1\\times10^{-5}$ (Wilcoxon signed-rank), "
                  "Cliff's $\\delta = 1.00$, distributions disjoint",
          fontsize=7.8, color="0.20", va="top")

# ---------------------------------------------------------------- panel (c)
HMAX = np.log(5)
ax_c.set_ylim(-1.78, -0.10)
ax_c.axhline(-HMAX, color="0.45", lw=0.9, ls=(0, (5, 3)), zorder=1)
ax_c.text(54, -HMAX + 0.05, "maximum entropy: uniform policy over the 5 actions "
                            "($-\\ln 5 = -1.609$)", fontsize=7.8, color="0.35",
          ha="center", va="bottom")
e1 = a1["train/entropy_loss"].values[PLAT]
e2 = a2["train/entropy_loss"].values[PLAT]
ax_c.text(72.0, -1.14, "plateau means  $-$%.3f (A1)  vs  $-$%.3f (A2)\n"
                       % (abs(e1.mean()), abs(e2.mean())) +
                       "difference not significant ($p = 0.18$)",
          ha="center", va="center", fontsize=7.8, color="0.20", linespacing=1.4,
          bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.80", lw=0.6, alpha=0.94))
ax_c.yaxis.set_major_locator(MultipleLocator(0.4))
ax_c.yaxis.set_minor_locator(MultipleLocator(0.1))
ax_c.set_xlabel("Environment interaction steps  ($\\times 10^{3}$)")
ax_c.xaxis.set_major_locator(MultipleLocator(20))
ax_c.xaxis.set_minor_locator(MultipleLocator(5))

sec_c = ax_c.secondary_yaxis("right", functions=(lambda v: -100 * v / HMAX,
                                                 lambda v: -v * HMAX / 100))
sec_c.set_ylabel("Policy entropy $\\mathcal{H}(\\pi)/\\mathcal{H}_{\\max}$  [%]", labelpad=6)
sec_c.set_yticks([0, 25, 50, 75, 100])
sec_c.spines["right"].set_linewidth(0.8)

# ---------------------------------------------------------------- labels & legend
for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
    ax.text(-0.105, 1.0, lab, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left")

handles = [
    Line2D([], [], color=C1, ls=LS1, lw=1.9, label="Approach 1 — InSAR only"),
    Line2D([], [], color=C2, ls=LS2, lw=1.9, label="Approach 2 — InSAR + rainfall"),
    Line2D([], [], color="0.55", lw=0.8, alpha=0.6, label="per-iteration value"),
    Line2D([], [], color="0.55", lw=1.9, label="5-iteration moving average"),
]
fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.515, 1.000),
           frameon=False, handlelength=2.4, columnspacing=2.2, handletextpad=0.7,
           labelspacing=0.45)

for ext in ("png", "pdf", "svg"):
    p = os.path.join(OUT, f"fig_training_progress_A1_vs_A2.{ext}")
    fig.savefig(p, dpi=600)
    print("saved:", p, f"({os.path.getsize(p)/1e6:.2f} MB)")
plt.close(fig)
