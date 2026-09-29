#!/usr/bin/env python3
"""Figure 1 (teaser), redrawn with larger text and directly labelled contrasts.

All values are copied from the paper's tables: Table 7 (prefix / repeat / suffix NLL, downstream checkpoint)
and Table 4 (decomposition of two checkpoints with 95% paired document-bootstrap intervals).
Palette: categorical slots 1-3 of the reference palette, validated for CVD separation (validate_palette.js).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "figs" / "fig0_teaser_v2.pdf"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"

# Table 7 (downstream checkpoint, step 4,000)
K = [1, 2, 4, 8]
prefix = [2.6426, 2.4197, 2.1178, 1.2173]
repeat = [1.9595, 1.9492, 1.9987, 1.2173]
suffix = [2.4405, 1.9376, 2.3579, 1.2173]

# Table 4: nats and shares [95% CI] for the downstream (step 4k) and pretraining (step 40k) checkpoints
comps = ["Application\ncount", "Distinct\niterations", "Inter-block\ncomposition", "Temp.-\ncorrectable"]
nats = {"Downstream (4k)": [0.6831, -0.0392, 0.7814, 0.3654],
        "Pretraining (40k)": [0.6272, 0.0079, 0.8461, 0.480]}
shares = {"Downstream (4k)": [(47.9, 46.6, 49.3), (-2.8, -3.2, -2.4), (54.8, 53.3, 56.3), (25.6, 24.5, 26.6)],
          "Pretraining (40k)": [(42.3, 40.8, 43.9), (0.5, 0.4, 0.6), (57.1, 55.6, 58.7), (32.4, 30.8, 33.8)]}

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.0, 1.15]})

# (a) truncation curves with the contrasts labelled on the plot
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
for y, col, mk in ((prefix, BLUE, "o"), (repeat, ORANGE, "s"), (suffix, AQUA, "^")):
    a.plot(K, y, color=col, lw=1.6, marker=mk, ms=5.5, mec="white", mew=1, zorder=3)
a.text(1.55, 2.60, "prefix", color=INK, fontsize=8, bbox=BOX, zorder=4)
a.text(4.35, 2.33, "suffix", color=INK, fontsize=8, bbox=BOX, zorder=4)
a.text(2.05, 1.86, "repeat (8 appl.)", color=INK, fontsize=8, bbox=BOX, zorder=4)
a.set_xscale("log", base=2)
a.set_xticks(K, [str(k) for k in K])
a.spines["bottom"].set_bounds(1, 8)
a.set_xlabel("Retained applications $k$")
a.set_ylabel("Held-out NLL (nats, lower = better)")
a.grid(axis="y", color=GRID, lw=0.8)
a.set_ylim(1.05, 2.95)
a.set_xlim(0.8, 17)


def bracket(ax, x, y0, y1, text, ha="left", ty=None):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3), zorder=5)
    ax.text(x * (1.07 if ha == "left" else 0.93), ty if ty else (y0 + y1) / 2, text, ha=ha, va="center",
            fontsize=7.8, color=INK, bbox=BOX, zorder=6)


bracket(a, 0.9, prefix[0], repeat[0], "application count: 0.68 nats", ha="left", ty=2.82)
a.text(1.06, 1.60, "distinct iterations\n(repeat $k$=1$\\to$4): $-0.04$ nats", fontsize=7.8, color=INK, bbox=BOX, zorder=6)
bracket(a, 8.6, repeat[2], prefix[3], "inter-block\ncomposition\n0.78 nats", ha="left", ty=1.6)
a.set_title("(a) Truncation curves (downstream ckpt.)", fontsize=9, color=INK, loc="left")

# (b) contrasts in nats; share and 95% CI printed on each bar
x = range(len(comps))
w = 0.36
for i, (name, col) in enumerate((("Downstream (4k)", BLUE), ("Pretraining (40k)", ORANGE))):
    xs = [j + (i - 0.5) * w for j in x]
    b.bar(xs, nats[name], width=w - 0.02, color=col, label=name, zorder=3)
    for xx, v, (s, lo, hi) in zip(xs, nats[name], shares[name]):
        b.text(xx, max(v, 0) + 0.015 + (0.07 * i if abs(v) < 0.05 else 0), f"{s:.1f}%", ha="center", va="bottom",
               fontsize=7, color=INK)
b.axhline(0, color=MUTED, lw=0.8)
b.set_xticks(list(x), comps)
b.set_ylabel("Contrast (nats)")
b.set_ylim(-0.1, 1.0)
b.grid(axis="y", color=GRID, lw=0.8)
b.legend(frameon=False, loc="upper left", fontsize=8)
b.set_title("(b) Contrasts in nats (share of naive gap)", fontsize=9, color=INK, loc="left")

fig.tight_layout()
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=110)
print(OUT)
