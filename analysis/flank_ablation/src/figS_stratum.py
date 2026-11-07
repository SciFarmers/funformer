"""Supplementary: flank usefulness by TSS stratum, per bin.

Separated from Figure 4 because it answers a narrower question -- whether the
flank the model loses is the proximal or the distal one -- and Figure 4 is
already carrying three panels.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.join(ROOT, "analysis", "flank_ablation")
sys.path.insert(0, os.path.join(ROOT, "manuscript", "analysis"))
from figutil import apply_style, panel_label          # noqa: E402

OUT = os.path.join(HERE, "out")
BINS = ["mammal-mammal", "mammal-chicken", "tetrapod-fish"]
BIN_LAB = {"mammal-mammal": "Mammal\n↔ mammal", "mammal-chicken": "Mammal\n→ chicken",
           "tetrapod-fish": "Tetrapod\n↔ fish"}
BIN_C = {"mammal-mammal": "#0072B2", "mammal-chicken": "#009E73",
         "tetrapod-fish": "#E69F00"}
BIN_MK = {"mammal-mammal": "o", "mammal-chicken": "s", "tetrapod-fish": "^"}


import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(ROOT, "analysis", "common"))
from ci import mean_ci  # noqa: E402

def main():
    apply_style()
    ch = pd.read_csv(os.path.join(OUT, "strata_split_channels.tsv"), sep="\t")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    rng = np.random.default_rng(11)
    for i, b in enumerate(BINS):
        sub = ch[ch.bin == b]
        for j, st in enumerate(("proximal", "distal")):
            v = sub[sub.stratum == st].usefulness_pp.to_numpy(float)
            if not len(v):
                continue
            x = i + (j - 0.5) * 0.34
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=9,
                       marker=BIN_MK[b], c=BIN_C[b] if j == 0 else "white",
                       edgecolors=BIN_C[b], linewidths=0.6, alpha=0.5, zorder=2)
            m, lo, hi = mean_ci(v)
            ax.plot([x - 0.14, x + 0.14], [m, m], color="0.15", lw=2.2, zorder=4)
            ax.plot([x, x], [lo, hi], color="0.15",
                    lw=1.1, zorder=4)
    ax.axhline(0, color="0.45", ls="--", lw=1.0, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([BIN_LAB[b] for b in BINS], fontsize=7.5)
    ax.set_ylabel("Flank usefulness (pp)\nintact $-$ target flank replaced")
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor="0.35",
               markeredgecolor="0.35", markersize=6, label="Proximal ($\\leq$2 kb)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="0.35", markersize=6, label="Distal ($>$10 kb)")],
        frameon=True, fontsize=7, loc="upper right", handletextpad=0.5)
    leg.get_frame().set_edgecolor("0.85"); leg.get_frame().set_linewidth(0.7)
    panel_label(ax, "", "Flank usefulness by TSS distance")
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    p = os.path.join(OUT, "figures", "figS_flank_stratum.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Wrote {p}")
    # also into the manuscript tree, which is what latex/compile.sh copies from
    mp = os.path.join(ROOT, "manuscript", "analysis", "out", "figures", "figS5.png")
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    fig.savefig(mp, dpi=300, bbox_inches="tight")
    print(f"Wrote {mp}")


if __name__ == "__main__":
    main()
