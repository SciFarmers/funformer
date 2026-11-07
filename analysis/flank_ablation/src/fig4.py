"""Figure 4: where the flank is used, and whether a source flank substitutes.

Styled as Figure 3 -- same bin palette, markers, jittered points over a bin
mean with 95% CI, and the same 720-relabelling permutation null.

Panel a  the Analysis A context margin split proximal vs distal, which is the
         TSS-stratified view of the Figure 3a quantity.
Panel b  transfer AUROC per bin under intact, random_same and random_source,
         with FunFormer-1k drawn as a reference line: the question is whether
         an ablated 20 kb model falls to, or below, a real 1 kb model.
Panel c  source-flank rescue per direction with its permutation null.
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
A_OUT = os.path.join(ROOT, "analysis", "tss_strata", "out")
BINS = ["mammal-mammal", "mammal-chicken", "tetrapod-fish"]
BIN_LAB = {"mammal-mammal": "Mammal\n↔ mammal", "mammal-chicken": "Mammal\n→ chicken",
           "tetrapod-fish": "Tetrapod\n↔ fish"}
BIN_C = {"mammal-mammal": "#0072B2", "mammal-chicken": "#009E73",
         "tetrapod-fish": "#E69F00"}
BIN_MK = {"mammal-mammal": "o", "mammal-chicken": "s", "tetrapod-fish": "^"}


import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(ROOT, "analysis", "common"))
from ci import mean_ci  # noqa: E402

def _ci(v):
    return mean_ci(v)


def panel_a(ax):
    """Context margin by bin, proximal vs distal (Analysis A)."""
    d = pd.read_csv(os.path.join(A_OUT, "directions_primary.tsv"), sep="\t")
    w = d.pivot_table(index=["source", "target", "bin"], columns="stratum",
                      values="margin_arch").reset_index()
    rng = np.random.default_rng(3)
    for i, b in enumerate(BINS):
        sub = w[w.bin == b]
        for j, (st, fill) in enumerate((("proximal", True), ("distal", False))):
            x = i + (j - 0.5) * 0.34
            v = sub[st].to_numpy(float)
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=22,
                       marker=BIN_MK[b], c=BIN_C[b] if fill else "white",
                       edgecolors=BIN_C[b], linewidths=0.9, alpha=0.85, zorder=3)
            m, lo, hi = _ci(v)
            ax.plot([x - 0.14, x + 0.14], [m, m], color="0.15", lw=2.2, zorder=4)
            ax.plot([x, x], [lo, hi], color="0.15", lw=1.1, zorder=4)
    ax.axhline(0, color="0.45", ls="--", lw=1.0, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([BIN_LAB[b] for b in BINS], fontsize=7)
    ax.set_ylabel("Context margin (pp)\nFunFormer-20k $-$ FunFormer-1k")
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor="0.35",
               markeredgecolor="0.35", markersize=6, label="Proximal ($\\leq$2 kb)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="0.35", markersize=6, label="Distal ($>$10 kb)")],
        frameon=True, fontsize=6.8, loc="lower left", handletextpad=0.5)
    leg.get_frame().set_edgecolor("0.85"); leg.get_frame().set_linewidth(0.7)
    panel_label(ax, "a")
    ax.set_title("Context margin by TSS distance", loc="center", fontsize=8.2)


def panel_b(ax):
    """Transfer AUROC per bin under each condition, FunFormer-1k as reference.

    One point per direction at each condition rather than a bin mean alone:
    the spread is the point in the fish bin, where the conditions separate by
    less than the directions do.
    """
    d = pd.read_csv(os.path.join(OUT, "directions_stage1.tsv"), sep="\t")
    d = d[d.bin != "within-species"]
    rng = np.random.default_rng(7)
    conds = [("intact", "Intact"), ("random_same", "Target flank replaced"),
             ("random_source", "Source flank")]
    face = {"intact": None, "random_same": "white", "random_source": "0.78"}
    for i, b in enumerate(BINS):
        sub = d[d.bin == b]
        means = []
        for k, (cond, _) in enumerate(conds):
            x = i + (k - 1) * 0.38
            if cond == "intact":
                v = sub.drop_duplicates(["source", "target"]).auroc_intact.to_numpy(float)
            else:
                v = sub[sub.condition == cond].auroc_ablated.to_numpy(float)
            if not len(v):
                continue
            fc = BIN_C[b] if face[cond] is None else face[cond]
            ax.scatter(x + rng.uniform(-0.085, 0.085, len(v)), v, s=16,
                       marker=BIN_MK[b], c=fc, edgecolors=BIN_C[b],
                       linewidths=0.8, alpha=0.75, zorder=3)
            ax.plot([x - 0.13, x + 0.13], [v.mean(), v.mean()], color="0.15",
                    lw=2.2, zorder=5)
            means.append((x, v.mean()))
        # join the three means: at the fish bin intact and source flank coincide,
        # which is the panel's point and is otherwise easy to miss
        if len(means) == 3:
            ax.plot([m[0] for m in means], [m[1] for m in means], color="0.3",
                    lw=0.9, zorder=4, alpha=0.8)
        ff = sub.auroc_ff1k.mean()
        ax.plot([i - 0.52, i + 0.52], [ff, ff], color="0.25", ls="--", lw=1.4,
                zorder=6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([BIN_LAB[b] for b in BINS], fontsize=7)
    ax.set_xlim(-0.68, 2.68)
    ax.set_ylabel("Transfer AUROC")
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor="0.35",
               markeredgecolor="0.35", markersize=6, label="Intact"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="0.35", markersize=6,
               label="Target flank replaced (random, same genome)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="0.78",
               markeredgecolor="0.35", markersize=6, label="Source flank"),
        Line2D([], [], color="0.25", ls="--", lw=1.4, label="FunFormer-1k")],
        frameon=True, fontsize=6.0, loc="upper right", handletextpad=0.5,
        labelspacing=0.35)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.85"); leg.get_frame().set_linewidth(0.7)
    # set the centred title separately: at panel_label's default size this
    # one is long enough to run into the bold panel letter
    panel_label(ax, "b")
    ax.set_title("Flank-ablated FunFormer-20k against FunFormer-1k",
                 loc="center", fontsize=8.2)


GROUPS = [("mammal-mammal", "Mammal\n\u2192 mammal", "#0072B2", "o"),
          ("mammal-chicken", "Mammal\n\u2192 chicken", "#009E73", "s"),
          ("tetrapod-fish-in", "Mammal\n\u2192 fish", "#E69F00", "^"),
          ("fish-tetrapod", "Fish\n\u2192 tetrapod", "#CC79A7", "v")]


def panel_c(ax):
    """Rescue per direction in four groups.

    The fish bin is split by which way the transfer runs, because that is where
    the effect lives: a source flank substitutes for the target's own only when
    a tetrapod model is reading a fish genome, not the reverse. Divergence time
    is not used here -- rescue does not track it (r = +0.34, P = 0.19), so a
    trend line would assert a result the data does not carry.
    """
    from divergence_times import CLADE
    w = pd.read_csv(os.path.join(OUT, "rescue_directions.tsv"), sep="\t")
    w["group"] = np.where(
        w.bin != "tetrapod-fish", w.bin,
        np.where(w.source.map(CLADE) == "fish", "fish-tetrapod", "tetrapod-fish-in"))
    rng = np.random.default_rng(5)
    for i, (key, lab, colour, mk) in enumerate(GROUPS):
        v = w[w.group == key].rescue_pp.to_numpy(float)
        if not len(v):
            continue
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=30, marker=mk,
                   c=colour, edgecolors="0.15", linewidths=0.5, alpha=0.9,
                   zorder=3)
        m, lo, hi = _ci(v)
        ax.plot([i - 0.3, i + 0.3], [m, m], color="0.15", lw=2.4, zorder=4)
        ax.plot([i, i], [lo, hi], color="0.15", lw=1.2, zorder=4)
        ax.annotate(f"{m:+.1f}", xy=(i + 0.34, m), fontsize=8,
                    fontweight="bold", va="center", ha="left", color="0.15")
    ax.axhline(0, color="0.45", ls="--", lw=1.0, zorder=1)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g[1] for g in GROUPS], fontsize=7)
    ax.set_xlim(-0.55, len(GROUPS) - 0.25)
    ax.set_ylabel("Source-flank rescue (pp)\nsource genome $-$ target genome")
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, "c")
    ax.set_title("Rescue by direction of transfer", loc="center", fontsize=8.2)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    panel_a(axes[0]); panel_b(axes[1]); panel_c(axes[2])
    fig.tight_layout(w_pad=2.6)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    p = os.path.join(OUT, "figures", "fig4.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Wrote {p}")
    # also into the manuscript tree, which is what latex/compile.sh copies from
    mp = os.path.join(ROOT, "manuscript", "analysis", "out", "figures", "fig4.png")
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    fig.savefig(mp, dpi=300, bbox_inches="tight")
    print(f"Wrote {mp}")


if __name__ == "__main__":
    main()
