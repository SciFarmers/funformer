"""Figure 5: eQTL variant-effect prediction, as per-tissue distributions.

Each model is drawn as the distribution of its random-forest AUROC across tissues
-- a kernel-density outline with every tissue plotted underneath and the median
marked -- rather than as a bar of the mean. The reason is that between-tissue
variance dwarfs between-model variance (human: SD 0.021 across the 49 GTEx
tissues against model-to-model differences of 0.003-0.018), so a bar chart of
means hides the quantity that actually governs whether a difference is real.
All model comparisons quoted in the text are paired across tissues, which
removes that shared tissue effect; the boxes here show what is being paired:
median, interquartile range and 1.5x IQR whiskers over tissues, with every
tissue drawn as a point.

Rows are grouped by what the model had access to:
  species-matched   trained on the target species
  1 kb baselines    trained on the target species, short context
  zero-shot         trained on another species, applied with no target training

Reads:  bench_results/rf_eqtl/*.csv,
        bench_results/rf_pig_qtl/{eqtl,sqtl}/*.csv
Writes: manuscript/analysis/out/figures/fig5.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from figutil import apply_style, panel_label

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG_DIR = os.path.join(HERE, "out", "figures")

C_SELF  = "#0072B2"   # trained on the target species
C_BASE  = "#9a9a9a"   # 1 kb baseline on the target species
C_ZERO  = "#E69F00"   # another species, zero-shot

FF = "funformer_{}_len20000_lr5e-4_b250_gac2.csv"
HUMAN_FF = "funformer_entex_423_len20000_lr5e-4_b250_gac2.csv"

HUMAN = [
    ("species-matched", [("FunFormer-20k (human)", HUMAN_FF, C_SELF)]),
    ("1 kb baselines",  [("FunFormer-1k", "funformer_entex_423_len1000_lr5e-4_b500_gac1.csv", C_BASE),
                         ("DeepATT", "DeepATT_5e-4_1000.csv", C_BASE),
                         ("DeepFormer", "DeepFormer_5e-4_1000.csv", C_BASE),
                         ("DeepSEA", "DeepSEA_1e-4_1000.csv", C_BASE)]),
    ("zero-shot",       [("cattle", FF.format("cattle"), C_ZERO),
                         ("pig", FF.format("pig"), C_ZERO),
                         ("chicken", FF.format("chicken"), C_ZERO),
                         ("salmon", FF.format("salmon"), C_ZERO),
                         ("zebrafish", FF.format("zebrafish"), C_ZERO)]),
]
PIG = [
    ("species-matched", [("FunFormer-20k (pig)", FF.format("pig"), C_SELF)]),
    ("1 kb baselines",  [("FunFormer-1k", "funformer_pig_len1000_lr5e-4_b500_gac1.csv", C_BASE),
                         ("DeepATT", "DeepATT_pig_5e-4_1000.csv", C_BASE),
                         ("DeepFormer", "DeepFormer_pig_5e-4_1000.csv", C_BASE),
                         ("DeepSEA", "DeepSEA_pig_5e-4_1000.csv", C_BASE)]),
    ("zero-shot",       [("cattle", FF.format("cattle"), C_ZERO),
                         ("human", "funformer_entex_423_len20000_lr5e-4_b250_gac2.csv", C_ZERO),
                         ("chicken", FF.format("chicken"), C_ZERO),
                         ("salmon", FF.format("salmon"), C_ZERO),
                         ("zebrafish", FF.format("zebrafish"), C_ZERO)]),
]


def _vals(base, fname):
    p = os.path.join(ROOT, "bench_results", base, fname)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)["mean_auc"].to_numpy(float)


def _panel(ax, base, groups, letter, title, ref_name, xlabel):
    # Groups are shown by the separator lines and named in the figure legend,
    # so no in-axes group labels: at this row pitch they collide with the row
    # names.
    rows, seps = [], []
    for gname, items in groups:
        got = [(n, f, c) for n, f, c in items if _vals(base, f) is not None]
        if not got:
            continue
        rows.extend(got)
        seps.append(len(rows) - 0.5)
    seps = seps[:-1]

    ref = _vals(base, dict((n, f) for n, f, _ in rows)[ref_name])
    ys = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(0)

    for y, (name, f, colour) in zip(ys, rows):
        v = _vals(base, f)
        bp = ax.boxplot(v, positions=[y], vert=False, widths=0.52,
                        patch_artist=True, showfliers=False,
                        manage_ticks=False, zorder=2)
        for b in bp["boxes"]:
            b.set(facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=0.8)
        for part in ("whiskers", "caps"):
            for el in bp[part]:
                el.set(color=colour, linewidth=0.8)
        for m in bp["medians"]:
            m.set(color="0.1", linewidth=1.6)
        # every tissue still shown, so the reader sees n and the spread the
        # box summarises
        ax.scatter(v, y + rng.uniform(-0.16, 0.16, len(v)), s=3.0,
                   color="0.15", alpha=0.40, linewidths=0, zorder=3)

    # Reference line at the MEDIAN of the species-matched model -- the same
    # statistic as the black bar on every row, so the line and the bars are
    # directly comparable.
    ax.axvline(np.median(ref), color=C_SELF, lw=1.0, ls="--", alpha=0.65, zorder=1)

    # The violins are marginal distributions, so their medians are NOT a paired
    # comparison: a row can sit further right than the reference and still lose
    # in most tissues, because the tissue at the middle of one distribution is
    # not the tissue at the middle of the other. Print the paired mean
    # difference against the species-matched model beside every row, which is
    # the quantity the text reports.
    for y, (name, f, colour) in zip(ys, rows):
        v = _vals(base, f)
        d = (v - ref).mean() * 100
        ax.annotate("" if abs(d) < 1e-9 else f"{d:+.1f}",
                    xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=6.8, va="center", ha="left",
                    color="0.25" if d >= 0 else "0.45")
    ax.annotate(r"$\Delta$ pp", xy=(1.0, ys[0]), xycoords=("axes fraction", "data"),
                xytext=(6, 13), textcoords="offset points", fontsize=6.5,
                va="center", ha="left", color="0.45")
    for s in seps:
        ax.axhline(len(rows) - 1 - s, color="0.85", lw=0.7, zorder=0)

    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=7.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="0.93", lw=0.6)
    panel_label(ax, letter, title)


def main():
    apply_style()
    # Stacked rather than side by side: the text block is ~6 in wide, so three
    # panels in a row would scale the labels below legibility. Stacking keeps
    # every panel the full width of the page.
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(7.0, 7.1))
    _panel(axA, "rf_eqtl", HUMAN, "a", "Human eQTL",
           "FunFormer-20k (human)", "eQTL random-forest AUROC")
    _panel(axB, "rf_pig_qtl/eqtl", PIG, "b", "Pig eQTL",
           "FunFormer-20k (pig)", "eQTL random-forest AUROC")
    _panel(axC, "rf_pig_qtl/sqtl", PIG, "c", "Pig sQTL",
           "FunFormer-20k (pig)", "sQTL random-forest AUROC")

    handles = [Line2D([], [], color=c, lw=6, alpha=0.45, label=l) for c, l in
               [(C_SELF, "trained on target species"), (C_BASE, "1 kb baseline"),
                (C_ZERO, "other species, zero-shot")]]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.005), columnspacing=1.6)
    fig.tight_layout(h_pad=1.4, rect=(0.02, 0.035, 0.965, 1))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig5.png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {FIG_DIR}/fig5.png")


if __name__ == "__main__":
    main()
