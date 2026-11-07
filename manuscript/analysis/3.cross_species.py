"""Figure 3: the context advantage decays with evolutionary divergence.

Panel a (panel_margin): the headline quantity. Every primary direction's
matched-channel margin, FunFormer-20k minus FunFormer-1k -- architecture and
parameter count held fixed, so what decays is attributable to context length
and not to architecture. Points are grouped into the three phylogenetic bins,
jittered within a bin, and sized by the number of matched channels behind the
direction mean; the bar and whisker are the bin mean and its 95% CI.
Panel b (panel_perm): the species-level exact test behind that decay. All
6! = 720 relabellings of the six taxa onto the tree, with the observed r
marked; the shaded tail is the count at least as extreme.
Panel c (panel_assay_pair): the same margin resolved by assay, drawn as
paired boxes (filled = FunFormer-20k, open = the reference) so both the gap
and the absolute level it sits at are visible. Per-assay absolutes exist only
at raw channel resolution, so this panel reads _raw_all() rather than the
direction-level margins file.
Panel d (panel_context_gradient): the within-architecture gradient for the
five human-source directions -- FunFormer at 1 and 20 kb, same architecture
and parameter count, so only context length varies. The intermediate 5 kb
point is deliberately not drawn: it is not monotone for the two fish, which
reads as a feature of the gradient when it is an artefact of sampling one
intermediate length.

panel_margin, panel_perm and panel_assay_pair are parameterised by margin
column, so s4.margin_baseline.py redraws all three against the best published
baseline (margin_all) for the companion supplementary figure. panel_absolute
is not used by this figure at all -- it draws absolute transfer AUROC against
divergence and is panel a of s2.absolute_transfer.py, kept here because it
shares this module's loaders.

Reads:  eval_results/cross_species/<src>_to_<tgt>/<model>_per_match.csv
        manuscript/analysis/out/cross_species_margins.csv
Writes: manuscript/analysis/out/figures/fig3.png
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from divergence_times import divergence, pair_bin, CLADE, SPECIES
from figutil import apply_style, panel_label, CLADE_COLOR
from stats_gradient import species_null_r

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
ROOT = os.path.dirname(os.path.dirname(HERE))
import sys as _sys
_sys.path.insert(0, os.path.join(ROOT, "analysis", "common"))
from ci import mean_ci  # noqa: E402
# CS_DIR redirects to a tree built under a different evaluation convention,
# matching aggregate_cross_species.py. Panels a and b read the margins CSV,
# which is already on the final convention; c and d read this tree directly,
# so without the override one figure would straddle two conventions.
CS = os.environ.get("CS_DIR") or os.path.join(ROOT, "eval_results", "cross_species")
FIG_DIR = os.path.join(HERE, "out", "figures")

BASE = ["DeepSEA", "DeepFormer", "DeepATT"]
FF20K, FF1K = "FunFormer_20k", "FunFormer_1k"
FF_COLOR, BASE_COLOR = "#0072B2", "#8a8a8a"
FF1K_COLOR = "#56B4E9"

BINS_ORDER = ["mammal-mammal", "mammal-chicken", "tetrapod-fish"]
BIN_LAB = {"mammal-mammal": "Mammal\n\u2194 mammal\n58\u201394 Mya",
           "mammal-chicken": "Mammal\n\u2192 chicken\n320 Mya",
           "tetrapod-fish": "Tetrapod\n\u2194 fish\n435 Mya"}
BIN_C = {"mammal-mammal": "#0072B2", "mammal-chicken": "#009E73",
         "tetrapod-fish": "#E69F00"}
BIN_MK = {"mammal-mammal": "o", "mammal-chicken": "s", "tetrapod-fish": "^"}
SEED = 0


def _load(direction, model):
    f = os.path.join(CS, direction, f"{model}_per_match.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def _mean(df, assay=None):
    if df is None:
        return np.nan
    df = df if assay is None else df[df.assay == assay]
    return df.auroc.mean() if len(df) else np.nan


def _primary():
    """Direction-level means (n=23), used only to fit panel a's trend lines."""
    rows = []
    for d in sorted(glob.glob(os.path.join(CS, "*_to_*"))):
        name = os.path.basename(d)
        src, tgt = name.split("_to_")
        if (src == "chicken" or pair_bin(src, tgt) == "fish-fish"
                or _load(name, "FunFormer_20k") is None):
            continue
        ff = _load(name, "FunFormer_20k")
        row = dict(name=name, src=src, tgt=tgt, div=divergence(src, tgt),
                   bin=pair_bin(src, tgt), ff=_mean(ff),
                   ff1k=_mean(_load(name, FF1K)),
                   best_base=np.nanmax([_mean(_load(name, b)) for b in BASE]))
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_all():
    """One row per (direction, assay, tissue) matched channel in the primary
    set, with both FunFormer-20k and best-baseline AUROC on that exact
    channel (no averaging across tissues)."""
    rows = []
    for d in sorted(glob.glob(os.path.join(CS, "*_to_*"))):
        name = os.path.basename(d)
        src, tgt = name.split("_to_")
        if src == "chicken" or pair_bin(src, tgt) == "fish-fish":
            continue
        ff = _load(name, "FunFormer_20k")
        if ff is None:
            continue
        base_frames = [b for b in (_load(name, m) for m in BASE) if b is not None]
        if not base_frames:
            continue
        base_df = pd.concat(base_frames, ignore_index=True)
        base_best = (base_df.groupby(["assay", "tissue"])["auroc"]
                     .max().rename("best_base_auroc"))
        merged = ff.merge(base_best, on=["assay", "tissue"], how="inner")
        merged = merged.rename(columns={"auroc": "ff_auroc"})
        merged["margin"] = merged["ff_auroc"] - merged["best_base_auroc"]
        # architecture-matched margin on the same channel, when FunFormer-1k
        # has been run for this direction
        ff1k = _load(name, FF1K)
        if ff1k is not None:
            ff1k_ch = (ff1k.groupby(["assay", "tissue"])["auroc"]
                       .mean().rename("ff1k_auroc"))
            merged = merged.merge(ff1k_ch, on=["assay", "tissue"], how="left")
            merged["margin_arch"] = merged["ff_auroc"] - merged["ff1k_auroc"]
        else:
            merged["ff1k_auroc"] = np.nan
            merged["margin_arch"] = np.nan
        merged["div"] = divergence(src, tgt)
        merged["bin"] = pair_bin(src, tgt)
        rows.append(merged[["assay", "tissue", "div", "bin", "ff_auroc",
                            "best_base_auroc", "ff1k_auroc",
                            "margin", "margin_arch"]])
    return pd.concat(rows, ignore_index=True)


def _fit_line(ax, x, y, color):
    m = ~(np.isnan(x) | np.isnan(y))
    b1, b0 = np.polyfit(x[m], y[m], 1)
    xs = np.array([x[m].min() - 20, x[m].max() + 20])
    ax.plot(xs, b0 + b1 * xs, color=color, lw=2.2, zorder=4)
    return np.corrcoef(x[m], y[m])[0, 1]


def _margins():
    return pd.read_csv(os.path.join(HERE, "out", "cross_species_margins.csv"))



def _channels_per_direction():
    """Matched (assay, tissue) channels behind each direction's mean.

    Directions rest on 8 to 40 channels, so the direction means they feed are
    not equally precise; panel a sizes its points by this count rather than
    drawing them all alike.
    """
    out = {}
    for d in sorted(glob.glob(os.path.join(CS, "*_to_*"))):
        name = os.path.basename(d)
        src, tgt = name.split("_to_")
        ff = _load(name, "FunFormer_20k")
        f1 = _load(name, FF1K)
        if ff is None or f1 is None:
            continue
        keys = set(map(tuple, ff[["assay", "tissue"]].values)) & \
               set(map(tuple, f1[["assay", "tissue"]].values))
        out[(src, tgt)] = len(keys)
    return out


def panel_margin(ax, letter="a", col="margin_arch_all", ref="FunFormer-1k"):
    """Context margin by phylogenetic bin, one point per direction.

    Points are jittered horizontally and sized by the number of matched
    channels behind that direction's mean (8 to 40). The mean and its 95%
    confidence interval sit beside the points rather than on top of them.

    Shown by bin rather than against continuous divergence time because
    divergence takes only four values, and a regression through them invites
    reading a crossing point out of the 226 Myr gap between 94 and 320 Mya
    where nothing was measured.
    """
    P = _margins()
    P = P[P.in_primary_set]
    nch = _channels_per_direction()
    for i, b in enumerate(BINS_ORDER):
        sub = P[P.bin == b].sort_values(col)
        v = sub[col].to_numpy(float) * 100
        n = len(v)
        counts = np.array([nch.get((r.source, r.target), 10)
                           for r in sub.itertuples()], float)
        sizes = 14 + 46 * (counts - counts.min()) / max(float(counts.max() - counts.min()), 1.0)
        rng = np.random.default_rng(4 + i)
        xs = i - 0.08 + rng.uniform(-0.20, 0.20, n)
        ax.scatter(xs, v, s=sizes, marker=BIN_MK[b], c=BIN_C[b],
                   edgecolors="0.15", linewidths=0.5, alpha=0.88, zorder=3)
        m, lo, hi = mean_ci(v)
        ax.plot([i + 0.22, i + 0.40], [m, m], color="0.15", lw=2.6, zorder=4)
        ax.plot([i + 0.31, i + 0.31], [lo, hi],
                color="0.15", lw=1.3, zorder=4)
        ax.annotate(f"{m:+.1f}", xy=(i + 0.44, m), fontsize=8.5,
                    fontweight="bold", va="center", ha="left", color="0.15")
    ax.axhline(0, color="0.45", ls="--", lw=1.0, zorder=1)
    ax.set_xticks([i - 0.08 for i in range(3)])
    counts_per_bin = [int((P.bin == b).sum()) for b in BINS_ORDER]
    ax.set_xticklabels([f"{BIN_LAB[b]}\nn={c}"
                        for b, c in zip(BINS_ORDER, counts_per_bin)],
                       fontsize=7)
    ax.set_xlim(-0.62, 2.9)
    ax.set_ylabel(f"Margin (pp)\nFunFormer-20k \u2212 {ref}")
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=BIN_C[b], marker=BIN_MK[b], linestyle="",
                      markeredgecolor="0.15", markeredgewidth=0.5, markersize=6,
                      label=" ".join(BIN_LAB[b].split("\n")[:2]))
               for b in BINS_ORDER]
    # the n counts moved into the tick labels, so the lower-left corner is
    # free; it is the emptiest region since only the fish bin reaches below -5
    leg = ax.legend(handles=handles, frameon=True, fontsize=6.8,
                    loc="lower left", handletextpad=0.5, borderpad=0.6,
                    labelspacing=0.45)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_linewidth(0.7)
    panel_label(ax, letter, "Context margin by divergence")


def panel_perm(ax, letter="b", col="margin_arch_all"):
    """The test itself: every relabelling of the six taxa onto the tree."""
    P = _margins()
    P = P[P.in_primary_set]
    rs, obs = species_null_r(P, col)
    # Two-sided on |r|, matching the test the statistics report runs.
    n_ex = int((np.abs(rs) >= abs(obs) - 1e-12).sum())
    bins = np.linspace(rs.min() - 0.02, rs.max() + 0.02, 44)
    ax.hist(rs, bins=bins, color="0.82", edgecolor="white", linewidth=0.5)
    ax.hist(rs[np.abs(rs) >= abs(obs) - 1e-12], bins=bins, color="#0072B2",
            edgecolor="white", linewidth=0.5)
    ax.axvline(obs, color="#a8402c", lw=1.8, zorder=5)
    ax.annotate(f"observed\n$r={obs:.3f}$", xy=(obs, ax.get_ylim()[1] * 0.92),
                xytext=(10, 0), textcoords="offset points", fontsize=7.5,
                color="#a8402c", va="top", ha="left")
    ax.annotate(f"{n_ex} of 720 relabelings\nat least as extreme\n"
                f"$P={n_ex/720:.3f}$",
                xy=(0.34, 0.97), xycoords="axes fraction", fontsize=7.5,
                va="top", ha="left", color="0.2")
    ax.set_xlabel("$r$(margin, divergence) under relabeling")
    ax.set_ylabel("Relabelings")
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, letter, "Species-level permutation null")


def panel_assay_pair(ax, raw, letter="c", ref_col="ff1k_auroc",
                     ref_lab="FunFormer-1k"):
    """Per assay and bin, paired boxes over the matched channels.

    Each box summarises 12 to 35 channels; the pooled means this panel used to
    show hid how wide those distributions are, particularly at the fish
    distance where channel AUROCs run from below 0.4 to above 0.9.
    """
    order = ["ATAC", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me3"]
    for j, a in enumerate(order):
        for i, b in enumerate(BINS_ORDER):
            sub = raw[(raw.assay == a) & (raw.bin == b)]
            if not len(sub):
                continue
            xx = j + (i - 1) * 0.28
            for col, filled in [("ff_auroc", True), (ref_col, False)]:
                v = sub[col].dropna().to_numpy()
                if not len(v):
                    continue
                px = xx + (-0.068 if filled else 0.068)
                bp = ax.boxplot(v, positions=[px], widths=0.105,
                                patch_artist=True, showfliers=False,
                                manage_ticks=False, zorder=4)
                for bx in bp["boxes"]:
                    bx.set(facecolor=BIN_C[b] if filled else "white",
                           alpha=0.55 if filled else 1.0,
                           edgecolor=BIN_C[b], linewidth=1.0)
                for part in ("whiskers", "caps"):
                    for el in bp[part]:
                        el.set(color=BIN_C[b], linewidth=0.9)
                for md in bp["medians"]:
                    md.set(color="0.12", linewidth=1.5)
                rng = np.random.default_rng(7)
                ax.scatter(px + rng.uniform(-0.052, 0.052, len(v)), v, s=4.0,
                           c=BIN_C[b], alpha=0.42, linewidths=0, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=7.5)
    ax.set_ylabel("Transfer AUROC")
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="s", linestyle="", markersize=7,
                      markerfacecolor="0.45", markeredgecolor="0.25",
                      alpha=0.6, label="FunFormer-20k"),
               Line2D([], [], marker="s", linestyle="", markersize=7,
                      markerfacecolor="white", markeredgecolor="0.45",
                      label=ref_lab)]
    leg = ax.legend(handles=handles, frameon=True, fontsize=6.8,
                    loc="lower right", handletextpad=0.5, borderpad=0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_linewidth(0.7)
    panel_label(ax, letter, "By assay (color as in a)")


def panel_context_gradient(ax, letter="d"):
    rng = np.random.default_rng(SEED)
    xs = [0, 1]
    jit = 0.06
    dy = {"pig": -0.02, "cattle": 0.02, "chicken": 0.0,
          "zebrafish": 0.013, "salmon": -0.013}

    # Two endpoints only: FunFormer at 1 kb and at 20 kb, one architecture and
    # one parameter count throughout, so the only thing that changes is context
    # length. The intermediate 5 kb point is deliberately not shown: it is not
    # monotone for the two fish (both rise slightly at 5 kb before falling at
    # 20 kb), which reads as a feature of the gradient when it is an artefact of
    # sampling one intermediate length, and the claim here is about the endpoints.
    TARGETS = ["pig", "cattle", "chicken", "salmon", "zebrafish"]
    use_ff1k = all(_load(f"human_to_{t}", FF1K) is not None for t in TARGETS)

    for tgt in TARGETS:
        name = f"human_to_{tgt}"
        if use_ff1k:
            first = _load(name, FF1K)["auroc"].values
        else:
            base_frames = [b for b in (_load(name, m) for m in BASE) if b is not None]
            base_df = pd.concat(base_frames, ignore_index=True)
            first = base_df.groupby(["assay", "tissue"])["auroc"].max().values
        ff20 = _load(name, FF20K)
        if ff20 is None:
            continue

        c = CLADE_COLOR[CLADE[tgt]]
        means = []
        for xi, vals in zip(xs, [first, ff20["auroc"].values]):
            jitter = rng.uniform(-jit, jit, size=len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals, s=8, color=c,
                       alpha=0.4, linewidths=0, zorder=2)
            means.append(vals.mean())
        ax.plot(xs, means, marker="o", ms=6, lw=1.8, color=c,
                markeredgecolor="0.15", markeredgewidth=0.5, zorder=3)
        ax.annotate(tgt.capitalize(), xy=(1, means[-1]),
                    xytext=(1.08, means[-1] + dy[tgt]), ha="left", va="center",
                    fontsize=8, color=c)

    ax.set_xlim(-0.2, 1.62)
    ax.set_xticks(xs)
    ax.set_xticklabels(["1 kb", "20 kb"] if use_ff1k
                       else ["1 kb\n(baseline)", "20 kb"])
    ax.set_xlabel("FunFormer context length" if use_ff1k else "Context length")
    ax.set_ylabel("Transfer AUROC (raw)")
    panel_label(ax, letter, title="Transfer from human (source species)")


def panel_absolute(ax, means):
    # Panel a of Supplementary Fig. S2. Direction-level means only (n=23) --
    # kept aggregated on purpose; panel b of that same figure refits the
    # regression on the raw matched channels.
    #
    # Divergence time takes only four distinct values, so all three series sit
    # on the same four verticals and overplot completely -- at 435 Mya that is
    # 14 directions x 3 models in one column. Each series is therefore dodged
    # into its own lane and jittered within it; the fits below are computed on
    # the true x, not the dodged one.
    x = means["div"].values.astype(float)
    has_ff1k = means["ff1k"].notna().all()
    rng = np.random.default_rng(0)
    lanes = [(-11.0, "ff", FF_COLOR, "o", 34, "FunFormer-20k"),
             (0.0, "best_base", BASE_COLOR, "s", 30, "Best 1 kb baseline")]
    if has_ff1k:
        lanes.append((11.0, "ff1k", FF1K_COLOR, "^", 32, "FunFormer-1k"))

    r = {}
    for off, col, colour, mk, size, _ in lanes:
        xd = x + off + rng.uniform(-3.5, 3.5, len(x))
        ax.scatter(xd, means[col], s=size, c=colour, marker=mk,
                   edgecolors="0.15", linewidths=0.5, alpha=0.9, zorder=3)
        r[col] = _fit_line(ax, x, means[col].values, colour)

    handles = [Line2D([], [], color=c, marker=mk, linestyle="-", lw=2.2,
                      markeredgecolor="0.15", markeredgewidth=0.5,
                      markersize=np.sqrt(sz) * 0.95,
                      label=f"{lab}  ($r={r[col]:.2f}$)")
               for _, col, c, mk, sz, lab in lanes]
    # Framed, on a solid ground: an unframed legend over an empty corner of a
    # scatter reads as data. The correlations live in the labels rather than in
    # a separate floating annotation, which was the other thing to mistake for
    # a data label.
    leg = ax.legend(handles=handles, frameon=True, fontsize=7.5,
                    loc="upper right", handletextpad=0.6, borderpad=0.6,
                    labelspacing=0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.85")
    leg.get_frame().set_linewidth(0.7)
    leg.set_zorder(6)

    ax.set_xlabel("Divergence time (Mya)")
    ax.set_ylabel("Transfer AUROC")
    ax.set_xlim(30, 470)
    ax.set_xticks([58, 94, 320, 435])
    ax.xaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, "a")


def main():
    apply_style()
    means = _primary()
    raw = _raw_all()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4))
    panel_margin(axes[0, 0], "a")
    panel_perm(axes[0, 1], "b")
    panel_assay_pair(axes[1, 0], raw, "c")
    panel_context_gradient(axes[1, 1], "d")
    fig.tight_layout(w_pad=3.0, h_pad=3.2)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig3.png"),
                dpi=300, bbox_inches="tight")
    print(f"Wrote {FIG_DIR}/fig3.png")


if __name__ == "__main__":
    main()
