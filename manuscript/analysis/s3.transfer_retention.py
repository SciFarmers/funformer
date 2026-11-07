"""Supplementary Figure S3: how much accuracy a transfer gives up against
the model
trained on the target genome itself.

The main text reports absolute transfer AUROC by phylogenetic bin, which mixes
two things: how lossy the transfer is, and how predictable the target genome is
in the first place. Chicken, for example, transfers to a low absolute AUROC
largely because chicken is the hardest genome for any model, including its own.

The reference here is the target species' own FunFormer-20k, scored on exactly
the channels that direction is evaluated on -- not its all-track test average.
For each matched (assay, tissue) group the target model's per-track AUROCs are
averaged over that group's target track indices, then averaged over groups, so
self and transfer are measured on the same channels and the same test
intervals.

Panel a: per direction, the target model's own AUROC and the transferred
AUROC, joined. Panel b: the retained fraction, transfer / self, against
divergence time.

Reads:  $TEST_SET_DIR/*_per_track.csv,
        meta_info/matched_tracks.json,
        manuscript/analysis/out/cross_species_margins.csv
Writes: manuscript/analysis/out/figures/figS3.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figutil import apply_style, panel_label

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
OUT = os.path.join(HERE, "out")
FIG_DIR = os.path.join(OUT, "figures")
ROOT = os.path.dirname(os.path.dirname(HERE))
# The panel is a ratio of a transferred AUROC to the target species' own model.
# Both sides must come from the same evaluation convention, so this honours
# TEST_SET_DIR exactly as t1 and 2.model_performance do.
TS = os.environ.get("TEST_SET_DIR") or os.path.join(ROOT, "eval_results", "test_set")
META = os.path.join(ROOT, "meta_info")
sys.path.insert(0, META)
import find_matched_tracks as FMT  # noqa: E402

CKPT = {"human": "funformer_entex_423_len20000_lr5e-4_b250_gac2",
        "pig": "funformer_pig_len20000_lr5e-4_b250_gac2",
        "cattle": "funformer_cattle_len20000_lr5e-4_b250_gac2",
        "chicken": "funformer_chicken_len20000_lr5e-4_b250_gac2",
        "salmon": "funformer_salmon_len20000_lr5e-4_b250_gac2",
        "zebrafish": "funformer_zebrafish_len20000_lr5e-4_b250_gac2"}
BINS = ["mammal-mammal", "mammal-chicken", "tetrapod-fish"]
BIN_LAB = {"mammal-mammal": "Mammal ↔ mammal",
           "mammal-chicken": "Mammal → chicken",
           "tetrapod-fish": "Tetrapod ↔ fish"}
C_SELF = "#8d8f94"
# Keyed to the clade of the non-mammal partner so the colours agree with the
# coverage figure: mammal blue, bird green, fish orange.
BIN_C = {"mammal-mammal": "#0072B2",
         "mammal-chicken": "#009E73",
         "tetrapod-fish": "#E69F00"}
BIN_MK = {"mammal-mammal": "o", "mammal-chicken": "s", "tetrapod-fish": "^"}
NICE = {"human": "human", "pig": "pig", "cattle": "cattle",
        "chicken": "chicken", "salmon": "salmon", "zebrafish": "zebrafish"}


def table():
    matched = json.load(open(os.path.join(META, "matched_tracks.json")))
    per = {s: pd.read_csv(os.path.join(TS, f"{c}_per_track.csv"))
                 .set_index("track_idx")["auroc"]
           for s, c in CKPT.items()}
    d = pd.read_csv(os.path.join(OUT, "cross_species_margins.csv"))
    rows = []
    for _, r in d[d.in_primary_set].iterrows():
        src, tgt = r["source"], r["target"]
        groups = matched[FMT.pair_key(src, tgt)]
        per_group = []
        for g in groups:
            v = [per[tgt].get(i, np.nan) for i in g[tgt]]
            v = [x for x in v if x == x]
            if v:
                per_group.append(float(np.mean(v)))
        self_auc = float(np.mean(per_group))
        rows.append({"source": src, "target": tgt, "div": float(r["divergence_mya"]),
                     "bin": r["bin"], "transfer": float(r["funformer20k_all"]),
                     "self": self_auc})
    t = pd.DataFrame(rows)
    t["retained"] = t["transfer"] / t["self"] * 100
    t["drop_pp"] = (t["self"] - t["transfer"]) * 100
    return t.sort_values(["div", "source", "target"]).reset_index(drop=True)


def panel_a(ax, t):
    order, seps, labels = [], [], []
    for b in BINS:
        sub = t[t.bin == b].sort_values("transfer")
        order.extend(sub.index.tolist())
        labels.extend(f"{NICE[r.source]} → {NICE[r.target]}"
                      for r in sub.itertuples())
        seps.append(len(order) - 0.5)
    seps = seps[:-1]
    y = np.arange(len(order))[::-1]
    s = t.loc[order]
    ax.hlines(y, s["transfer"], s["self"], color="0.75", lw=1.4, zorder=1)
    ax.scatter(s["self"], y, s=26, c=C_SELF, zorder=3,
               label="target's own model")
    # The transferred point carries the bin's colour and marker, the same
    # encoding as panel b, so a row can be placed without reading the labels.
    for b in BINS:
        mask = (s["bin"] == b).to_numpy()
        ax.scatter(s["transfer"][mask], y[mask], s=32, marker=BIN_MK[b],
                   c=BIN_C[b], edgecolors="0.15", linewidths=0.5, zorder=3,
                   label=BIN_LAB[b])
    for yy, dp in zip(y, s["drop_pp"]):
        ax.annotate(f"−{dp:.1f}", xy=(1.0, yy),
                    xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=6.5, va="center", ha="left", color="0.35")
    ax.annotate("drop (pp)", xy=(1.0, y[0]), xycoords=("axes fraction", "data"),
                xytext=(6, 14), textcoords="offset points", fontsize=6.5,
                va="center", ha="left", color="0.45")
    for ss in seps:
        ax.axhline(len(order) - 1 - ss, color="0.85", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel("Matched-channel AUROC")
    ax.set_xlim(0.50, 1.0)
    ax.xaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left",
              handletextpad=0.3, labelspacing=0.35)
    panel_label(ax, "a")


def panel_b(ax, t):
    for b in BINS:
        sub = t[t.bin == b]
        ax.scatter(sub["div"] + np.random.default_rng(0).uniform(-6, 6, len(sub)),
                   sub["retained"], s=32, marker=BIN_MK[b], c=BIN_C[b],
                   label=BIN_LAB[b], edgecolors="0.15", linewidths=0.5, zorder=3)
    for b in BINS:
        sub = t[t.bin == b]
        ax.plot([sub["div"].iloc[0] - 16, sub["div"].iloc[0] + 16],
                [sub["retained"].mean()] * 2, color="0.25", lw=1.6, zorder=4)
    ax.set_xlabel("Divergence time (Mya)")
    ax.set_ylabel("Accuracy retained (%)")
    ax.set_xlim(30, 465)
    ax.set_ylim(55, 102)
    ax.axhline(100, color="0.6", ls="--", lw=0.8, zorder=1)
    ax.yaxis.grid(True, color="0.93", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    # Labels sit to the left of each bin's mean line: the fish directions are
    # tightly clustered at 435 Mya, so an offset above the line lands inside
    # the points.
    for b in BINS:
        sub = t[t.bin == b]
        ax.annotate(f"{sub['retained'].mean():.0f}%",
                    xy=(sub["div"].iloc[0] - 18, sub["retained"].mean()),
                    xytext=(-3, 0), textcoords="offset points",
                    fontsize=8, ha="right", va="center",
                    color="0.2", fontweight="bold")
    panel_label(ax, "b")


def main():
    apply_style()
    t = table()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    panel_a(ax1, t)
    panel_b(ax2, t)
    fig.tight_layout(w_pad=3.0, rect=(0, 0, 0.985, 1))
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "figS3.png"), dpi=300, bbox_inches="tight")
    g = t.groupby("bin")[["transfer", "self", "drop_pp", "retained"]].mean()
    print(g.round(3).to_string())
    print(f"\nWrote {FIG_DIR}/figS3.png")


if __name__ == "__main__":
    main()
