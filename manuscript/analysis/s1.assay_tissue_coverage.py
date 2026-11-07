"""Supplementary Figure S1: which assays and tissues each species actually
has, and
how many matched channels each species pair therefore shares.

Cross-species transfer is scored only on (assay, tissue) combinations that
exist in both the source and the target, so the comparison set differs from
pair to pair. Nothing in the main text shows what that set is. This figure
does.

Panel a: the coverage grid. One row per (tissue, assay) combination present in
at least two species, one column per species, the cell giving the number of
tracks. Empty cells are combinations the species does not have, and they are
the reason some pairs share far fewer channels than others.

Panel b: matched (assay, tissue) groups per species pair, read off
meta_info/matched_tracks.json -- the same file the evaluation uses, so the
figure cannot drift from what was actually scored.

Reads:  meta_info/*_track_names.txt, meta_info/metadata_entex.tsv,
        meta_info/matched_tracks.json
Writes: manuscript/analysis/out/figures/figS1.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figutil import apply_style, panel_label

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
FIG_DIR = os.path.join(HERE, "out", "figures")
META = os.path.join(os.path.dirname(os.path.dirname(HERE)), "meta_info")
sys.path.insert(0, META)
import find_matched_tracks as FMT  # noqa: E402

SP = FMT.SPECIES_ORDER
NICE = {"human": "Human", "pig": "Pig", "cattle": "Cattle",
        "chicken": "Chicken", "salmon": "Salmon", "zebrafish": "Zebrafish"}
ASSAYS = ["ATAC", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me3"]
CLADE_C = {"human": "#0072B2", "pig": "#0072B2", "cattle": "#0072B2",
           "chicken": "#009E73", "salmon": "#E69F00", "zebrafish": "#E69F00"}


def records():
    # find_matched_tracks resolves its inputs relative to the repository root,
    # so parse from there regardless of where this script is invoked.
    cwd = os.getcwd()
    os.chdir(os.path.dirname(META))
    try:
        return _records()
    finally:
        os.chdir(cwd)


def _records():
    f = FMT.TRACK_FILES
    return {"human": FMT.parse_human_tracks(f["human"], FMT.ENTEX_METADATA),
            "pig": FMT.parse_simple_tracks(f["pig"]),
            "cattle": FMT.parse_simple_tracks(f["cattle"]),
            "chicken": FMT.parse_chicken_tracks(f["chicken"]),
            "salmon": FMT.parse_salmon_tracks(f["salmon"]),
            "zebrafish": FMT.parse_zebrafish_tracks(f["zebrafish"])}


def panel_a(ax, T):
    tissues = sorted({r["tissue"] for recs in T.values() for r in recs})
    grid, labels, seps = [], [], []
    for t in tissues:
        block = []
        for a in ASSAYS:
            counts = [sum(1 for r in T[s] if r["assay"] == a and r["tissue"] == t)
                      for s in SP]
            if sum(1 for c in counts if c) >= 2:      # shared by >= 2 species
                block.append((f"{t}  ·  {a}", counts))
        if block:
            grid.extend(c for _, c in block)
            labels.extend(l for l, _ in block)
            seps.append(len(grid) - 0.5)
    M = np.array(grid, float)
    seps = seps[:-1]

    # Shade by presence, not by count: the count is printed, and a colour ramp
    # over 1..12 would imply the number of replicates matters for matching.
    ax.imshow(np.where(M > 0, 1.0, 0.0), aspect="auto", cmap="Blues",
              vmin=0, vmax=2.6, interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            ax.text(j, i, v if v else "·", ha="center", va="center",
                    fontsize=6.5, color="0.15" if v else "0.65")
    ax.set_xticks(range(len(SP)))
    ax.set_xticklabels([NICE[s] for s in SP], fontsize=7.5)
    for tick, s in zip(ax.get_xticklabels(), SP):
        tick.set_color(CLADE_C[s])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.2)
    for y in seps:
        ax.axhline(y, color="white", lw=1.6)
    ax.set_xticks(np.arange(len(SP) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel_label(ax, "a", "Tracks per assay and tissue")
    return len(labels)


def panel_b(ax):
    m = json.load(open(os.path.join(META, "matched_tracks.json")))
    n = len(SP)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(SP):
        for j, b in enumerate(SP):
            if i >= j:
                continue
            key = FMT.pair_key(a, b)
            if key in m:
                M[i, j] = M[j, i] = len(m[key])
    # "equal" ties the matrix width to its height, leaving panel b far
    # narrower than panel a; "auto" lets it fill the same width
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=60,
              interpolation="nearest")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="0.6")
            elif not np.isnan(M[i, j]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        fontsize=7.5,
                        color="white" if M[i, j] > 30 else "0.15")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([NICE[s] for s in SP], fontsize=7.5, rotation=40, ha="right")
    ax.set_yticklabels([NICE[s] for s in SP], fontsize=7.5)
    for tick, s in zip(ax.get_xticklabels(), SP): tick.set_color(CLADE_C[s])
    for tick, s in zip(ax.get_yticklabels(), SP): tick.set_color(CLADE_C[s])
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    # The matrix axes is square and narrow, so a centred title overruns the
    # left-aligned panel letter; place the letter in figure coordinates instead.
    ax.set_title("Matched (assay, tissue) groups per pair", fontsize=9, pad=8)
    ax.figure.text(0.055, ax.get_position().y1 + 0.012, "b",
                   fontweight="bold", fontsize=12, va="bottom")


def main():
    apply_style()
    T = records()
    fig = plt.figure(figsize=(7.2, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.05, 1.25], hspace=0.20)
    n_rows = panel_a(fig.add_subplot(gs[0]), T)
    panel_b(fig.add_subplot(gs[1]))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.96, bottom=0.06)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "figS1.png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {FIG_DIR}/figS1.png  ({n_rows} shared assay-tissue rows)")


if __name__ == "__main__":
    main()
