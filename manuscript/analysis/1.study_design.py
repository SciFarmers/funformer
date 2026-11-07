"""Figure 1: study design.

Panel A: a time-calibrated phylogeny of the six species, with node ages read
from divergence_times.py (single source of truth), tips coloured by clade
(mammal / bird / fish), and nested clade brackets naming the three groupings
the text uses -- the mammals sit inside the tetrapods, and the fish are the
outgroup. Panel B: the cross-species evaluation pipeline, from
per-species training to the transfer margin against divergence time.

Writes: manuscript/analysis/out/figures/fig1.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from divergence_times import divergence, CLADE
from figutil import apply_style, panel_label, CLADE_COLOR

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
FIG_DIR = os.path.join(HERE, "out", "figures")

# (tracks trained on, tracks evaluated). They differ where a track has no
# positive or no negative interval in the test split, or carries fewer than 50
# positives among the intervals that survive the N-flank exclusion.
N_TRACKS = {"human": (423, 401), "pig": (80, 80), "cattle": (95, 95),
            "chicken": (97, 97), "salmon": (301, 211), "zebrafish": (37, 37)}

# node ages (Mya), from the divergence module
T_PC = divergence("pig", "cattle")        # 58
T_HPC = divergence("human", "pig")        # 94
T_AMNIOTE = divergence("human", "chicken")  # 320
T_ROOT = divergence("human", "salmon")    # 435
T_FISH = divergence("salmon", "zebrafish")  # 206 -- Otophysi/Protacanthopterygii

# tip y-positions (top to bottom) and the fixed 6-taxon topology
TIP_Y = {"zebrafish": 6, "salmon": 5, "chicken": 4, "human": 3, "pig": 2, "cattle": 1}
Y_PC = (TIP_Y["pig"] + TIP_Y["cattle"]) / 2
Y_HPC = (TIP_Y["human"] + Y_PC) / 2
Y_AMN = (Y_HPC + TIP_Y["chicken"]) / 2
Y_FISH = (TIP_Y["salmon"] + TIP_Y["zebrafish"]) / 2
Y_ROOT = (Y_AMN + Y_FISH) / 2


def draw_phylogeny(ax):
    # internal nodes: vertical connector + horizontal branch to each child
    def node(nx, children):
        ys = [c[2] for c in children]
        ax.plot([nx, nx], [min(ys), max(ys)], color="0.2", lw=1.4, zorder=2)
        for cx, _, cy in children:
            ax.plot([nx, cx], [cy, cy], color="0.2", lw=1.4, zorder=2)

    node(T_ROOT, [(T_AMNIOTE, None, Y_AMN), (T_FISH, None, Y_FISH)])
    node(T_FISH, [(0, "salmon", TIP_Y["salmon"]), (0, "zebrafish", TIP_Y["zebrafish"])])
    node(T_AMNIOTE, [(T_HPC, None, Y_HPC), (0, "chicken", TIP_Y["chicken"])])
    node(T_HPC, [(0, "human", TIP_Y["human"]), (T_PC, None, Y_PC)])
    node(T_PC, [(0, "pig", TIP_Y["pig"]), (0, "cattle", TIP_Y["cattle"])])

    # Node-age labels sit just past the node, away from the branch. The
    # pig-cattle node is only 36 Myr from its parent, so a label placed on that
    # side runs straight through the 94 Mya connector; it goes on the tip side
    # instead, into the empty wedge between the pig and cattle branches.
    for nx, ny in [(T_HPC, Y_HPC), (T_AMNIOTE, Y_AMN),
                   (T_FISH, Y_FISH), (T_ROOT, Y_ROOT)]:
        ax.annotate(f"{nx:.0f} Mya", xy=(nx, ny), xytext=(nx + 8, ny + 0.14),
                    ha="right", va="bottom", fontsize=7.5, color="0.35")
    ax.annotate(f"{T_PC:.0f} Mya", xy=(T_PC - 6, Y_PC), ha="left", va="center",
                fontsize=7.5, color="0.35")

    # tips: coloured marker + species name + track count
    tip_texts = []
    for sp, y in TIP_Y.items():
        c = CLADE_COLOR[CLADE[sp]]
        ax.scatter([0], [y], s=52, color=c, edgecolors="0.15", linewidths=0.6,
                   zorder=3, clip_on=False)
        n_in, n_ev = N_TRACKS[sp]
        lab = (f"  {sp.capitalize()} ({n_in} tracks)" if n_in == n_ev
               else f"  {sp.capitalize()} ({n_in} tracks, {n_ev} evaluated)")
        tip_texts.append(
            ax.annotate(lab, xy=(0, y), ha="left", va="center", fontsize=9, color=c))

    # Clade brackets, nested to match the grouping used throughout the text:
    # the mammals sit inside the tetrapods, and the fish are the outgroup.
    # Two columns rather than one so the nesting is visible as nesting.
    def bracket(bx, y0, y1, label, color, arm=13, lw=1.2):
        ax.plot([bx, bx], [y0, y1], color=color, lw=lw, clip_on=False, zorder=4)
        for y in (y0, y1):
            ax.plot([bx, bx + arm], [y, y], color=color, lw=lw,
                    clip_on=False, zorder=4)
        ax.annotate(label, xy=(bx - 10, (y0 + y1) / 2), ha="center",
                    va="center", rotation=90, fontsize=8.5, color=color,
                    clip_on=False, zorder=4)

    # Place the brackets from the MEASURED extent of the tip labels rather than
    # from a guessed offset: the labels change length whenever the track counts
    # do, and a fixed offset has twice let them collide with the brackets.
    ax.set_xlim(T_ROOT + 30, -420)          # provisional, wide enough to measure in
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    label_edge = min(inv.transform((t.get_window_extent(renderer=rend).x1, 0))[0]
                     for t in tip_texts)
    BX_IN = label_edge - 28
    BX_OUT = BX_IN - 60
    bracket(BX_IN, TIP_Y["zebrafish"], TIP_Y["salmon"], "Fish",
            CLADE_COLOR["fish"])
    bracket(BX_IN, TIP_Y["human"], TIP_Y["cattle"], "Mammals",
            CLADE_COLOR["mammal"])
    bracket(BX_OUT, TIP_Y["chicken"], TIP_Y["cattle"], "Tetrapods", "0.35")

    ax.set_xlim(T_ROOT + 30, BX_OUT - 40)   # past on the left, present on right
    ax.set_ylim(0.4, 6.7)
    ax.set_yticks([])
    ax.set_xticks([0, 100, 200, 300, 400])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Divergence time (Mya)")
    panel_label(ax, "a")


BOX_W = 0.205


def _box(ax, x, text, h=0.52, fc="#eef2f5"):
    ax.add_patch(FancyBboxPatch(
        (x - BOX_W / 2, 0.5 - h / 2), BOX_W, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9, edgecolor="0.35", facecolor=fc, zorder=2))
    ax.text(x, 0.5, text, ha="center", va="center", fontsize=7.5, zorder=3)


def draw_pipeline(ax):
    xs = [0.12, 0.373, 0.627, 0.88]
    texts = [
        "Train per species,\nFunFormer at 1 and 20 kb\nand three 1 kb baselines",
        "Apply the source\nmodel to the target\nspecies genome",
        "Score matched\n(assay, tissue)\nchannels by AUROC",
        "Transfer margin vs\ndivergence time\n(Figs. 3, 5)",
    ]
    fcs = ["#eef2f5", "#eef2f5", "#eef2f5", "#fdf1e7"]
    for x, t, fc in zip(xs, texts, fcs):
        _box(ax, x, t, fc=fc)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch(
            (x0 + BOX_W / 2 + 0.006, 0.5), (x1 - BOX_W / 2 - 0.006, 0.5),
            arrowstyle="-|>", mutation_scale=12, lw=1.1, color="0.35", zorder=1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b")


def main():
    apply_style()
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(7.0, 5.4), gridspec_kw={"height_ratios": [3.1, 1.0]})
    draw_phylogeny(axA)
    draw_pipeline(axB)
    fig.tight_layout(h_pad=1.2)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "fig1.png"),
                dpi=300, bbox_inches="tight")
    print(f"Wrote {FIG_DIR}/fig1.png")


if __name__ == "__main__":
    main()
