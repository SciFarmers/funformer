"""Supplementary Figure S2: absolute transfer AUROC, and whether the regression
behind it depends on being fitted to direction means.

Panel a was the lead panel of Figure 3 until the figure was reorganised around
the margin. The absolute level is worth reporting, but it makes a poorer lead:
it conflates how lossy a transfer is with how predictable the target genome is
at all (Supplementary Fig. S3 separates those), and its most visible feature,
the point where the FunFormer-20k line crosses the two 1 kb references, falls
in the 226 Myr gap between 94 and 320 Mya where no direction was measured.

Panel b refits that same regression on all 359 raw (direction, assay, tissue)
matched channels instead of the 23 direction-level means. Channels within a
direction share a source genome, a target genome and a model checkpoint, so
they are not independent draws and the main analysis averages them away before
fitting. Both fits agree on sign and magnitude; the raw-point r is weaker,
which is the expected signature of averaging away channel-level noise rather
than evidence the two analyses disagree.

Both panels use the primary set: chicken-source and fish-fish directions
excluded, as everywhere else.

Reads:  eval_results/cross_species/<src>_to_<tgt>/<model>_per_match.csv
Writes: manuscript/analysis/out/figures/figS2.png
"""
import importlib.util
import os

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

_spec = importlib.util.spec_from_file_location(
    "cs", os.path.join(HERE, "3.cross_species.py"))
CS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CS)

JIT, SEED = 7.0, 0


def _fit(ax, x, y, color, ls):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    b1, b0 = np.polyfit(x[m], y[m], 1)
    xs = np.array([x[m].min() - 20, x[m].max() + 20])
    ax.plot(xs, b0 + b1 * xs, color=color, lw=2.0, ls=ls, zorder=4)
    return np.corrcoef(x[m], y[m])[0, 1]


def panel_b(ax):
    """Raw matched channels against the direction-level fit."""
    raw = CS._raw_all()                      # primary set: 23 directions, 359 channels
    means = CS._primary()
    rng = np.random.default_rng(SEED)
    xj = raw["div"].to_numpy(float) + rng.uniform(-JIT, JIT, len(raw))
    ax.scatter(xj, raw["ff_auroc"], s=8, c=CS.FF_COLOR, alpha=0.30,
               linewidths=0, zorder=1)
    ax.scatter(xj, raw["best_base_auroc"], s=8, c=CS.BASE_COLOR, marker="s",
               alpha=0.30, linewidths=0, zorder=1)
    r_ff_m = _fit(ax, means["div"], means["ff"], CS.FF_COLOR, "-")
    r_bb_m = _fit(ax, means["div"], means["best_base"], CS.BASE_COLOR, "-")
    r_ff_r = _fit(ax, raw["div"], raw["ff_auroc"], CS.FF_COLOR, "--")
    r_bb_r = _fit(ax, raw["div"], raw["best_base_auroc"], CS.BASE_COLOR, "--")
    ax.plot([], [], color="0.2", lw=2.0, ls="-",
            label=f"fit on direction means (n={len(means)})")
    ax.plot([], [], color="0.2", lw=2.0, ls="--",
            label=f"fit on raw channels (n={len(raw)})")
    ax.scatter([], [], c=CS.FF_COLOR, label="FunFormer-20k")
    ax.scatter([], [], c=CS.BASE_COLOR, marker="s", label="Best 1 kb baseline")
    ax.set_xlabel("Divergence time (Mya)")
    ax.set_ylabel("Transfer AUROC")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax.annotate(f"means  $r={r_ff_m:.2f}$ / ${r_bb_m:.2f}$\n"
                f"raw    $r={r_ff_r:.2f}$ / ${r_bb_r:.2f}$",
                xy=(0.97, 0.97), xycoords="axes fraction", fontsize=7,
                ha="right", va="top", color="0.25")
    panel_label(ax, "b", "Raw channels against the direction-level fit")
    return r_ff_m, r_bb_m, r_ff_r, r_bb_r


def main():
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    CS.panel_absolute(ax1, CS._primary())
    panel_label(ax1, "a", "Absolute transfer AUROC")
    rs = panel_b(ax2)
    fig.tight_layout(w_pad=3.0)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "figS2.png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {FIG_DIR}/figS2.png   "
          f"means r={rs[0]:.3f}/{rs[1]:.3f}  raw r={rs[2]:.3f}/{rs[3]:.3f}")


if __name__ == "__main__":
    main()
