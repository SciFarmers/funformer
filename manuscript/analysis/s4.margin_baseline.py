"""Supplementary Figure S4: the same analysis as Figure 3a-c, with the
margin taken
against the best published 1 kb baseline instead of FunFormer-1k.

The main text uses the architecture- and parameter-matched FunFormer-1k as the
1 kb reference, so that the margin isolates context length. This figure shows
the originally published comparison -- against the best of DeepSEA, DeepFormer
and DeepATT -- to demonstrate the gradient does not depend on that choice.

Reads:  manuscript/analysis/out/cross_species_margins.csv and the raw
        matched-channel table built by 3.cross_species.py
Writes: manuscript/analysis/out/figures/figS4.png
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

REF = "best published baseline"


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0),
                             gridspec_kw={"width_ratios": [1.1, 1.0, 1.15]})
    CS.panel_margin(axes[0], "a", col="margin_all", ref=REF)
    CS.panel_perm(axes[1], "b", col="margin_all")
    CS.panel_assay_pair(axes[2], CS._raw_all(), "c",
                        ref_col="best_base_auroc", ref_lab=REF)
    fig.tight_layout(w_pad=3.0)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "figS4.png"), dpi=300, bbox_inches="tight")
    print(f"Wrote {FIG_DIR}/figS4.png")


if __name__ == "__main__":
    main()
