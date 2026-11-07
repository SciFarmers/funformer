"""Supplementary Table S3: is the cross-species gradient an artifact of which 1 kb
model is used as the reference, or of including chromatin accessibility?

The main text reports the margin of FunFormer-20k over the architecture- and
parameter-matched FunFormer-1k, on all five assays. Two choices there are
arguable. The 1 kb reference could instead be the best of the three published
convolutional baselines, or the best of all four 1 kb models (the strictest
comparison, since it takes whichever 1 kb model happens to win). And chromatin
accessibility (ATAC-seq, DNase-seq for chicken) could be excluded, leaving the
four histone marks.

This table crosses those two choices: three references x two assay sets. For
each it gives the mean margin in the three phylogenetic bins, the pooled
Pearson correlation with divergence time, and the species-level exact P.

Significance uses the same test as the main analysis: the 23 directions are
not independent, so the six taxa are relabelled onto the tree and every
divergence time recomputed, enumerated over all 6! = 720 relabellings.

Reads:  manuscript/analysis/out/cross_species_margins.csv
Writes: manuscript/analysis/out/tables/tabS3_margin_sensitivity.tex
"""

import itertools
import os

import numpy as np
import pandas as pd

from divergence_times import SPECIES, divergence

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
OUT = os.path.join(HERE, "out")
BINS = ["mammal-mammal", "mammal-chicken", "tetrapod-fish"]

ROWS = [
    ("FunFormer-1k (main text)",      "margin_arch_{}"),
    ("best published baseline",       "margin_{}"),
    ("best of all four 1 kb models",  "margin_best1kb_{}"),
]
ASSAY_SETS = [("all five assays", "all"), ("histone marks only", "noatac")]


# The exact test lives in stats_gradient so this table, Figure 3b and the
# statistics report cannot disagree.
from stats_gradient import species_exact_p_r as species_exact_p   # noqa: E402


def main():
    d = pd.read_csv(os.path.join(OUT, "cross_species_margins.csv"))
    p = d[d.in_primary_set]

    lines = [r"\begin{tabular}{l l r r r r r}", r"\toprule",
             r" &  & \multicolumn{3}{c}{Mean margin by bin (pp)} &  &  \\",
             r"\cmidrule(lr){3-5}",
             r"1 kb reference & Assays & Mammal & Chicken & Fish & $r$ & $P$ \\",
             r"\midrule"]
    for label, tmpl in ROWS:
        for i, (aset_label, aset) in enumerate(ASSAY_SETS):
            col = tmpl.format(aset)
            vals = [p[p.bin == b][col].mean() * 100 for b in BINS]
            r, pv = species_exact_p(p, col)
            shown = label if i == 0 else ""
            lines.append(
                f"{shown} & {aset_label} & "
                + " & ".join(f"${v:+.1f}$" for v in vals)
                + f" & ${r:+.2f}$ & {pv:.3f} \\\\")
        if label != ROWS[-1][0]:
            lines.append(r"\addlinespace")
    # Sensitivity: the original convention, before the N-flank exclusion and
    # the 50-positive track floor were applied. Kept so the effect of the two
    # filters on the headline gradient is visible in the table itself.
    raw = os.path.join(OUT, "cross_species_margins_nofloor.csv")
    if os.path.exists(raw):
        dr = pd.read_csv(raw)
        pr = dr[dr.in_primary_set]
        lines.append(r"\addlinespace")
        for i, (aset_label, aset) in enumerate(ASSAY_SETS):
            col = "margin_arch_{}".format(aset)
            vals = [pr[pr.bin == b][col].mean() * 100 for b in BINS]
            r, pv = species_exact_p(pr, col)
            shown = ("no filters (sensitivity)" if i == 0 else "")
            lines.append(
                f"{shown} & {aset_label} & "
                + " & ".join(f"${v:+.1f}$" for v in vals)
                + f" & ${r:+.2f}$ & {pv:.3f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    path = os.path.join(OUT, "tables", "tabS3_margin_sensitivity.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
