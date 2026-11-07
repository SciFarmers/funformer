"""Supplementary Table S4: absolute AUROC behind the stratified context margin.

Figure 4a reports a margin, FunFormer-20k minus FunFormer-1k, by phylogenetic
bin and TSS stratum. A margin cannot be read without its baseline, so this
table gives the two absolute AUROCs it is the difference of, and splits the
tetrapod-fish bin by transfer direction to show that the within-direction
contrast the figure relies on does not depend on pooling them.

Reads : analysis/tss_strata/out/directions_primary.tsv
Writes: manuscript/analysis/out/tables/tabS4_stratum_absolute.tex
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
OUT = os.path.join(HERE, "out")
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from divergence_times import CLADE  # noqa: E402

SRC = os.path.join(ROOT, "analysis", "tss_strata", "out", "directions_primary.tsv")
BINS = ["mammal-mammal", "mammal-chicken", "tetrapod-fish"]
LAB = {"mammal-mammal": r"Mammal $\leftrightarrow$ mammal",
       "mammal-chicken": r"Mammal $\rightarrow$ chicken",
       "tetrapod-fish": r"Tetrapod $\leftrightarrow$ fish"}


def rows(d):
    out = []
    for b in BINS:
        sub = d[d["bin"] == b]
        out.append((LAB[b], sub, ""))
        if b == "tetrapod-fish":
            for key, lab in (("tetrapod->fish", r"\quad into fish"),
                             ("fish->tetrapod", r"\quad out of fish")):
                out.append((lab, sub[sub["kind"] == key], r"\itshape"))
    return out


def main():
    d = pd.read_csv(SRC, sep="\t")
    d["kind"] = ["fish->tetrapod" if CLADE[s] == "fish" else "tetrapod->fish"
                 for s in d["source"]]
    lines = [r"\begin{tabular}{l c r r r r r}", r"\toprule",
             r" & & \multicolumn{2}{c}{Proximal ($\leq$2 kb)} & "
             r"\multicolumn{2}{c}{Distal ($>$10 kb)} & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Phylogenetic bin & $n$ & FF-20k & FF-1k & FF-20k & FF-1k & "
             r"Distal $-$ proximal \\", r"\midrule"]
    for lab, sub, style in rows(d):
        g = sub.groupby("stratum")[["auroc_ff20k", "auroc_ff1k", "margin_arch"]].mean()
        n = sub["stratum"].value_counts().max()
        contrast = g.loc["distal", "margin_arch"] - g.loc["proximal", "margin_arch"]
        cells = [f"{g.loc['proximal', 'auroc_ff20k']:.3f}",
                 f"{g.loc['proximal', 'auroc_ff1k']:.3f}",
                 f"{g.loc['distal', 'auroc_ff20k']:.3f}",
                 f"{g.loc['distal', 'auroc_ff1k']:.3f}",
                 f"${contrast:+.1f}$"]
        lines.append(f"{style} {lab} & {n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    p = os.path.join(OUT, "tables", "tabS4_stratum_absolute.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {p}")
    print(d.groupby(["bin", "stratum"])[["auroc_ff20k", "auroc_ff1k"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
