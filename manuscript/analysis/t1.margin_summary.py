"""Table 1: cross-species transfer margins by phylogenetic bin, broken out per
assay.

Reports the mean transfer margin in each phylogenetic bin, per assay
(chromatin accessibility plus the four histone marks). The margin is
FunFormer-20k minus FunFormer-1k where the architecture- and parameter-matched
1 kb control is available, so the per-assay decay reflects context length
rather than architecture; it falls back to FunFormer-20k minus the best 1 kb
CNN baseline otherwise. Which definition was used is printed on stdout and
stated in the table's note. The pooled all-assay trend statistics (Pearson /
Jonckheere-Terpstra, with and without the salmon->chicken outlier) are
computed in stats_gradient.py and reported in the Results text and
out/stats_report.txt, not duplicated here.

Reads:  manuscript/analysis/out/cross_species_margins.csv
Writes: manuscript/analysis/out/tables/tab1_margin_summary.tex
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
MARGINS = os.path.join(HERE, "out", "cross_species_margins.csv")
OUT = os.path.join(HERE, "out", "tables", "tab1_margin_summary.tex")

BIN_ROWS = [
    ("mammal-mammal", r"Mammal $\leftrightarrow$ mammal", "58-94"),
    ("mammal-chicken", r"Mammal $\rightarrow$ chicken", "320"),
    ("tetrapod-fish", r"Tetrapod $\leftrightarrow$ fish", "435"),
]

ASSAYS = ["ATAC", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me3"]


def _assay_cols(prim):
    """Prefer the architecture-matched per-assay margins when complete.

    Returns (columns, label, matched) where `matched` says whether the
    architecture-matched definition (FunFormer-20k minus FunFormer-1k) is in
    use, as opposed to the published one (vs the best 1 kb CNN baseline).
    """
    arch = [f"margin_arch_{a}" for a in ASSAYS]
    have_control = ("margin_arch_all" in prim.columns
                    and prim["margin_arch_all"].notna().all())
    if have_control and all(c in prim.columns for c in arch):
        return list(zip(arch, ASSAYS)), "FunFormer-20k $-$ FunFormer-1k", True
    return ([(f"margin_{a}", a) for a in ASSAYS],
            "FunFormer-20k $-$ best 1 kb baseline", False)


def render(df):
    prim = df[df["in_primary_set"]].copy()
    ASSAY_COLS, margin_label, matched = _assay_cols(prim)
    print(f"Margin definition: {margin_label}"
          f"{' (architecture-matched)' if matched else ' (fallback: 1 kb control absent)'}")

    n_assay = len(ASSAY_COLS)
    n_cols = 3 + n_assay
    header_cells = " & ".join(label for _, label in ASSAY_COLS)
    lines = [
        r"\begin{tabular}{l c c " + "r " * n_assay + "}",
        r"\toprule",
        r" & & & " + r"\multicolumn{%d}{c}{Mean margin by assay (pp), %s} \\"
        % (n_assay, margin_label),
        r"\cmidrule(lr){4-%d}" % n_cols,
        r"Phylogenetic bin & Divergence (Mya) & $n$ & " + header_cells + r" \\",
        r"\midrule",
    ]
    for key, label, div in BIN_ROWS:
        sub = prim[prim["bin"] == key]
        n = len(sub)
        def cell(col):
            v = sub[col].mean() * 100
            if v != v:                      # NaN => assay absent for this bin
                return "--"
            # The tetrapod-fish H3K27me3 cell averages two halves that disagree:
            # seven salmon-source channels near zero, and five salmon-target
            # channels strongly negative on tracks carrying 115-200 positives in
            # 299,099 intervals. Flagged rather than silently averaged.
            mark = r"$^{\dagger}$" if (key == "tetrapod-fish"
                                       and col.endswith("H3K27me3")) else ""
            return f"${v:+.1f}$" + mark
        vals = " & ".join(cell(col) for col, _ in ASSAY_COLS)
        lines.append(f"{label} & {div} & {n} & " + vals + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    df = pd.read_csv(MARGINS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(render(df))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
