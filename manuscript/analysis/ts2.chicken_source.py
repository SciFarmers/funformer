"""Supplementary Table S2: chicken-as-source cross-species transfer margins,
reported as a sensitivity check outside the primary 23-direction set.

Chicken is used target-side only in the primary analysis (see Methods):
chicken's within-species FunFormer AUROC (0.876) was the weakest of the six
species, and chicken-source directions showed uniformly negative margins in
every case, which we attribute to a training-quality confound rather than a
phylogenetic signal. This table reports all five such directions
(chicken->human, chicken->pig, chicken->cattle, chicken->salmon,
chicken->zebrafish) for completeness, since they were computed as part of the
same pipeline; they are not used for the headline permutation/trend statistics.

Reads:  manuscript/analysis/out/cross_species_margins.csv
Writes: manuscript/analysis/out/tables/tabS2_chicken_source.tex
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
MARGINS = os.path.join(HERE, "out", "cross_species_margins.csv")
OUT = os.path.join(HERE, "out", "tables", "tabS2_chicken_source.tex")

# Every target for which a chicken-source direction was computed. Both fish
# are included: dropping zebrafish here would silently omit a direction that
# exists in the margins CSV.
TARGET_ORDER = ["human", "pig", "cattle", "salmon", "zebrafish"]


def render(df):
    chick = df[df["source"] == "chicken"].set_index("target").loc[TARGET_ORDER]

    # Include the architecture-matched control (FunFormer-1k) once it exists
    # for these directions, so the sensitivity table separates the same two
    # effects as the primary analysis.
    has_ff1k = ("funformer1k_all" in chick.columns
                and chick["funformer1k_all"].notna().all())

    if has_ff1k:
        lines = [
            r"\begin{tabular}{l c c c c l r r}",
            r"\toprule",
            r"Target & Divergence (Mya) & FunFormer-20k & FunFormer-1k & Best baseline AUROC "
            r"& Best baseline & Margin vs baseline (pp) & Margin vs FunFormer-1k (pp) \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{tabular}{l c c c l r}",
            r"\toprule",
            r"Target & Divergence (Mya) & FunFormer AUROC & Best baseline AUROC & Best baseline & Margin (pp) \\",
            r"\midrule",
        ]
    for tgt, row in chick.iterrows():
        div = int(row["divergence_mya"])
        ff = row["funformer20k_all"]
        bb = row["best_baseline_all"]
        bb_model = row["best_baseline_model"]
        margin = row["margin_all"] * 100
        if has_ff1k:
            f1 = row["funformer1k_all"]
            m_arch = row["margin_arch_all"] * 100
            lines.append(
                f"{tgt.capitalize()} & {div} & {ff:.3f} & {f1:.3f} & {bb:.3f} & {bb_model} "
                f"& ${margin:+.1f}$ & ${m_arch:+.1f}$ " + r"\\"
            )
        else:
            lines.append(
                f"{tgt.capitalize()} & {div} & {ff:.3f} & {bb:.3f} & {bb_model} & ${margin:+.1f}$ " + r"\\"
            )
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
