"""Aggregate cross-species transfer results into per-direction margins.

For every directed species pair (source -> target) we compute, per model, the
mean matched-track AUROC (i) over all assays and (ii) excluding the chromatin
accessibility assay group ("ATAC" -- which for chicken is silently remapped
DNase-seq, a technology confound). The headline quantity is the *margin*:

    margin = AUROC(FunFormer_20k)  -  max AUROC over the three 1 kb baselines

joined to the source<->target divergence time. Positive margin = the
long-context model wins the transfer; negative = it loses.

Because that margin compares a 20 kb transformer against 1 kb CNN/CNN-LSTM
baselines, it confounds architecture with context length. FunFormer_1k -- the
same architecture and the same parameter count (n_tokens is pinned at 500 for
every context length), trained with the same optimization at the baselines'
own 1 kb context -- separates the two, so three margins are emitted per
convention:

    margin_all       = FunFormer_20k - best of the three 1 kb baselines
                       (unchanged; the originally published quantity)
    margin_arch_all  = FunFormer_20k - FunFormer_1k
                       (context-length effect, architecture held fixed)
    margin_best1kb_all = FunFormer_20k - best of ALL FOUR 1 kb models
                       (strictest: is 20 kb better than any 1 kb model?)

plus arch_gain_all = FunFormer_1k - best baseline, the architecture effect at
matched 1 kb context. Every one of these also has a _noatac variant and a
per-assay variant. Columns are NaN when FunFormer_1k is not present for a
direction, so the file stays readable before those jobs land.

Primary analysis set (n = 23 of the 30 computed directions):
  * 6 mammal <-> mammal directed pairs (human/pig/cattle with each other)
  * 3 mammal -> chicken directions (chicken is used target-side only)
  * 14 tetrapod <-> fish directions (both fish: salmon and zebrafish)
Two groups are held out, both flagged in the CSV rather than deleted:
  * 5 chicken -> X directions (chicken_source), excluded from the primary claim
    because chicken's source model is the weakest of the six (a training-quality
    confound, not a phylogenetic signal); reported in Supplementary Table S1.
  * 2 salmon <-> zebrafish directions (fish_fish), a single ~206 Mya species
    pair whose two directions disagree in sign; pooling it with the 435 Mya
    pairs would also place a 206 Mya comparison at the far end of the
    divergence axis.

Reads:  eval_results/cross_species/<src>_to_<tgt>/<model>_per_match.csv
Writes: analysis/out/cross_species_margins.csv
"""
import os
import glob
import pandas as pd

from divergence_times import divergence, pair_bin, CLADE

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root is two levels up now: manuscript/analysis -> manuscript -> repo.
# REPO_ROOT overrides it when this runs from a frozen code snapshot, so inputs
# and outputs resolve to the repository rather than to the snapshot's copy.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(HERE))
# Which per-match tables to aggregate, and what to call the result. The
# 50-positive floor is applied upstream by analysis/floor50, which writes a
# parallel tree of per-match CSVs; pointing CS_DIR at it makes the floored
# convention primary without changing anything downstream.
CS_DIR = os.environ.get("CS_DIR") or os.path.join(ROOT, "eval_results", "cross_species")
OUT_DIR = os.path.join(ROOT, "manuscript", "analysis", "out")
OUT_NAME = os.environ.get("MARGINS_NAME", "cross_species_margins.csv")

BASELINES = ["DeepSEA", "DeepFormer", "DeepATT"]
FF20K = "FunFormer_20k"
FF1K = "FunFormer_1k"   # architecture- and parameter-matched 1 kb control
ACCESSIBILITY_ASSAY = "ATAC"  # DNase-seq is remapped to this label upstream
ASSAYS = ["ATAC", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me3"]


def _mean_auroc(per_match_csv, exclude_atac=False, assay=None):
    """Mean matched-track AUROC for one (direction, model) per_match file."""
    df = pd.read_csv(per_match_csv)
    if assay is not None:
        df = df[df["assay"] == assay]
    elif exclude_atac:
        df = df[df["assay"] != ACCESSIBILITY_ASSAY]
    if len(df) == 0:
        return float("nan")
    return df["auroc"].mean()


def _model_means(direction_dir, exclude_atac=False, assay=None):
    """Return {model: mean_auroc} for every model present in a direction dir."""
    means = {}
    for path in glob.glob(os.path.join(direction_dir, "*_per_match.csv")):
        model = os.path.basename(path).replace("_per_match.csv", "")
        means[model] = _mean_auroc(path, exclude_atac=exclude_atac, assay=assay)
    return means


def _row_for_direction(direction_dir):
    name = os.path.basename(direction_dir)
    src, tgt = name.split("_to_")

    all_means = _model_means(direction_dir, exclude_atac=False)
    noa_means = _model_means(direction_dir, exclude_atac=True)
    if FF20K not in all_means:
        return None  # nothing to compare

    def margin(means):
        """Return every margin flavour for one {model: mean_auroc} dict."""
        ff = means.get(FF20K, float("nan"))
        ff1k = means.get(FF1K, float("nan"))
        base = [means[b] for b in BASELINES if b in means]
        best = max(base) if base else float("nan")
        best_model = BASELINES[base.index(best)] if base else None
        # strictest 1 kb reference: best of the three baselines AND FunFormer-1k
        cands = list(base) + ([ff1k] if ff1k == ff1k else [])   # NaN-safe
        best1kb = max(cands) if cands else float("nan")
        return {
            "ff": ff, "ff1k": ff1k, "best_base": best, "best_base_model": best_model,
            "margin": ff - best,              # published: vs best CNN baseline
            "margin_arch": ff - ff1k,         # context effect, architecture fixed
            "margin_best1kb": ff - best1kb,   # vs best of all four 1 kb models
            "arch_gain": ff1k - best,         # architecture effect at 1 kb
        }

    m_all = margin(all_means)
    m_noa = margin(noa_means)

    # chicken is used target-side only in the primary set; chicken as source is
    # excluded (weak source model). The single fish-fish pair
    # (salmon<->zebrafish, ~206 Mya) is also excluded: one species pair, two
    # directions that disagree in sign, six matched channels -- too thin to
    # report or to let influence the trend statistics.
    chicken_source = (src == "chicken")
    fish_fish = pair_bin(src, tgt) == "fish-fish"
    in_primary = not chicken_source and not fish_fish

    row = {
        "source": src,
        "target": tgt,
        "src_clade": CLADE[src],
        "tgt_clade": CLADE[tgt],
        "divergence_mya": divergence(src, tgt),
        "bin": pair_bin(src, tgt),
        "chicken_target": tgt == "chicken",
        "chicken_source": chicken_source,
        "fish_fish": fish_fish,
        "in_primary_set": in_primary,
        "funformer20k_all": m_all["ff"],
        "funformer1k_all": m_all["ff1k"],
        "best_baseline_all": m_all["best_base"],
        "best_baseline_model": m_all["best_base_model"],
        "margin_all": m_all["margin"],
        "margin_arch_all": m_all["margin_arch"],
        "margin_best1kb_all": m_all["margin_best1kb"],
        "arch_gain_all": m_all["arch_gain"],
        "funformer20k_noatac": m_noa["ff"],
        "funformer1k_noatac": m_noa["ff1k"],
        "best_baseline_noatac": m_noa["best_base"],
        "margin_noatac": m_noa["margin"],
        "margin_arch_noatac": m_noa["margin_arch"],
        "margin_best1kb_noatac": m_noa["margin_best1kb"],
        "arch_gain_noatac": m_noa["arch_gain"],
    }

    # Per-assay margins (one column per assay) for the assay-resolved table.
    for a in ASSAYS:
        m_a = margin(_model_means(direction_dir, assay=a))
        row[f"margin_{a}"] = m_a["margin"]
        row[f"margin_arch_{a}"] = m_a["margin_arch"]
        row[f"margin_best1kb_{a}"] = m_a["margin_best1kb"]

    return row


def build():
    rows = []
    for d in sorted(glob.glob(os.path.join(CS_DIR, "*_to_*"))):
        if os.path.isdir(d):
            row = _row_for_direction(d)
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        ["in_primary_set", "divergence_mya", "source", "target"],
        ascending=[False, True, True, True],
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, OUT_NAME)
    df.to_csv(out, index=False)
    return df, out


if __name__ == "__main__":
    df, out = build()
    prim = df[df["in_primary_set"]]
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"Wrote {out}  ({len(df)} directions, {len(prim)} in primary set)\n")
    cols = ["source", "target", "divergence_mya", "bin",
            "margin_all", "margin_arch_all", "margin_best1kb_all", "margin_noatac"]
    print("PRIMARY SET (n=%d):" % len(prim))
    print(prim[cols].to_string(index=False))
    print("\nGroup means (primary set):")
    summary_cols = ["margin_all", "margin_arch_all", "margin_best1kb_all",
                    "arch_gain_all", "margin_noatac", "margin_arch_noatac"]
    print(prim.groupby("bin")[summary_cols].mean()
          .reindex(["mammal-mammal", "mammal-chicken", "tetrapod-fish"]).to_string())
