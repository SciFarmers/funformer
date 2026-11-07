"""Supplementary Table S1: within-species test-set AUROC and AUPRC for five
models across six species.

For each (species, model) we take the best learning-rate run present in
eval_results/test_set/*_summary.csv (selected by AUROC). AUPRC for the same
run is reported alongside. The five method columns (DeepSEA, DeepFormer,
DeepATT, FunFormer-1k, FunFormer-20k) are shared by both metrics; each
species spans two rows, one for AUROC and one for AUPRC, with the row maximum
bolded independently per metric.

FunFormer-1k is the architecture- and parameter-matched control at the
baselines' own 1 kb context, so the table decomposes the comparison that a
FunFormer-20k-vs-1 kb-baseline column alone would confound (mean over the six
species: architecture +0.02 pp, context +1.81 pp):
    FunFormer-1k  vs the three baselines -> architecture effect (context fixed)
    FunFormer-20k vs FunFormer-1k        -> context-length effect (architecture fixed)
Establishes that the long-context model is the strongest backbone within
every genome, and why, before any cross-species claim is made.

Reads:  eval_results/test_set/*_summary.csv
Writes: manuscript/analysis/out/tables/tabS1_within_species.tex  (a booktabs
        tabular, \\input by the table float in main.tex)
"""
import os
import glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
ROOT = os.path.dirname(os.path.dirname(HERE))
# TEST_SET_DIR redirects to a tree built under a different evaluation
# convention (e.g. the 50-positive floor plus the N-flank exclusion).
SUMM = os.environ.get("TEST_SET_DIR") or os.path.join(ROOT, "eval_results", "test_set")
OUT = os.path.join(HERE, "out", "tables", "tabS1_within_species.tex")

# display order (roughly increasing divergence from human) and detection keys
SPECIES = [("Human", None), ("Pig", "pig"), ("Cattle", "cattle"),
           ("Chicken", "chicken"), ("Salmon", "salmon"),
           ("Zebrafish", "zebrafish")]
MODELS = [("DeepSEA", "DeepSEA"), ("DeepFormer", "DeepFormer"),
          ("DeepATT", "DeepATT"), ("FunFormer-1k", "len1000_"),
          ("FunFormer-20k", "len20000")]


def _species_of(name):
    n = name.lower()
    for label, key in SPECIES:
        if key and key in n:
            return label
    return "Human"


def _model_of(name):
    """Map an exp_name to a table column.

    FunFormer runs are matched on their context length, not on the bare string
    "funformer": every ablation length shares that prefix, so a len5000 or
    len1000 run would otherwise be counted as the 20 kb model of record (and,
    because the best run per cell is chosen by AUROC, could silently displace
    it). Lengths outside the table -- 5k/10k/30k/50k -- return None.
    """
    n = name.lower()
    if "funformer" in n:
        for label, key in MODELS:
            if key.startswith("len") and key in n:
                return label
        return None
    for label, key in MODELS:
        if not key.startswith("len") and key.lower() in n:
            return label
    return None


def collect():
    rows = []
    for f in glob.glob(os.path.join(SUMM, "*_summary.csv")):
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            sp = _species_of(r["exp_name"])
            md = _model_of(r["exp_name"])
            if md is None:
                continue
            # n_tracks_eval, not the trained track count: under the final
            # convention a target track needs 50 positives among the intervals
            # that survive the N-flank exclusion (human 401 of 423, salmon
            # 211 of 301).
            rows.append((sp, md, int(r["n_tracks_eval"]),
                         float(r["auroc_mean"]), float(r["auprc_mean"])))
    df = pd.DataFrame(rows, columns=["species", "model", "n", "auroc", "auprc"])
    # best learning-rate run per (species, model), selected by AUROC
    idx = df.groupby(["species", "model"])["auroc"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def _metric_block(sub, model_labels, metric):
    cells = {r["model"]: r[metric] for _, r in sub.iterrows()}
    best_val = max(cells.values())
    vals = []
    for m in model_labels:
        v = cells.get(m)
        if v is None:
            vals.append("--")
        elif abs(v - best_val) < 1e-9:
            vals.append(r"\textbf{%.3f}" % v)
        else:
            vals.append("%.3f" % v)
    return vals


def render(best):
    model_labels = [m for m, _ in MODELS]
    n_m = len(model_labels)
    lines = [
        r"\begin{tabular}{l r l " + "c " * n_m + "}",
        r"\toprule",
        r"Species & Tracks & Metric & " + " & ".join(model_labels) + r" \\",
        r"\midrule",
    ]
    for sp, _ in SPECIES:
        sub = best[best["species"] == sp]
        if sub.empty:
            continue
        n = int(sub["n"].iloc[0])
        auroc_vals = _metric_block(sub, model_labels, "auroc")
        auprc_vals = _metric_block(sub, model_labels, "auprc")
        lines.append(
            f"{sp} & {n} & AUROC & " + " & ".join(auroc_vals) + r" \\"
        )
        lines.append(
            " & & AUPRC & " + " & ".join(auprc_vals) + r" \\"
        )
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    best = collect()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(render(best))
    print(f"Wrote {OUT}")
    print(best.pivot(index="species", columns="model", values="auroc")
          .reindex([s for s, _ in SPECIES]).round(3).to_string())
    print(best.pivot(index="species", columns="model", values="auprc")
          .reindex([s for s, _ in SPECIES]).round(3).to_string())


if __name__ == "__main__":
    main()
