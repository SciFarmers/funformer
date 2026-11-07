"""The (direction, model) -> checkpoint map, derived, not retyped.

"Same checkpoints as the main analysis" is the hard requirement, so the
checkpoint for each (species, model) is selected exactly the way
10d.cross_species_zebrafish.sh selects it: the best test-AUROC learning-rate
run in eval_results/test_set/*_summary.csv. Retyping the paths would let this
analysis drift from the main one silently; deriving them cannot.

Selection is cross-checked against the paths 8a.cross_species_eval.sh hard-codes
(see check_against_8a), which is the one place the repo states them literally.

Jobs emitted, 75 in total:
  * 23 primary directions x 3 models -- FunFormer-20k, FunFormer-1k, and the
    one published baseline that won that direction unstratified (held fixed,
    so proximal and distal margins share a reference)
  * 6 within-species FunFormer-20k runs, the self-reference for retention
"""
import csv
import glob
import os
import re

import pandas as pd

# REPO_ROOT lets a frozen snapshot of this code run against the real
# repository, where the checkpoints and datasets live.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
TAG = {"human": "entex_423", "pig": "pig", "cattle": "cattle",
       "chicken": "chicken", "salmon": "salmon", "zebrafish": "zebrafish"}
DATA = {sp: f"/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_{t}_beds"
        for sp, t in {**TAG, "human": "ENTEX_423"}.items()}
GENOME = {
    "human": "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    "pig": "Sus_scrofa.Sscrofa11.1.dna.toplevel.fa",
    "cattle": "Bos_taurus.ARS-UCD1.2.dna.toplevel.fa",
    "chicken": "Gallus_gallus.GRCg6a.dna.toplevel.fa",
    "salmon": "Salmo_salar.Ssal_v3.1.dna.toplevel.fa",
    "zebrafish": "Danio_rerio.GRCz10.dna_sm.toplevel.fa",
}
GENOME = {k: f"/cluster/projects/nn9780k/dat/training_data/genomes/{v}"
          for k, v in GENOME.items()}


def _summaries():
    rows = []
    for f in glob.glob(os.path.join(ROOT, "eval_results/test_set/*_summary.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def _matches(exp, sp, model):
    tag = TAG[sp]
    if model.startswith("FunFormer"):
        want = "len1000_" if model == "FunFormer_1k" else "len20000"
        return "funformer" in exp and want in exp and f"_{tag}_" in exp
    if not exp.startswith(model + "_"):
        return False
    # human baselines carry no species infix: "DeepATT_5e-4_1000"
    if sp == "human":
        return re.fullmatch(rf"{model}_[0-9]e-[0-9]+_1000", exp) is not None
    return f"_{tag}_" in exp


def checkpoint(sp, model):
    best, best_a = None, -1.0
    for r in _summaries():
        if _matches(r["exp_name"], sp, model) and float(r["auroc_mean"]) > best_a:
            best_a, best = float(r["auroc_mean"]), r["exp_name"]
    if best is None:
        raise SystemExit(f"no test-set summary for {sp} {model}")
    return f"checkpoints/{best}/best_model", best_a


ALL_MODELS = ("FunFormer_20k", "FunFormer_1k", "DeepSEA", "DeepFormer", "DeepATT")


def jobs_all_models():
    """Every model on every computed direction, for the floor recompute.

    Applying a per-track positive floor changes which (source, target) replicate
    pairs enter each channel mean, so the group AUROCs have to be recomputed
    from predictions -- the committed per-match CSVs store only the already
    averaged value. That needs all five models on all 30 directions, including
    the chicken-source and fish-fish ones that Tables S1 and the sensitivity
    row rest on.
    """
    m = pd.read_csv(os.path.join(ROOT, "manuscript/analysis/out/cross_species_margins.csv"))
    out = []
    for _, r in m.iterrows():
        for model in ALL_MODELS:
            try:
                ck, _ = checkpoint(r["source"], model)
            except SystemExit:
                continue
            out.append(dict(source=r["source"], target=r["target"], model=model,
                            checkpoint=ck, kind="transfer"))
    for sp in TAG:
        ck, _ = checkpoint(sp, "FunFormer_20k")
        out.append(dict(source=sp, target=sp, model="FunFormer_20k",
                        checkpoint=ck, kind="self"))
    return pd.DataFrame(out)


def jobs():
    m = pd.read_csv(os.path.join(ROOT, "manuscript/analysis/out/cross_species_margins.csv"))
    out = []
    for _, r in m[m.in_primary_set].iterrows():
        for model in ("FunFormer_20k", "FunFormer_1k", r["best_baseline_model"]):
            ck, _ = checkpoint(r["source"], model)
            out.append(dict(source=r["source"], target=r["target"], model=model,
                            checkpoint=ck, kind="transfer"))
    for sp in TAG:                       # self-reference for retention
        ck, _ = checkpoint(sp, "FunFormer_20k")
        out.append(dict(source=sp, target=sp, model="FunFormer_20k",
                        checkpoint=ck, kind="self"))
    return pd.DataFrame(out)


def check_against_8a():
    """8a states the baseline and FunFormer-20k paths literally. Derived
    selection must reproduce them, or the two analyses are on different runs."""
    text = open(os.path.join(ROOT, "8a.cross_species_eval.sh")).read()
    stated = set(re.findall(r"checkpoints/(\S+?)/best_model", text))
    derived = {c.split("/")[1] for c in
               (checkpoint(sp, m)[0] for sp in TAG
                for m in ("DeepSEA", "DeepFormer", "DeepATT", "FunFormer_20k"))}
    missing = stated - derived
    return stated, derived, missing


if __name__ == "__main__":
    if "--tsv" in __import__("sys").argv:          # machine-readable, for submit_cache.sh
        j = jobs_all_models() if "--all-models" in __import__("sys").argv else jobs()
        print("source\ttarget\tmodel\tcheckpoint\tkind")
        for _, r in j.iterrows():
            print(f"{r.source}\t{r.target}\t{r.model}\t{r.checkpoint}\t{r.kind}")
        raise SystemExit

    stated, derived, missing = check_against_8a()
    print(f"8a states {len(stated)} checkpoints; derived set covers all: {not missing}")
    if missing:
        print("  NOT reproduced by derivation:", sorted(missing))
    j = jobs()
    print(f"\n{len(j)} jobs  ({(j.kind=='transfer').sum()} transfer, {(j.kind=='self').sum()} self)")
    print(j.groupby("model").size().to_string())
    print("\nsample:")
    print(j.head(4).to_string(index=False))
