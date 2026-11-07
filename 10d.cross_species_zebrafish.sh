#!/bin/bash
# Cross-species transfer for the zebrafish WGD control.
#
# Zebrafish shares the ~435 Mya tetrapod/ray-finned-fish split with Atlantic
# salmon and the teleost WGD, but lacks the salmonid-specific Ss4R duplication.
# Transfers to the two fish are therefore matched on divergence time, so
# comparing them separates "transfer fails at this phylogenetic distance" from
# "transfer fails because the salmon genome was re-duplicated".
#
# Directions: zebrafish -> each of the other five, and each of the other five
# -> zebrafish, for all five models (3 baselines + FunFormer-20k + FunFormer-1k).
# Best-LR checkpoint per (species, model) is selected automatically from
# eval_results/test_set, the same rule used everywhere else in the paper.
#
# Usage: bash 10d.cross_species_zebrafish.sh
set -euo pipefail

declare -A DATA=(
  [human]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_ENTEX_423_beds"
  [pig]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_pig_beds"
  [cattle]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_cattle_beds"
  [salmon]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_salmon_beds"
  [chicken]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_chicken_beds"
  [zebrafish]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_zebrafish_beds"
)
declare -A GENOME=(
  [human]="/cluster/projects/nn9780k/dat/training_data/genomes/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
  [pig]="/cluster/projects/nn9780k/dat/training_data/genomes/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"
  [cattle]="/cluster/projects/nn9780k/dat/training_data/genomes/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
  [salmon]="/cluster/projects/nn9780k/dat/training_data/genomes/Salmo_salar.Ssal_v3.1.dna.toplevel.fa"
  [chicken]="/cluster/projects/nn9780k/dat/training_data/genomes/Gallus_gallus.GRCg6a.dna.toplevel.fa"
  [zebrafish]="/cluster/projects/nn9780k/dat/training_data/genomes/Danio_rerio.GRCz10.dna_sm.toplevel.fa"
)
SPECIES=(human pig cattle chicken salmon zebrafish)
MODEL_TAGS=(DeepSEA DeepFormer DeepATT FunFormer_20k FunFormer_1k)

# --- best-LR checkpoint per (species, model), from the test-set summaries ----
pick() {  # pick <species> <model_tag> -> checkpoint dir or empty
    /usr/bin/python3.12 - "$1" "$2" <<'PY'
import csv, glob, sys, os
sp, model = sys.argv[1], sys.argv[2]
tag = {"human":"entex_423"}.get(sp, sp)
def matches(exp):
    e = exp.lower()
    if model.startswith("FunFormer"):
        if "funformer" not in e: return False
        want = "len1000_" if model == "FunFormer_1k" else "len20000"
        return want in e and f"_{tag}_" in e
    return e.startswith(model.lower()+"_") and (f"_{tag}_" in e or (sp=="human" and "_zebrafish_" not in e and e.count("_")==2))
best, best_a = None, -1.0
for f in glob.glob("eval_results/test_set/*_summary.csv"):
    for r in csv.DictReader(open(f)):
        if matches(r["exp_name"]) and float(r["auroc_mean"]) > best_a:
            best_a, best = float(r["auroc_mean"]), r["exp_name"]
print(best or "")
PY
}

declare -A CKPT
for SP in "${SPECIES[@]}"; do
    for M in "${MODEL_TAGS[@]}"; do
        name="$(pick "$SP" "$M")"
        if [[ -n "$name" && -d "checkpoints/${name}/best_model" ]]; then
            CKPT["${SP}|${M}"]="checkpoints/${name}/best_model"
            printf "  %-10s %-14s -> %s\n" "$SP" "$M" "$name"
        else
            printf "  %-10s %-14s -> MISSING (skipped)\n" "$SP" "$M"
        fi
    done
done
echo

submit() {
    local SRC=$1 TGT=$2 M=$3
    local key="${SRC}|${M}"
    [[ -n "${CKPT[$key]:-}" ]] || { echo "skip ${SRC}->${TGT} ${M}: no checkpoint"; return; }
    SOURCE_SPECIES="${SRC}" SOURCE_MODEL_PATH="${CKPT[$key]}" \
    TARGET_SPECIES="${TGT}" TARGET_DATA_DIR="${DATA[$TGT]}" TARGET_GENOME="${GENOME[$TGT]}" \
    MODEL_TAG="${M}" sbatch --export=ALL olivia_submiter/evaluate_cross_species.sh >/dev/null
    echo "submitted ${SRC} -> ${TGT}  (${M})"
}

n=0
for M in "${MODEL_TAGS[@]}"; do
    for TGT in human pig cattle chicken salmon; do submit zebrafish "$TGT" "$M"; n=$((n+1)); done
    for SRC in human pig cattle chicken salmon; do submit "$SRC" zebrafish "$M"; n=$((n+1)); done
done
echo ""
echo "${n} zebrafish cross-species jobs submitted."
