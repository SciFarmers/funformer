#!/bin/bash
# Cross-species transfer evaluation for FunFormer 1 kb — the architecture- and
# parameter-matched control for the context-length claim.
#
# With FunFormer_1k present in every direction, the regression analysis gains
# two fair comparisons that the published FunFormer_20k-vs-CNN-baseline margin
# could not make:
#   FunFormer_20k - FunFormer_1k  -> context-length effect, architecture held fixed
#   FunFormer_1k  - best baseline -> architecture effect, context held fixed
#
# Jobs: 16 primary directions (all ordered pairs among human/pig/cattle/salmon,
# plus X -> chicken) + 4 chicken-source directions for Supplementary Table S1.
#
# The source checkpoint per species is the best-AUROC learning-rate run, chosen
# automatically from eval_results/test_set/ (same rule as every other model in
# the paper). Run 9b first.
#
# Output: eval_results/cross_species/<src>_to_<tgt>/FunFormer_1k_{per_match,summary}.csv
#
# Usage: bash 9c.cross_species_funformer1k.sh
set -euo pipefail

MODEL_TAG_NAME="FunFormer_1k"

declare -A DATA=(
  [human]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_ENTEX_423_beds"
  [pig]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_pig_beds"
  [cattle]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_cattle_beds"
  [salmon]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_salmon_beds"
  [chicken]="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_chicken_beds"
)
declare -A GENOME=(
  [human]="/cluster/projects/nn9780k/dat/training_data/genomes/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
  [pig]="/cluster/projects/nn9780k/dat/training_data/genomes/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"
  [cattle]="/cluster/projects/nn9780k/dat/training_data/genomes/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
  [salmon]="/cluster/projects/nn9780k/dat/training_data/genomes/Salmo_salar.Ssal_v3.1.dna.toplevel.fa"
  [chicken]="/cluster/projects/nn9780k/dat/training_data/genomes/Gallus_gallus.GRCg6a.dna.toplevel.fa"
)
declare -A TAG=(
  [human]="entex_423" [cattle]="cattle" [pig]="pig" [salmon]="salmon" [chicken]="chicken"
)

# --- pick the best-AUROC LR run per species from the within-species summaries --
declare -A BEST_CKPT
for SP in human pig cattle salmon chicken; do
    CKPT=$(python3 - "$SP" "${TAG[$SP]}" <<'PY'
import csv, glob, sys, os
sp, tag = sys.argv[1], sys.argv[2]
best, best_auroc = None, -1.0
for f in glob.glob(f"eval_results/test_set/funformer_{tag}_len1000_lr*_summary.csv"):
    with open(f) as fh:
        for row in csv.DictReader(fh):
            a = float(row["auroc_mean"])
            if a > best_auroc:
                best_auroc, best = a, row["exp_name"]
print(f"{best}\t{best_auroc:.4f}" if best else "")
PY
)
    if [[ -z "$CKPT" ]]; then
        echo "ERROR: no eval_results/test_set summary for ${SP} FunFormer-1k. Run 9b first." >&2
        exit 1
    fi
    NAME="${CKPT%%$'\t'*}"; AUROC="${CKPT##*$'\t'}"
    BEST_CKPT[$SP]="checkpoints/${NAME}/best_model"
    printf "best FunFormer-1k for %-8s : %s  (test AUROC %s)\n" "$SP" "$NAME" "$AUROC"
    [[ -d "${BEST_CKPT[$SP]}" ]] || { echo "ERROR: missing ${BEST_CKPT[$SP]}" >&2; exit 1; }
done
echo

submit() {
    local SRC_SP=$1 TGT_SP=$2
    SOURCE_SPECIES="${SRC_SP}" \
    SOURCE_MODEL_PATH="${BEST_CKPT[$SRC_SP]}" \
    TARGET_SPECIES="${TGT_SP}" \
    TARGET_DATA_DIR="${DATA[$TGT_SP]}" \
    TARGET_GENOME="${GENOME[$TGT_SP]}" \
    MODEL_TAG="${MODEL_TAG_NAME}" \
    sbatch --export=ALL olivia_submiter/evaluate_cross_species.sh
    echo "submitted ${SRC_SP} -> ${TGT_SP}"
}

# --- 16 primary directions --------------------------------------------------
for SRC in human pig cattle salmon; do
    for TGT in human pig cattle salmon chicken; do
        [[ "$SRC" == "$TGT" ]] && continue
        submit "$SRC" "$TGT"
    done
done

# --- 4 chicken-source directions (Supplementary Table S1 only) --------------
for TGT in human pig cattle salmon; do
    submit chicken "$TGT"
done

echo ""
echo "20 FunFormer-1k cross-species jobs submitted (16 primary + 4 chicken-source)."
