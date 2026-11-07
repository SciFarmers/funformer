#!/bin/bash
# =============================================================================
# Cross-species regulatory sequence classification — SLURM GPU job
#
# Runs src_evaluate.evaluate_cross_species:
#   loads a model from SOURCE_SPECIES, runs inference on TARGET_SPECIES test
#   set, computes AUROC for matched (assay, tissue) track pairs.
#
# ── HOW TO SUBMIT ────────────────────────────────────────────────────────────
# Use 8a.cross_species_eval.sh to submit all 48 jobs at once, or manually:
#
#   SOURCE_SPECIES=human \
#   SOURCE_MODEL_PATH=checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model \
#   TARGET_SPECIES=pig \
#   TARGET_DATA_DIR=/cluster/.../hf_pig_beds \
#   TARGET_GENOME=/cluster/.../Sus_scrofa.Sscrofa11.1.dna.toplevel.fa \
#   MODEL_TAG=FunFormer_20k \
#   sbatch --export=ALL olivia_submiter/evaluate_cross_species.sh
# =============================================================================
#SBATCH --account=nn9780k
#SBATCH --job-name=cross_species
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cross_species_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# =============================================================================
# PARAMETERIZED INPUTS
# =============================================================================
SOURCE_SPECIES="${SOURCE_SPECIES:-human}"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model}"
TARGET_SPECIES="${TARGET_SPECIES:-pig}"
TARGET_DATA_DIR="${TARGET_DATA_DIR:-/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_pig_beds}"
TARGET_GENOME="${TARGET_GENOME:-/cluster/projects/nn9780k/dat/training_data/genomes/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa}"
MODEL_TAG="${MODEL_TAG:-FunFormer_20k}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MATCHED_TRACKS="${MATCHED_TRACKS:-meta_info/matched_tracks.json}"

OUT_DIR="eval_results/cross_species/${SOURCE_SPECIES}_to_${TARGET_SPECIES}"

# =============================================================================
# FIXED PATHS
# =============================================================================
CONTAINER_PATH="/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif"

mkdir -p logs "${OUT_DIR}"

echo "============================================================"
echo " Cross-species Evaluation"
echo "  Source species  : ${SOURCE_SPECIES}"
echo "  Source model    : ${SOURCE_MODEL_PATH}"
echo "  Target species  : ${TARGET_SPECIES}"
echo "  Target data     : ${TARGET_DATA_DIR}"
echo "  Model tag       : ${MODEL_TAG}"
echo "  Output dir      : ${OUT_DIR}"
echo "============================================================"

apptainer exec --nv \
    --pwd /workspace \
    --bind "$PWD":/workspace \
    --bind "/cluster":/cluster \
    "${CONTAINER_PATH}" \
    python -m src_evaluate.evaluate_cross_species \
        --source-model-path "${SOURCE_MODEL_PATH}" \
        --source-species    "${SOURCE_SPECIES}" \
        --target-species    "${TARGET_SPECIES}" \
        --target-data-dir   "${TARGET_DATA_DIR}" \
        --target-genome     "${TARGET_GENOME}" \
        --matched-tracks    "${MATCHED_TRACKS}" \
        --model-tag         "${MODEL_TAG}" \
        --batch-size        "${BATCH_SIZE}" \
        --num-workers       "${NUM_WORKERS}" \
        --out-dir           "${OUT_DIR}"

echo "Done. Results in ${OUT_DIR}/"
