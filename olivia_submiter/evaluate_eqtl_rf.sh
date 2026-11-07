#!/bin/bash
# =============================================================================
# RF Evaluation of VEP scores — SLURM CPU job
#
# Runs after vep_eqtl_finemap.sh has produced all score CSVs.
# No GPU needed.
#
# ── HOW TO SUBMIT ────────────────────────────────────────────────────────────
# DeepFormer (default):
#   sbatch olivia_submiter/evaluate_eqtl_rf.sh
#
# Other models:
#   MODEL_TAG=DeepSEA_5e-4_1000 sbatch --export=ALL olivia_submiter/evaluate_eqtl_rf.sh
#   MODEL_TAG=DanQ_5e-4_1000    sbatch --export=ALL olivia_submiter/evaluate_eqtl_rf.sh
# =============================================================================
#SBATCH --account=nn9780k
#SBATCH --job-name=rf_eqtl
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/rf_eqtl_%j.out

cd "${SLURM_SUBMIT_DIR}"

# =============================================================================
# PARAMETERIZED INPUTS (override via --export=ALL,VAR=value)
# =============================================================================
MODEL_TAG="${MODEL_TAG:-DeepFormer_5e-4_1000}"
SCORE_DIR="${SCORE_DIR:-bench_results/vep_eqtl/${MODEL_TAG}}"
OUT="${OUT:-bench_results/rf_eqtl/${MODEL_TAG}.csv}"
N_REPEATS="${N_REPEATS:-10}"
CV_FOLDS="${CV_FOLDS:-10}"

# =============================================================================
# FIXED PATHS
# =============================================================================
CONTAINER_PATH="/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif"

mkdir -p logs "$(dirname "${OUT}")"

echo "============================================================"
echo " RF Evaluation — eQTL Fine-mapping"
echo "  Score dir   : ${SCORE_DIR}"
echo "  Output      : ${OUT}"
echo "  Repeats     : ${N_REPEATS}   Folds: ${CV_FOLDS}"
echo "============================================================"

N_CSV=$(ls -1 "${SCORE_DIR}"/*.csv.gz 2>/dev/null | wc -l)
if [[ "${N_CSV}" -eq 0 ]]; then
    echo "ERROR: No CSV files found in ${SCORE_DIR}"
    echo "  Run vep_eqtl_finemap.sh first."
    exit 1
fi
echo "Found ${N_CSV} score CSV files."

apptainer exec \
    --pwd /workspace \
    --bind "$PWD":/workspace \
    --bind "/cluster":/cluster \
    "${CONTAINER_PATH}" \
    python -m src_evaluate.evaluate_eqtl_rf \
        --score-dir  "${SCORE_DIR}" \
        --out        "${OUT}" \
        --n-repeats  "${N_REPEATS}" \
        --cv-folds   "${CV_FOLDS}"

echo "RF evaluation complete. Results: ${OUT}"
