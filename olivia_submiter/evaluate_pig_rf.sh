#!/bin/bash
# =============================================================================
# RF evaluation on the pig QTL fine-mapping benchmark.
#
# One model per job. SCORE_TAGS names the model, resolved under
# bench_results/vep_pig_qtl/${KIND}/.
#
#   KIND         eqtl | sqtl   (default eqtl)
#   SCORE_TAGS   the model tag
#   TAG          output name (default: the model tag)
#
# ── EXAMPLES ─────────────────────────────────────────────────────────────────
#   SCORE_TAGS="funformer_pig_len20000_lr5e-4_b250_gac2" \
#     sbatch --export=ALL olivia_submiter/evaluate_pig_rf.sh
#
#   KIND=sqtl \
#   SCORE_TAGS="funformer_pig_len20000_lr5e-4_b250_gac2 funformer_entex_423_len20000_lr5e-4_b250_gac2" \
#   TAG=combined_pig_human \
#     sbatch --export=ALL olivia_submiter/evaluate_pig_rf.sh
# =============================================================================
#SBATCH --account=nn9780k
#SBATCH --job-name=rf_pig_qtl
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_pig_qtl_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

KIND="${KIND:-eqtl}"   # labels come from the _pos/_neg score files, not a variant dir
N_REPEATS="${N_REPEATS:-10}"
CV_FOLDS="${CV_FOLDS:-10}"
CONTAINER_PATH="/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif"

if [[ -z "${SCORE_TAGS:-}" ]]; then
    echo "ERROR: set SCORE_TAGS to a model tag."; exit 1
fi

SCORE_ARG="--score-dir bench_results/vep_pig_qtl/${KIND}/${SCORE_TAGS}"
OUT_TAG="${TAG:-${SCORE_TAGS}}"
OUT="bench_results/rf_pig_qtl/${KIND}/${OUT_TAG}.csv"
mkdir -p logs "$(dirname "${OUT}")"

# evaluate_eqtl_rf is species-agnostic: it reads whatever per-track features
# the score dir holds.
MODULE=src_evaluate.evaluate_eqtl_rf

echo "============================================================"
echo " RF — pig ${KIND}"
echo "  QTL kind : ${KIND}"
echo "  Features : ${SCORE_TAGS}"
echo "  Module   : ${MODULE}"
echo "  Output   : ${OUT}"
echo "============================================================"

apptainer exec \
    --pwd /workspace \
    --bind "$PWD":/workspace \
    --bind "/cluster":/cluster \
    "${CONTAINER_PATH}" \
    python -m ${MODULE} \
        ${SCORE_ARG} \
        --out        "${OUT}" \
        --n-repeats  "${N_REPEATS}" \
        --cv-folds   "${CV_FOLDS}"

echo "RF complete -> ${OUT}"
