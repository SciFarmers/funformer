#!/bin/bash
# =============================================================================
# VEP eQTL Fine-mapping — sequential loop over all VCFs in one SLURM job
#
# ── HOW TO SUBMIT ────────────────────────────────────────────────────────────
# DeepFormer (default test model):
#   sbatch olivia_submiter/vep_eqtl_finemap.sh
#
# Other models:
#   MODEL_CLASS=DeepSEA  MODEL_PATH=checkpoints/DeepSEA_5e-4_1000/best_model  \n#     sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh
#   MODEL_CLASS=DanQ     MODEL_PATH=checkpoints/DanQ_5e-4_1000/best_model      \n#     sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh
#   MODEL_CLASS=FunFormer MODEL_PATH=checkpoints/funformer_.../best_model WINDOW_SIZE=10000 \n#     sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh
# =============================================================================
#SBATCH --account=nn9780k
#SBATCH --job-name=vep_eqtl
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vep_eqtl_%j.out

cd "${SLURM_SUBMIT_DIR}"

# =============================================================================
# PARAMETERIZED INPUTS (override via --export=ALL,VAR=value)
# =============================================================================
MODEL_CLASS="${MODEL_CLASS:-DeepFormer}"
MODEL_PATH="${MODEL_PATH:-checkpoints/DeepFormer_5e-4_1000/best_model}"
WINDOW_SIZE="${WINDOW_SIZE:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# =============================================================================
# FIXED PATHS
# =============================================================================
# Defaults are the human GTEx benchmark; override GENOME_PATH, VCF_DIR and
# OUT_ROOT to score any other Enformer-format variant set with the same code.
GENOME_PATH="${GENOME_PATH:-/cluster/projects/nn9780k/dat/training_data/genomes/Homo_sapiens.GRCh38.dna.primary_assembly.fa}"
VCF_DIR="${VCF_DIR:-bench_data/enformer_gtex_fine_vcf}"
OUT_ROOT="${OUT_ROOT:-bench_results/vep_eqtl}"
CONTAINER_PATH="/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif"

MODEL_TAG="$(basename "${MODEL_PATH%/best_model}")"
OUTPUT_DIR="${OUT_ROOT}/${MODEL_TAG}"
mkdir -p "${OUTPUT_DIR}" logs

echo "============================================================"
echo " VEP eQTL Fine-mapping"
echo "  Variants    : ${VCF_DIR}"
echo "  Genome      : $(basename ${GENOME_PATH})"
echo "  Model class : ${MODEL_CLASS}"
echo "  Model path  : ${MODEL_PATH}"
echo "  Window size : ${WINDOW_SIZE}"
echo "  Batch size  : ${BATCH_SIZE}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "============================================================"

# =============================================================================
# LOOP OVER ALL VCF FILES
# =============================================================================
for VCF_FILE in $(ls -1 "${VCF_DIR}"/*.vcf | sort); do
    BASENAME="$(basename "${VCF_FILE}" .vcf)"
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}.csv.gz"

    if [[ -f "${OUTPUT_FILE}" ]]; then
        echo "[SKIP] Already exists: ${BASENAME}.csv.gz"
        continue
    fi

    echo "[RUN]  ${BASENAME}"

    apptainer exec --nv \
        --pwd /workspace \
        --bind "$PWD":/workspace \
        --bind "/cluster":/cluster \
        "${CONTAINER_PATH}" \
        python -m src_evaluate.vep_eqtl_finemap \
            --genome-path  "${GENOME_PATH}" \
            --variant-path "${VCF_FILE}" \
            --output-path  "${OUTPUT_FILE}" \
            --model-path   "${MODEL_PATH}" \
            --model-class  "${MODEL_CLASS}" \
            --window-size  "${WINDOW_SIZE}" \
            --batch-size   "${BATCH_SIZE}"

    echo "[DONE] ${BASENAME}"
done

echo "All VCFs processed for ${MODEL_CLASS}."
