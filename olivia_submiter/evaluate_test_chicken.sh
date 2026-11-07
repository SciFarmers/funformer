#!/bin/bash
#SBATCH --account=nn9780k
#SBATCH --job-name=eval_test_chicken
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j_eval_test_chicken.out

cd "${SLURM_SUBMIT_DIR}"

# PARAMETERIZED: CHECKPOINT_DIR passed via --export
CHECKPOINT_DIR="${CHECKPOINT_DIR:?ERROR: CHECKPOINT_DIR must be set}"
CHECKPOINT_TAG="$(basename "${CHECKPOINT_DIR}")"

DATA_DIR="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_chicken_beds"
GENOME_PATH="/cluster/projects/nn9780k/dat/training_data/genomes/Gallus_gallus.GRCg6a.dna.toplevel.fa"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/test_set}"

CONTAINER_PATH="/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif"

mkdir -p "${OUTPUT_DIR}" logs

echo "==================================================="
echo " Test-set Evaluation — chicken"
echo "   CHECKPOINT_DIR : ${CHECKPOINT_DIR}"
echo "   OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "==================================================="

apptainer exec --nv \
    --pwd /workspace \
    --bind $PWD:/workspace \
    --bind "/cluster":/cluster \
    "${CONTAINER_PATH}" \
    python -m src_evaluate.evaluate_test_set \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --genome-path "${GENOME_PATH}" \
        --batch-size "${BATCH_SIZE:-1000}" \
        --output-dir "${OUTPUT_DIR}"

echo "Evaluation finished: ${CHECKPOINT_TAG}"
