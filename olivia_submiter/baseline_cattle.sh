#!/bin/bash
#SBATCH --account=nn9780k
#SBATCH --job-name=baseline_cattle_train
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm_%j_baseline_cattle_train.out

cd "${SLURM_SUBMIT_DIR}"

# =============================================================================
# PARAMETERIZED INPUTS (passed via --export or environment variables)
#   MODEL_CLASS   — e.g. DeepSEA, DeepFormer, DeepATT
#   LEARNING_RATE — e.g. 1e-5, 5e-5, 1e-4, 5e-4
# =============================================================================
MODEL_CLASS="${MODEL_CLASS:-DeepSEA}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"

# =============================================================================
# FIXED PARAMETERS
# =============================================================================
DATASET_TAG="cattle"
SEQUENCE_LENGTH=1000
NUM_TRACKS=95
DATA_DIR="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_cattle_beds"
GENOME_PATH="/cluster/projects/nn9780k/dat/training_data/genomes/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
BATCH_SIZE=1000
MAX_EPOCHS=50
PATIENCE=5
PRECISION="bf16"
SEED=42
LOGGING_STEPS=500
NUM_WORKERS=8
GRADIENT_ACCUMULATION_STEPS=1
USE_WANDB=True
WANDB_PROJECT="cattle"
EXP_NAME="${MODEL_CLASS}_${DATASET_TAG}_${LEARNING_RATE}_${SEQUENCE_LENGTH}"

# =============================================================================
# PATHS
# =============================================================================
CONTAINER_PATH="/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif"

BASE_OUTDIR='checkpoints'
mkdir -p ${BASE_OUTDIR}

# =============================================================================
# APPTAINER EXEC
# =============================================================================
apptainer exec --nv \
    --pwd /workspace \
    --bind $PWD:/workspace \
    --bind "/cluster":/cluster \
    "${CONTAINER_PATH}" \
    python -m src.seq2fun.trainer \
        --model_class=${MODEL_CLASS} \
        --sequence_length=${SEQUENCE_LENGTH} \
        --batch_size=${BATCH_SIZE} \
        --max_epochs=${MAX_EPOCHS} \
        --patience=${PATIENCE} \
        --lr=${LEARNING_RATE} \
        --lr_scheduler_type=${LR_SCHEDULER_TYPE} \
        --precision=${PRECISION} \
        --seed=${SEED} \
        --logging_steps=${LOGGING_STEPS} \
        --num_workers=${NUM_WORKERS} \
        --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        --data_dir=${DATA_DIR} \
        --genome_path=${GENOME_PATH} \
        --num_tracks=${NUM_TRACKS} \
        --output_dir=${BASE_OUTDIR} \
        --use_wandb=${USE_WANDB} \
        --wandb_project=${WANDB_PROJECT} \
        --exp_name=${EXP_NAME}

echo "Training job finished: MODEL_CLASS=${MODEL_CLASS} LEARNING_RATE=${LEARNING_RATE} DATASET=${DATASET_TAG}"
