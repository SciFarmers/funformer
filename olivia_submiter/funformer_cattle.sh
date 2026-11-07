#!/bin/bash
#SBATCH --account=nn9780k
#SBATCH --job-name=funformer_cattle
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm_%j_funformer_cattle.out

cd "${SLURM_SUBMIT_DIR}"
set -euo pipefail

# =============================================================================
# PARAMETERIZED INPUTS (passed via --export or environment variables)
#   LEARNING_RATE — e.g. 1e-5, 5e-5, 1e-4, 5e-4
# =============================================================================
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# =============================================================================
# FIXED PARAMETERS — FunFormer 20k, cattle
# =============================================================================
DATASET_TAG="cattle"
SEQ_LEN="${SEQ_LEN:-20000}"

# Physical batch size + GAC per sequence length. Effective batch is held at
# 1000 for every length (per-device batch x 2 GPUs x GAC), matching both the
# rest of the FunFormer ablation family and the 1 kb baselines.
case "${SEQ_LEN}" in
     1000) BATCH_SIZE=500; GRADIENT_ACCUMULATION_STEPS=1 ;;
     5000) BATCH_SIZE=500; GRADIENT_ACCUMULATION_STEPS=1 ;;
    10000) BATCH_SIZE=500; GRADIENT_ACCUMULATION_STEPS=1 ;;
    20000) BATCH_SIZE=250; GRADIENT_ACCUMULATION_STEPS=2 ;;
    30000) BATCH_SIZE=250; GRADIENT_ACCUMULATION_STEPS=2 ;;
    50000) BATCH_SIZE=250; GRADIENT_ACCUMULATION_STEPS=2 ;;
    *)
        echo "ERROR: Unknown SEQ_LEN=${SEQ_LEN}. Supported: 1000 5000 10000 20000 30000 50000" >&2
        exit 1
        ;;
esac
NUM_TRACKS=95
DATA_DIR="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_cattle_beds"
GENOME_PATH="/cluster/projects/nn9780k/dat/training_data/genomes/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
MAX_EPOCHS=30
PATIENCE=5
PRECISION="bf16"
SEED=42
LOGGING_STEPS=500
NUM_WORKERS=16

# Model architecture — matches entex_423 FunFormer config
NUM_HIDDEN_LAYERS=1
HIDDEN_SIZE=512
NUM_ATTENTION_HEADS=8
INTERMEDIATE_SIZE=1024
PROJECTION_DIM=256
ROPE_THETA=10000.0

USE_WANDB=True
WANDB_PROJECT="cattle"
EXP_NAME="funformer_${DATASET_TAG}_len${SEQ_LEN}_lr${LEARNING_RATE}_b${BATCH_SIZE}_gac${GRADIENT_ACCUMULATION_STEPS}"

# =============================================================================
# PATHS
# =============================================================================
CONTAINER_PATH="/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif"
BASE_OUTDIR='checkpoints'
mkdir -p ${BASE_OUTDIR}

echo "==================================================="
echo " FunFormer 20k — cattle"
echo "   DATASET_TAG  : ${DATASET_TAG}"
echo "   SEQ_LEN      : ${SEQ_LEN}"
echo "   LEARNING_RATE: ${LEARNING_RATE}"
echo "   BATCH_SIZE   : ${BATCH_SIZE}"
echo "   GAC          : ${GRADIENT_ACCUMULATION_STEPS}"
echo "   Effective BS : $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2))"  # per-device x 2 GPUs x gac
echo "   WANDB_PROJECT: ${WANDB_PROJECT}"
echo "   EXP_NAME     : ${EXP_NAME}"
echo "==================================================="

# =============================================================================
# APPTAINER + ACCELERATE LAUNCH (2-GPU DDP)
# lr_scheduler_type not set — uses HF Trainer default (linear with warmup)
# =============================================================================
apptainer exec --nv \
    --pwd /workspace \
    --bind $PWD:/workspace \
    --bind "/cluster":/cluster \
    "${CONTAINER_PATH}" \
    accelerate launch --num_machines=1 --num_processes=2 -m src.seq2fun.trainer \
        --model_class=FunFormer \
        --sequence_length=${SEQ_LEN} \
        --batch_size=${BATCH_SIZE} \
        --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        --lr=${LEARNING_RATE} \
        --max_epochs=${MAX_EPOCHS} \
        --patience=${PATIENCE} \
        --precision=${PRECISION} \
        --seed=${SEED} \
        --logging_steps=${LOGGING_STEPS} \
        --num_workers=${NUM_WORKERS} \
        --num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        --hidden_size=${HIDDEN_SIZE} \
        --num_attention_heads=${NUM_ATTENTION_HEADS} \
        --intermediate_size=${INTERMEDIATE_SIZE} \
        --projection_dim=${PROJECTION_DIM} \
        --rope_theta=${ROPE_THETA} \
        --data_dir=${DATA_DIR} \
        --genome_path=${GENOME_PATH} \
        --num_tracks=${NUM_TRACKS} \
        --output_dir=${BASE_OUTDIR} \
        --use_wandb=${USE_WANDB} \
        --wandb_project=${WANDB_PROJECT} \
        --exp_name=${EXP_NAME}

echo "Job finished: SEQ_LEN=${SEQ_LEN} LEARNING_RATE=${LEARNING_RATE} DATASET=${DATASET_TAG}"
