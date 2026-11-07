#!/bin/bash
# Submit FunFormer ablation jobs: 4 sequence lengths × 4 learning rates = 16 jobs.
# Effective batch size = 1000 (matches baseline standard models).
# Uses shared trainer: src.seq2fun.trainer --model_class=FunFormer
#
# Usage:
#   bash 1b.human_funformer.sh

set -euo pipefail

SEQ_LENS=(5000 10000 20000 30000 50000)

LRS=(1e-5 5e-5 1e-4 5e-4)

# Dataset tag is fixed to entex_423 in the job file.
# For other species/datasets, create a new pair of job + wrapper scripts.

SUBMITTER="olivia_submiter/funformer_entex423.sh"

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting: SEQ_LEN=${SEQ_LEN}  LEARNING_RATE=${LR}"
        sbatch --export=ALL,SEQ_LEN="${SEQ_LEN}",LEARNING_RATE="${LR}" \
               --job-name="funFormer_${SEQ_LEN}_${LR}" \
               --output="slurm_%j_funformer_len${SEQ_LEN}_lr${LR}.out" \
               "${SUBMITTER}"
    done
done

echo ""
echo "All $((${#SEQ_LENS[@]} * ${#LRS[@]})) jobs submitted."
echo "Sequence lengths : ${SEQ_LENS[*]}"
echo "Learning rates   : ${LRS[*]}"
echo "WANDB_PROJECT    : FunFormer_entex_423"
