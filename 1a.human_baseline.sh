#!/bin/bash
# Submit baseline model training jobs for 3 models x 4 learning rates (12 total jobs).
# Usage: bash 1a.submit_baseline_standard.sh

set -euo pipefail

MODELS=(DeepSEA DeepFormer DeepATT)
LRS=(1e-5 5e-5 1e-4 5e-4)

SUBMITTER="olivia_submiter/baseline_entex423.sh"

for MODEL in "${MODELS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting: MODEL_CLASS=${MODEL}  LEARNING_RATE=${LR}"
        sbatch --export=ALL,MODEL_CLASS="${MODEL}",LEARNING_RATE="${LR}" \
               --job-name="${MODEL}_${LR}" \
               --output="slurm_%j_${MODEL}_${LR}_train.out" \
               "${SUBMITTER}"
    done
done

echo "All 12 jobs submitted."
