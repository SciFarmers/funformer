#!/bin/bash
# Zebrafish baselines: 3 models x 4 LRs = 12 jobs.
#
# Zebrafish is added as a second ray-finned fish to control for the
# salmonid-specific whole-genome duplication (Ss4R, ~80-100 Mya) carried by
# Atlantic salmon. Zebrafish shares the teleost WGD (Ts3R) and the same
# tetrapod-fish divergence (~435 Mya) but lacks Ss4R, so it separates
# "transfer fails at this phylogenetic distance" from "transfer fails because
# the salmon genome was re-duplicated".
#
# Identical optimization to every other species' baselines: 4 LRs, effective
# batch 1000, 50 max epochs, patience 5, bf16, seed 42.
#
# Usage: bash 10a.zebrafish_jobs.sh
set -euo pipefail

MODELS=(DeepSEA DeepFormer DeepATT)
LRS=(1e-5 5e-5 1e-4 5e-4)
SUBMITTER="olivia_submiter/baseline_zebrafish.sh"
mkdir -p logs

n=0
for MODEL in "${MODELS[@]}"; do
    for LR in "${LRS[@]}"; do
        echo "Submitting ${MODEL} zebrafish lr=${LR}"
        sbatch --export=ALL,MODEL_CLASS="${MODEL}",LEARNING_RATE="${LR}" \
               --job-name="zf_${MODEL}_${LR}" \
               --output="logs/slurm_%j_${MODEL}_zebrafish_${LR}.out" \
               "${SUBMITTER}"
        n=$((n + 1))
    done
done
echo ""
echo "${n} zebrafish baseline jobs submitted (${MODELS[*]} x ${LRS[*]})."
