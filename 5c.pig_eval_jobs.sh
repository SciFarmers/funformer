#!/bin/bash
# Submit test-set evaluation jobs for pig.
# model_class/sequence_length/n_targets read from best_model/model_config.json.
#
# Usage: bash 5c.pig_eval_jobs.sh

set -euo pipefail

SUBMITTER="olivia_submiter/evaluate_test_pig.sh"

CHECKPOINT_DIRS=(
    "checkpoints/DeepSEA_pig_5e-4_1000"
    "checkpoints/DeepFormer_pig_5e-4_1000"
    "checkpoints/DeepATT_pig_5e-4_1000"
    "checkpoints/funformer_pig_len20000_lr5e-4_b250_gac2"
)

for CKPT in "${CHECKPOINT_DIRS[@]}"; do
    TAG="$(basename "${CKPT}")"
    echo "Submitting eval: ${TAG}"
    sbatch --export=ALL,CHECKPOINT_DIR="${CKPT}" \
           --job-name="eval_pig_${TAG}" \
           --output="logs/eval_pig_${TAG}_%j.out" \
           "${SUBMITTER}"
done

echo ""
echo "${#CHECKPOINT_DIRS[@]} evaluation job(s) submitted for pig."
echo "  Output: eval_results/test_set/"
