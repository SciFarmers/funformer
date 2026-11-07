#!/bin/bash
# Submit test-set evaluation jobs for salmon.
# model_class/sequence_length/n_targets read from best_model/model_config.json.
#
# Usage: bash 5d.salmon_eval_jobs.sh

set -euo pipefail

SUBMITTER="olivia_submiter/evaluate_test_salmon.sh"

CHECKPOINT_DIRS=(
    "checkpoints/DeepSEA_salmon_5e-5_1000"
    "checkpoints/DeepFormer_salmon_5e-5_1000"
    "checkpoints/DeepATT_salmon_5e-4_1000"
    "checkpoints/funformer_salmon_len20000_lr5e-4_b250_gac2"
)

for CKPT in "${CHECKPOINT_DIRS[@]}"; do
    TAG="$(basename "${CKPT}")"
    echo "Submitting eval: ${TAG}"
    sbatch --export=ALL,CHECKPOINT_DIR="${CKPT}" \
           --job-name="eval_salmon_${TAG}" \
           --output="logs/eval_salmon_${TAG}_%j.out" \
           "${SUBMITTER}"
done

echo ""
echo "${#CHECKPOINT_DIRS[@]} evaluation job(s) submitted for salmon."
echo "  Output: eval_results/test_set/"
