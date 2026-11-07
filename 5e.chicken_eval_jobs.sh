#!/bin/bash
# Submit test-set evaluation for all chicken checkpoints (16 jobs: 12 baselines + 4 FunFormer).
# Evaluate all LR variants to identify the best checkpoint per model class.
#
# Run AFTER training (3d + 4d) completes.
# Results → eval_results/test_set/<checkpoint>_{summary,per_track}.csv
#
# After this runs, pick the best LR per model (highest mean AUROC in *_summary.csv),
# then update checkpoint paths in 8c.cross_species_chicken.sh before submitting.
#
# Usage: bash 5e.chicken_eval_jobs.sh

set -euo pipefail
mkdir -p logs

SUBMITTER="olivia_submiter/evaluate_test_chicken.sh"

# --- Baselines: all 3 models × 4 LRs ---
BASELINE_CHECKPOINTS=(
    "checkpoints/DeepSEA_chicken_1e-5_1000"
    "checkpoints/DeepSEA_chicken_5e-5_1000"
    "checkpoints/DeepSEA_chicken_1e-4_1000"
    "checkpoints/DeepSEA_chicken_5e-4_1000"
    "checkpoints/DeepFormer_chicken_1e-5_1000"
    "checkpoints/DeepFormer_chicken_5e-5_1000"
    "checkpoints/DeepFormer_chicken_1e-4_1000"
    "checkpoints/DeepFormer_chicken_5e-4_1000"
    "checkpoints/DeepATT_chicken_1e-5_1000"
    "checkpoints/DeepATT_chicken_5e-5_1000"
    "checkpoints/DeepATT_chicken_1e-4_1000"
    "checkpoints/DeepATT_chicken_5e-4_1000"
)

# --- FunFormer 20k: 4 LRs ---
FUNFORMER_CHECKPOINTS=(
    "checkpoints/funformer_chicken_len20000_lr1e-5_b250_gac2"
    "checkpoints/funformer_chicken_len20000_lr5e-5_b250_gac2"
    "checkpoints/funformer_chicken_len20000_lr1e-4_b250_gac2"
    "checkpoints/funformer_chicken_len20000_lr5e-4_b250_gac2"
)

ALL_CHECKPOINTS=("${BASELINE_CHECKPOINTS[@]}" "${FUNFORMER_CHECKPOINTS[@]}")

for CKPT in "${ALL_CHECKPOINTS[@]}"; do
    TAG="$(basename "${CKPT}")"
    echo "Submitting eval: ${TAG}"
    sbatch --export=ALL,CHECKPOINT_DIR="${CKPT}" \
           --job-name="eval_chicken_${TAG}" \
           --output="logs/eval_chicken_${TAG}_%j.out" \
           "${SUBMITTER}"
done

echo ""
echo "${#ALL_CHECKPOINTS[@]} evaluation jobs submitted for chicken."
echo "  Results: eval_results/test_set/"
echo ""
echo "After results are ready, pick best LR per model with:"
echo "  grep '' eval_results/test_set/*chicken*_summary.csv | sort -t, -k3 -rn | head -20"
