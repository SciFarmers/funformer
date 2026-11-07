#!/bin/bash
# Cross-species evaluation — all directions involving chicken (32 jobs).
#
# Prerequisites:
#   1. Chicken models trained (3d + 4d scripts) and best checkpoints identified.
#   2. meta_info/matched_tracks.json regenerated with chicken included:
#        python meta_info/find_matched_tracks.py
#
# Best chicken checkpoints (update after training + eval):
#   DeepSEA    : checkpoints/DeepSEA_chicken_<lr>_1000/best_model
#   DeepFormer : checkpoints/DeepFormer_chicken_<lr>_1000/best_model
#   DeepATT    : checkpoints/DeepATT_chicken_<lr>_1000/best_model
#   FunFormer  : checkpoints/funformer_chicken_len20000_lr5e-4_b250_gac2/best_model
#
# Jobs: 8 directions × 4 models = 32 jobs
# Output: eval_results/cross_species/<src>_to_<tgt>/

set -euo pipefail
mkdir -p logs

# =============================================================================
# Paths
# =============================================================================
HUMAN_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_ENTEX_423_beds"
PIG_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_pig_beds"
CATTLE_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_cattle_beds"
CHICKEN_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_chicken_beds"
SALMON_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_salmon_beds"

HUMAN_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
PIG_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"
CATTLE_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
CHICKEN_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Gallus_gallus.GRCg6a.dna.toplevel.fa"
SALMON_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Salmo_salar.Ssal_v3.1.dna.toplevel.fa"

# =============================================================================
# Best chicken checkpoints — UPDATE THESE after training
# =============================================================================
CHICKEN_DEEPSEA="checkpoints/DeepSEA_chicken_1e-4_1000/best_model"
CHICKEN_DEEPFORMER="checkpoints/DeepFormer_chicken_5e-4_1000/best_model"
CHICKEN_DEEPATT="checkpoints/DeepATT_chicken_5e-4_1000/best_model"
CHICKEN_FUNFORMER="checkpoints/funformer_chicken_len20000_lr5e-4_b250_gac2/best_model"

# Best checkpoints for other species (already trained)
HUMAN_DEEPSEA="checkpoints/DeepSEA_1e-4_1000/best_model"
HUMAN_DEEPFORMER="checkpoints/DeepFormer_5e-4_1000/best_model"
HUMAN_DEEPATT="checkpoints/DeepATT_5e-4_1000/best_model"
HUMAN_FUNFORMER="checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model"

PIG_DEEPSEA="checkpoints/DeepSEA_pig_5e-4_1000/best_model"
PIG_DEEPFORMER="checkpoints/DeepFormer_pig_5e-4_1000/best_model"
PIG_DEEPATT="checkpoints/DeepATT_pig_5e-4_1000/best_model"
PIG_FUNFORMER="checkpoints/funformer_pig_len20000_lr5e-4_b250_gac2/best_model"

CATTLE_DEEPSEA="checkpoints/DeepSEA_cattle_5e-5_1000/best_model"
CATTLE_DEEPFORMER="checkpoints/DeepFormer_cattle_5e-4_1000/best_model"
CATTLE_DEEPATT="checkpoints/DeepATT_cattle_5e-4_1000/best_model"
CATTLE_FUNFORMER="checkpoints/funformer_cattle_len20000_lr5e-4_b250_gac2/best_model"

SALMON_DEEPSEA="checkpoints/DeepSEA_salmon_5e-5_1000/best_model"
SALMON_DEEPFORMER="checkpoints/DeepFormer_salmon_5e-5_1000/best_model"
SALMON_DEEPATT="checkpoints/DeepATT_salmon_5e-4_1000/best_model"
SALMON_FUNFORMER="checkpoints/funformer_salmon_len20000_lr5e-4_b250_gac2/best_model"

# =============================================================================
# Helper
# =============================================================================
submit() {
    local SRC_SP=$1 SRC_PATH=$2 TGT_SP=$3 TGT_DATA=$4 TGT_GENOME=$5 TAG=$6
    SOURCE_SPECIES="${SRC_SP}" \
    SOURCE_MODEL_PATH="${SRC_PATH}" \
    TARGET_SPECIES="${TGT_SP}" \
    TARGET_DATA_DIR="${TGT_DATA}" \
    TARGET_GENOME="${TGT_GENOME}" \
    MODEL_TAG="${TAG}" \
    sbatch --export=ALL olivia_submiter/evaluate_cross_species.sh
}

# =============================================================================
# chicken → human / pig / cattle / salmon
# =============================================================================
submit chicken "$CHICKEN_DEEPSEA"    human  "$HUMAN_DATA"   "$HUMAN_GENOME"   DeepSEA
submit chicken "$CHICKEN_DEEPFORMER" human  "$HUMAN_DATA"   "$HUMAN_GENOME"   DeepFormer
submit chicken "$CHICKEN_DEEPATT"    human  "$HUMAN_DATA"   "$HUMAN_GENOME"   DeepATT
submit chicken "$CHICKEN_FUNFORMER"  human  "$HUMAN_DATA"   "$HUMAN_GENOME"   FunFormer_20k

submit chicken "$CHICKEN_DEEPSEA"    pig    "$PIG_DATA"     "$PIG_GENOME"     DeepSEA
submit chicken "$CHICKEN_DEEPFORMER" pig    "$PIG_DATA"     "$PIG_GENOME"     DeepFormer
submit chicken "$CHICKEN_DEEPATT"    pig    "$PIG_DATA"     "$PIG_GENOME"     DeepATT
submit chicken "$CHICKEN_FUNFORMER"  pig    "$PIG_DATA"     "$PIG_GENOME"     FunFormer_20k

submit chicken "$CHICKEN_DEEPSEA"    cattle "$CATTLE_DATA"  "$CATTLE_GENOME"  DeepSEA
submit chicken "$CHICKEN_DEEPFORMER" cattle "$CATTLE_DATA"  "$CATTLE_GENOME"  DeepFormer
submit chicken "$CHICKEN_DEEPATT"    cattle "$CATTLE_DATA"  "$CATTLE_GENOME"  DeepATT
submit chicken "$CHICKEN_FUNFORMER"  cattle "$CATTLE_DATA"  "$CATTLE_GENOME"  FunFormer_20k

submit chicken "$CHICKEN_DEEPSEA"    salmon "$SALMON_DATA"  "$SALMON_GENOME"  DeepSEA
submit chicken "$CHICKEN_DEEPFORMER" salmon "$SALMON_DATA"  "$SALMON_GENOME"  DeepFormer
submit chicken "$CHICKEN_DEEPATT"    salmon "$SALMON_DATA"  "$SALMON_GENOME"  DeepATT
submit chicken "$CHICKEN_FUNFORMER"  salmon "$SALMON_DATA"  "$SALMON_GENOME"  FunFormer_20k

# =============================================================================
# human / pig / cattle / salmon → chicken
# =============================================================================
submit human  "$HUMAN_DEEPSEA"    chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepSEA
submit human  "$HUMAN_DEEPFORMER" chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepFormer
submit human  "$HUMAN_DEEPATT"    chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepATT
submit human  "$HUMAN_FUNFORMER"  chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" FunFormer_20k

submit pig    "$PIG_DEEPSEA"      chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepSEA
submit pig    "$PIG_DEEPFORMER"   chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepFormer
submit pig    "$PIG_DEEPATT"      chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepATT
submit pig    "$PIG_FUNFORMER"    chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" FunFormer_20k

submit cattle "$CATTLE_DEEPSEA"   chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepSEA
submit cattle "$CATTLE_DEEPFORMER" chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepFormer
submit cattle "$CATTLE_DEEPATT"   chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepATT
submit cattle "$CATTLE_FUNFORMER" chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" FunFormer_20k

submit salmon "$SALMON_DEEPSEA"   chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepSEA
submit salmon "$SALMON_DEEPFORMER" chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepFormer
submit salmon "$SALMON_DEEPATT"   chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" DeepATT
submit salmon "$SALMON_FUNFORMER" chicken "$CHICKEN_DATA" "$CHICKEN_GENOME" FunFormer_20k

echo "Submitted 32 cross-species chicken evaluation jobs."
