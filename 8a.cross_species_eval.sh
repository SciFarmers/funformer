# Cross-species regulatory sequence classification — submit all 48 jobs
#
# Prerequisites:
#   1. Run meta_info/find_matched_tracks.py first (login node, no GPU):
#        python meta_info/find_matched_tracks.py
#      Generates meta_info/matched_tracks.json used by every job.
#   2. All 16 species×model checkpoints must exist in checkpoints/.
#
# Jobs: 12 directions × 4 models (3 best baselines + FunFormer 20k)
# Output: eval_results/cross_species/<src>_to_<tgt>/<model_tag>_{per_match,summary}.csv

# =============================================================================
# Paths
# =============================================================================

HUMAN_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_ENTEX_423_beds"
PIG_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_pig_beds"
CATTLE_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_cattle_beds"
SALMON_DATA="/cluster/projects/nn9780k/dat/training_data/seq2fun/binary/hf_salmon_beds"

HUMAN_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
PIG_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa"
CATTLE_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa"
SALMON_GENOME="/cluster/projects/nn9780k/dat/training_data/genomes/Salmo_salar.Ssal_v3.1.dna.toplevel.fa"

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
# human → pig / cattle / salmon
# =============================================================================

submit human checkpoints/DeepSEA_1e-4_1000/best_model                              pig    "$PIG_DATA"    "$PIG_GENOME"    DeepSEA
submit human checkpoints/DeepFormer_5e-4_1000/best_model                            pig    "$PIG_DATA"    "$PIG_GENOME"    DeepFormer
submit human checkpoints/DeepATT_5e-4_1000/best_model                               pig    "$PIG_DATA"    "$PIG_GENOME"    DeepATT
submit human checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model   pig    "$PIG_DATA"    "$PIG_GENOME"    FunFormer_20k

submit human checkpoints/DeepSEA_1e-4_1000/best_model                              cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepSEA
submit human checkpoints/DeepFormer_5e-4_1000/best_model                            cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepFormer
submit human checkpoints/DeepATT_5e-4_1000/best_model                               cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepATT
submit human checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model   cattle "$CATTLE_DATA" "$CATTLE_GENOME" FunFormer_20k

submit human checkpoints/DeepSEA_1e-4_1000/best_model                              salmon "$SALMON_DATA" "$SALMON_GENOME" DeepSEA
submit human checkpoints/DeepFormer_5e-4_1000/best_model                            salmon "$SALMON_DATA" "$SALMON_GENOME" DeepFormer
submit human checkpoints/DeepATT_5e-4_1000/best_model                               salmon "$SALMON_DATA" "$SALMON_GENOME" DeepATT
submit human checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model   salmon "$SALMON_DATA" "$SALMON_GENOME" FunFormer_20k

# =============================================================================
# pig → human / cattle / salmon
# =============================================================================

submit pig checkpoints/DeepSEA_pig_5e-4_1000/best_model                            human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepSEA
submit pig checkpoints/DeepFormer_pig_5e-4_1000/best_model                          human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepFormer
submit pig checkpoints/DeepATT_pig_5e-4_1000/best_model                             human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepATT
submit pig checkpoints/funformer_pig_len20000_lr5e-4_b250_gac2/best_model           human  "$HUMAN_DATA"  "$HUMAN_GENOME"  FunFormer_20k

submit pig checkpoints/DeepSEA_pig_5e-4_1000/best_model                            cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepSEA
submit pig checkpoints/DeepFormer_pig_5e-4_1000/best_model                          cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepFormer
submit pig checkpoints/DeepATT_pig_5e-4_1000/best_model                             cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepATT
submit pig checkpoints/funformer_pig_len20000_lr5e-4_b250_gac2/best_model           cattle "$CATTLE_DATA" "$CATTLE_GENOME" FunFormer_20k

submit pig checkpoints/DeepSEA_pig_5e-4_1000/best_model                            salmon "$SALMON_DATA" "$SALMON_GENOME" DeepSEA
submit pig checkpoints/DeepFormer_pig_5e-4_1000/best_model                          salmon "$SALMON_DATA" "$SALMON_GENOME" DeepFormer
submit pig checkpoints/DeepATT_pig_5e-4_1000/best_model                             salmon "$SALMON_DATA" "$SALMON_GENOME" DeepATT
submit pig checkpoints/funformer_pig_len20000_lr5e-4_b250_gac2/best_model           salmon "$SALMON_DATA" "$SALMON_GENOME" FunFormer_20k

# =============================================================================
# cattle → human / pig / salmon
# =============================================================================

submit cattle checkpoints/DeepSEA_cattle_5e-5_1000/best_model                      human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepSEA
submit cattle checkpoints/DeepFormer_cattle_5e-4_1000/best_model                    human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepFormer
submit cattle checkpoints/DeepATT_cattle_5e-4_1000/best_model                       human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepATT
submit cattle checkpoints/funformer_cattle_len20000_lr5e-4_b250_gac2/best_model     human  "$HUMAN_DATA"  "$HUMAN_GENOME"  FunFormer_20k

submit cattle checkpoints/DeepSEA_cattle_5e-5_1000/best_model                      pig    "$PIG_DATA"    "$PIG_GENOME"    DeepSEA
submit cattle checkpoints/DeepFormer_cattle_5e-4_1000/best_model                    pig    "$PIG_DATA"    "$PIG_GENOME"    DeepFormer
submit cattle checkpoints/DeepATT_cattle_5e-4_1000/best_model                       pig    "$PIG_DATA"    "$PIG_GENOME"    DeepATT
submit cattle checkpoints/funformer_cattle_len20000_lr5e-4_b250_gac2/best_model     pig    "$PIG_DATA"    "$PIG_GENOME"    FunFormer_20k

submit cattle checkpoints/DeepSEA_cattle_5e-5_1000/best_model                      salmon "$SALMON_DATA" "$SALMON_GENOME" DeepSEA
submit cattle checkpoints/DeepFormer_cattle_5e-4_1000/best_model                    salmon "$SALMON_DATA" "$SALMON_GENOME" DeepFormer
submit cattle checkpoints/DeepATT_cattle_5e-4_1000/best_model                       salmon "$SALMON_DATA" "$SALMON_GENOME" DeepATT
submit cattle checkpoints/funformer_cattle_len20000_lr5e-4_b250_gac2/best_model     salmon "$SALMON_DATA" "$SALMON_GENOME" FunFormer_20k

# =============================================================================
# salmon → human / pig / cattle
# =============================================================================

submit salmon checkpoints/DeepSEA_salmon_5e-5_1000/best_model                      human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepSEA
submit salmon checkpoints/DeepFormer_salmon_5e-5_1000/best_model                    human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepFormer
submit salmon checkpoints/DeepATT_salmon_5e-4_1000/best_model                       human  "$HUMAN_DATA"  "$HUMAN_GENOME"  DeepATT
submit salmon checkpoints/funformer_salmon_len20000_lr5e-4_b250_gac2/best_model     human  "$HUMAN_DATA"  "$HUMAN_GENOME"  FunFormer_20k

submit salmon checkpoints/DeepSEA_salmon_5e-5_1000/best_model                      pig    "$PIG_DATA"    "$PIG_GENOME"    DeepSEA
submit salmon checkpoints/DeepFormer_salmon_5e-5_1000/best_model                    pig    "$PIG_DATA"    "$PIG_GENOME"    DeepFormer
submit salmon checkpoints/DeepATT_salmon_5e-4_1000/best_model                       pig    "$PIG_DATA"    "$PIG_GENOME"    DeepATT
submit salmon checkpoints/funformer_salmon_len20000_lr5e-4_b250_gac2/best_model     pig    "$PIG_DATA"    "$PIG_GENOME"    FunFormer_20k

submit salmon checkpoints/DeepSEA_salmon_5e-5_1000/best_model                      cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepSEA
submit salmon checkpoints/DeepFormer_salmon_5e-5_1000/best_model                    cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepFormer
submit salmon checkpoints/DeepATT_salmon_5e-4_1000/best_model                       cattle "$CATTLE_DATA" "$CATTLE_GENOME" DeepATT
submit salmon checkpoints/funformer_salmon_len20000_lr5e-4_b250_gac2/best_model     cattle "$CATTLE_DATA" "$CATTLE_GENOME" FunFormer_20k
