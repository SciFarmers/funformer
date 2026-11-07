# VEP — main pipeline (src_evaluate.vep_eqtl_finemap)
# Output: bench_results/vep_eqtl/<MODEL_TAG>/
# Run 2b.run_vep.sh for RF evaluation after these jobs complete.

MODEL_CLASS=DeepSEA MODEL_PATH=checkpoints/DeepSEA_1e-4_1000/best_model \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

# MODEL_CLASS=DanQ MODEL_PATH=checkpoints/DanQ_1e-4_1000/best_model \
#   sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

MODEL_CLASS=DeepATT MODEL_PATH=checkpoints/DeepATT_5e-4_1000/best_model \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

MODEL_CLASS=DeepFormer MODEL_PATH=checkpoints/DeepFormer_5e-4_1000/best_model \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

MODEL_CLASS=FunFormer MODEL_PATH=checkpoints/funformer_entex_423_len20000_lr5e-4_b250_gac2/best_model WINDOW_SIZE=20000 \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh
