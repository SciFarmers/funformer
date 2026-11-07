# VEP — apply every species-trained FunFormer model to the human GTEx VCFs.
#
# All five non-human species are run, so that "human + all species" means all
# five and not a subset. Chicken and zebrafish were added after the first pass;
# before that, combined_human_all_species covered only pig, cattle and salmon,
# which made the "+ All" label wrong.
#
# The submitter skips any VCF whose output already exists, so re-running this
# script is safe and only computes what is missing.
#
# Output: bench_results/vep_eqtl/funformer_<species>_len20000_lr5e-4_b250_gac2/
# Run 2b.run_vep_rf.sh after all five jobs complete.

MODEL_CLASS=FunFormer \
MODEL_PATH=checkpoints/funformer_pig_len20000_lr5e-4_b250_gac2/best_model \
WINDOW_SIZE=20000 \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

MODEL_CLASS=FunFormer \
MODEL_PATH=checkpoints/funformer_cattle_len20000_lr5e-4_b250_gac2/best_model \
WINDOW_SIZE=20000 \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

MODEL_CLASS=FunFormer \
MODEL_PATH=checkpoints/funformer_salmon_len20000_lr5e-4_b250_gac2/best_model \
WINDOW_SIZE=20000 \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

MODEL_CLASS=FunFormer \
MODEL_PATH=checkpoints/funformer_chicken_len20000_lr5e-4_b250_gac2/best_model \
WINDOW_SIZE=20000 \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh

MODEL_CLASS=FunFormer \
MODEL_PATH=checkpoints/funformer_zebrafish_len20000_lr5e-4_b250_gac2/best_model \
WINDOW_SIZE=20000 \
  sbatch --export=ALL olivia_submiter/vep_eqtl_finemap.sh
