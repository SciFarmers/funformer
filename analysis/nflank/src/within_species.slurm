#!/bin/bash
#SBATCH --account=nn9780k
#SBATCH --job-name=nflank_ws
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nflank_ws_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${SNAP:?SNAP not set}"
export REPO_ROOT="${SLURM_SUBMIT_DIR}"
export SNAPSHOT_ID="$(sed -n 's/^short *//p' "${SNAP}/COMMIT")"
echo "code snapshot: ${SNAP}  (${SNAPSHOT_ID})"
apptainer exec --nv --pwd /workspace \
    --bind "$PWD":/workspace --bind /cluster:/cluster \
    /cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif \
    env PYTHONPATH="${SNAP}/code" REPO_ROOT="${REPO_ROOT}" SNAPSHOT_ID="${SNAPSHOT_ID}" \
        python "${SNAP}/code/analysis/nflank/src/within_species.py" \
        --species "${SPECIES}" --model "${MODEL}" --checkpoint "${CKPT}" \
        --batch-size "${BATCH_SIZE:-256}" --num-workers "${NUM_WORKERS:-4}"
