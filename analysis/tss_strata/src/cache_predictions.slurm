#!/bin/bash
# One (direction, model) prediction cache. Submitted in bulk by submit_cache.sh.
#SBATCH --account=nn9780k
#SBATCH --job-name=tss_cache
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/tss_cache_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Run the frozen copy of the code, never the working tree: a queued job starts
# long after submission and must not be changed by an edit in between. SNAP is
# set by the submitter; REPO_ROOT keeps checkpoints and datasets resolving to
# the real repository.
: "${SNAP:?SNAP not set -- submit through the submitter, which snapshots the code}"
export REPO_ROOT="${SLURM_SUBMIT_DIR}"
export SNAPSHOT_ID="$(sed -n 's/^short *//p' "${SNAP}/COMMIT")"
echo "code snapshot: ${SNAP}  (${SNAPSHOT_ID})"

apptainer exec --nv --pwd /workspace \
    --bind "$PWD":/workspace --bind /cluster:/cluster \
    /cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun.sif \
    env PYTHONPATH="${SNAP}/code" REPO_ROOT="${REPO_ROOT}" SNAPSHOT_ID="${SNAPSHOT_ID}" \
        python "${SNAP}/code/analysis/tss_strata/src/cache_predictions.py" \
        --source "${SRC}" --target "${TGT}" \
        --model "${MODEL}" --checkpoint "${CKPT}" \
        --batch-size "${BATCH_SIZE:-256}" --num-workers "${NUM_WORKERS:-4}"
