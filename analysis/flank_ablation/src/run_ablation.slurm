#!/bin/bash
#SBATCH --account=nn9780k
#SBATCH --job-name=flank_abl
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/flank_abl_%j.out
# 80G and 8 cpus: random_source holds two whole genomes in RAM at once, and the
# dinucleotide shuffle is CPU-bound in the dataloader workers (~6.4 ms per 19 kb
# flank), so the GPU starves with fewer.
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
        python "${SNAP}/code/analysis/flank_ablation/src/run_ablation.py" \
        --source "${SRC}" --target "${TGT}" --condition "${COND}" \
        --keep-bp "${KEEP:-1000}" --seed "${SEED:-0}" --checkpoint "${CKPT}" \
        --batch-size "${BATCH_SIZE:-256}" --num-workers "${NUM_WORKERS:-8}" \
        ${EXTRA:-}
