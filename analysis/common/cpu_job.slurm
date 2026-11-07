#!/bin/bash
# Generic CPU batch job for analysis scripts. Submitted by run_cpu.sh.
#
# Analysis runs inside the amd64 companion image seq2fun_cpu.sif, built from
# the same pinned recipe as the GPU image, so analysis sees the same library
# versions as training. Override the image with CPU_SIF.
#SBATCH --account=nn9780k
#SBATCH --job-name=analysis
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/analysis_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

: "${SNAP:?SNAP not set -- submit through analysis/common/run_cpu.sh}"
: "${SCRIPT:?SCRIPT not set}"
CPU_SIF="${CPU_SIF:-/cluster/projects/nn9780k/dat/olivia_containers/builds/seq2fun_cpu.sif}"
[[ -f "${CPU_SIF}" ]] || { echo "no CPU image at ${CPU_SIF}" >&2; exit 1; }

export REPO_ROOT="${SLURM_SUBMIT_DIR}"
export SNAPSHOT_ID="$(sed -n 's/^short *//p' "${SNAP}/COMMIT")"
export PYTHONPATH="${SNAP}/code"
echo "code snapshot: ${SNAP}  (${SNAPSHOT_ID})"
echo "script       : ${SCRIPT}"
echo "image        : ${CPU_SIF}"

apptainer exec --bind /cluster:/cluster "${CPU_SIF}" \
    python -u "${SNAP}/code/${SCRIPT}" ${SCRIPT_ARGS:-}
