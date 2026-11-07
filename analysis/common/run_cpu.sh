#!/usr/bin/env bash
# Run an analysis script on a CPU compute node, from a frozen code snapshot.
#
# Analysis scripts must not run on the login node. These aggregations take tens
# of minutes and tens of GB, which is not what a shared login node is for.
# The job runs inside the amd64 image seq2fun_cpu.sif; override with CPU_SIF.
#
#   bash analysis/common/run_cpu.sh analysis/floor50/src/make_per_match.py
#   MEM=96G TIME=08:00:00 bash analysis/common/run_cpu.sh <script> [args...]
#
# Honours MEM, CPUS, TIME, PARTITION and CPU_SIF. Waits for the job and tails its log
# unless NOWAIT=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"; mkdir -p logs
SCRIPT="${1:?usage: run_cpu.sh <script-path-relative-to-repo> [args...]}"; shift || true
[[ -f "$SCRIPT" ]] || { echo "no such script: $SCRIPT" >&2; exit 1; }

# Snapshot name: the analysis directory the script belongs to. A
# manuscript/analysis/... path would otherwise yield the literal "analysis".
case "$SCRIPT" in
    analysis/*) NAME="$(echo "$SCRIPT" | cut -d/ -f2)" ;;
    manuscript/*) NAME="manuscript" ;;
    *) NAME="misc" ;;
esac
SNAP="${SNAP:-$(bash analysis/common/snapshot.sh "$NAME")}"
echo "code snapshot: ${SNAP#$ROOT/}  ($(sed -n 's/^short *//p' "$SNAP/COMMIT"))"

JID=$(SNAP="$SNAP" SCRIPT="$SCRIPT" SCRIPT_ARGS="$*" sbatch --parsable \
        --mem="${MEM:-48G}" --cpus-per-task="${CPUS:-4}" --time="${TIME:-04:00:00}" \
        --partition="${PARTITION:-small}" --export=ALL \
        analysis/common/cpu_job.slurm)
echo "submitted job $JID -> logs/analysis_${JID}.out"
[[ -n "${NOWAIT:-}" ]] && exit 0
while squeue -j "$JID" -h -o %T 2>/dev/null | grep -q .; do sleep 20; done
echo "--- logs/analysis_${JID}.out ---"
sed -e 's/\r/\n/g' "logs/analysis_${JID}.out" | grep -vE "^(Job|Submitted|Requested|Elapsed|Task and|ID |[0-9]+\.?[a-z]* +[0-9]|Used CPU|Unused CPU|Memory stat|$)"
