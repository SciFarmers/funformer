#!/usr/bin/env bash
# TSS-distance stratified transfer. CPU stages run anywhere; the GPU stage is
# SLURM and is skipped unless the cache is missing.
#
#   bash analysis/tss_strata/run.sh            # CPU stages (needs the cache)
#   bash analysis/tss_strata/run.sh --submit   # also queue the GPU cache jobs
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3.12}"

echo "== 1. GTFs (assembly-pinned) =="
bash "$HERE/src/fetch_gtf.sh" >/dev/null

echo "== 2. TSS distance per test interval =="
"$PY" "$HERE/src/tss_distance.py"

if [[ "${1:-}" == "--submit" ]]; then
    echo "== 3. Queue prediction-cache jobs =="
    bash "$HERE/src/submit_cache.sh"
    echo "Jobs queued. Re-run without --submit once they finish."
    exit 0
fi

echo "== 3. Regression checks =="
"$PY" "$HERE/src/validate.py"

echo "== 4. Aggregate + stats =="
"$PY" "$HERE/src/aggregate.py"
echo "Done. Artifacts in $HERE/out/"
