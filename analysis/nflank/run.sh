#!/usr/bin/env bash
# N-flank exclusion: measure it, recompute everything it touches.
#
#   bash analysis/nflank/run.sh             # measure + report (needs the GPU pass)
#   bash analysis/nflank/run.sh --submit    # queue the 30 within-species GPU jobs
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"

echo "== 1. flank N content per species =="
MEM=32G TIME=01:00:00 bash analysis/common/run_cpu.sh analysis/common/n_content.py

echo "== 2. what the excluded intervals contain =="
MEM=48G TIME=01:00:00 bash analysis/common/run_cpu.sh analysis/common/excluded_profile.py

if [[ "${1:-}" == "--submit" ]]; then
    echo "== 3. queue within-species evaluation, both conventions =="
    bash analysis/nflank/src/submit.sh
    echo "Jobs queued. Re-run without --submit once they finish."
    exit 0
fi

echo "== 3. transfer margins, both conventions =="
MEM=64G TIME=04:00:00 bash analysis/common/run_cpu.sh analysis/nflank/src/transfer_into.py

echo "== 4. final-convention per-track tables and the diff =="
MEM=32G TIME=01:00:00 bash analysis/common/run_cpu.sh analysis/nflank/src/build_final_test_set.py
MEM=32G TIME=01:00:00 bash analysis/common/run_cpu.sh analysis/nflank/src/diff_report.py
echo "Done. Artifacts in $ROOT/analysis/nflank/out/"
