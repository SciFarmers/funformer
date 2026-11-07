#!/usr/bin/env bash
# Final evaluation convention: N-flank exclusion + 50-positive track floor.
# Every stage runs on a CPU compute node, never the login node.
#
#   bash analysis/floor50/run.sh            # rebuild the per-match tables
#   bash analysis/floor50/run.sh --regen    # also make it primary and rebuild figures
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"

echo "== 1. per-match AUROCs under both filters =="
MEM=64G TIME=04:00:00 bash analysis/common/run_cpu.sh analysis/floor50/src/make_per_match.py

if [[ "${1:-}" == "--regen" ]]; then
    echo "== 2. make it primary and rebuild the artifacts =="
    MEM=64G TIME=04:00:00 bash analysis/common/run_cpu.sh analysis/nflank/src/final_regen.py
fi
echo "Done. Artifacts in $ROOT/analysis/floor50/out/"
