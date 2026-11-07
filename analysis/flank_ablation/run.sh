#!/usr/bin/env bash
# Flank ablation at inference. Staged exactly as the design orders it: stage 1
# reports before stage 2 is run.
#
#   bash analysis/flank_ablation/run.sh                 # aggregate what exists
#   bash analysis/flank_ablation/run.sh --submit 1      # queue stage 1
#   bash analysis/flank_ablation/run.sh --submit check  # intact-window check
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3.12}"

if [[ "${1:-}" == "--submit" ]]; then
    STAGE="${2:-1}" bash "$HERE/src/submit_ablation.sh"
    exit 0
fi
"$PY" "$HERE/src/aggregate.py"
echo "Done. Artifacts in $HERE/out/"
