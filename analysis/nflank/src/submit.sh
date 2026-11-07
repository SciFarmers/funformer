#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ROOT"; mkdir -p logs
SNAP="${SNAP:-$(bash analysis/common/snapshot.sh nflank)}"; export SNAP
echo "code snapshot: ${SNAP#$ROOT/}"
n=0
while IFS=$'\t' read -r SP M CK; do
    out="analysis/nflank/out/${SP}__${M}.csv"
    if [[ -s "$out" ]]; then echo "have   $SP $M"; continue; fi
    if [[ -n "${DRY:-}" ]]; then echo "would  $SP $M"; n=$((n+1)); continue; fi
    SPECIES="$SP" MODEL="$M" CKPT="$CK" sbatch --export=ALL \
        analysis/nflank/src/within_species.slurm >/dev/null
    echo "submit $SP $M"; n=$((n+1))
done < <(python3.12 - <<'PY'
import sys
sys.path.insert(0, "analysis/tss_strata/src")
from manifest import TAG, checkpoint, ALL_MODELS
for sp in sorted(TAG):
    for m in ALL_MODELS:
        try: ck, _ = checkpoint(sp, m)
        except SystemExit: continue
        print(f"{sp}\t{m}\t{ck}")
PY
)
echo "${n} job(s)."
