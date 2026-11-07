#!/usr/bin/env bash
# Submit every prediction-cache job the manifest lists (75: 69 transfer + 6 self).
# Idempotent -- cache_predictions.py skips a direction whose .npz already exists,
# so re-running after a partial batch costs one short job per missing file.
#
#   bash analysis/tss_strata/src/submit_cache.sh          # submit
#   DRY=1 bash analysis/tss_strata/src/submit_cache.sh    # list, submit nothing
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# Freeze the code this batch will run. Jobs execute from the snapshot, so an
# edit made while they sit in the queue cannot change what they run.
SNAP="${SNAP:-$(bash analysis/common/snapshot.sh tss_strata)}"
export SNAP
echo "code snapshot: ${SNAP#$ROOT/}  ($(sed -n 's/^short *//p' "$SNAP/COMMIT"))"
mkdir -p logs

python3.12 analysis/tss_strata/src/manifest.py --tsv ${ALL_MODELS:+--all-models} > /tmp/tss_jobs.$$.tsv
n=0
while IFS=$'\t' read -r SRC TGT MODEL CKPT KIND; do
    [[ "$SRC" == "source" ]] && continue
    out="analysis/tss_strata/data/cache/${SRC}_to_${TGT}/${MODEL}.npz"
    if [[ -s "$out" ]]; then echo "have   ${SRC}->${TGT} ${MODEL}"; continue; fi
    if [[ -n "${DRY:-}" ]]; then echo "would  ${SRC}->${TGT} ${MODEL}  ${CKPT}"; n=$((n+1)); continue; fi
    SRC="$SRC" TGT="$TGT" MODEL="$MODEL" CKPT="$CKPT" \
        sbatch --export=ALL analysis/tss_strata/src/cache_predictions.slurm >/dev/null
    echo "submit ${SRC}->${TGT} ${MODEL}"
    n=$((n+1))
done < /tmp/tss_jobs.$$.tsv
rm -f /tmp/tss_jobs.$$.tsv
echo "${n} job(s)."
