#!/usr/bin/env bash
# Submit flank-ablation jobs, staged the way the spec orders them.
#
#   STAGE=1 : shuffled + random_same, k=1kb, seed 0, every direction  (58 jobs)
#             -- report before anything else is run
#   STAGE=2 : seeds 1 and 2 for those, plus random_source x3 seeds    (185 jobs)
#   STAGE=3 : graded k in {1,2.5,5,10,20} kb, random_same only        (conditional)
#   STAGE=check  : assert the intact window matches Seq2FunDataset, 6 short jobs
#   STAGE=intact : actually run intact through C's path, so verify_intact.py can
#                  compare it to the A cache BITWISE. Normally intact is not run
#                  at all -- A already cached it -- so this exists only as the
#                  end-to-end proof that the two paths are the same file.
#
# DEP=<jobid[:jobid...]> gates a stage on earlier jobs finishing OK, which is
# how stage 1 is held behind STAGE=check.
#
# The intact condition is never submitted: analysis/tss_strata already cached
# exactly those predictions (same model, channels, intervals).
#
#   DRY=1 STAGE=1 bash analysis/flank_ablation/src/submit_ablation.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# Freeze the code this batch will run. Jobs execute from the snapshot, so an
# edit made while they sit in the queue cannot change what they run.
SNAP="${SNAP:-$(bash analysis/common/snapshot.sh flank_ablation)}"
export SNAP
echo "code snapshot: ${SNAP#$ROOT/}  ($(sed -n 's/^short *//p' "$SNAP/COMMIT"))"; mkdir -p logs
STAGE="${STAGE:-1}"

python3.12 analysis/tss_strata/src/manifest.py --tsv \
  | awk -F'\t' 'NR>1 && $3=="FunFormer_20k"' > /tmp/abl_jobs.$$.tsv

submit () {  # src tgt cond keep seed ckpt [extra]
    local out="analysis/flank_ablation/data/cache/${1}_to_${2}/FunFormer_20k__${3}_k${4}_s${5}.npz"
    if [[ -z "${7:-}" && -s "$out" ]]; then echo "have   $1->$2 $3 k$4 s$5"; return; fi
    if [[ -n "${DRY:-}" ]]; then echo "would  $1->$2 $3 k$4 s$5"; N=$((N+1)); return; fi
    local dep=""
    # plain `[[ -n x ]] && dep=...` would return non-zero on the empty case and
    # kill the script under `set -e`
    if [[ -n "${DEP:-}" ]]; then dep="--dependency=afterok:${DEP}"; fi
    SRC="$1" TGT="$2" COND="$3" KEEP="$4" SEED="$5" CKPT="$6" EXTRA="${7:-}" \
        sbatch ${dep} --export=ALL analysis/flank_ablation/src/run_ablation.slurm >/dev/null
    echo "submit $1->$2 $3 k$4 s$5"; N=$((N+1))
}

N=0
while IFS=$'\t' read -r SRC TGT MODEL CKPT KIND; do
    case "$STAGE" in
      check) if [[ "$KIND" == "self" ]]; then submit "$SRC" "$TGT" intact 1000 0 "$CKPT" "--check-intact"; fi ;;
      intact) submit "$SRC" "$TGT" intact 1000 0 "$CKPT" ;;
      1) for C in shuffled random_same; do submit "$SRC" "$TGT" "$C" 1000 0 "$CKPT"; done ;;
      2) # random_source, one seed, transfer directions only. The extra seeds for
         # shuffled/random_same were dropped: stage 1 already answered what they
         # were there to check.
         if [[ "$KIND" == "transfer" ]]; then
             submit "$SRC" "$TGT" random_source 1000 0 "$CKPT"
         fi ;;
      3) for K in 1000 2500 5000 10000 20000; do submit "$SRC" "$TGT" random_same "$K" 0 "$CKPT"; done ;;
      *) echo "unknown STAGE=$STAGE" >&2; exit 1 ;;
    esac
done < /tmp/abl_jobs.$$.tsv
rm -f /tmp/abl_jobs.$$.tsv
echo "${N} job(s) for STAGE=${STAGE}."
