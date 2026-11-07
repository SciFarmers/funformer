#!/usr/bin/env bash
# Freeze the code a batch will run, and print the directory holding it.
#
# Jobs must never read a working-tree script. A queued job starts minutes or
# hours after submission, so an edit made in between silently changes what it
# runs, and a job can fail after writing part of its output because the code
# moved under it.
#
# Each submission therefore copies the code it needs into
# analysis/_runs/<name>/<UTC stamp>-<short hash>[-dirty]/code/ and the jobs run
# from that copy. Only code is copied (a few MB); checkpoints, datasets and
# outputs stay where they are and are reached through REPO_ROOT.
#
# A dirty tree is allowed but marked, and the exact diff is saved beside the
# snapshot, so a run is always reconstructible even when it was not committed.
#
#   SNAP=$(bash analysis/common/snapshot.sh flank_ablation)
#
set -euo pipefail
NAME="${1:?usage: snapshot.sh <analysis-name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HASH="$(git rev-parse --short=12 HEAD)"
DIRTY=""
if ! git diff --quiet HEAD -- src src_evaluate meta_info analysis; then DIRTY="-dirty"; fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
# Kept outside analysis/<name>/ so snapshotting a directory that would
# otherwise contain its own snapshots cannot recurse into itself.
SNAP="$ROOT/analysis/_runs/$NAME/${STAMP}-${HASH}${DIRTY}"

mkdir -p "$SNAP/code"
for d in src src_evaluate meta_info manuscript/analysis analysis/common "analysis/$NAME/src" analysis/tss_strata/src; do
    [[ -d "$d" ]] || continue
    mkdir -p "$SNAP/code/$(dirname "$d")"
    cp -r "$d" "$SNAP/code/$d"
done

{
    echo "commit      $(git rev-parse HEAD)"
    echo "short       ${HASH}${DIRTY}"
    echo "branch      $(git rev-parse --abbrev-ref HEAD)"
    echo "snapshot_at ${STAMP}"
    echo "analysis    ${NAME}"
} > "$SNAP/COMMIT"
if [[ -n "$DIRTY" ]]; then
    git diff HEAD -- src src_evaluate meta_info analysis > "$SNAP/uncommitted.diff"
fi

echo "$SNAP"
