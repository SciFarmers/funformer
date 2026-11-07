#!/usr/bin/env bash
# Reproduce every analysis artifact for the manuscript, in dependency order,
# then (optionally) build the PDF. Run from anywhere:
#
#   bash manuscript/analysis/run_all.sh          # regenerate figures + tables
#   bash manuscript/analysis/run_all.sh --pdf    # also run ../latex/compile.sh
#
# Naming: every runnable script emits exactly one artifact into
# out/{figures,tables}, and its prefix says which --
#
#   N.*.py    -> Figure N                 (main text)
#   tN.*.py   -> Table N                  (main text)
#   sN.*.py   -> Supplementary Figure SN
#   tsN.*.py  -> Supplementary Table SN
#
# Everything without a numeric prefix (divergence_times.py,
# aggregate_cross_species.py, stats_gradient.py, figutil.py) is a library
# imported by those scripts and is never run directly.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Exploratory analyses live outside this pipeline and are OFF by default. They
# write only to their own analysis/<name>/out/ and touch nothing the manuscript
# reads, so they short-circuit here rather than running after the figures.
#   bash run_all.sh --tss-strata       TSS-distance stratified transfer
#   bash run_all.sh --flank-ablation   flank ablation at inference
#   bash run_all.sh --nflank           N-flank exclusion: measure and recompute
#   bash run_all.sh --final-convention N-flank exclusion + 50-positive floor
case "${1:-}" in
    --tss-strata)       exec bash "$HERE/../../analysis/tss_strata/run.sh" "${2:-}" ;;
    --flank-ablation)   exec bash "$HERE/../../analysis/flank_ablation/run.sh" "${2:-}" "${3:-}" ;;
    --nflank)           exec bash "$HERE/../../analysis/nflank/run.sh" "${2:-}" ;;
    --final-convention) exec bash "$HERE/../../analysis/floor50/run.sh" "${2:-}" ;;
esac

PY="${PYTHON:-python}"
mkdir -p out/figures out/tables

# The final evaluation convention -- N-flank exclusion plus the 50-positive
# floor -- lives in its own result trees. Every script that reads a tree
# directly honours one of these two variables, so exporting them here is what
# makes this pipeline reproduce the committed artifacts. Unset them to rebuild
# the paper against the unfiltered trees instead.
ROOT_DIR="$(cd "$HERE/../.." && pwd)"
export CS_DIR="${CS_DIR:-$ROOT_DIR/eval_results/cross_species_final}"
export TEST_SET_DIR="${TEST_SET_DIR:-$ROOT_DIR/eval_results/test_set_final}"

echo "== 1. Data + statistics (dependencies) =="
"$PY" aggregate_cross_species.py                 # -> out/cross_species_margins.csv
"$PY" stats_gradient.py | tee out/stats_report.txt

echo "== 2. Figures =="
"$PY" 1.study_design.py
"$PY" 2.model_performance.py
"$PY" 3.cross_species.py
"$PY" 5.vep.py

echo "== 3. Tables =="
"$PY" t1.margin_summary.py

echo "== 4. Supplement =="
# Glob the numbered supplement scripts only. A bare "s*.py" would also match
# stats_gradient.py, which then re-runs the whole 200k-resample permutation
# battery a second time -- harmless but it roughly doubles the pipeline's
# wall-clock and writes its report to stdout instead of out/stats_report.txt.
for s in ts[0-9]*.py s[0-9]*.py; do
    [ -e "$s" ] || continue
    "$PY" "$s"
done

echo "== 5. Flank ablation figures =="
# Figure 4 and Figure S5 are built from analysis/flank_ablation/out/, not from
# the trees above, so they live outside this directory but belong to the paper.
REPO_ROOT="$ROOT_DIR" "$PY" "$ROOT_DIR/analysis/flank_ablation/src/fig4.py"
REPO_ROOT="$ROOT_DIR" "$PY" "$ROOT_DIR/analysis/flank_ablation/src/figS_stratum.py"

if [[ "${1:-}" == "--pdf" ]]; then
    if [[ -f ../latex/compile.sh ]]; then
        echo "== 6. Build PDF =="
        bash ../latex/compile.sh
    else
        echo "no LaTeX source here; the figures and tables above are the output"
    fi
fi

