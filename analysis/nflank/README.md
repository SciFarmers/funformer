# N-flank exclusion: measurement and recomputation

Quantifies how much of each species' test set sits beside unplaced sequence,
and recomputes every affected number with those intervals removed.

## The finding

Fraction of test intervals whose 20 kb window is more than 50% N on at least
one side:

| species | test chrom | intervals | either side | both sides |
|---|---|---|---|---|
| cattle | 21 | 349,314 | 0.01% | 0.00% |
| chicken | 21 | 34,224 | 0.15% | 0.00% |
| **human** | 21 | 233,549 | **14.80%** | **13.56%** |
| pig | 16 | 399,721 | 0.01% | 0.00% |
| salmon | 21 | 299,099 | 0.02% | 0.00% |
| zebrafish | 21 | 229,478 | 0.02% | 0.00% |

Human is the outlier by three orders of magnitude: the acrocentric p-arm of
chromosome 21 is 14.2% unplaced N. "Both sides" being nearly as large as
"either side" shows these intervals sit *inside* the block, not beside it —
and 95.8% of them have a central 1 kb that is itself mostly N, so the model
sees N in the window centre too.

They are also almost entirely label-free: 99.3% of the excluded human intervals
are negative in all 423 tracks, and together they hold 0.31% of human's
positives. Excluding them therefore removes easy negatives, which lowers AUROC
and raises AUPRC — both observed.

This is a property of the main evaluation, not only of the flank ablation, and
belongs in Methods on that basis.

## Scripts

```
src/within_species.py     GPU: per-track AUROC/AUPRC with and without the exclusion,
                          computed in one pass so the pair is exactly comparable
src/submit.sh             submit those, 5 models x 6 species
src/transfer_into.py      transfer margins under both conventions
src/build_final_test_set.py  per-track tables under the final convention,
                          shaped like eval_results/test_set/ so TEST_SET_DIR
                          can redirect Table 1 and Figure 2 at them
src/final_regen.py        regenerate every manuscript artifact under it
src/diff_report.py        old vs new for every number that appears in main.tex
```

## Caveat on the companion measurement

`analysis/common/n_content.py` defines flanks with the ablation geometry
(20 kb window, central 1 kb kept, so 9.5 kb per side). The Methods number and
the ablation exclusion are therefore the same quantity, which is deliberate.
