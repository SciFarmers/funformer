# Final evaluation convention

**The directory name is historical.** It began as the 50-positive track floor
alone; it now applies the *final* convention, which is two filters together:

| filter | unit | drops |
|---|---|---|
| N-flank exclusion | interval | any test interval whose 20 kb flank is more than 50% N on either side |
| 50-positive floor | target track | any track with fewer than 50 positives **among the intervals that survive the first filter** |

Both are applied identically to every species and to both within-species and
cross-species evaluation. Output goes to `eval_results/cross_species_final/`,
in the same shape as `eval_results/cross_species/`, so the manuscript's own
aggregator reads it unchanged via `CS_DIR`.

## Why the order matters

Positives are counted *after* the interval exclusion. Counting them on the full
set would admit a track whose positives all sit in the excluded N regions —
which is exactly the salmon H3K27me3 case the floor exists to catch.

## Scripts

```
src/make_per_match.py   rebuild per-match AUROCs under both filters
```

Runs on a CPU node:

```bash
MEM=64G bash analysis/common/run_cpu.sh analysis/floor50/src/make_per_match.py
```

The old-vs-new comparison lives in `analysis/nflank/src/diff_report.py`, which
reads the two margin tables directly; the separate comparison script here was
removed once its floor-only input tree stopped being produced.

## What the filters changed

No channel is dropped entirely: 48 target tracks fall below the floor across 10
channels, but every channel keeps at least one usable track, so the floor
removes replicate pairs rather than channels.

| | no filters | final |
|---|---|---|
| margin_arch mammal-mammal | +3.22 | +3.37 |
| margin_arch mammal-chicken | +0.41 | +0.39 |
| margin_arch tetrapod-fish | −7.07 | −7.80 |
| Pearson r | −0.758 | −0.764 |
| P species-level | 0.0167 | 0.0167 |

The species-level P is unchanged because 0.0167 is the resolution floor of a
720-relabelling exact test, not because nothing moved.

The no-filter margins stay at
`manuscript/analysis/out/cross_species_margins_nofloor.csv` and are the
sensitivity row of Supplementary Table S2.
