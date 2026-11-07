# Analysis A — TSS-distance stratified transfer

Is the FunFormer-20k advantage, and its decay with divergence, concentrated in
distal or proximal regulatory elements?

Exploratory. Writes only to `analysis/tss_strata/out/`; nothing here is read by
the manuscript pipeline.

## Status

Code complete and dry-tested. **Numbers pending**: all 75 prediction-cache jobs
are queued and the Olivia cluster is in maintenance (every partition `down`).
Re-run `bash run.sh` once they finish.

| check | state |
|---|---|
| A aggregation — direction means reproduce `cross_species_margins.csv` | **passes**, 69/69 |
| B unstratified — recomputed AUROC matches the committed per-match values | pending cache |
| C pooling — three exhaustive strata cover every positive | pending cache |
| AUROC implementation vs brute-force definition | **exact**, 200/200 incl. ties and fp16 |

## How to read the result

**The comparison is within direction: distal minus proximal.** Each direction
supplies both strata over the same channels and the same negatives, so the
difference is internally controlled.

**A single stratum must not be compared across bins.** "Distal > 10 kb" means a
different thing in each species, because the test split is one chromosome and
those chromosomes differ sharply in gene density — 88% of human chr21 intervals
are more than 10 kb from a TSS against 50% for chicken. A bin contrast on the
distal stratum alone therefore mixes the context effect with how gene-poor each
test chromosome happens to be. The distal-minus-proximal difference is the
quantity that survives this; the species-level permutation test is run on it.

## Design

Strata on **positives only**; negatives are shared, so the two AUROCs differ in
which positives must outrank the same background, not in the background.

* primary: proximal ≤ 2 kb, distal > 10 kb, 2–10 kb dropped
* sensitivity: ≤ 1 kb / > 20 kb
* exhaustive (for the pooling check): ≤ 2 kb / 2–10 kb / > 10 kb

TSS are protein-coding transcript 5′ ends from Ensembl GTFs **pinned to the
assembly each model was trained on** — zebrafish GRCz10 needs release 91, chicken
GRCg6a release 105, cattle ARS-UCD1.2 release 110. A newer release would decouple
TSS coordinates from the sequence the model saw.

## Decisions taken

1. **Baseline reference held fixed** at each direction's unstratified winner
   (DeepFormer 10, DeepATT 9, DeepSEA 4 of 23). Re-selecting per stratum would
   let proximal and distal reference different models, so their difference would
   partly be a change of comparator.
2. **50-positive floor applied in both strata jointly.** A channel enters both or
   neither, so the comparison is paired on one channel set.

   Measured cost, written per direction to `out/direction_survival_*.tsv`:
   **354 of 359 matched channels survive (98.6%), and no direction loses as
   much as half** — under either the primary or the sensitivity strata. The
   floor bites harder on individual target tracks (19 of 88 salmon tracks, 1 of
   79 cattle) than on channels, because a channel survives if any of its
   replicate tracks does.
3. **Pooling check restated.** The spec asked that proximal + distal reproduce
   +3.2 / +0.4 / −7.2 pp. That cannot hold while the 2–10 kb band is excluded —
   those positives are absent. The identity is checked against the three
   exhaustive strata instead; the unstratified regression check (B) is the gate.

## Constraint worth knowing

The test split is a **single held-out chromosome** per species — 21, or 16 for
pig — at 200 bp per interval. "Distal > 10 kb" is therefore relative to that one
chromosome's gene density, which differs sharply between species: 88% of human
chr21 intervals are >10 kb from a TSS against 50% for chicken. Per-stratum
counts are written to `out/stratum_sizes_*.tsv` before any aggregate.

Positives concentrate near TSS regardless, so power is adequate: median
proximal/distal positives per channel are 1524/1257 (human) to 2780/1977
(zebrafish).

## Layout

```
src/fetch_gtf.sh          assembly-pinned Ensembl GTFs
src/tss_distance.py       midpoint -> nearest protein-coding TSS, per species
src/manifest.py           (direction, model) -> checkpoint, DERIVED not retyped
src/cache_predictions.py  GPU: cache matched-channel probs (fp16)
src/submit_cache.sh       queue the 75 jobs; idempotent
src/strata.py             stratified AUROC (Mann-Whitney, no sklearn needed)
src/validate.py           checks A, B, C
src/aggregate.py          channel -> direction -> bin TSVs + permutation stats
```

`data/` (GTFs, ~2 GB of cached predictions) is gitignored and regenerable.
