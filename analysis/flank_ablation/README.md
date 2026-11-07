# Analysis C — flank ablation at inference

Does FunFormer-20k use sequence beyond the central 1 kb, and is that flank what
fails to transfer across species?

Confirmatory, not exploratory: the conditions, the donor-matching rule and
the verification criteria below were fixed before any ablation run, and the
analysis rests on predictions that did not exist when the gradient was
observed. The splits of rescue by source clade and by TSS stratum came after
the bin result and are descriptive. Writes only to
`analysis/flank_ablation/out/`.

## Status

Code complete and dry-tested. **Numbers pending**: 6 intact-check jobs and 58
stage-1 jobs are queued, the 58 gated on the checks passing. Cluster is in
maintenance.

| check | state |
|---|---|
| dinucleotide shuffle preserves length, dinucleotide counts, N positions | **passes**, 500/500 |
| shuffle preserves GC exactly | **passes** (0.38037 → 0.38037 on 19 kb) |
| intact window token-identical to `Seq2FunDataset` | queued (`STAGE=check`) |
| intact predictions **bitwise identical** to the A cache | queued (`STAGE=intact`, then `src/verify_intact.py`) |

A difference at float precision — max |delta| ≤ 1e-3, roughly two fp16 ULP near
0.5 — is GPU non-determinism, from a different card or cuDNN algorithm between
the two runs. It is recorded in `out/intact_bitwise_report.md` and accepted.
Only a larger or structured delta fails and warrants investigation. Magnitude is
the discriminator, not how many entries differ: 1-ULP jitter is expected to
touch many of them.

The report gives **two** deltas per direction: the prediction-level maximum, and
the maximum **channel-level AUROC** difference over the matched channels. The
first says the bits differ; only the second says whether that reaches a number
anyone reads. A channel AUROC moving by less than 1e-4 changes nothing at the
precision any result is quoted to, and the report states outright whether the
largest such movement clears that bar.
| three seeds agree | pending stage 2 |

## Conditions

| | flank replaced with |
|---|---|
| `intact` | nothing — **reuses Analysis A's cache**, same model, channels and intervals, so it costs no GPU time and is identical by construction |
| `shuffled` | the flank's own dinucleotide shuffle (Altschul-Erikson) |
| `random_same` | a real, GC-matched locus from the target genome |
| `random_source` | a real locus from the *source* genome, around the target's real central k bp |

N is never used as filler, and donor loci carrying >1% N are rejected: 19 kb of N
is far outside the training distribution, which is what made the earlier
N-padding attempt uninterpretable.

The reverse strand is recomputed from the modified forward sequence. Reusing the
genome's original reverse strand would score a different sequence on one of the
two strands the pipeline averages.

## Donor sampling

Must-hold, as revised: **replaced flanks match the original in length and in GC
within the stated tolerance, with achieved deltas reported.** A real flank from a
random locus carries that locus's GC, so exact matching is impossible; without
GC matching, `random_same` would confound "the flank is foreign" with "the flank
has different GC", which is the contrast it exists to make. `shuffled` preserves
GC exactly by construction and needs none of this.

Both `random_same` and `random_source` match the GC of **the flank being
replaced** — for `random_source` that is the *target's* flank, matched by a donor
drawn from the *source* genome, so the only thing distinguishing the two
conditions is which genome supplied real sequence.

Tolerance escalates rather than dropping intervals:

| | |
|---|---|
| tier 1 | ±2% absolute GC, up to 40 draws |
| tier 2 | ±3%, up to 40 more |
| fallback | closest candidate seen is kept; no interval is ever dropped |

Rejected outright: donors on the donor genome's own held-out chromosome, donors
more than 1% N, and donors containing an N run of 50 bp or more (assembly gaps).

Per run, the fraction of flanks satisfied at each tier is printed and the
achieved GC deltas are written with the donor table.

### Chromosome exclusion

The test split tiles a single chromosome, so **every** donor drawn from it would
overlap some evaluated interval's own 20 kb window. Barring that chromosome
outright is the same rule stated once, and it replaces the original
"same chromosome where possible", which pointed the opposite way. For
`random_source` the bar falls on the *source* species' held-out chromosome, which
keeps donors inside sequence the source model actually trained on.

### Reproducibility

Donor provenance is written per interval and per seed to
`data/cache/<direction>/donors__<condition>_k<k>_s<seed>.tsv.gz`:
row, side, donor chromosome, donor start, original GC, donor GC, tier satisfied.
The ablation can be reconstructed from that table without rerunning the sampler.

Each dataloader worker writes its own log and the job concatenates them —
appending to a list on the dataset object would be silently discarded, since
every worker holds its own copy.

## Run order

As specified: stage 1 (`shuffled` + `random_same`, k = 1 kb, one seed, 58 jobs)
reports before anything else runs. Stage 2 adds seeds 1–2 and `random_source`
(185 jobs). Stage 3, the graded k ∈ {1, 2.5, 5, 10, 20} kb series for
`random_same` only (145 jobs), runs only if stage 1 shows signal. k = 20 kb
replaces the whole window and doubles as a sanity point.

```
STAGE=check  bash src/submit_ablation.sh   # 6 jobs: window arithmetic
DEP=<ids> STAGE=1 bash src/submit_ablation.sh
STAGE=intact bash src/submit_ablation.sh   # 29 jobs, then:
python3.12 src/verify_intact.py            # bitwise vs the A cache
```

`verify_intact.py` compares the arrays with `np.array_equal`, not a tolerance:
the two paths run the same checkpoint over the same intervals, so the files
should be identical bit for bit. A tolerance check would pass even if the
ablation dataset mis-centred the window by a few bases. If they differ, the max
absolute difference is printed so a genuine fault is distinguishable from GPU
non-determinism.

## Validity checks

`src/verify_holdout.py` confirms the three properties the ablation rests on and
does not itself check. The split must be unsharded, so that reading one arrow
shard really gives the test chromosome. That chromosome must be absent from the
training split, or donors would come from sequence the source model has already
seen and rescue would measure recall rather than transferable information. And
the donors drawn must actually lie on it.

```bash
bash analysis/common/run_cpu.sh analysis/flank_ablation/src/verify_holdout.py
```

## Cost note

The dinucleotide shuffle is CPU-bound at 6.4 ms per 19 kb flank, so the GPU
starves on 4 dataloader workers; jobs request 8 CPUs and 80 GB, the memory being
for `random_source`, which holds two whole genomes in RAM at once.

## GC distance is not an independent variable here

A correlation between source-flank rescue and the GC distance between the
source genome and the target flanks is **not** evidence that composition drives
rescue, because the design already equalises composition: `random_source`
donors are GC-matched to the flank they replace (±2%, widening to ±3%). A
genome-wide GC distance therefore measures a property the sampler is actively
removing.

What it does predict is how well that matching succeeded. Within the
tetrapod-fish bin, over 14 directions:

| | r |
|---|---|
| GC distance vs rescue, unadjusted | −0.628 (P = 0.016) |
| GC distance vs rescue, transfer direction held fixed | −0.657 (P = 0.011) |
| the same as partial Spearman | −0.476 (P = 0.086) |
| GC distance vs fraction of flanks matched within 2% | −0.678 |
| GC distance vs achieved GC gap of the donors used | +0.835 |
| **achieved GC gap vs rescue** | **−0.772 (P = 0.0012)** |

The achieved gap predicts rescue better than the genome-wide distance does, and
the genome-wide distance predicts the achieved gap. The chain that fits is
mechanical rather than biological: a more distant genome offers fewer
GC-compatible donors, the sampler settles for a worse match, and a worse-matched
flank rescues less. Read as a biological result it would be an artefact of
imperfect matching.

Descriptive, within the six tetrapod-to-fish directions: rescue into salmon
+5.50 pp (3 directions, +3.48 to +7.92), into zebrafish +3.89 pp (3 directions,
+2.20 to +5.14).

Within-group correlations are unstable at this size: tetrapod→fish n=6 gives
r = −0.42 (P = 0.41), fish→tetrapod n=8 gives r = −0.75 (P = 0.034).

None of this is in the manuscript, and none of it has a figure or table.
