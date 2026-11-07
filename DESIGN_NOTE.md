# Cross-Species Regulatory Sequence Classification

## Motivation

Within-species evaluation shows that FunFormer with a longer input beats the
1 kb models in every species. This analysis asks how far that advantage
travels. Can a model trained on one species' regulatory peaks rank another
species' intervals, and does the longer context still help once the genome
changes?

The comparison that answers it is FunFormer-20k against FunFormer-1k, since
those two share an architecture and a parameter count and differ only in how
much sequence they see.

---

## Design

### Models evaluated

Five models per source species, each the checkpoint whose learning rate won on
the validation split.

| Model | Context | Role |
|---|---|---|
| DeepSEA | 1 kb | published baseline |
| DeepFormer | 1 kb | published baseline |
| DeepATT | 1 kb | published baseline |
| FunFormer-1k | 1 kb | architecture- and parameter-matched control |
| FunFormer-20k | 20 kb | the long-context model |

Baseline tags follow `<Model>_<species>_<lr>_1000`, and FunFormer follows
`funformer_<tag>_len20000_lr5e-4_b250_gac2` at 20 kb and
`funformer_<tag>_len1000_lr5e-4_b500_gac1` at 1 kb.

FunFormer-1k is what makes the result interpretable. Without it the comparison
confounds a transformer against convolutions with 20 kb against 1 kb, and the
two effects pull in different directions.

### Evaluation directions

Six species give 30 ordered directions with a usable matched-channel set, and
the paper reports 23 of them as the primary set.

Seven are set aside for reasons fixed before the results were seen. The five
chicken-source directions are excluded because the chicken model is the
weakest within its own genome, so a low transfer score would measure the source
model rather than the distance between genomes. The two salmon-zebrafish
directions are excluded because both are teleosts, so they carry no information
about the tetrapod-fish boundary the analysis is about, and salmon's
salmonid-specific duplication makes the pair unlike every other.

Each direction is run for all five models, and the margin reported per
direction is FunFormer-20k minus FunFormer-1k over the channels both models
cover.

### Inference logic

For direction Source → Target:
1. Load source backbone (reads `model_config.json` automatically → correct seq_len).
2. Wrap in `DualStrandWrapper` (fwd + rev strand averaging, same as within-species).
3. Build `Seq2FunDataset(split="test")` from **target species** HF dataset + genome.
   The dataset provides target genomic intervals and binary peak labels.
4. Run forward pass → predictions `(N, n_source_tracks)`.
5. For each matched (assay, tissue) group (see below), compute AUROC over every
   `source_idx × target_idx` pair, then average → one AUROC per group.
6. Report mean AUROC across all matched groups in the summary CSV.

The model never sees target species training data. Sequence input comes from the
target genome at the target species' test intervals.

---

## Track Matching

### Why matching is needed

Source model outputs `n_source_tracks` channels (e.g., 423 for human FunFormer).
Target labels have `n_target_tracks` channels (e.g., 80 for pig).  Only channels
corresponding to the **same assay in the same tissue** are comparable.

### Script

```
python meta_info/find_matched_tracks.py
```

Run on the login node before submitting evaluation jobs. Outputs:
- `meta_info/matched_tracks.json` — machine-readable index lookup
- `meta_info/matched_tracks.csv`  — human-readable flat table

### Matched assays

CTCF is excluded (not present in pig).  Species-specific assays (DMC1,
H3K36me3, POLR2A) are excluded.  Remaining shared assays:

**ATAC, H3K27ac, H3K27me3, H3K4me1, H3K4me3**

### Matched tissues

Human ENTEX uses UBERON free-text terms; other species use simple labels.
Only tissues with unambiguous orthologues are kept:

| Human ENTEX term | Canonical |
|---|---|
| subcutaneous adipose tissue | Adipose |
| right lobe of liver | Liver |
| upper lobe of left lung | Lung |
| gastrocnemius medialis | Muscle |
| spleen | Spleen |

Brain regions (Cerebellum, Cortex, Hypothalamus) are present in pig, cattle and
chicken but absent from human ENTEX, so they are matched only among those three
and dropped whenever human or salmon is one side of the pair.
Salmon-specific tissues (Brain, Gill, Gonad, Sperm) have no match → excluded;
only **Liver** and **Muscle** match salmon.
Zebrafish matches **Liver**, **Muscle** and **Spleen**, and carries only three
of the five shared assays (ATAC, H3K27ac, H3K4me3 — no H3K27me3 or H3K4me1).

### Matched groups per species pair

Generated from `meta_info/matched_tracks.json`; regenerate with
`meta_info/find_matched_tracks.py` after adding a species. All fifteen pairs of
the six species are listed.

When adding a species, `SPECIES_ORDER` in `find_matched_tracks.py` and
`_SPECIES_ORDER` in `evaluate_cross_species.py` must be appended to and never
inserted into, and the two must stay identical. The pair keys derive from that
order, so a mismatch yields zero matched channels without raising anything.

| Pair | Tissues | Assays | Groups |
|------|---------|--------|--------|
| human ↔ pig | Adipose, Liver, Muscle, Spleen | 5 | 14 |
| human ↔ cattle | Adipose, Liver, Muscle, Spleen | 5 | 14 |
| human ↔ chicken | Adipose, Liver, Muscle, Spleen | 5 | 14 |
| human ↔ salmon | Liver, Muscle | 5 | 8 |
| human ↔ zebrafish | Liver, Muscle, Spleen | 3 | 9 |
| pig ↔ cattle | Adipose, Cerebellum, Cortex, Hypothalamus, Liver, Lung, Muscle, Spleen | 5 | 40 |
| pig ↔ chicken | Adipose, Cerebellum, Cortex, Hypothalamus, Liver, Lung, Muscle, Spleen | 5 | 40 |
| pig ↔ salmon | Liver, Muscle | 5 | 10 |
| pig ↔ zebrafish | Liver, Muscle, Spleen | 3 | 9 |
| cattle ↔ chicken | Adipose, Cerebellum, Cortex, Hypothalamus, Liver, Lung, Muscle, Spleen | 5 | 40 |
| cattle ↔ salmon | Liver, Muscle | 5 | 10 |
| cattle ↔ zebrafish | Liver, Muscle, Spleen | 3 | 9 |
| chicken ↔ salmon | Liver, Muscle | 5 | 10 |
| chicken ↔ zebrafish | Liver, Muscle, Spleen | 3 | 9 |
| salmon ↔ zebrafish | Liver, Muscle | 3 | 6 |
| **total** | | | **242** |

Two things drive the spread in group counts. Human's EN-TEx track set matches
only four tissues against another mammal or chicken (no Lung), whereas pig,
cattle and chicken share eight among themselves including the three brain
tissues. And zebrafish carries only three of the five shared assays (ATAC,
H3K27ac, H3K4me3 — no H3K27me3 or H3K4me1), so every zebrafish pair is capped
at three assays regardless of tissue overlap.

Each group may contain multiple replicate track indices per species (e.g., pig has
2 individuals per assay×tissue). AUROC is computed for every source×target index
pair and averaged.

### JSON format

```json
{
  "human_pig": [
    {"assay": "H3K27ac", "tissue": "Liver", "human": [12, 45], "pig": [24, 25]},
    ...
  ],
  "human_cattle": [...],
  "human_salmon": [...],
  "pig_cattle":   [...],
  "pig_salmon":   [...],
  "cattle_salmon":[...]
}
```

Keys follow canonical ordering: `human < pig < cattle < salmon`.
For direction pig → human, the evaluation script uses the `"human_pig"` key and
reads `entry["pig"]` as source indices and `entry["human"]` as target indices.

---

## Evaluation convention

Two filters apply to the cross-species AUROCs exactly as they apply within
species, in this order.

1. **N-flank exclusion.** Any test interval whose 20 kb window is more than
   50% N on either side of the central 1 kb is dropped. The filter runs on the
   target genome, so both models in a comparison see the same interval set.
2. **50-positive floor.** Any target track with fewer than 50 positives among
   the surviving intervals is dropped. Counting positives before the exclusion
   would admit a track whose positives all sit in dropped regions and then
   score it on almost nothing.

The filtered results are written to `eval_results/cross_species_final/` by
`manuscript/analysis/run_all.sh --final-convention`, and those are the numbers
the paper reports. The unfiltered tree stays for the sensitivity comparison.
Across the primary set this leaves 359 matched channels.

---

## How to Run

### Step 1 — Generate track index table (once, login node)

```bash
python meta_info/find_matched_tracks.py
```

### Step 2 — Submit the evaluation jobs

```bash
bash 8a.cross_species_eval.sh        # the mammal and salmon directions
bash 8c.cross_species_chicken.sh     # every direction involving chicken
bash 10d.cross_species_zebrafish.sh  # every direction involving zebrafish
bash 9c.cross_species_funformer1k.sh # the architecture-matched control
```

Each job takes under an hour on a gh200 GPU node.

---

## Outputs

```
eval_results/cross_species/
  <source>_to_<target>/
    <model_tag>_per_match.csv    # one row per matched (assay, tissue) group
    <model_tag>_summary.csv      # one-row summary across all groups
```

### `*_per_match.csv` columns

`model, source_species, target_species, assay, tissue, n_pairs, auroc, auprc`

`n_pairs` = number of source×target replicate index combinations evaluated.

### `*_summary.csv` columns

`model, source_species, target_species, n_matched, auroc_mean, auroc_median, auroc_std, auprc_mean, auprc_std`

---

## Comparison with within-species results

Within-species AUROC is in `eval_results/test_set_final/`. Cross-species AUROC
is lower, since the model is reading a genome it never trained on, but the
question is not the absolute level. It is whether the long-context model keeps
its advantage over the matched 1 kb control, and over what phylogenetic range
it does so.
