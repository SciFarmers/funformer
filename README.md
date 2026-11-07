# FunFormer

A long-context classifier for regulatory activity, and the pipeline that uses
it to measure how far regulatory sequence information transfers between
vertebrate genomes.

FunFormer predicts, for a genomic interval, whether that interval overlaps a
peak in each of several hundred epigenomic tracks. A convolutional encoder
compresses any input from 1 to 50 kb into a fixed 500 tokens, so context length
can be varied while the parameter count stays identical. That separates what
the architecture contributes from what the extra sequence contributes, within a
genome and across six vertebrates spanning 435 million years.

**Reproducing the figures and tables needs no GPU and no download.** The
per-track evaluation results are stored here, and one command rebuilds every
figure, table and statistic from them in about three minutes. Training and
inference are the expensive half and need the weights and the source genomes,
which are published separately because they are too large for a code host.

## Quick start

```bash
apptainer build seq2fun_cpu.sif containers/seq2fun_cpu.def   # on an x86 machine
apptainer exec --bind "$PWD" seq2fun_cpu.sif \
    bash manuscript/analysis/run_all.sh
```

Writes ten figures to `manuscript/analysis/out/figures/`, five LaTeX tables to
`out/tables/`, and the full statistics report to `out/stats_report.txt`.

## Environment

Both images are defined in `containers/`. Architecture follows the machine that
runs the build, so the GPU image is arm64 for gh200 nodes and the analysis
image is amd64, and they cannot be built on the same node. `containers/README.md`
has the details.

```bash
apptainer build seq2fun.sif     containers/seq2fun.def       # GPU node
apptainer build seq2fun_cpu.sif containers/seq2fun_cpu.def   # x86 node
```

The jobs here are written for SLURM. `analysis/common/run_cpu.sh` submits an
analysis script to a CPU node from a frozen copy of the code, so an edit made
while a job is queued cannot change what runs.

## Source data

All data are public and none of it is redistributed here.

| Species | Assembly | Tracks trained / evaluated | Source |
|---|---|---|---|
| Human | GRCh38 | 423 / 401 | EN-TEx, Rozowsky et al. 2023 |
| Pig | Sscrofa11.1 | 80 / 80 | FAANG, Nguyen et al. 2025 |
| Cattle | ARS-UCD1.2 | 95 / 95 | FAANG, Nguyen et al. 2025 |
| Chicken | GRCg6a | 97 / 97 | FAANG, Nguyen et al. 2025 |
| Atlantic salmon | Ssal_v3.1 | 301 / 211 | AQUA-FAANG, Nguyen et al. 2025 |
| Zebrafish | GRCz10 | 37 / 37 | GEO **GSE134055**, Yang et al. 2020 |

Tracks are binary peak calls for chromatin accessibility and histone marks.
H3K27ac, H3K27me3, H3K4me1 and H3K4me3 are shared by every species except
zebrafish, which has H3K27ac and H3K4me3 only. CTCF and H3K36me3 are not shared
by all six and are excluded from every cross-species comparison.

**Zebrafish must use GRCz10**, the assembly the GSE134055 peak calls were built
on. Pairing those coordinates with GRCz11 decouples the labels from the
sequence and drives every model to chance.

Variant benchmarks are GTEx v8 fine-mapped eQTL in human, and PigGTEx eQTL and
sQTL in pig (Teng et al. 2024), the latter built by
`6a0.build_pig_qtl_benchmark.sh`. Divergence times are TimeTree 5.

Each species is a HuggingFace Arrow dataset of intervals and per-track labels,
with sequence fetched from the genome FASTA at run time. Splits are by whole
chromosome, autosomes only, one held out for validation and one for test. Test
is chromosome 21 in human, cattle, chicken, salmon and zebrafish and 16 in pig.
Validation is 22 in human, 17 in pig and 25 in the other four.

## Models

| Model | Context | Description |
|---|---|---|
| DeepSEA | 1 kb | three-layer CNN plus a two-layer classifier |
| DeepFormer | 1 kb | CNN with batch normalisation plus flow attention |
| DeepATT | 1 kb | wide convolution, BiLSTM, cross-attention with a per-track query |
| FunFormer | 1 to 50 kb | convolutional encoder to 500 tokens, then transformer layers with rotary position encodings, bidirectional attention and SwiGLU |

All are trained with unweighted per-track binary cross-entropy and wrapped in
`DualStrandWrapper`, which picks one strand per training step and averages both
at evaluation.

`FunFormerConfig` asserts `n_tokens == 500` at every context length, and the
only length-dependent parameter block is the projection head, so **every
context length has the same parameter count** for a given track count. Two
configurations are reported, FunFormer-20k and FunFormer-1k, the latter being
the architecture- and parameter-matched control at the baselines' own context.
Context length and learning rate were both selected on the validation split,
over four learning rates for every model and five context lengths in human.
The test split reports final performance only.

## Evaluation convention

Two filters, applied identically everywhere, in this order.

1. **N-flank exclusion.** Drop any test interval whose 20 kb window is over 50%
   N on either side of the central 1 kb. This removes 14.8% of human intervals
   and under 0.2% elsewhere.
2. **50-positive floor.** Drop any target track with fewer than 50 positives
   among the intervals that survive step 1. Counting before the exclusion would
   admit tracks whose positives all sit in dropped regions.

Results under this convention are in `eval_results/test_set_final/` and
`eval_results/cross_species_final/`. The unfiltered trees are kept for the
sensitivity row of Supplementary Table S3.

Transfer is never scored as bare accuracy in the target. A source model's
prediction for one (assay, tissue) channel is scored by AUROC against the
target species' labels for the matching channel, then compared against a model
trained on the target over the same channels. The index is
`meta_info/matched_tracks.json` and the design is in
`DESIGN_NOTE.md`.

## Running the pipeline

The numbered scripts in the root run in prefix order and submit SLURM jobs.

**Train.** Baselines and FunFormer per species, then the 1 kb control.

```bash
bash 1a.human_baseline.sh && bash 1b.human_funformer.sh
bash 3a.cattle_jobs.sh   && bash 4a.cattle_funformer_jobs.sh      # 3b/4b pig, 3c/4c salmon,
bash 3d.chicken_baseline_jobs.sh && bash 4d.chicken_funformer.sh  # 10a/10b zebrafish
bash 9a.funformer1k_jobs.sh
```

**Evaluate within species.** `5a` to `5e` and `10c` per species, `9b` for the
1 kb control, and `5f` for the human context sweep. Writes
`eval_results/test_set/`.

**Transfer.** Build the matched-channel index once with
`python meta_info/find_matched_tracks.py`, then `8a`, `8c`, `10d` and `9c`.
Writes `eval_results/cross_species/`.

**Variant effects.** `2a` and `7a` score human GTEx, `6a0`, `6a` score the pig
benchmark, and `2b`, `6b` fit the random forests. Writes `bench_results/`.

**Apply the convention and build the artifacts.**

```bash
bash manuscript/analysis/run_all.sh --final-convention
bash manuscript/analysis/run_all.sh
```

Each analysis script writes exactly one artifact and its prefix says which, so
`3.cross_species.py` builds Figure 3 and `ts4.stratum_absolute.py` builds
Supplementary Table S4. The exploratory analyses are separate and off by
default.

```bash
bash manuscript/analysis/run_all.sh --tss-strata      # transfer by distance to the nearest TSS
bash manuscript/analysis/run_all.sh --flank-ablation  # replace or shuffle the flank at inference
```

## Key results

**Within species**, mean test AUROC.

| Species | Tracks | best 1 kb baseline | FunFormer-1k | FunFormer-20k |
|---|---|---|---|---|
| Human | 401 | 0.957 | 0.957 | **0.967** |
| Pig | 80 | 0.889 | 0.892 | **0.927** |
| Cattle | 95 | 0.928 | 0.929 | **0.949** |
| Chicken | 97 | 0.865 | 0.868 | **0.876** |
| Salmon | 211 | 0.942 | 0.943 | **0.955** |
| Zebrafish | 37 | 0.925 | 0.929 | **0.950** |

At matched context the transformer adds **+0.19 pp** over the best published
baseline, while extending it to 20 kb adds **+1.8 pp**. The advantage is
context, not architecture.

**Across species**, the context margin over 23 directions and 359 matched
channels.

| Bin | Divergence | Directions | Margin |
|---|---|---|---|
| Mammal to mammal | 58 to 94 Mya | 6 | **+3.4 pp** |
| Mammal to chicken | 320 Mya | 3 | **+0.4 pp** |
| Tetrapod to fish | 435 Mya | 14 | **−7.8 pp** |

The margin decays with divergence (r = −0.764) and changes sign between the
amniote and tetrapod-fish splits. Significance comes from relabelling the six
species onto the tree, enumerated exactly over all 6! = 720 relabellings
because the 23 directions are drawn from six genomes and span only four
divergence values. 12 of 720 are at least as extreme, so **P = 0.017**.

**What the context carries.** Replacing the flank outside the central 1 kb with
GC-matched sequence from the same genome costs 8.1 pp among mammals and 9.6 pp
into chicken, against 2.4 pp across the fish split where little is left to
remove. Drawing the replacement from the source genome instead rescues +0.6 pp
among mammals but +4.8 pp transferring a tetrapod model into a fish genome.

## Checkpoints

Every model was trained at four learning rates, 1e-5, 5e-5, 1e-4 and 5e-4,
and the reported run is the one selected on the validation split. That came
out at 5e-4 for all twelve FunFormer cells and varies among the baselines,
where 5e-4 wins twelve of eighteen, 5e-5 four and 1e-4 two. The names carry
the selected rate, so FunFormer runs read
`funformer_<tag>_len20000_lr5e-4_b250_gac2` at 20 kb and
`funformer_<tag>_len1000_lr5e-4_b500_gac1` at 1 kb, with `<tag>` one of
`entex_423`, `cattle`, `pig`, `salmon`, `chicken`, `zebrafish`, while a
baseline reads for instance `DeepSEA_cattle_5e-5_1000`.

The archive carries the 30 selected runs. The full grid, 92 FunFormer runs and
72 baseline runs, stays on the cluster.

`save_pretrained` writes `model_config.json` with the model class and its
backbone arguments, plus `model.safetensors`. Load with
`ModelClass.from_pretrained(path)`.

## The data archive

Weights and the intermediate prediction caches are published separately as a
single zip of 7.7 GB, because they are too large for a code host.

<https://drive.google.com/drive/folders/1iE-ZELm6COiinYTudsgknbIuF6aXqc4i>

That folder holds `funformer_data.zip` and its checksum. It is a working
location while the paper is under review, and the archive will be deposited
with a permanent identifier on publication.

Every path inside the zip is relative to the repository root, so unzip it
inside your clone and each file lands where the code already looks for it.
Nothing is overwritten, because none of those paths exist in a fresh clone.

```bash
git clone https://github.com/SciFarmers/funformer.git
cd funformer
# download funformer_data.zip and funformer_data.zip.sha256 from the link above
sha256sum -c funformer_data.zip.sha256
unzip funformer_data.zip
```

| what it adds | what that lets you run |
|---|---|
| `checkpoints/<run>/best_model/` | inference, variant effect prediction, any new evaluation |
| `analysis/tss_strata/data/cache/` | the TSS-stratified analysis |
| `analysis/flank_ablation/data/` | the flank ablation behind Figure 4 |

The figures and tables need none of it.

## Layout

```
src/              models, dataset, genome reader, training CLI
src_evaluate/     within-species and cross-species evaluation, VEP, random forests
meta_info/        track lists, EN-TEx metadata, the matched-channel index
containers/       Apptainer definitions for both images
analysis/         common helpers and one directory per analysis, each with its own README
manuscript/analysis/   run_all.sh and one script per figure or table
olivia_submiter/  SLURM job templates
0-12*.sh          the numbered workflow, run in prefix order
```

## Citation

If you use this code or the FunFormer models, please cite the paper. The
reference will be added here on publication.
