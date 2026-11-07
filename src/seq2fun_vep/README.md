# Variant Effect Prediction (VEP) for Seq2Fun Models

This directory contains tools for predicting variant impacts using trained Seq2Fun, BinTEN, and baseline models.

## Overview

The VEP pipeline computes the functional impact of genetic variants by comparing model predictions between reference and alternate alleles. The impact score quantifies how much a variant changes the predicted regulatory activity.

## Files

- `vep.py` - Main VEP script for all model types
- `example.py` - Example VEP implementation (Borzoi model)
- `utils.py` - Utility functions for sequence handling and data loading

## Supported Models

- **BinTEN1, BinTEN2, BinTEN3** - Transformer-based models
- **DeepSEA** - CNN baseline
- **DanQ** - CNN + LSTM baseline
- **DeepFormer** - Flow attention baseline

## Installation

Required packages:
```bash
pip install torch transformers datasets pandas numpy biopython pysam
```

## Usage

### Basic Usage

```bash
python vep.py \
    --genome-path /path/to/hg38.fa \
    --variant-path variants.vcf \
    --output-path results.csv \
    --model-path /path/to/checkpoint \
    --model-class BinTEN1 \
    --window-size 10000
```

### Example: BinTEN Model

```bash
python vep.py \
    --genome-path data/hg38.fa \
    --variant-path data/variants.vcf \
    --output-path results/binten1_vep.csv \
    --model-path checkpoints/binten1/best_model \
    --model-class BinTEN1 \
    --window-size 10000 \
    --num-tracks 423 \
    --per-device-batch-size 16
```

### Example: DeepSEA Model

```bash
python vep.py \
    --genome-path data/hg38.fa \
    --variant-path data/variants.vcf \
    --output-path results/deepsea_vep.csv \
    --model-path checkpoints/deepsea.pt \
    --model-class DeepSEA \
    --window-size 1000 \
    --num-tracks 423
```

## Arguments

### Required Arguments

- `--genome-path`: Path to reference genome FASTA file (must be indexed with .fai)
- `--variant-path`: Path to variant file (VCF, TSV, or compressed)
- `--output-path`: Output CSV file path
- `--model-path`: Path to model checkpoint (directory or file)
- `--model-class`: Model architecture (BinTEN1, BinTEN2, BinTEN3, DeepSEA, DanQ, DeepFormer)

### Model Configuration

- `--window-size`: Sequence window size around variant (default: 1000)
  - BinTEN models: typically 5000-50000
  - DeepSEA/DanQ: typically 1000
- `--num-tracks`: Number of output tracks (default: 423)

### Scoring Options

- `--score-method`: Method to compute impact score
  - `l2_lfc` (default): L2 norm of log2 fold change
  - `abs_diff`: Mean absolute difference
  - `squared_diff`: Mean squared difference
  - `max_abs_diff`: Maximum absolute difference
  - `per_track_diff`: Per-track differences (returns all tracks)
- `--use-sigmoid`: Apply sigmoid to logits before scoring

### Performance Options

- `--per-device-batch-size`: Batch size (default: 8)
- `--chunk-size`: Variants per chunk (default: 30000)
- `--device`: Device to use (default: cuda if available)
- `--limit`: Limit number of variants (for testing)

## Input Format

### VCF Format
```
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
chr1    12345   .       A       G       .       .       .
chr1    67890   .       C       T       .       .       .
```

### TSV Format
```
chrom   pos     ref     alt
chr1    12345   A       G
chr1    67890   C       T
```

Required columns: `chrom`, `pos`, `ref`, `alt`

## Output Format

The output CSV contains:
- Variant ID (chrom:pos:ref:alt)
- Impact score(s)

### Single Score Output (default)
```csv
variant_id,impact_score
chr1:12345:A:G,0.523
chr1:67890:C:T,1.245
```

### Per-Track Output (--score-method per_track_diff)
```csv
variant_id,track_0,track_1,track_2,...
chr1:12345:A:G,0.12,0.45,0.23,...
chr1:67890:C:T,0.89,0.34,0.56,...
```

## Score Methods

### L2 Log Fold Change (l2_lfc) - Default
Computes log2 fold change for each track, then takes L2 norm:
```python
lfc = log2(1 + y_alt) - log2(1 + y_ref)
score = ||lfc||_2
```
Good for: Overall impact across all regulatory features

### Absolute Difference (abs_diff)
Mean absolute difference across tracks:
```python
score = mean(|y_alt - y_ref|)
```
Good for: Average change magnitude

### Maximum Absolute Difference (max_abs_diff)
Maximum change across any track:
```python
score = max(|y_alt - y_ref|)
```
Good for: Finding variants with strong impact on any feature

### Per-Track Difference (per_track_diff)
Returns differences for all tracks:
```python
scores = |y_alt - y_ref|  # shape (n_variants, n_tracks)
```
Good for: Detailed analysis of which features are affected

## Performance Tips

1. **Batch Size**: Increase `--per-device-batch-size` for faster inference (limited by GPU memory)
2. **Chunk Size**: Adjust `--chunk-size` based on memory constraints
3. **GPU Memory**: For large models or long sequences, reduce batch size
4. **CPU Usage**: Set `--device cpu` if GPU is unavailable

## Genome Preparation

The genome FASTA must be indexed:
```bash
samtools faidx hg38.fa
```

This creates `hg38.fa.fai` index file required by pysam.

## Troubleshooting

### Reference Allele Mismatch
If you see warnings about reference allele mismatches:
- Verify your variant coordinates match the genome build
- Check that VCF positions are 1-based (standard)
- Ensure genome FASTA matches the variant file

### Out of Memory
If you run out of GPU memory:
- Reduce `--per-device-batch-size`
- Reduce `--window-size` (if model allows)
- Use `--device cpu` (slower but uses system RAM)

### Model Loading Errors
If model fails to load:
- Verify `--model-class` matches the trained model
- Check `--num-tracks` matches the model's output dimension
- Ensure `--window-size` is compatible with model architecture

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```
