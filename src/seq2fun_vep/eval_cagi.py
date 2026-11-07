#!/usr/bin/env python
"""
Evaluate VEP predictions against CAGI ground truth using Pearson correlation.

Usage:
    python eval_cagi.py \
        --vep-path bench_results/CAGI/vep_F9.csv \
        --cagi-path bench_data/CAGI/challenge_F9.tsv.gz
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from utils import load_cagi


def eval_cagi(vep_path, cagi_path, agg='mean'):
    """Evaluate VEP predictions against CAGI ground truth."""
    # Load VEP predictions
    vep_df = pd.read_csv(vep_path, index_col=0)

    # Aggregate across tracks
    if agg == 'max':
        # Max magnitude with sign preserved
        max_col = vep_df.abs().idxmax(axis=1)
        vep_df['score'] = vep_df.values[np.arange(len(vep_df)), vep_df.columns.get_indexer(max_col)]
    else:
        vep_df['score'] = vep_df.mean(axis=1)

    # Load CAGI ground truth
    cagi_df = load_cagi(cagi_path)
    cagi_df['variant_id'] = cagi_df.apply(
        lambda r: f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}", axis=1
    )

    # Merge on variant_id
    merged = cagi_df.set_index('variant_id').join(vep_df[['score']], how='inner')

    # Compute Pearson correlation
    r, p = pearsonr(merged['score'], merged['value'])

    return {
        'n_vep': len(vep_df),
        'n_cagi': len(cagi_df),
        'n_matched': len(merged),
        'pearson_r': r,
        'p_value': p,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VEP predictions against CAGI")
    parser.add_argument("--vep-path", required=True, help="Path to VEP output CSV")
    parser.add_argument("--cagi-path", required=True, help="Path to CAGI challenge file")
    parser.add_argument("--gene", help="Gene name for CSV output")
    parser.add_argument("--csv-row", action="store_true", help="Output as CSV row")
    parser.add_argument("--agg", choices=['mean', 'max'], default='mean',
                        help="Aggregation method across tracks (default: mean)")
    args = parser.parse_args()

    result = eval_cagi(args.vep_path, args.cagi_path, agg=args.agg)

    if args.csv_row:
        gene = args.gene or "unknown"
        print(f"{gene},{result['n_vep']},{result['n_cagi']},{result['n_matched']},{result['pearson_r']:.6f},{result['p_value']:.2e}")
    else:
        print(f"VEP variants: {result['n_vep']}")
        print(f"CAGI variants: {result['n_cagi']}")
        print(f"Matched: {result['n_matched']}")
        print(f"\nPearson r: {result['pearson_r']:.4f}")
        print(f"P-value: {result['p_value']:.2e}")


if __name__ == "__main__":
    main()
