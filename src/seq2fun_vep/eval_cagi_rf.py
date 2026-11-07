#!/usr/bin/env python
"""
Evaluate VEP predictions using Random Forest trained on release data.

Train RF on release data (423 track features → CAGI value),
test on challenge data, report Pearson correlation.

Usage:
    python eval_cagi_rf.py \
        --release-vep bench_results/CAGI_RF/BinTEN2/release_F9.csv \
        --release-cagi bench_data/CAGI/release_F9.tsv.gz \
        --challenge-vep bench_results/CAGI_RF/BinTEN2/challenge_F9.csv \
        --challenge-cagi bench_data/CAGI/challenge_F9.tsv.gz
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from utils import load_cagi


def load_and_merge(vep_path, cagi_path):
    """Load VEP predictions and merge with CAGI ground truth."""
    vep_df = pd.read_csv(vep_path, index_col=0)

    cagi_df = load_cagi(cagi_path)
    cagi_df['variant_id'] = cagi_df.apply(
        lambda r: f"{r['chrom']}:{r['pos']}:{r['ref']}:{r['alt']}", axis=1
    )

    # Merge
    merged = cagi_df.set_index('variant_id').join(vep_df, how='inner')

    # Get track columns
    track_cols = [c for c in vep_df.columns if c.startswith('track_')]

    return merged, track_cols


def eval_cagi_rf(release_vep_path, release_cagi_path,
                 challenge_vep_path, challenge_cagi_path,
                 n_estimators=100, random_state=42):
    """Train RF on release, evaluate on challenge."""

    # Load release data (training)
    train_df, track_cols = load_and_merge(release_vep_path, release_cagi_path)
    X_train = train_df[track_cols].values
    y_train = train_df['value'].values

    # Load challenge data (testing)
    test_df, _ = load_and_merge(challenge_vep_path, challenge_cagi_path)
    X_test = test_df[track_cols].values
    y_test = test_df['value'].values

    # Train RF
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluate
    r, p = pearsonr(y_pred, y_test)

    # Get feature importances
    importances = rf.feature_importances_
    top_k = 10
    top_indices = np.argsort(importances)[-top_k:][::-1]
    top_features = [(track_cols[i], importances[i]) for i in top_indices]

    return {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'pearson_r': r,
        'p_value': p,
        'top_features': top_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VEP with RF trained on release data")
    parser.add_argument("--release-vep", required=True, help="Path to release VEP CSV")
    parser.add_argument("--release-cagi", required=True, help="Path to release CAGI file")
    parser.add_argument("--challenge-vep", required=True, help="Path to challenge VEP CSV")
    parser.add_argument("--challenge-cagi", required=True, help="Path to challenge CAGI file")
    parser.add_argument("--gene", help="Gene name for output")
    parser.add_argument("--csv-row", action="store_true", help="Output as CSV row")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of RF trees")
    args = parser.parse_args()

    result = eval_cagi_rf(
        args.release_vep, args.release_cagi,
        args.challenge_vep, args.challenge_cagi,
        n_estimators=args.n_estimators
    )

    if args.csv_row:
        gene = args.gene or "unknown"
        print(f"{gene},{result['n_train']},{result['n_test']},{result['pearson_r']:.6f},{result['p_value']:.2e}")
    else:
        print(f"Training samples: {result['n_train']}")
        print(f"Test samples: {result['n_test']}")
        print(f"\nPearson r: {result['pearson_r']:.4f}")
        print(f"P-value: {result['p_value']:.2e}")
        print(f"\nTop 10 features:")
        for feat, imp in result['top_features']:
            print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
