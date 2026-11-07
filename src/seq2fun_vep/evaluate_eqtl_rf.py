"""
Random Forest Classification Evaluation for eQTL Fine-mapping Results

This script trains Random Forest classifiers to distinguish causal (pos) from non-causal (neg) 
variants based on variant effect prediction scores across 423 genomic tracks.
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def load_tissue_data(model_dir: Path, tissue_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load negative and positive variant data for a specific tissue.
    
    Args:
        model_dir: Directory containing model predictions
        tissue_name: Name of the tissue (without _neg/_pos suffix)
    
    Returns:
        X: Feature matrix (N_samples, 423)
        y: Labels (0 for neg, 1 for pos)
    """
    neg_file = model_dir / f"{tissue_name}_neg.csv"
    pos_file = model_dir / f"{tissue_name}_pos.csv"
    
    # Load data
    neg_df = pd.read_csv(neg_file, index_col='variant_id')
    pos_df = pd.read_csv(pos_file, index_col='variant_id')
    
    # Combine features and labels
    X_neg = neg_df.values
    X_pos = pos_df.values
    X = np.vstack([X_neg, X_pos])
    
    y_neg = np.zeros(len(X_neg))
    y_pos = np.ones(len(X_pos))
    y = np.hstack([y_neg, y_pos])
    
    return X, y

def train_rf_single_run(X: np.ndarray, y: np.ndarray, seed: int, cv_folds: int = 10) -> np.ndarray:
    """
    Train Random Forest classifier with cross-validation for a single seed.
    
    Args:
        X: Feature matrix
        y: Labels
        seed: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
    
    Returns:
        Array of AUC scores from cross-validation
    """
    # Train Random Forest classifier with specific seed
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=seed,
        n_jobs=1  # Single job per run, parallelized across runs
    )
    
    # Cross-validated AUC scores
    cv_scores = cross_val_score(rf, X, y, 
                                 cv=cv_folds, 
                                 scoring='roc_auc',
                                 n_jobs=1)
    
    return cv_scores

def train_rf_classifier(X: np.ndarray, y: np.ndarray, 
                        n_repeats: int = 10,
                        cv_folds: int = 10,
                        n_jobs: int = -1) -> Dict[str, float]:
    """
    Train Random Forest classifier with repeated cross-validation in parallel.
    
    Args:
        X: Feature matrix
        y: Labels
        n_repeats: Number of repetitions with different seeds (1-10)
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 for all cores)
    
    Returns:
        Dictionary with mean and std of AUC scores across all repetitions
    """
    # Run CV in parallel for seeds 1 to n_repeats
    all_cv_scores = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(train_rf_single_run)(X, y, seed, cv_folds) 
        for seed in range(1, n_repeats + 1)
    )
    
    # Flatten all CV scores (n_repeats * cv_folds total scores)
    all_scores = np.concatenate(all_cv_scores)
    
    return {
        'mean_auc': all_scores.mean(),
        'std_auc': all_scores.std(),
        'all_scores': all_scores,
        'n_scores': len(all_scores)
    }

def evaluate_all_tissues(model_dir: Path, n_repeats: int = 10, cv_folds: int = 10) -> pd.DataFrame:
    """
    Evaluate Random Forest performance across all tissues with repeated CV.
    
    Args:
        model_dir: Directory containing model predictions
        n_repeats: Number of repetitions with different seeds
        cv_folds: Number of cross-validation folds
    
    Returns:
        DataFrame with tissue names and AUC scores
    """
    # Get list of tissues
    csv_files = sorted(model_dir.glob("*_neg.csv"))
    tissues = [f.stem.replace("_neg", "") for f in csv_files]
    
    results = []
    
    print(f"\nEvaluating {model_dir.name} model on {len(tissues)} tissues...")
    print(f"Using {n_repeats} repetitions × {cv_folds} folds = {n_repeats * cv_folds} total CV runs per tissue")
    print("-" * 80)
    
    for i, tissue in enumerate(tissues, 1):
        try:
            # Load data
            X, y = load_tissue_data(model_dir, tissue)
            
            # Train and evaluate with repeated CV
            metrics = train_rf_classifier(X, y, n_repeats=n_repeats, cv_folds=cv_folds)
            
            results.append({
                'tissue': tissue,
                'mean_auc': metrics['mean_auc'],
                'std_auc': metrics['std_auc'],
                'n_scores': metrics['n_scores']
            })
            
            print(f"[{i}/{len(tissues)}] {tissue:40s} AUC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f} (n={metrics['n_scores']})")
            
        except Exception as e:
            print(f"[{i}/{len(tissues)}] {tissue:40s} ERROR: {str(e)}")
            continue
    
    return pd.DataFrame(results)


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='Evaluate variant effect predictions using Random Forest classification'
    )
    parser.add_argument(
        '--score-dir',
        type=str,
        required=True,
        help='Directory containing prediction scores (e.g., DeepSEA, BinTEN1)'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output CSV file path for results'
    )
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=10,
        help='Number of repetitions with different seeds (default: 10)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=10,
        help='Number of cross-validation folds (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    score_dir = Path(args.score_dir)
    output_file = Path(args.out)
    
    # Check that directory exists
    if not score_dir.exists():
        print(f"ERROR: Directory not found: {score_dir}")
        return
    
    # Print configuration
    print("="*80)
    print("Random Forest Evaluation with Repeated Cross-Validation")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Score directory: {score_dir}")
    print(f"  - Output file: {output_file}")
    print(f"  - Number of repetitions: {args.n_repeats} (seeds 1-{args.n_repeats})")
    print(f"  - Cross-validation folds: {args.cv_folds}")
    print(f"  - Total CV runs per tissue: {args.n_repeats * args.cv_folds}")
    print("="*80)
    
    # Evaluate
    results = evaluate_all_tissues(score_dir, n_repeats=args.n_repeats, cv_folds=args.cv_folds)
    
    # Save results to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"  Number of tissues: {len(results)}")
    print(f"  Mean AUC:          {results['mean_auc'].mean():.4f} ± {results['mean_auc'].std():.4f}")
    print(f"  Median AUC:        {results['mean_auc'].median():.4f}")
    print(f"  Min AUC:           {results['mean_auc'].min():.4f} ({results.loc[results['mean_auc'].idxmin(), 'tissue']})")
    print(f"  Max AUC:           {results['mean_auc'].max():.4f} ({results.loc[results['mean_auc'].idxmax(), 'tissue']})")
    print("=" * 80)
    print("\nEVALUATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
