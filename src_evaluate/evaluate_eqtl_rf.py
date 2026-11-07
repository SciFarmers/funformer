"""
Random Forest Evaluation for eQTL Fine-mapped Variants.

Trains Random Forest classifiers to distinguish causal (pos) from non-causal
(neg) variants using VEP feature vectors produced by
src_evaluate/vep_eqtl_finemap.py.

Species-agnostic: it reads whatever per-track features the score dir holds,
so the same code serves the human and pig benchmarks.

Each tissue has two CSV files:
    <score_dir>/<tissue>_pos.csv.gz   — causal variants (label=1)
    <score_dir>/<tissue>_neg.csv.gz   — non-causal variants (label=0)

Classification performance is measured with AUC-ROC using 10-fold
cross-validation over ten forest seeds (100 fits per tissue).

Folds are not shuffled, so the partition is the same for every model and
tissue and the repeats vary only the forest seed.

Usage (run from workspace root):
    python -m src_evaluate.evaluate_eqtl_rf \\
        --score-dir bench_results/vep/DanQ_5e-4_1000 \\
        --out        bench_results/rf/DanQ_5e-4_1000.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tissue_data(score_dir: Path, tissue: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load neg/pos CSV files for one tissue.

    Returns
    -------
    X : ndarray, shape (N, n_tracks)
    y : ndarray, shape (N,)  — 0=neg, 1=pos
    """
    neg_df = pd.read_csv(score_dir / f"{tissue}_neg.csv.gz", index_col="variant_id")
    pos_df = pd.read_csv(score_dir / f"{tissue}_pos.csv.gz", index_col="variant_id")

    X = np.vstack([neg_df.values, pos_df.values])
    y = np.hstack([np.zeros(len(neg_df)), np.ones(len(pos_df))])
    return X, y


# ---------------------------------------------------------------------------
# RF training
# ---------------------------------------------------------------------------

def _cv_single_seed(X: np.ndarray, y: np.ndarray, seed: int, cv_folds: int) -> np.ndarray:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=seed,
        n_jobs=1,
    )
    return cross_val_score(rf, X, y, cv=cv_folds, scoring="roc_auc", n_jobs=1)


def train_rf_repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    cv_folds: int = 10,
    n_jobs: int = -1,
) -> dict:
    """
    Repeated k-fold CV with different seeds, parallelised across seeds
    using concurrent.futures.ProcessPoolExecutor.

    Returns dict with keys: mean_auc, std_auc, all_scores, n_scores.
    """
    max_workers = None if n_jobs == -1 else max(1, n_jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_cv_single_seed, X, y, seed, cv_folds)
            for seed in range(1, n_repeats + 1)
        ]
        all_cv = [f.result() for f in futures]
    all_scores = np.concatenate(all_cv)
    return {
        "mean_auc": float(all_scores.mean()),
        "std_auc": float(all_scores.std()),
        "all_scores": all_scores,
        "n_scores": len(all_scores),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all_tissues(
    score_dir: Path,
    n_repeats: int = 10,
    cv_folds: int = 10,
) -> pd.DataFrame:
    """
    Evaluate RF performance on every tissue present in score_dir.

    Discovers tissues by looking for *_neg.csv.gz files.
    """
    tissues = sorted(
        f.name.replace("_neg.csv.gz", "")
        for f in score_dir.glob("*_neg.csv.gz")
    )

    print(f"\nModel: {score_dir.name}")
    print(f"Tissues found: {len(tissues)}")
    print(f"CV scheme: {n_repeats} repeats × {cv_folds} folds = {n_repeats * cv_folds} runs/tissue")
    print("-" * 80)

    results = []
    for i, tissue in enumerate(tissues, 1):
        try:
            X, y = load_tissue_data(score_dir, tissue)
            metrics = train_rf_repeated_cv(X, y, n_repeats=n_repeats, cv_folds=cv_folds)
            results.append({
                "tissue": tissue,
                "mean_auc": metrics["mean_auc"],
                "std_auc": metrics["std_auc"],
                "n_pos": int(y.sum()),
                "n_neg": int((y == 0).sum()),
                "n_scores": metrics["n_scores"],
            })
            print(
                f"[{i:3d}/{len(tissues)}] {tissue:<45s} "
                f"AUC {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}"
            )
        except Exception as exc:
            print(f"[{i:3d}/{len(tissues)}] {tissue:<45s} ERROR: {exc}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random Forest AUC evaluation of VEP scores for eQTL fine-mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--score-dir", required=True,
                        help="Directory with <tissue>_{neg,pos}.csv.gz files from vep_eqtl_finemap.py")
    parser.add_argument("--out", required=True,
                        help="Output CSV path for per-tissue AUC results.")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Number of repeated CV seeds.")
    parser.add_argument("--cv-folds", type=int, default=10,
                        help="Number of cross-validation folds per seed.")

    args = parser.parse_args()
    score_dir = Path(args.score_dir)
    out_path = Path(args.out)

    if not score_dir.exists():
        print(f"ERROR: score directory not found: {score_dir}")
        return

    print("=" * 80)
    print("Random Forest Evaluation — eQTL Fine-mapping")
    print("=" * 80)
    print(f"  Score dir   : {score_dir}")
    print(f"  Output      : {out_path}")
    print(f"  Repeats     : {args.n_repeats}  |  Folds: {args.cv_folds}")
    print("=" * 80)

    results = evaluate_all_tissues(score_dir, n_repeats=args.n_repeats, cv_folds=args.cv_folds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n✓ Saved results → {out_path}")

    # Summary statistics
    if len(results):
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"  Tissues evaluated : {len(results)}")
        print(f"  Mean AUC          : {results['mean_auc'].mean():.4f} ± {results['mean_auc'].std():.4f}")
        print(f"  Median AUC        : {results['mean_auc'].median():.4f}")
        best = results.loc[results["mean_auc"].idxmax()]
        worst = results.loc[results["mean_auc"].idxmin()]
        print(f"  Best tissue       : {best['tissue']}  ({best['mean_auc']:.4f})")
        print(f"  Worst tissue      : {worst['tissue']}  ({worst['mean_auc']:.4f})")
        print("=" * 80)

    print("\nDONE.")


if __name__ == "__main__":
    main()
