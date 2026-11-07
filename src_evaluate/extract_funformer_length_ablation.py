"""
Extract test-set performance for the FunFormer sequence-length ablation.

Pulls all FunFormer runs from the "FunFormer_entex_423" W&B project. Test-set
metrics (test/*) are logged once per run, after training, using the
best-on-validation checkpoint — so there is a single test/auroc_mean etc. per
run (no history search needed there, unlike eval/loss which is logged at every
validation step during training). Validation metrics are also included for
reference, taken at the step with the lowest eval/loss (the early-stopping
criterion), since HF Trainer's early stopping can log a few more evaluations
past the best checkpoint before it actually stops.

Usage:
    /usr/bin/python3.12 -m src_evaluate.extract_funformer_length_ablation \
        --out-csv eval_results/funformer_length_ablation.csv

Note: run with python3.12 specifically (the login node's default `python3` is
3.6 and doesn't have wandb installed; also avoid running from a directory that
contains a local wandb/ folder, e.g. the repo root, since Python resolves the
local directory as the "wandb" module before the installed package).
"""

import argparse

import pandas as pd
import wandb

ENTITY = "datngu"
PROJECT = "FunFormer_entex_423"

HISTORY_KEYS = [
    "_step",
    "eval/loss",
    "eval/auroc_mean",
    "eval/auroc_median",
    "eval/auprc_mean",
    "eval/auprc_std",
]


def extract_run(run) -> dict:
    cfg = run.config
    row = {
        "run_name": run.name,
        "run_id": run.id,
        "state": run.state,
        "sequence_length": cfg.get("sequence_length"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
        "test_loss": run.summary.get("test/loss"),
        "test_auroc_mean": run.summary.get("test/auroc_mean"),
        "test_auroc_median": run.summary.get("test/auroc_median"),
        "test_auprc_mean": run.summary.get("test/auprc_mean"),
        "test_auprc_std": run.summary.get("test/auprc_std"),
    }

    hist = run.history(keys=HISTORY_KEYS, pandas=True)
    if hist.empty or "eval/loss" not in hist:
        row["note"] = "no eval history"
        return row

    best = hist.loc[hist["eval/loss"].idxmin()]
    row.update(
        {
            "best_step": int(best["_step"]),
            "best_eval_loss": best["eval/loss"],
            "eval_auroc_mean_at_best": best["eval/auroc_mean"],
            "eval_auroc_median_at_best": best["eval/auroc_median"],
            "eval_auprc_mean_at_best": best["eval/auprc_mean"],
            "final_eval_loss_summary": run.summary.get("eval/loss"),
        }
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default="eval_results/funformer_length_ablation.csv")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"config.model_class": "FunFormer"})

    rows = [extract_run(r) for r in runs]
    df = pd.DataFrame(rows).sort_values(["sequence_length", "lr"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)

    print(df.to_string(index=False))

    if "test_loss" in df:
        print("\n=== Best LR per sequence length (lowest test loss — fairest criterion) ===")
        best_test_per_len = df.loc[df.groupby("sequence_length")["test_loss"].idxmin()]
        print(
            best_test_per_len[
                ["sequence_length", "lr", "test_loss", "test_auroc_mean", "test_auprc_mean"]
            ].to_string(index=False)
        )
        overall_best = df.loc[df["test_loss"].idxmin()]
        print(
            f"\nBest overall (lowest test loss across all lengths/LRs): "
            f"len={int(overall_best['sequence_length'])}, lr={overall_best['lr']}, "
            f"test_loss={overall_best['test_loss']:.5f}, "
            f"test_auroc_mean={overall_best['test_auroc_mean']:.4f}"
        )

    if "best_eval_loss" in df:
        print("\n=== Best LR per sequence length (lowest validation loss, for reference) ===")
        best_per_len = df.loc[df.groupby("sequence_length")["best_eval_loss"].idxmin()]
        print(
            best_per_len[
                ["sequence_length", "lr", "best_eval_loss", "eval_auroc_mean_at_best"]
            ].to_string(index=False)
        )

    print(f"\nWrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
