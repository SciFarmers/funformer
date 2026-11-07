import torch
import numpy as np
import wandb
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import logging
import subprocess
from sklearn.metrics import roc_auc_score, average_precision_score
logger = logging.getLogger(__name__)


# Load environment variables from .env file
load_dotenv()

def get_git_info():
    """
    Get current git branch and commit hash.
    Returns tuple of (branch, commit_hash) or (None, None) if not in git repo.
    """
    try:
        # Get current branch
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        # Get commit hash (first 7 characters)
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        return branch, commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None

def login_wandb():
    """
    Login to Wandb and initialize a run.
    Expects environment variables:
    - WANDB_API_KEY (optional, for automatic login)
    - WANDB_PROJECT
    - WANDB_ENTITY
    """
    try:
        logged_in = wandb.login()
        if not logged_in:
            print("Wandb login required. Please login to Weights & Biases to continue.")
            sys.exit(1)
    except Exception as e:
        print(f"Wandb login failed with error: {e}")
        print("Please ensure you are logged in to Weights & Biases to continue.")
        sys.exit(1)

    # Get git info for prefix
    branch, commit = get_git_info()
    if branch and commit:
        git_prefix = f"{branch}_{commit}"
    else:
        git_prefix = "no-git"

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_name = f"{git_prefix}_{timestamp}"

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "bsqHiC"),
        entity=os.getenv("WANDB_ENTITY", None),
        name=run_name,
        reinit=True
    )
    return run_name

# def compute_metrics(eval_pred):
#     """
#     Compute AUROC and AUPRC metrics for multi-label classification.
#     """
#     predictions, labels = eval_pred
    
#     # Apply sigmoid to get probabilities
#     predictions = torch.sigmoid(torch.from_numpy(predictions)).numpy()
    
#     # Compute metrics for each track
#     auroc_scores = []
#     auprc_scores = []
    
#     for i in range(labels.shape[1]):
#         # Skip if all labels are the same (no positive or negative samples)
#         if len(np.unique(labels[:, i])) > 1:
#             try:
#                 auroc = roc_auc_score(labels[:, i], predictions[:, i])
#                 auprc = average_precision_score(labels[:, i], predictions[:, i])
#                 auroc_scores.append(auroc)
#                 auprc_scores.append(auprc)
#             except:
#                 continue
    
#     return {
#         "auroc_mean": np.mean(auroc_scores) if auroc_scores else 0.0,
#         "auprc_mean": np.mean(auprc_scores) if auprc_scores else 0.0,
#         "auroc_std": np.std(auroc_scores) if auroc_scores else 0.0,
#         "auprc_std": np.std(auprc_scores) if auprc_scores else 0.0,
#     }


def compute_metrics(eval_pred):
    """
    Compute AUROC, AUPRC, and BCE loss for multi-label classification.
    """
    predictions, labels = eval_pred
    
    predictions_tensor = torch.from_numpy(predictions)
    labels_tensor = torch.from_numpy(labels).float()
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(predictions_tensor).numpy()
    
    # Compute BCE loss
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions_tensor, labels_tensor).item()
    
    # Compute metrics for each track
    auroc_scores = []
    auprc_scores = []
    
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            try:
                auroc = roc_auc_score(labels[:, i], probs[:, i])
                auprc = average_precision_score(labels[:, i], probs[:, i])
                auroc_scores.append(auroc)
                auprc_scores.append(auprc)
            except:
                continue
    
    return {
        "auroc_mean": np.mean(auroc_scores) if auroc_scores else 0.0,
        "auroc_median": np.median(auroc_scores) if auroc_scores else 0.0,
        "auprc_mean": np.mean(auprc_scores) if auprc_scores else 0.0,
        "auroc_std": np.std(auroc_scores) if auroc_scores else 0.0,
        "auprc_std": np.std(auprc_scores) if auprc_scores else 0.0,
        "loss": bce_loss
    }