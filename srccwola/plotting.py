"""Functions and utilities for creating plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_closure(
    inputs: np.ndarray, labels: np.ndarray, outputs: np.ndarray, output_path: Path
):
    """Plot a ROC comparing the predictions on the template vs background."""
    # Get the combined outputs
    tem_outs = outputs[labels == 0]
    bkg_outs = outputs[inputs[:, -1] == 0]
    preds = np.hstack((tem_outs, bkg_outs))
    real_labels = np.hstack((np.zeros(len(tem_outs)), np.ones(len(bkg_outs))))

    # Get the AUC
    fpr, tpr, _ = roc_curve(real_labels, preds)
    auc_score = auc(fpr, tpr)
    print("[--] Closure AUC: ", auc_score)

    # Plot and save the ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"Template AUC: {auc_score:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_path / "closure_roc.png")


def plot_svb(
    inputs: np.ndarray, outputs: np.ndarray, extra_preds: np.ndarray, output_path: Path
) -> None:
    """Plot a ROC comparing the signal vs background predictions."""
    # Get the combined outputs
    real_mask = inputs[:, -1] > -1  # template data has -1 label
    outs = outputs[real_mask]
    labs = inputs[:, -1][real_mask]

    # Also include the extra predictions
    outs = np.hstack((outs, extra_preds))
    labs = np.hstack((labs, np.ones(len(extra_preds))))

    # Get the AUC
    fpr, tpr, _ = roc_curve(labs, outs)
    auc_score = auc(fpr, tpr)
    print("[--] Sig vs Bkg AUC: ", auc_score)

    # Plot and save the ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"Template AUC: {auc_score:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output_path / "svb_roc.png")
