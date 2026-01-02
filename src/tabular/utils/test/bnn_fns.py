"""BNN specific functions for tabular model evaluation."""

# Imports
import json
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score



def is_bnn_model(model):
    """Check if model supports uncertainty quantification."""
    return hasattr(model, 'predict_with_uncertainty')


def plot_uncertainty_distribution(uncertainties, y_test, save_path):
    """Plot distribution of uncertainties split by correct/incorrect predictions."""

    uncertainties = np.asarray(uncertainties)
    mean_unc = np.mean(uncertainties)
    median_unc = np.median(uncertainties)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall distribution
    ax = axes[0]
    ax.hist(uncertainties, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

    ax.axvline(mean_unc, color='red', linestyle='--', label=f'Mean: {mean_unc:.3f}')
    ax.axvline(median_unc, color='green', linestyle='--', label=f'Median: {median_unc:.3f}')

    ax.set_xlabel('Uncertainty (Std)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Uncertainties')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot by class
    df_unc = pd.DataFrame({
        'Uncertainty': uncertainties,
        'True Class': np.where(np.asarray(y_test) == 1, 'CKD', 'No CKD')
    })

    sns.boxplot(data=df_unc, x='True Class', y='Uncertainty', ax=axes[1])
    axes[1].set_title('Uncertainty by True Class')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved uncertainty distribution to {save_path}")



def plot_uncertainty_vs_confidence(predictions, uncertainties, y_test, save_path):
    """Plot uncertainty vs prediction confidence, colored by correctness."""
    confidence = np.abs(predictions - 0.5) * 2  # 0 = uncertain, 1 = confident
    correct = (predictions >= 0.5) == y_test

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(confidence, uncertainties, 
                         c=correct, cmap='RdYlGn', alpha=0.6, edgecolors='black')
    plt.colorbar(scatter, label='Correct Prediction')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Uncertainty (Std)')
    plt.title('Uncertainty vs Confidence')
    plt.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(confidence, uncertainties, 1)
    p = np.poly1d(z)
    plt.plot(confidence, p(confidence), "r--", alpha=0.8, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved uncertainty vs confidence plot to {save_path}")


def plot_calibration_with_uncertainty(predictions, uncertainties, y_test, save_path, n_bins=10):
    """Plot calibration curve with uncertainty bars."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_accs = []
    bin_confs = []
    bin_uncs = []

    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(y_test[mask].mean())
            bin_confs.append(predictions[mask].mean())
            bin_uncs.append(uncertainties[mask].mean())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(bin_centers[i])
            bin_uncs.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_uncs = np.array(bin_uncs)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.errorbar(bin_confs, bin_accs, yerr=bin_uncs, fmt='o-', 
                capsize=5, label='Model Calibration', color='steelblue')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot with Uncertainty')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved calibration plot to {save_path}")


def uncertainty_based_metrics(predictions, uncertainties, y_test, thresholds=None):
    """Calculate metrics at different uncertainty thresholds."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3]

    results = {}

    for thresh in thresholds:
        mask = uncertainties <= thresh
        if mask.sum() == 0:
            continue

        y_pred_thresh = (predictions[mask] >= 0.5).astype(int)
        acc = accuracy_score(y_test[mask], y_pred_thresh)
        coverage = mask.sum() / len(y_test)

        results[f'uncertainty_{thresh}'] = {
            'accuracy': float(acc),
            'coverage': float(coverage),
            'n_samples': int(mask.sum())
        }

    return results

def bnn_analysis(details):
    """BNN plots and analysis"""
    logger.info("Generating BNN uncertainty analysis...")

    total_unc = details["total_unc"]
    mc_unc = details["mc_unc"]
    ensemble_unc = details["ensemble_unc"]
    y_test = details["y_test"]
    predictions = details["predictions"]
    eval_dir = details["eval_dir"]

    # Uncertainty distribution
    plot_uncertainty_distribution(
        total_unc, y_test,
        eval_dir / "test_uncertainty_distribution.png"
    )

    # Uncertainty vs confidence
    plot_uncertainty_vs_confidence(
        predictions, total_unc, y_test,
        eval_dir / "test_uncertainty_vs_confidence.png"
    )

    # Calibration with uncertainty
    plot_calibration_with_uncertainty(
        predictions, total_unc, y_test,
        eval_dir / "test_calibration_with_uncertainty.png"
    )

    # Uncertainty-based metrics
    unc_metrics = uncertainty_based_metrics(
        predictions, total_unc, y_test,
        thresholds=[0.05, 0.1, 0.15, 0.2]
    )

    # Save comprehensive uncertainty summary
    uncertainty_summary = {
        'mean_uncertainty': float(total_unc.mean()),
        'std_uncertainty': float(total_unc.std()),
        'median_uncertainty': float(np.median(total_unc)),
        'max_uncertainty': float(total_unc.max()),
        'min_uncertainty': float(total_unc.min()),
        'metrics_by_uncertainty_threshold': unc_metrics,
    }

    # Add ensemble-specific metrics if available
    if ensemble_unc is not None and mc_unc is not None:
        uncertainty_summary['mean_ensemble_uncertainty'] = float(ensemble_unc.mean())
        uncertainty_summary['mean_mc_uncertainty'] = float(mc_unc.mean())
        uncertainty_summary['ensemble_to_total_ratio'] = float(
            ensemble_unc.mean() / total_unc.mean()
        )

    with open(eval_dir / "test_uncertainty_metrics.json", "w", encoding="utf-8") as f:
        json.dump(uncertainty_summary, f, indent=4)
    logger.success(f"BNN uncertainty analysis complete. Mean uncertainty: {total_unc.mean():.4f}")
