"""Testing utility functions for tabular data models with BNN uncertainty and SHAP support."""

# Imports
import json
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from src.config import INTERIM_DATA_DIR

def is_bnn_model(model):
    """Check if model supports uncertainty quantification."""
    return hasattr(model, 'predict_with_uncertainty')


def plot_uncertainty_distribution(uncertainties, y_test, save_path):
    """Plot distribution of uncertainties split by correct/incorrect predictions."""

    uncertainties = np.asarray(uncertainties)
    mean_unc = np.mean(uncertainties)
    median_unc = np.median(uncertainties)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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


# SHAP
def compute_shap(model, X_test, is_bnn=False):
    """
    Compute SHAP values for model interpretability.
    """
    logger.info("Computing SHAP values for model interpretability...")

    try:
        # For BNN/neural models, use Kernel or Sampling explainer
        if is_bnn:
            # Use a subset for background (faster computation)
            background = shap.sample(X_test, min(100, len(X_test)))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1],
                background
            )
            shap_values = explainer.shap_values(X_test, nsamples=100)
        else:
            # Try TreeExplainer for tree-based models
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                # Handle multi-output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            except (AttributeError, TypeError, RuntimeError):
                # Fallback to KernelExplainer
                background = shap.sample(X_test, min(100, len(X_test)))
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1],
                    background
                )
                shap_values = explainer.shap_values(X_test, nsamples=100)

        logger.info("SHAP values computed successfully")
        return shap_values, explainer

    except (AttributeError, TypeError, RuntimeError) as e:
        logger.error(f"Error computing SHAP values: {e}")
        return None, None


def plot_shap_summary(shap_values, X_test, feature_names, save_path):
    """Create SHAP summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {save_path}")


def plot_shap_bar(shap_values, feature_names, save_path):
    """Create SHAP bar plot showing mean absolute importance."""
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, 
                     plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP bar plot to {save_path}")


def create_shap_force_plots(shap_values, X_test, feature_names, explainer, save_dir):
    """Generate SHAP force plot for each case."""
    sfp_dir = Path(save_dir / "shap_force_plots")
    sfp_dir.mkdir(parents=True, exist_ok=True)

    pred_count = len(X_test)

    for i in range(0, pred_count):
        plt.figure(figsize=(12, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            X_test[i],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        plt.savefig(sfp_dir / f"sfp_{i}.png", dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved SHAP force plots for {len(X_test)} cases")


def results_dataframe(
    X_test_sel, X_test_full, y_test, predictions, y_pred,
    selected_features, all_features,
    uncertainties=None, ensemble_unc=None, mc_unc=None,
    shap_values=None
):
    """Create results dataframe for clinical review."""

    # Base dataframe
    results_df = pd.DataFrame({
        'sample_id': np.arange(len(y_test)),
        'true_label': y_test,
        'predicted_label': y_pred,
        'predicted_probability': predictions,
        'prediction_confidence': np.abs(predictions - 0.5) * 2,
        'correct_prediction': y_pred == y_test,
    })

    # Uncertainty fields
    if uncertainties is not None:
        results_df['total_uncertainty'] = uncertainties
        results_df['confidence_category'] = pd.cut(
            uncertainties,
            bins=[0, 0.1, 0.2, np.inf],
            labels=['High Confidence', 'Medium Confidence', 'Low Confidence']
        )

        if ensemble_unc is not None:
            results_df['ensemble_uncertainty'] = ensemble_unc
            results_df['mc_dropout_uncertainty'] = mc_unc

    # Selected feature values
    feature_df = pd.DataFrame(
        X_test_sel,
        columns=[f'feature_{f}' for f in selected_features]
    )
    results_df = pd.concat([results_df, feature_df], axis=1)

    # SHAP values
    if shap_values is not None:
        shap_df = pd.DataFrame(
            shap_values,
            columns=[f'shap_{f}' for f in selected_features]
        )
        results_df = pd.concat([results_df, shap_df], axis=1)

        # Total SHAP importance
        results_df['total_shap_importance'] = np.abs(shap_values).sum(axis=1)

        # Top 3 features per sample
        abs_shap = np.abs(shap_values)
        top_idx = np.argsort(abs_shap, axis=1)[:, -3:][:, ::-1]
        results_df['top_3_features'] = [
            ', '.join([selected_features[j] for j in row])
            for row in top_idx
        ]

    # Add clinical flags
    results_df['needs_review'] = False

    if uncertainties is not None:
        # Flag high uncertainty cases
        results_df.loc[uncertainties > np.percentile(uncertainties, 90), 'needs_review'] = True

    # Flag incorrect predictions
    results_df.loc[~results_df['correct_prediction'], 'needs_review'] = True

    # Flag borderline predictions
    results_df.loc[
        (predictions > 0.4) & (predictions < 0.6), 
        'needs_review'
    ] = True

    # Add all features (not just selected ones) for complete record
    for feat in all_features:
        if feat not in selected_features and feat in X_test_full.columns:
            results_df[f'other_{feat}'] = X_test_full[feat].values

    return results_df


def evaluate_model(specs):
    """
    Evaluate a trained model on hidden test data and save results.
    Enhanced with BNN uncertainty quantification and SHAP interpretability.

    Parameters
    specs : dict with model directory, target, test data name, and SHAP options

    Saves
    - test_results_detailed.csv: Comprehensive results for clinical review
    - classification_report.json
    - confusion_matrix.png
    - roc_curve.png
    - precision_recall_curve.png
    - [BNN] uncertainty_distribution.png
    - [BNN] uncertainty_vs_confidence.png
    - [BNN] calibration_with_uncertainty.png
    - [BNN] uncertainty_metrics.json
    - [SHAP] shap_summary.png
    - [SHAP] shap_bar.png
    - [SHAP] shap_force_plot_case_*.png
    - [SHAP] shap_feature_importance.json
    """

    model_dir = Path(specs["model_dir"])
    eval_dir = Path(model_dir / "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    df_test = pd.read_csv(INTERIM_DATA_DIR / f"tabular_data/{specs['test_file']}.csv")
    X_test = df_test.drop(specs["target"], axis=1)
    y_test = df_test[specs["target"]].values
    logger.info(f"Loaded test dataset with shape: {df_test.shape}")

    # Load model + provenance
    model_file = model_dir / "model.joblib"
    prov_file = model_dir / "provenance_training.json"

    model = joblib.load(model_file)
    with open(prov_file, "r", encoding="utf-8") as f:
        provenance = json.load(f)

    selected_features = provenance["selected_features"]
    logger.info(f"Using selected features: {selected_features}")

    # Prepare test data
    X_test_sel = X_test[selected_features].values

    # Check if BNN model
    is_bnn = is_bnn_model(model)

    # Initialize uncertainty variables
    total_unc = None
    ensemble_unc = None
    mc_unc = None

    if is_bnn:
        logger.info("Detected BNN model - computing predictions with uncertainty...")

        # Get predictions with uncertainty
        result = model.predict_with_uncertainty(X_test_sel)

        # Handle different return formats (single BNN vs ensemble)
        if len(result) == 4:
            predictions, total_unc, ensemble_unc, mc_unc = result
        else:
            predictions, total_unc, _ = result
            ensemble_unc = None
            mc_unc = total_unc

        y_proba = predictions
        y_pred = (predictions >= 0.5).astype(int)

        logger.info(f"Mean uncertainty: {total_unc.mean():.4f} (std: {total_unc.std():.4f})")
        logger.info(f"Max uncertainty: {total_unc.max():.4f}, Min: {total_unc.min():.4f}")

    else:
        logger.info("Standard model detected - computing predictions...")
        y_pred = model.predict(X_test_sel)
        y_proba = model.predict_proba(X_test_sel)[:, 1]
        predictions = y_proba

    # Standard metrics
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test ROC-AUC: {auc_score:.4f}")

    # SHAP Analysis
    shap_values = None
    shap_explainer = None

    if specs.get("compute_shap", True):
        logger.info("Computing SHAP values for interpretability...")
        shap_values, shap_explainer = compute_shap(model, X_test_sel, is_bnn=is_bnn)

        if shap_values is not None:
            # Summary plot
            plot_shap_summary(
                shap_values, X_test_sel, selected_features,
                eval_dir / "shap_summary.png"
            )

            # Bar plot
            plot_shap_bar(
                shap_values, selected_features,
                eval_dir / "shap_bar.png"
            )

            # Force plots for specific cases
            if shap_explainer is not None:
                create_shap_force_plots(
                    shap_values, X_test_sel, selected_features,
                    shap_explainer, eval_dir
                )

            # Save feature importance ranking
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = {
                feat: float(imp) 
                for feat, imp in zip(selected_features, mean_abs_shap)
            }
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            with open(eval_dir / "shap_feature_importance.json", "w", encoding="utf-8") as f:
                json.dump(feature_importance, f, indent=4)

            logger.success("SHAP analysis complete")

    # Create results df
    logger.info("Creating detailed results dataframe...")
    results_df = results_dataframe(
        X_test_sel=X_test_sel,
        X_test_full=X_test,
        y_test=y_test,
        predictions=predictions,
        y_pred=y_pred,
        selected_features=selected_features,
        all_features=X_test.columns.tolist(),
        uncertainties=total_unc,
        ensemble_unc=ensemble_unc,
        mc_unc=mc_unc,
        shap_values=shap_values
    )

    # Save detailed results
    results_csv_path = eval_dir / "test_results_detailed.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved detailed results to {results_csv_path}")

    # Create summary for cases needing review
    review_cases = results_df[results_df['needs_review'] == True]
    if len(review_cases) > 0:
        review_csv_path = eval_dir / "cases_for_review.csv"
        review_cases.to_csv(review_csv_path, index=False)
        logger.warning(f"{len(review_cases)} cases need review. Saved to {review_csv_path}")

    # Save classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(eval_dir / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    logger.info("Saved classification report.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No CKD", "CKD"], yticklabels=["No CKD", "CKD"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(eval_dir / "test_confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(eval_dir / "test_roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="green", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(eval_dir / "test_precision_recall_curve.png")
    plt.close()

    # BNN-specific analysis
    if is_bnn and total_unc is not None:
        logger.info("Generating BNN uncertainty analysis...")

        # 1. Uncertainty distribution
        plot_uncertainty_distribution(
            total_unc, y_test,
            eval_dir / "test_uncertainty_distribution.png"
        )

        # 2. Uncertainty vs confidence
        plot_uncertainty_vs_confidence(
            predictions, total_unc, y_test,
            eval_dir / "test_uncertainty_vs_confidence.png"
        )

        # 3. Calibration with uncertainty
        plot_calibration_with_uncertainty(
            predictions, total_unc, y_test,
            eval_dir / "test_calibration_with_uncertainty.png"
        )

        # 4. Uncertainty-based metrics
        unc_metrics = uncertainty_based_metrics(
            predictions, total_unc, y_test,
            thresholds=[0.05, 0.1, 0.15, 0.2]
        )

        # 5. Save comprehensive uncertainty summary
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

    # Final summary
    logger.success("Testing complete. Generated outputs:")
    logger.success(f"  - Cases for review: {len(review_cases)} / {len(results_df)}")
