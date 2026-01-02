"""Standard utility functions for results dataframe creation and classification results plotting."""

# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve
)


def results_dataframe(details):
    """Create results dataframe for clinical review."""

    X_test_sel = details["X_test_sel"]
    X_test_full = details["X_test_full"]
    y_test = details["y_test"]
    predictions = details["predictions"]
    y_pred = details["y_pred"]
    selected_features = details["selected_features"]
    all_features = details["X_test_full"].columns.tolist()
    uncertainties = details["total_unc"]
    ensemble_unc = details["ensemble_unc"]
    mc_unc = details["mc_unc"]
    shap_values = details["shap_values"]

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


def cls_plots(details):
    """Plots confusion matrix, AUC ROC curve, precision recall curve"""
    y_test = details["y_test"]
    y_pred = details["y_pred"]
    y_proba = details["y_proba"]
    eval_dir = details["eval_dir"]

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
