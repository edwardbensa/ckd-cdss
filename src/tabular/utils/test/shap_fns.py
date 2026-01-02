"""SHAP functions for model evaluation."""

# Imports
import json
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import shap


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


def shap_analysis(details: dict):
    """SHAP plots and analysis."""
    logger.info("Computing SHAP values for interpretability...")

    model = details["model"]
    X_test = details["X_test_sel"]
    features = details["selected_features"]
    eval_dir = details["eval_dir"]
    is_bnn = details["is_bnn"]

    shap_values, shap_explainer = compute_shap(model, X_test, is_bnn=is_bnn)

    if shap_values is not None:
        # Summary plot
        plot_shap_summary(
            shap_values, X_test, features,
            eval_dir / "shap_summary.png"
        )

        # Bar plot
        plot_shap_bar(
            shap_values, features,
            eval_dir / "shap_bar.png"
        )

        # Force plots for specific cases
        if shap_explainer is not None:
            create_shap_force_plots(
                shap_values, X_test, features,
                shap_explainer, eval_dir
            )

        # Save feature importance ranking
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = {
            feat: float(imp) 
            for feat, imp in zip(features, mean_abs_shap)
        }
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        with open(eval_dir / "shap_feature_importance.json", "w", encoding="utf-8") as f:
            json.dump(feature_importance, f, indent=4)

        logger.success("SHAP analysis complete")

    return shap_values, shap_explainer
