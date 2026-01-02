"""Testing utility functions for tabular data models with BNN uncertainty and SHAP support."""

# Imports
import json
from pathlib import Path
from loguru import logger
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from src.config import INTERIM_DATA_DIR
from .bnn_fns import is_bnn_model, bnn_analysis
from .shap_fns import shap_analysis
from .std_fns import results_dataframe, cls_plots


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
    details = {
        "X_test_sel": X_test_sel,
        "X_test_full": X_test,
        "y_test": y_test,
        "selected_features": selected_features,
        "model": model,
        "is_bnn": is_bnn,
        "mc_unc": mc_unc,
        "total_unc": total_unc,
        "ensemble_unc": ensemble_unc,
        "predictions": predictions,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "acc": acc,
        "auc_score": auc_score,
        "shap_values": None,
        "shap_explainer": None,
        "eval_dir": eval_dir
    }

    if specs.get("compute_shap", True):
        shap_values, shap_explainer = shap_analysis(details)
        details["shap_values"] = shap_values
        details["shap_explainer"] = shap_explainer

    # Create results df
    logger.info("Creating detailed results dataframe...")
    results_df = results_dataframe(details)

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

    # Confusion Matrix, ROC Curve, PR Curve
    cls_plots(details)

    # BNN-specific analysis
    if is_bnn and total_unc is not None:
        bnn_analysis(details)

    # Final summary
    logger.success("Testing complete. Generated outputs:")
    logger.success(f"  - Cases for review: {len(review_cases)} / {len(results_df)}")
