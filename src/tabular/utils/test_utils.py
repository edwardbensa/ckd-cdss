"""Testing utility functions for tabular data models."""

# Imports
import json
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from src.config import INTERIM_DATA_DIR


def evaluate_model(specs):
    """
    Evaluate a trained model on hidden CKD test data and save results.

    Parameters
    specs : dict with model directory, target, and test data name.

    Saves
    - classification_report.json
    - confusion_matrix.png
    - roc_curve.png
    - precision_recall_curve.png
    """

    model_dir = Path(specs["model_dir"])
    eval_dir = Path(model_dir / "eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    df_test = pd.read_csv(INTERIM_DATA_DIR / f"tabular_data/{specs["test_file"]}.csv")
    X_test = df_test.drop(specs["target"], axis=1)
    y_test = df_test[specs["target"]]
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

    # Evaluate model
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test ROC-AUC: {auc_score:.4f}")

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

    logger.success("Testing complete. Results saved to model_dir.")
