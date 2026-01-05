"""Model utility functions"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from src.cdss.config import MODEL_FOLDER_PATH, TEST_DATA


@st.cache_resource
def load_diagnostic_model(model_folder_path):
    """Load the trained diagnostic model and its preprocessor."""
    # Define paths
    model_path = Path(model_folder_path) / "model.joblib"
    prov_path = Path(model_folder_path) / "provenance_training.json"
    preprocessor_path = Path(model_folder_path).parent / "preprocessor.joblib"

    # Load model, provenance and preprocessor
    bnn_model = joblib.load(model_path)
    bnn_preprocessor = joblib.load(preprocessor_path)
    with open(prov_path, "r", encoding="utf-8") as f:
        bnn_provenance = json.load(f)

    return bnn_model, bnn_preprocessor, bnn_provenance

# Load model and store in session state
try:
    model, preprocessor, provenance = load_diagnostic_model(MODEL_FOLDER_PATH)
    selected_features = provenance["selected_features"]

    # Store in session state
    st.session_state["model"] = model
    st.session_state["preprocessor"] = preprocessor
    st.session_state["provenance"] = provenance

except (FileNotFoundError, KeyError, ValueError) as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure the model path is correct in config.py")
    st.stop()


def as_scalar(x):
    """Convert to scalar"""
    if x is None:
        return 0
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def predict_single(patient_data):
    """Make prediction for a single patient with preprocessor."""
    patient_df = pd.DataFrame([patient_data])

    # Transform and get feature names
    X_transformed = preprocessor.transform(patient_df)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    X_test = X_df[selected_features].values

    # Ensure 2D array
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    # Get prediction with uncertainty
    result = model.predict_with_uncertainty(X_test)

    if len(result) == 4:
        pred, total_unc, ensemble_unc, mc_unc = result
    else:
        pred, total_unc, _ = result
        ensemble_unc = None
        mc_unc = None

    return {
        'probability': as_scalar(pred),
        'uncertainty': as_scalar(total_unc),
        'ensemble_uncertainty': as_scalar(ensemble_unc),
        'mc_uncertainty': as_scalar(mc_unc)
    }


def predict_batch(patients_data):
    """Make predictions for multiple patients."""
    results = []

    for patient_data in patients_data:
        pred_result = predict_single(patient_data)
        pred_result['patient_id'] = patient_data.get('patient_id', 'Unknown')
        results.append(pred_result)

    return results


# SHAP
@st.cache_data
def get_background(test_data_path):
    """Get background data for SHAP from test set."""
    df = pd.read_csv(test_data_path)
    X_background = df[selected_features].values

    # Sample if too large (for performance)
    if len(X_background) > 100:
        indices = np.random.choice(len(X_background), 100, replace=False)
        X_background = X_background[indices]

    return X_background


@st.cache_resource
def get_explainer(test_data_path):
    """Create and cache the SHAP explainer."""
    background = get_background(test_data_path)
    explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)

    return explainer


def plot_css():
    """Adaptive CSS for light and dark mode."""
    adaptive_css = """
    <style>
        body {
            background-color: white !important;
            padding: 5px;
        }
        text {
            fill: #333 !important;
        }
        .tick text {
            fill: #333 !important;
        }
        svg {
            background-color: white !important;
        }
        .label {
            color: #333 !important;
        }
    </style>
    """
    return adaptive_css


def shap_force_plot(patient_data):
    """Generate interactive SHAP force plot (HTML) for a single patient."""
    explainer = get_explainer(TEST_DATA)

    # Process specific patient
    patient_df = pd.DataFrame([patient_data])
    X_transformed = preprocessor.transform(patient_df)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    X_patient = X_df[selected_features].values

    if X_patient.ndim == 1:
        X_patient = X_patient.reshape(1, -1)

    # Compute patient SHAP values 
    shap_values = explainer.shap_values(X_patient, nsamples=100)

    # Generate force plot for this patient
    shap.initjs()
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_patient[0],
        feature_names=selected_features
    )

    shap_html = f"<head>{shap.getjs()}{plot_css()}</head><body>{force_plot.html()}</body>"

    return shap_html
