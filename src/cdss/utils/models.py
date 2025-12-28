"""Model utility functions"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_diagnostic_model(model_folder_path):
    """Load the trained diagnostic model and its preprocessor."""
    # Define paths
    model_path = Path(model_folder_path) / "model.joblib"
    prov_path = Path(model_folder_path) / "provenance_training.json"
    preprocessor_path = Path(model_folder_path).parent / "preprocessor.joblib"

    # Load model, provenance and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(prov_path, "r", encoding="utf-8") as f:
        provenance = json.load(f)

    return model, preprocessor, provenance


def as_scalar(x):
    if x is None:
        return 0
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def predict_single(model, preprocessor, provenance, patient_data):
    """
    Make prediction for a single patient with preprocessor.
    
    Args:
        model: Trained BNN model
        preprocessor: Feature preprocessor (can be None)
        patient_data: Dict with patient features
    
    Returns:
        dict with probability, uncertainty, and other metrics
    """
    patient_df = pd.DataFrame([patient_data])

    # Transform and get feature names
    X_transformed = preprocessor.transform(patient_df)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    selected_features = provenance["selected_features"]
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


def predict_batch(model, preprocessor, provenance, patients_data):
    """
    Make predictions for multiple patients.
    
    Args:
        model: Trained BNN model
        preprocessor: Feature preprocessor (can be None)
        patients_data: List of dicts with patient features
    
    Returns:
        list of prediction dicts
    """
    results = []

    for patient_data in patients_data:
        pred_result = predict_single(model, preprocessor, provenance, patient_data)
        pred_result['patient_id'] = patient_data.get('patient_id', 'Unknown')
        results.append(pred_result)

    return results
