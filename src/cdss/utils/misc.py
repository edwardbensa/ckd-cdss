"""Miscellaneous utility functions."""

# Imports
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
from src.cdss.utils.db import update_patient
from src.cdss.utils.models import predict_single
from src.cdss.utils.recommendations import generate_recommendations
from src.cdss.utils.data import convert_dipstick_readings
from src.dipstick_clf.read.utils import read_dipstick
from src.cdss.config import DIPSTICK_IMAGE_FOLDER


def display_results(probability, uncertainty, recommendations):
    """Display results and recommendations."""
    # Alert box
    alert_type = recommendations["alert_level"]
    if alert_type == "critical":
        st.error(f"{recommendations['decision']}", icon=":material/error:")
    elif alert_type == "warning":
        st.warning(f"{recommendations['decision']}", icon=":material/warning:")
    elif alert_type == "success":
        st.success(f"{recommendations['decision']}", icon=":material/diagnosis:")
    else:
        st.info(f"{recommendations['decision']}", icon=":material/diagnosis:")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CKD Probability", f"{probability*100:.1f}%")
    with col2:
        st.metric("Model Uncertainty", f"{uncertainty*100:.2f}%")
    with col3:
        st.metric("Confidence Level", recommendations["confidence"])

    # Clinical recommendations
    st.markdown("---")
    st.subheader("Clinical Recommendations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommended Action:**")
        st.info(recommendations["action"])
    with col2:
        st.markdown("**Follow-up Plan:**")
        st.info(recommendations["follow_up"])

# Check if patient has all required model features
def has_required_features(patient, required_features):
    return all(
        feature in patient and patient[feature] not in [None, ""]
        for feature in required_features
    )


def run_batch_predictions(patients, fields):
    # Filter out incomplete records
    valid_patients = [
        p for p in patients
        if has_required_features(p, fields)
    ]

    skipped_patients = [
        p for p in patients
        if p not in valid_patients
    ]

    if skipped_patients:
        st.warning(
            f"Skipping {len(skipped_patients)} patients due to missing required features: "
            f"{', '.join(fields)}"
        )

    if not valid_patients:
        st.error("No valid patients to test after filtering.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    for i, patient in enumerate(valid_patients):
        pid = patient.get("patient_id", f"ID_{i}")
        status_text.text(f"Testing patient {pid}... ({i+1}/{len(valid_patients)})")

        result = predict_single(patient)
        probability = result["probability"]
        uncertainty = result["uncertainty"]
        recommendations = generate_recommendations(probability, uncertainty)

        # Save to DB if patient has a real ID
        if "patient_id" in patient:
            update_data = {
                "predicted_diagnosis": "ckd" if probability > 0.5 else "notckd",
                "predicted_probability": probability,
                "uncertainty": uncertainty,
                "recommendation": recommendations["action"],
                "confidence_level": recommendations["confidence"],
                "test_date": datetime.now()
            }
            update_patient(patient["patient_id"], update_data)

        results.append({
            "patient_id": pid,
            "diagnosis": "CKD" if probability > 0.5 else "No CKD",
            "probability": f"{probability*100:.1f}%",
            "uncertainty": f"{uncertainty*100:.2f}%",
            "confidence": recommendations["confidence"],
            "alert": recommendations["alert_level"]
        })

        progress_bar.progress((i + 1) / len(valid_patients))

    status_text.text("Batch testing complete!")
    st.success(f"Tested **{len(results)}** patients successfully!")

    # Display results
    st.markdown("---")
    st.subheader("Batch Results")

    df_results = pd.DataFrame(results)

    def highlight_alert(row):
        colors = {
            'critical': 'background-color: #ffcccc',
            'warning': 'background-color: #fff4cc',
            'success': 'background-color: #ccffcc',
            'info': 'background-color: #cce5ff'
        }
        return [colors.get(row['alert'], '')] * len(row)

    styled_df = df_results.style.apply(highlight_alert, axis=1)
    st.dataframe(styled_df, width="stretch")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CKD Cases", sum(r["diagnosis"] == "CKD" for r in results))
    with col2:
        st.metric("High Risk Cases", sum(r["alert"] in ["critical", "warning"] for r in results))
    with col3:
        st.metric("Needs Review", sum(r["confidence"] in ["Low", "Uncertain"] for r in results))

    st.balloons()

def read_dipstick_image(uploaded):
    """Read a urine dipstick image and return the mapped results."""
    st.image(uploaded, caption="Uploaded Dipstick Image", width="content")

    # Save to temp file
    temp_path = Path(f"{DIPSTICK_IMAGE_FOLDER}/temp_dipstick.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    # Run model
    results = read_dipstick(temp_path)

    # Map to patient fields
    mapped = convert_dipstick_readings(results)

    # Build results table
    table_rows = []
    for pad, info in results.items():
        table_rows.append({
            "Test": pad,
            "Value": info.get("value", "N/A"),
            "Confidence": info.get("confidence", "N/A"),
            "ΔE": info.get("delta_e", "N/A"),
        })

    df = pd.DataFrame(table_rows)

    st.markdown("### Dipstick Results")
    st.dataframe(df, width="stretch")

    st.success("Dipstick analysis complete.")

    return mapped


def decode_stream(response):
    """
    Decodes a streaming response from the FastAPI backend.
    Ensures that newlines and spaces are preserved for Markdown rendering.
    """
    try:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                token = chunk.decode("utf-8")
                yield token
    except ValueError as e:
        yield f"\n\n**Stream Error:** {str(e)}"
