"""
Diagnostic prediction script
"""

# Imports
from datetime import datetime
import streamlit as st
import pandas as pd
from src.cdss.utils.db import get_all_patients, get_patient, update_patient
from src.cdss.utils.models import predict_single, preprocessor, provenance
from src.cdss.utils.recommendations import generate_recommendations
from src.cdss.utils.data import FIELD_REGISTRY, render_field
from src.cdss.utils.misc import display_results, run_batch_predictions

st.title("CKD Diagnostic Prediction")
st.markdown("---")

# Load model
try:
    selected_features = provenance["selected_features"]
    DISPLAY_FIELDS = [feat.split("__", 1)[1] for feat in selected_features]

    st.sidebar.success("Model loaded successfully!")
    with st.sidebar.expander("Model Details"):
        st.json({
            "features": selected_features,
            "n_features": len(selected_features),
            "model_type": provenance.get("model_key", "bnn")
        })

except (FileNotFoundError, KeyError, ValueError) as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure the model path is correct in config.py")
    st.stop()


# Single Patient Prediction
mode = st.radio("Select Testing Mode:", ["Single Patient", "Batch Testing"], horizontal=True)

if mode == "Single Patient":
    st.subheader("Single Patient Diagnostic Prediction")

    input_method = st.radio("Input Method:",["Load from Database", "Manual Entry"], horizontal=True)

    # Load from database
    if input_method == "Load from Database":
        patients = get_all_patients()
        patient_ids = [p["patient_id"] for p in patients]

        if not patient_ids:
            st.info("No patients in database. Please add patients first.")
            st.stop()

        selected_patient = st.selectbox("Select Patient", patient_ids)
        patient_data = get_patient(selected_patient)

        if patient_data:
            st.subheader("Patient Information")

            display_fields = {k: v for k, v in FIELD_REGISTRY.items() if k in DISPLAY_FIELDS}
            cols = st.columns(3)

            for i, (key, config) in enumerate(display_fields.items()):
                with cols[i % 3]:
                    value = patient_data.get(key, "N/A")

                    if config["widget"] == "select" and value != "N/A":
                        display_value = config.get("format_func", str)(value)
                    elif isinstance(value, float) and config.get("format"):
                        display_value = config["format"] % value
                    else:
                        display_value = str(value)

                    st.metric(config["label"], display_value)

            st.markdown("---")

            # Run Prediction Button
            if st.button("Run Diagnostic Prediction", type="primary", icon=":material/play_arrow:"):
                with st.spinner("Running diagnostic analysis..."):
                    result = predict_single(patient_data)

                # Store results in session_state
                st.session_state["last_prediction"] = {
                    "probability": result["probability"],
                    "uncertainty": result["uncertainty"],
                    "recommendations": generate_recommendations(
                        result["probability"], result["uncertainty"]
                    ),
                    "patient_id": selected_patient
                }

            # Display Results if Available
            if "last_prediction" in st.session_state:
                pred = st.session_state["last_prediction"]

                st.markdown("---")
                st.subheader("Diagnostic Prediction Results")
                display_results(pred["probability"], pred["uncertainty"], pred["recommendations"])

                # Save Results Button
                if st.button("Save Results to Patient Record", icon=":material/save:", type="secondary"):
                    update_data = {
                        "predicted_diagnosis": "ckd" if pred["probability"] > 0.5 else "notckd",
                        "predicted_probability": pred["probability"],
                        "uncertainty": pred["uncertainty"],
                        "recommendation": pred["recommendations"]["action"],
                        "follow_up": pred["recommendations"]["follow_up"],
                        "confidence_level": pred["recommendations"]["confidence"],
                        "test_date": datetime.now()
                    }

                    update_patient(pred["patient_id"], update_data)
                    st.success("Results saved to patient record!", icon=":material/check:")
                    st.balloons()

    # Manual entry
    else:
        st.info("Enter patient data manually for immediate diagnostic testing")

        with st.form("manual_test_form"):
            cols = st.columns(3)

            # Initialize values with defaults
            values = {
                feature: FIELD_REGISTRY.get(feature, {}).get("default")
                for feature in preprocessor.feature_names_in_
            }

            # Render only selected model features
            for i, (key, config) in enumerate(FIELD_REGISTRY.items()):
                if key not in DISPLAY_FIELDS:
                    continue
                with cols[i % 3]:
                    values[key] = render_field(key, config)

            test_submitted = st.form_submit_button("Run Test", type="primary")

            if test_submitted:
                with st.spinner("Running diagnostic analysis..."):
                    result = predict_single(values)

                st.session_state["last_prediction"] = {
                    "probability": result["probability"],
                    "uncertainty": result["uncertainty"],
                    "recommendations": generate_recommendations(
                        result["probability"], result["uncertainty"]
                    ),
                    "patient_id": None
                }

        # Display Results if Available
        if "last_prediction" in st.session_state:
            pred = st.session_state["last_prediction"]

            st.markdown("---")
            st.subheader("Diagnostic Prediction Results")
            display_results(pred["probability"], pred["uncertainty"], pred["recommendations"])


# Batch Diagnostic Testing
else:
    st.subheader("Batch Diagnostic Testing")

    batch_mode = st.radio(
        "Batch Mode:",
        ["Test All Pending", "Test ID Range", "Upload CSV"],
        horizontal=True
    )

    # Test all pending
    if batch_mode == "Test All Pending":
        pending_patients = get_all_patients({"diagnosis": {"$exists": False}})

        if not pending_patients:
            st.info("No pending patients to test.")
            st.stop()

        st.write(f"Found **{len(pending_patients)}** patients pending diagnosis")

        with st.expander("Preview Pending Patients"):
            df_preview = pd.DataFrame(pending_patients).drop("_id", axis=1, errors="ignore")
            preview_cols = [c for c in FIELD_REGISTRY.keys() if c in df_preview.columns][:5]
            st.dataframe(df_preview.loc[:, preview_cols], use_container_width=True)

        if st.button("Run Batch Diagnostics", type="primary", use_container_width=True):
            run_batch_predictions(pending_patients, DISPLAY_FIELDS)

    # Define range of patient IDs
    elif batch_mode == "Test ID Range":
        st.info("Select a range of patient IDs to test")

        all_patients = get_all_patients()
        all_ids = sorted([
            int(p["patient_id"]) for p in all_patients
            if str(p.get("patient_id")).isdigit()
        ])

        if not all_ids:
            st.warning("No valid numeric patient IDs found.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            start_id = st.number_input("Start ID", min_value=min(all_ids), max_value=max(all_ids), value=min(all_ids))
        with col2:
            end_id = st.number_input("End ID", min_value=min(all_ids), max_value=max(all_ids), value=max(all_ids))

        if start_id > end_id:
            st.error("Start ID must be less than or equal to End ID.")
            st.stop()

        selected_patients = [
            p for p in all_patients
            if str(p.get("patient_id")).isdigit() and start_id <= int(p["patient_id"]) <= end_id
        ]

        st.write(f"Found **{len(selected_patients)}** patients in selected range")

        if selected_patients:
            with st.expander("Preview Selected Patients"):
                df_preview = pd.DataFrame(selected_patients).drop("_id", axis=1, errors="ignore")
                preview_cols = [c for c in FIELD_REGISTRY.keys() if c in df_preview.columns][:5]
                st.dataframe(df_preview.loc[:, preview_cols], use_container_width=True)

            if st.button("Run Diagnostics for Selected Range", type="primary", use_container_width=True):
                run_batch_predictions(selected_patients, DISPLAY_FIELDS)
        else:
            st.info("No patients found in this ID range.")

    # 3. UPLOAD CSV
    else:
        st.info("Upload a CSV file with patient data for batch testing")

        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df)}** patients from CSV")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Test Uploaded Patients", type="primary", use_container_width=True):
                patients_data = df.to_dict("records")
                run_batch_predictions(patients_data, DISPLAY_FIELDS)
