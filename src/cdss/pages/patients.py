"""
Patient CRUD operations page
"""

from datetime import datetime
import streamlit as st
import pandas as pd
from src.cdss.utils.misc import read_dipstick_image
from src.cdss.utils.db import (
    get_all_patients, add_patient, update_patient,
    delete_patient, get_patient, next_patient_id
)
from src.cdss.utils.data import (
    FIELD_REGISTRY, render_field, compute_derived_fields,
    validate_dates, group_fields_by_section
)


st.title("Patient Management")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Add Patient",
    "View Patients",
    "Update Patient",
    "Delete Patient"
])

# Create
with tab1:
    st.subheader("Add New Patient")

    sections = group_fields_by_section(FIELD_REGISTRY)

    with st.form("add_patient_form"):
        values = {}
        # Auto increment ID
        values["patient_id"] = next_patient_id()

        for section_name, fields in sections.items():
            with st.expander(section_name, expanded=False):
                cols = st.columns(2)
                for i, (key, cfg) in enumerate(fields):
                    if cfg.get("computed"):
                        continue
                    with cols[i % 2]:
                        values[key] = render_field(key, cfg)

        with st.expander("Dipstick Image Upload", expanded=False):
            uploaded = st.file_uploader("Upload Dipstick Image", type=["jpg", "jpeg", "png"])

            if uploaded:
                values["dipstick_image"] = uploaded.getvalue()
                mapped = read_dipstick_image(uploaded)
                for k, v in mapped.items():
                    values[k] = v

                st.info("Dipstick values have been pre-filled into the form.")

        submitted = st.form_submit_button("Add Patient", type="primary")

        if submitted:
            # Date validation
            date_errors = validate_dates(values)
            if date_errors:
                for err in date_errors:
                    st.error(err)
                st.stop()

            # Compute derived fields and add logging
            values = compute_derived_fields(values)
            values["created_at"] = datetime.now()
            values["updated_at"] = datetime.now()

            add_patient(values)
            st.success(f"Patient {values['patient_id']} added successfully!")
            st.balloons()

# Read
with tab2:
    st.subheader("View All Patients")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search by Patient ID", placeholder="Enter patient ID...")
    with col2:
        filter_diagnosed = st.selectbox("Filter", ["All", "Diagnosed", "Pending"])

    query = {}
    if search:
        query["patient_id"] = {"$regex": search, "$options": "i"}
    if filter_diagnosed == "Diagnosed":
        query["predicted_diagnosis"] = {"$exists": True}
    elif filter_diagnosed == "Pending":
        query["predicted_diagnosis"] = {"$exists": False}

    patients = get_all_patients(query)

    if patients:
        st.write(f"Found {len(patients)} patient(s)")

        df: pd.DataFrame = pd.DataFrame(patients).drop("_id", axis=1)

        # Order columns based on registry
        base_cols = list(FIELD_REGISTRY.keys())
        extra_cols = [c for c in ["predicted_diagnosis", "predicted_probability", "uncertainty"]
                      if c in df.columns]

        display_cols = [c for c in base_cols + extra_cols if c in df.columns]

        st.dataframe(df.loc[:, display_cols], width='stretch')

        patients_with_images = [p for p in patients if "dipstick_image" in p]
        st.markdown("### Dipstick Image Viewer")

        if patients_with_images:
            selected = st.selectbox(
                "Select a patient to view their dipstick image:",
                [""] + [p["patient_id"] for p in patients_with_images]
            )

            if selected:
                patient = next(p for p in patients_with_images if p["patient_id"] == selected)
                st.image(
                    patient["dipstick_image"],
                    caption=f"Dipstick Image for Patient {selected}",
                    width='stretch'
                )
                st.caption("Click the image to enlarge it.")
        else:
            st.info("No dipstick images found for any patients.")


# Update
with tab3:
    st.subheader("Update Patient Information")

    patients = get_all_patients()
    patient_ids = [p["patient_id"] for p in patients]

    if patient_ids:
        selected_id = st.selectbox("Select Patient to Update", patient_ids)
        patient = get_patient(selected_id)

        if patient:
            sections = group_fields_by_section(FIELD_REGISTRY)

            with st.form("update_patient_form"):
                updated = {}

                for section_name, fields in sections.items():
                    with st.expander(section_name, expanded=False):
                        cols = st.columns(2)
                        for i, (key, cfg) in enumerate(fields):
                            if cfg.get("computed"):
                                continue
                            with cols[i % 2]:
                                updated[key] = render_field(key, cfg, value=patient.get(key))

                with st.expander("Dipstick Image Upload", expanded=False):
                    uploaded = st.file_uploader("Upload Dipstick Image",type=["jpg", "jpeg", "png"],
                                                key="update_dipstick")

                    if uploaded:
                        updated["dipstick_image"] = uploaded.getvalue()
                        mapped = read_dipstick_image(uploaded)
                        for k, v in mapped.items():
                            updated[k] = v

                        st.info("Dipstick values have been updated in the form.")

                if st.form_submit_button("Update Patient", type="primary"):
                    # Date validation
                    date_errors = validate_dates(updated)
                    if date_errors:
                        for err in date_errors:
                            st.error(err)
                        st.stop()

                    # Compute derived fields
                    updated = compute_derived_fields(updated)
                    updated["updated_at"] = datetime.now()

                    update_patient(selected_id, updated)
                    st.success(f"Patient {selected_id} updated successfully!")
                    st.rerun()

    else:
        st.info("No patients in database.")


# Delete
with tab4:
    st.subheader("Delete Patient")
    st.warning("This action cannot be undone!")

    patients = get_all_patients()
    patient_ids = [p["patient_id"] for p in patients]

    if patient_ids:
        delete_id = st.selectbox("Select Patient to Delete", [""] + patient_ids)

        if delete_id:
            patient = get_patient(delete_id)

            if patient:
                st.json({k: v for k, v in patient.items() if k != "_id"})

                if st.button("Confirm Delete", type="primary"):
                    delete_patient(delete_id)
                    st.success(f"Patient {delete_id} deleted successfully!")
                    st.rerun()
    else:
        st.info("No patients in database.")
