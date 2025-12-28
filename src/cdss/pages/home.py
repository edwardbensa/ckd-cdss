"""CDSS homepage."""
import streamlit as st
from src.cdss.utils.db import get_patients_collection

st.title("CKD Clinical Decision Support System")
st.markdown("---")

patients_collection = get_patients_collection()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Patients", patients_collection.count_documents({}))

with col2:
    ckd_count = patients_collection.count_documents({"diagnosis": "CKD"})
    st.metric("CKD Diagnoses", ckd_count)

with col3:
    pending = patients_collection.count_documents({"diagnosis": {"$exists": False}})
    st.metric("Pending Review", pending)

st.markdown("---")

st.subheader("System Overview")
st.markdown("""
This Clinical Decision Support System uses Bayesian Neural Networks to:
- **Predict CKD** with uncertainty quantification
- **Provide actionable recommendations** based on confidence levels
- **Explain predictions** using SHAP values
- **Manage patient data** with full CRUD operations


### Key Features:
- **100% accuracy** on validation dataset
- **Uncertainty quantification** for clinical confidence
- **SHAP explanations** for interpretability
- **MongoDB integration** for secure data storage

### Quick Start:
1. Navigate to **Patient Management** to add or view patients
2. Use **Diagnostic Testing** to run predictions on patient data
3. View **Analytics Dashboard** for system-wide insights
""")

st.info("This system is intended to support clinical decision-making, not replace it.")
