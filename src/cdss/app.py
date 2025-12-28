"""Streamlit CDSS app."""
# Imports
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CKD Clinical Decision Support",
    page_icon=":material/nephrology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

home_page = st.Page("pages/home.py", title="Homepage", icon=":material/home:")
patients_page = st.Page("pages/patients.py", title="Patient Management", icon=":material/patient_list:")
predictions_page = st.Page("pages/predictions.py", title="Diagnostic Prediction", icon=":material/diagnosis:")
analytics_page = st.Page("pages/analytics.py", title="Analytics Dashboard", icon=":material/analytics:")
chat_page = st.Page("pages/chat.py", title="Guideline Chat", icon=":material/chat:")

st.navigation([home_page, patients_page, predictions_page, analytics_page, chat_page]).run()
