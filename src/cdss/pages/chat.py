"""RAG agent chat interface"""

# Imports
import uuid
import datetime
import requests
import streamlit as st
from src.cdss.utils.db import get_all_patients, get_patient
from src.cdss.utils.misc import decode_stream

API_URL = "http://localhost:8000"

st.title("Clinical Guidance Chat (NICE NG203)")

# Create or load session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# Mode selection
mode = st.radio(
    "Conversation Mode",
    ["Free conversation", "Patient-focused"],
    horizontal=True
)

patient = None

# Patient Selection
if mode == "Patient-focused":
    patients = get_all_patients()
    patient_ids = [p["patient_id"] for p in patients]

    selected_id = st.selectbox("Select Patient", [""] + patient_ids)

    if selected_id:
        patient = get_patient(selected_id)

        with st.expander("Patient Summary", expanded=True):
            st.json({k: v for k, v in patient.items() if k != "_id"}) # type: ignore


# Chat Input
user_query = st.text_area("Ask a clinical question:")

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Streaming Response
    if mode == "Free conversation":
        endpoint = f"{API_URL}/chat/{session_id}"
        payload = {"query": user_query}

    else:  # Patient-focused
        if patient is None:
            st.warning("Please select a patient first.")
            st.stop()

        endpoint = f"{API_URL}/patient_chat/{session_id}"
        clean_patient = {}
        for k, v in patient.items():
            if isinstance(v, datetime.datetime):
                clean_patient[k] = v.isoformat()  # Converts to "1987-01-08T00:00:00"
            elif k == "_id":
                clean_patient[k] = str(v)
            else:
                clean_patient[k] = v
        payload = {"query": user_query, "patient": clean_patient}

    try:
        with st.spinner("Thinking..."):
            response = requests.post(endpoint, json=payload, stream=True, timeout=30)

            # Check for HTTP errors
            if response.status_code != 200:
                st.error(f"API Error: {response.status_code}")
                st.code(response.text)
                st.stop()

            st.markdown("### Recommendation")

            answer = st.write_stream(decode_stream(response))

        # Fetch Citations
        citations_response = requests.post(
            f"{API_URL}/citations",
            json={"query": user_query},
            timeout=30
        )

        if citations_response.status_code == 200:
            citations_data = citations_response.json()
            citations = citations_data.get("citations", [])

            if citations:
                st.markdown("### Citations")
                for c in citations:
                    st.markdown(
                        f"- [{c['id']}]({c['url']}) — {c['section']} ({c['type']})"
                    )
        else:
            st.warning(f"Could not fetch citations: {citations_response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Make sure RAG API is running at http://localhost:8000")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The query may be taking too long to process.")
    except requests.exceptions.JSONDecodeError as e:
        st.error(f"Invalid JSON response from API: {e}")
    except (ValueError, TypeError) as e:
        st.error(f"An unexpected error occurred: {e}")
