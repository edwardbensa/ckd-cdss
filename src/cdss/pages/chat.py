"""RAG agent chat interface"""

# Imports
import uuid
import datetime
import requests
import streamlit as st
from src.cdss.utils.db import get_all_patients, get_patient
from src.cdss.utils.misc import decode_stream

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CKD Clinical Assistant", layout="wide")
st.title("Clinical Guidance Chat (NICE NG203)")

# Initialise session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # Store visual chat history

# Sidebar: mode and patient selection
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Conversation Mode",
        ["Free conversation", "Patient-focused"],
        index=0
    )

    patient = None
    if mode == "Patient-focused":
        patients = get_all_patients()
        patient_ids = [p["patient_id"] for p in patients]
        selected_id = st.selectbox("Select Patient", [""] + patient_ids)

        if selected_id:
            patient = get_patient(selected_id)
            st.success(f"Patient {selected_id} selected.")
            with st.expander("View Patient Summary"):
                st.json({k: v for k, v in patient.items() if k != "_id"}) # type: ignore

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("Citations"):
                for c in message["citations"]:
                    st.markdown(f"- [{c['id']}]({c['url']}) — {c['section']} ({c['type']})")

# Chat Input & Response Logic
if prompt := st.chat_input("Ask a clinical question..."):

    # Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare Payload
    if mode == "Free conversation":
        endpoint = f"{API_URL}/chat/{st.session_state.session_id}"
        payload = {"query": prompt}
    else:
        if not patient:
            st.error("Please select a patient in the sidebar first.")
            st.stop()

        endpoint = f"{API_URL}/patient_chat/{st.session_state.session_id}"
        # Clean patient data for JSON serialization
        clean_patient = {
            k: (v.isoformat() if isinstance(v, datetime.datetime)
                else (str(v) if k == "_id" else v))
            for k, v in patient.items()
        }
        payload = {"query": prompt, "patient": clean_patient}

    # Generate Assistant Response
    with st.chat_message("assistant"):
        try:
            # Container for streaming text
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner("Searching guidelines..."):
                response = requests.post(endpoint, json=payload, stream=True, timeout=30)
                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code}")
                    st.stop()

                # Stream response into the placeholder
                full_response = st.write_stream(decode_stream(response))

            # Fetch Citations (Post-response)
            citations = []
            citations_resp = requests.post(f"{API_URL}/citations", json={"query": prompt}, timeout=30)
            if citations_resp.status_code == 200:
                citations = citations_resp.json().get("citations", [])
                if citations:
                    with st.expander("Citations"):
                        for c in citations:
                            st.markdown(f"- [{c['id']}]({c['url']}) — {c['section']} ({c['type']})")

            # Save Assistant Message to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "citations": citations
            })

        except (TimeoutError, TypeError, ValueError) as e:
            st.error(f"An unexpected error occurred: {e}")
