"""
NICE NG203 RAG agent: retrieve guidelines + answer clinician queries
based on patient data and model predictions.
"""

# Imports
from src.rag.utils.agent_tools import retrieve_for_agent, build_patient_summary
from src.rag.utils.agent_response import synthesise_answer, ChatSession
from src.rag.utils.parsers import extract_citations

history = []


def answer_query(query, session: ChatSession, mode="free", patient=None):
    """Answer query."""
    session.add_user_message(query)

    # Patient-focused mode
    patient_summary = None
    if mode == "patient" and patient is not None:
        patient_summary = build_patient_summary(patient)
    if mode == "patient":
        chat_history = None
    else:
        chat_history = session.messages


    # Retrieve guideline context
    retrieved = retrieve_for_agent(query)

    # Build streaming generator
    answer_gen = synthesise_answer(
        query,
        retrieved,
        chat_history=chat_history,
        patient_summary=patient_summary
    )

    # Consume the stream
    answer = "".join(token for token in answer_gen)

    # Save assistant message
    session.add_assistant_message(answer)

    # Extract citations
    citations = extract_citations(retrieved)

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "history": session.messages,
        "summary": session.summary,
        "patient_summary": patient_summary
    }



Q1 = "How should CKD patients be monitored for progression?"
Q2 = "What's the rationale behind the renal ultrasound?"
Q3 = "How should I assess the risk of adverse outcomes in adults?"

new_session = ChatSession()
print(answer_query(Q1, new_session))
