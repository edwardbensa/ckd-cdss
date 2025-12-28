"""FastAPI endpoints"""

# Imports
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from src.rag.utils.agent_response import ChatSession, synthesise_answer
from src.rag.utils.agent_tools import retrieve_for_agent, build_patient_summary
from src.rag.utils.parsers import extract_citations

app = FastAPI()

# In-memory session store
sessions = {}

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: Request):
    data = await request.json()
    query = data["query"]

    # Create session if new
    if session_id not in sessions:
        sessions[session_id] = ChatSession()

    session = sessions[session_id]

    # Add user message
    session.add_user_message(query)

    # Retrieve guideline context
    retrieved = retrieve_for_agent(query)

    # Build streaming generator
    answer_gen = synthesise_answer(
        query=query,
        retrieved=retrieved,
        chat_history=session.messages,
        patient_summary=None
    )

    # Stream + capture
    async def stream_and_capture():
        full_answer = ""
        for token in answer_gen:
            full_answer += token
            yield str(token)
        session.add_assistant_message(full_answer)

    return StreamingResponse(stream_and_capture(), media_type="text/plain")


@app.post("/patient_chat/{session_id}")
async def patient_chat(session_id: str, request: Request):
    data = await request.json()
    query = data["query"]
    patient = data["patient"]  # full patient doc from MongoDB

    # Create session if new
    if session_id not in sessions:
        sessions[session_id] = ChatSession()

    session = sessions[session_id]

    # Add user message
    session.add_user_message(query)

    # Build patient summary
    patient_summary = build_patient_summary(patient)

    # Retrieve guideline context
    retrieved = retrieve_for_agent(query)

    # Patient mode: ignore chat history
    answer_gen = synthesise_answer(
        query=query,
        retrieved=retrieved,
        chat_history=None,
        patient_summary=patient_summary
    )

    async def stream_and_capture():
        full_answer = ""
        for token in answer_gen:
            full_answer += token
            yield str(token)
        session.add_assistant_message(full_answer)

    return StreamingResponse(stream_and_capture(), media_type="text/plain")


@app.post("/citations")
async def citations(request: Request):
    data = await request.json()
    query = data["query"]

    retrieved = retrieve_for_agent(query)
    citations_ = extract_citations(retrieved)

    return JSONResponse({"citations": citations_})
