"""Agent response functions"""

from openai import OpenAI, APIError
from loguru import logger
from src.config import OPENAI_API_KEY
from src.rag.utils.parsers import build_context

client = OpenAI(api_key=OPENAI_API_KEY)


class ChatSession:
    def __init__(self, max_messages=20, summary_threshold=12):
        self.messages = []          # full chat messages
        self.summary = None         # running summary of older turns
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        self._maybe_summarise()

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})
        self._maybe_summarise()

    def _maybe_summarise(self):
        """
        If the chat history gets too long, summarise the oldest part
        and keep only the most recent messages.
        """
        if len(self.messages) <= self.max_messages:
            return

        # Extract portion to summarise and build text block for LLM
        to_summarise = self.messages[:-self.summary_threshold]
        text_block = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_summarise
        )

        # Ask model to summarise older turns
        summary_prompt = (
            "Summarise the following conversation history so that the key "
            "intent, decisions, and context are preserved. Keep it concise.\n\n"
            f"{text_block}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarise conversation history."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.2,
        )

        new_summary = response.choices[0].message.content.strip() # type: ignore

        # Merge with existing summary if present
        if self.summary:
            self.summary = self.summary + "\n" + new_summary
        else:
            self.summary = new_summary

        # Keep only the most recent messages
        self.messages = self.messages[-self.summary_threshold:]

    def build_prompt_messages(self, system_prompt, user_prompt):
        """
        Build the final message list for the OpenAI call.
        Includes:
        - system prompt
        - summary (if exists)
        - recent messages
        - the new user prompt containing retrieved context
        """
        msgs = [{"role": "system", "content": system_prompt}]

        if self.summary:
            msgs.append({"role": "system", "content": f"Conversation summary:\n{self.summary}"})

        msgs.extend(self.messages)
        msgs.append({"role": "user", "content": user_prompt})

        return msgs



def synthesise_answer(query, retrieved, chat_history=None, patient_summary=None):
    """
    Generate a final clinical answer using OpenAI, based ONLY on the retrieved context.
    The retrieved dict should contain:
        - recommendations (markdown string)
        - rationales (markdown string)
        - tables (markdown string)
    """

    # If nothing was retrieved, return a fallback
    if not retrieved or all(not v for v in retrieved.values()):
        return (
            "I could not find any relevant information in the NICE NG203 guidelines "
            "to answer your question."
        )

    # System prompt
    system_prompt = (
        "You are a clinical assistant specialising in chronic kidney disease (CKD). "
        "Your job is to answer the user's question using ONLY the provided NICE NG203 "
        "guideline context. Do not use outside knowledge. "
        "Do not hallucinate. "
        "If the context does not contain enough information, say so explicitly.\n\n"
        "When answering:\n"
        "- Synthesise information from recommendations, rationales, and tables.\n"
        "- Provide a clear, concise clinical answer.\n"
        "- Cite [Recommendation {rec_id}](Source url) "
        "or [Table {table_id}](Source url) "
        "or [Rationale {rationale_id}](Source url) when relevant.\n"
        "- Include URLs when available.\n"
        "- If tables are provided, interpret them rather than repeating them verbatim.\n"
        "- If patient information is provided, incorporate it into your reasoning. "
        "Tailor the answer to the details provided.\n"
        "- Do NOT invent patient details.\n"
        "- Do NOT invent guideline content.\n"
    )

    # Prepare the context bundle
    context = build_context(retrieved, patient_summary=patient_summary)

    user_prompt = f"""
    Below is the context retrieved from the NICE NG203 guideline database.
    Use ONLY this information to answer the question.

    ---------------- CONTEXT START ----------------
    {context}
    ---------------- CONTEXT END ----------------

    Question: {query}

    Provide a clinically accurate answer with citations.
    """

    # Message list
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    # Call OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages, # type: ignore
            stream=True,
            temperature=0.2,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        return (
            "There was an error generating the clinical answer. "
            "Please try again or refine your question."
        )
