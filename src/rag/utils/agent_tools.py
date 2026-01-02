"""RAG functions"""

# Imports
import os
import json
import cohere
import weaviate
import numpy as np
from openai import OpenAI
from loguru import logger
from sentence_transformers import SentenceTransformer
from src.config import COHERE_API_KEY, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT
from src.rag.etl.load import COLLECTION_NAME
from src.rag.utils.parsers import trim_chunks

# Config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
co = cohere.Client(COHERE_API_KEY)
client = OpenAI()


def build_patient_summary(patient: dict):
    """
    Turn a patient document into a compact summary string.
    """
    gender_map = {
        True: "male",
        False: "female",
        None: None
    }

    summary_dict = {
        "Patient ID": patient.get("patient_id"),
        "Age": patient.get("age"),
        "Sex": gender_map[patient.get("male")],
        "eGFR": patient.get("egfr"),
        "ACR": patient.get("acr"),
        "GFR stage": patient.get("gfr_stage"),
        "ACR stage": patient.get("acr_stage"),
        "Predicted Diagnosis": patient.get("predicted_diagnosis")
    }

    clean_data = {k: v for k, v in summary_dict.items() if v not in (None, "", [])}

    # Build markdown
    lines = ["### Patient Summary", ""]
    for key, value in clean_data.items():
        lines.append(f"- **{key}:** {value}")

    return "\n".join(lines)


def cohere_rerank(query_text, candidates, top_k=10):
    """Rerank vector embeddings with Cohere."""
    results = co.rerank(
        query=query_text,
        documents=candidates,
        top_n=top_k,
        model="rerank-english-v3.0"
    )
    return results

def search_by_type(query_text, obj_type, top_k=10, hybrid=True, cutoff=0.05):
    """
    Search NICE guideline chunks by type ('recommendation', 'rationale', 'table')
    using hybrid search (BM25 + vector) and Cohere reranking.
    """

    # Embed query for vector search
    query_vector = embedder.encode(query_text).astype(np.float32)

    # Connect to Weaviate
    weaviate_client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST, #type: ignore
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST, #type: ignore
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False
    )

    try:
        collection = weaviate_client.collections.get(COLLECTION_NAME)

        # Choose hybrid or pure vector search
        if hybrid:
            results = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                limit=top_k * 3,
                alpha=0.5,
                filters=(
                    weaviate.classes.query.Filter.by_property("type").equal(obj_type)
                ),
                return_properties=[
                    "text",
                    "full_text",
                    "full_context",
                    "section",
                    "subsection",
                    "type",
                    "guideline_id",
                    "source_url",
                    "rec_id",
                    "rationale_id",
                    "table_id"
                ],
                return_metadata=["score"]
            )
        else:
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=(
                    weaviate.classes.query.Filter.by_property("type").equal(obj_type)
                ),
                return_properties=[
                    "text",
                    "full_text",
                    "full_context",
                    "section",
                    "subsection",
                    "type",
                    "guideline_id",
                    "source_url",
                    "rec_id",
                    "rationale_id",
                    "table_id"
                ],
                return_metadata=["distance"]
            )

        objects = results.objects

        # Cohere reranking
        texts = [obj.properties.get("text", "") for obj in objects]
        co_results = co.rerank(
            query=query_text,
            documents=texts, # type: ignore
            top_n=top_k,
            model="rerank-english-v3.0"
        )

        id_map = {
            "recommendation": "rec_id",
            "table": "table_id",
            "rationale": "rationale_id"
        }
        obj_id = id_map[obj_type]

        # Reorder objects according to Cohere ranking
        ranked = []
        for r in co_results.results:
            ranked.append({
                obj_id: objects[r.index].properties.get(obj_id),
                "section": objects[r.index].properties.get("section"),
                "subsection": objects[r.index].properties.get("subsection"),
                "text": objects[r.index].properties.get("text"),
                "full_text": objects[r.index].properties.get("full_text"),
                "source_url": objects[r.index].properties.get("source_url"),
                "type": objects[r.index].properties.get("type"),
                "score": r.relevance_score,
                "distance": getattr(objects[r.index].metadata, "distance", None),
            })

        ranked = [r for r in ranked if r["score"] >= cutoff]
        logger.info(f"Retrieved {len(ranked)} NG203 chunks for query: {query_text!r}")

        return ranked
    finally:
        weaviate_client.close()



def search_recommendations(query, top_k=10):
    """
    Retrieve top-k recommendations most relevant to the query.
    """
    recs = search_by_type(query, obj_type="recommendation", top_k=top_k)
    recs = trim_chunks(recs, obj_type="recommendation")

    # Build markdown
    md_lines = []
    md_lines.append(f"## Top {len(recs)} Recommendations for: **{query}**\n")

    for i, rec in enumerate(recs, start=1):
        rec_id = rec.get("rec_id")
        section = rec.get("section") or "N/A"
        subsection = rec.get("subsection") or "N/A"
        url = rec.get("source_url") or ""
        score = rec.get("score")
        text = rec.get("full_text")

        md_lines.append(f"### {i}. Recommendation {rec_id} — {section} / {subsection}")
        if url:
            md_lines.append(f"[Source link]({url})")
        md_lines.append(f"**Relevance score:** {score:.4f}")
        md_lines.append("")
        md_lines.append(text.strip())
        md_lines.append("\n---\n")

    markdown = "\n".join(md_lines)

    return {
        "items": recs,
        "markdown": markdown
        }


def search_rationales(query, top_k=10):
    """
    Retrieve top-k rationale sections most relevant to the query.
    """
    rats = search_by_type(query, obj_type="rationale", top_k=top_k)

    md_lines = []
    md_lines.append(f"## Top {len(rats)} Rationales for: **{query}**\n")

    for i, rat in enumerate(rats, start=1):
        rat_id = rat.get("rationale_id")
        section = rat.get("section") or "N/A"
        subsection = rat.get("subsection") or "N/A"
        url = rat.get("source_url") or ""
        score = rat.get("score")
        text = rat.get("text") or ""

        md_lines.append(f"### {i}. Rationale {rat_id} — {section} / {subsection}")
        if url:
            md_lines.append(f"[Source link]({url})")
        md_lines.append(f"**Relevance score:** {score:.4f}")
        md_lines.append("")
        md_lines.append(text.strip())
        md_lines.append("\n---\n")

    markdown = "\n".join(md_lines)

    return {
        "items": rats,
        "markdown": markdown
        }


def search_tables(query, top_k=10):
    """
    Retrieve top-k tables most relevant to the query.
    """
    tables = search_by_type(query, obj_type="table", top_k=top_k)
    tables = trim_chunks(tables, obj_type="table")
    logger.info(f"Found {len(tables)} relevant tables.")
    if len(tables) == 0:
        return None

    md_lines = []
    md_lines.append(f"## Top {len(tables)} Tables for: **{query}**\n")

    for i, tbl in enumerate(tables, start=1):
        table_id = tbl.get("table_id")
        section = tbl.get("section") or "N/A"
        subsection = tbl.get("subsection") or "N/A"
        url = tbl.get("source_url") or ""
        score = tbl.get("score")
        text = tbl.get("full_text")

        md_lines.append(f"### {i}. Table {table_id} — {section} / {subsection}")
        if url:
            md_lines.append(f"[Source link]({url})")
        md_lines.append(f"**Relevance score:** {score:.4f}")
        md_lines.append("")
        md_lines.append(text.strip())
        md_lines.append("\n---\n")

    markdown = "\n".join(md_lines)

    return {
        "items": tables,
        "markdown": markdown
        }


with open("src/rag/utils/tools.json", "r", encoding="utf-8") as f:
    tools = json.load(f)

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_recommendations",
            "description": "Retrieve clinical recommendations from NICE NG203 guidelines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The clinical search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rationales",
            "description": "Retrieve the 'why' and evidence behind recommendations. Use if the user asks for reasons or explanations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The clinical search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tables",
            "description": "Retrieve staging, risk, or monitoring tables. Use for numbers, stages, or frequency of checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The clinical search query."}
                },
                "required": ["query"],
            },
        },
    }
]

def retrieve_for_agent(query: str):
    """
    Uses OpenAI Tool Calling to determine which search functions to run,
    then executes them and returns the results.
    """

    # Ask the model which tools to use
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a clinical coordinator."
            "Determine which guideline tools are needed to answer the user's query."},
            {"role": "user", "content": query}
        ],
        tools=tools, # type: ignore
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    results = {}

    # If the model didn't call any tools, default to recommendations
    if not tool_calls:
        results["recommendations"] = search_recommendations(query)
        return results

    # Map tool names to actual Python functions
    available_functions = {
        "search_recommendations": search_recommendations,
        "search_rationales": search_rationales,
        "search_tables": search_tables,
    }

    # Execute the calls
    for tool_call in tool_calls:
        function_name = tool_call.function.name # type: ignore
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments) # type: ignore
        results[function_name.replace("search_", "")] = function_to_call(
            query=function_args.get("query")
        )

    return results

def generate_context(query: str, retrieved: dict):
    """
    Combine recommendations, rationales, and tables into a single answer.
    """
    md = []
    md.append(f"## Answer to: **{query}**\n")

    # Recommendations
    if "recommendations" in retrieved:
        md.append("### Recommendations")
        md.append(retrieved["recommendations"])

    # Rationales
    if "rationales" in retrieved:
        md.append("### Rationale and Explanation")
        md.append(retrieved["rationales"])

    # Tables
    if "tables" in retrieved:
        md.append("### Relevant Tables")
        md.append(retrieved["tables"])

    return "\n".join(md)
