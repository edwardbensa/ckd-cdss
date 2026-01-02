"""Parsing utility functions."""

# Imports
import json
from loguru import logger


def extract_base_id(rec_id: str):
    """Extract chunked id base id."""
    return rec_id.split("_part")[0]

def get_unique_base_ids(chunks, obj_id):
    """Generate list of unique base ids from chunked rec ids."""
    ids = []
    for chunk in chunks:
        idx = chunk.get(obj_id)
        if not idx:
            continue
        base_id = extract_base_id(idx)
        ids.append(base_id)
    return set(ids)


def trim_chunks(recs, obj_type="recommendation"):
    """
    Remove duplicate chunked results by collapsing all _partX entries
    into a single representative entry (the highest scoring chunk).
    """
    id_map = {
        "recommendation": "rec_id",
        "table": "table_id",
        "rationale": "rationale_id"
    }
    obj_id = id_map[obj_type]

    # Separate chunked vs unchunked
    chunked = [r for r in recs if "_part" in r.get(obj_id, "")]
    full = [r for r in recs if "_part" not in r.get(obj_id, "")]

    if not chunked:
        return recs

    merged = []
    unique_base_ids = get_unique_base_ids(chunked, obj_id)

    for base in unique_base_ids:
        logger.info(f"Processing base {obj_id}: {base}")

        # Pick highest‑scoring chunk for this base ID
        best = max(
            (r for r in chunked if extract_base_id(r[obj_id]) == base),
            key=lambda x: x["score"]
        )

        # Replace chunked ID with base ID (remove _partX)
        best_clean = best.copy()
        best_clean[obj_id] = base

        merged.append(best_clean)

    # Combine unchunked + merged chunked
    combined = full + merged

    # Sort by score descending
    combined.sort(key=lambda r: r.get("score", 0), reverse=True)

    return combined


def extract_citations(retrieved):
    """Extract citations from retrieved context."""
    citations = []

    for _, bundle in retrieved.items():
        if not bundle or "items" not in bundle:
            continue

        for item in bundle["items"]:
            citations.append({
                "id": item.get("rec_id") or item.get("rationale_id") or item.get("table_id"),
                "type": item.get("type"),
                "section": item.get("section"),
                "subsection": item.get("subsection"),
                "url": item.get("source_url"),
            })

    # dedupe
    unique = {(c["id"], c["type"]): c for c in citations}
    return list(unique.values())


def build_context(retrieved: dict, patient_summary=None):
    """Build markdown context as JSON."""
    context = {}

    if patient_summary:
        context["patient_summary"] = patient_summary
        
    for key, value in retrieved.items():
        # Check if value is a dict and has the markdown key
        if value and isinstance(value, dict) and "markdown" in value:
            context[key] = value["markdown"]

    return json.dumps(context, indent=2)
