"""Chunking and embedding functions for NICE NG203 guidelines."""

# Imports
import os
import re
import copy
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag.etl.utils import save_json, load_json
from src.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR


GUIDELINES_INT_DIR = INTERIM_DATA_DIR / "nice_guidelines"
os.makedirs(GUIDELINES_INT_DIR, exist_ok=True)
GUIDELINES_DIR = PROCESSED_DATA_DIR / "nice_guidelines"

rec_json_path = GUIDELINES_DIR / "nice_ng203_recommendations.json"
rat_json_path = GUIDELINES_DIR / "nice_ng203_rationale-and-impact.json"

def make_splitter(chunk_size=500):
    """
    Creates a splitter optimized for MiniLM (approx 256 tokens).
    Since 1 token ~= 4 chars, 600-700 chars is a safe conservative limit.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(0.2 * chunk_size),
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def chunk_guidelines(data):
    """
    Iterates through the extracted JSON list and splits only entries that are too long.
    """
    char_limit = 600
    noise_pattern = re.compile(r'^(\[Return to recommendations\]|\#{1,6}\s.*|\[.*\]\(.*\))$',
                               re.IGNORECASE)

    splitter = make_splitter(chunk_size=char_limit)
    chunked_dataset = []

    for entry in data:
        text = entry.get('text', '')

        # If text is short enough, keep as is
        if len(text) < char_limit:
            chunked_dataset.append(entry)
            continue

        # If text is too long, split
        chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(chunks):
            # Clean chunk text
            clean_text = chunk_text.strip()
            if len(clean_text) < 20:
                continue
            if noise_pattern.match(clean_text):
                continue

            new_entry = copy.deepcopy(entry)

            # Update the text
            new_entry['text'] = chunk_text

            # Rebuild full_context for embedding, adding (Part X) to distinguish vectors
            base_context = f"{entry['section']} > {entry.get('subsection', 'General')}"
            new_entry['full_context'] = f"{base_context} (Part {i+1}): {chunk_text}"

            # Update id
            for id_key in ['rec_id', 'rationale_id']:
                if id_key in new_entry:
                    new_entry[id_key] = f"{new_entry[id_key]}_part{i+1}"

            chunked_dataset.append(new_entry)

    return chunked_dataset

# Execution
if __name__ == "__main__":
    # Load extracted JSON files
    recommendations = load_json(rec_json_path)
    rationales = load_json(rat_json_path)

    # Chunk guidelines
    chunked_recommendations = chunk_guidelines(recommendations)
    chunked_rationales = chunk_guidelines(rationales)

    # Save chunked JSON files
    chunked_rec_path = GUIDELINES_INT_DIR / "nice_ng203_recommendations_chunked.json"
    chunked_rat_path = GUIDELINES_INT_DIR / "nice_ng203_rationale-and-impact_chunked.json"

    save_json(chunked_rec_path, chunked_recommendations)
    save_json(chunked_rat_path, chunked_rationales)

    logger.info(f"Saved chunked recommendations to {chunked_rec_path}")
    logger.info(f"Saved chunked rationales to {chunked_rat_path}")
