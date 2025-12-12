"""
Ingestion script: Loads chunked JSONs into Weaviate.
"""
import weaviate
import weaviate.classes.config as wc
from loguru import logger
from src.rag.etl.utils import load_json
from src.config import INTERIM_DATA_DIR

# Connect to Weaviate (Adjust URL/Port as needed)
weaviate_client = weaviate.connect_to_local() 

def create_unified_collection(client):
    """
    Creates a single 'NiceGuideline' collection that can store 
    Recommendations, Rationales, AND Tables.
    """
    # 1. Reset for fresh ingestion (Be careful in prod!)
    if client.collections.exists("NiceGuideline"):
        client.collections.delete("NiceGuideline")
        logger.info("Deleted existing 'NiceGuideline' collection.")

    # 2. Create the unified schema
    client.collections.create(
        name="NiceGuideline",
        description="NICE NG203 guidelines: Recommendations, Rationales, and Tables.",

        # MiniLM Configuration
        vectorizer_config=wc.Configure.Vectorizer.text2vec_transformers(),
        generative_config=wc.Configure.Generative.openai(),

        properties=[
            # --- The Searchable Content ---
            wc.Property(
                name="full_context",
                data_type=wc.DataType.TEXT,
                vectorize_property_name=False, # Pure content embedding
                tokenization=wc.Tokenization.WORD
            ),

            # --- Metadata (Filters & Display) ---
            wc.Property(name="text", data_type=wc.DataType.TEXT,
                        skip_vectorization=True),
            wc.Property(name="section", data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD, skip_vectorization=True),
            wc.Property(name="subsection", data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.WORD, skip_vectorization=True),

            # --- Type Discriminator ---
            wc.Property(
                name="type", 
                data_type=wc.DataType.TEXT, 
                tokenization=wc.Tokenization.FIELD, # Allows filter: where type="table"
                skip_vectorization=True
            ),

            # --- Flexible ID Storage (All optional, populate based on type) ---
            wc.Property(name="rec_id", data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD, skip_vectorization=True),
            wc.Property(name="rationale_id", data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD, skip_vectorization=True),
            wc.Property(name="table_id", data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD, skip_vectorization=True),
            wc.Property(name="guideline_id", data_type=wc.DataType.TEXT,
                        skip_vectorization=True),
            wc.Property(name="source_url", data_type=wc.DataType.TEXT,
                        skip_vectorization=True)
        ]
    )
    logger.success("Created unified 'NiceGuideline' collection.")

def ingest_data(client, data_list):
    """Batched ingestion of data into Weaviate."""
    collection = client.collections.get("NiceGuideline")

    with collection.batch.dynamic() as batch:
        for item in data_list:
            # Weaviate expects a dict of properties matching the schema
            # Your JSON keys mostly match, but we ensure safety here
            properties = {
                "full_context": item.get("full_context"),
                "text": item.get("text"),
                "section": item.get("section"),
                "subsection": item.get("subsection"),
                "type": item.get("type"), # 'recommendation', 'rationale', or 'table'
                "guideline_id": item.get("guideline_id"),
                "source_url": item.get("source_url"),

                # Conditional IDs
                "rec_id": item.get("rec_id"),
                "rationale_id": item.get("rationale_id"),
                "table_id": item.get("table_id"),
            }

            # Remove None values to avoid errors
            properties = {k: v for k, v in properties.items() if v is not None}

            batch.add_object(properties=properties)

    if len(collection.batch.failed_objects) > 0:
        logger.error(f"Failed to import {len(collection.batch.failed_objects)} objects.")
        for fail in collection.batch.failed_objects[:3]:
            logger.error(f"Error: {fail.message}")
    else:
        logger.success(f"Successfully ingested {len(data_list)} objects.")

if __name__ == "__main__":
    # Define Paths
    CHUNKED_GUIDELINES_DIR = INTERIM_DATA_DIR / "nice_guidelines"
    chunked_rec_path = CHUNKED_GUIDELINES_DIR / "nice_ng203_recommendations_chunked.json"
    chunked_rat_path = CHUNKED_GUIDELINES_DIR / "nice_ng203_rationale-and-impact_chunked.json"

    # Setup DB
    create_unified_collection(weaviate_client)

    # Ingest Recommendations
    recs_data = load_json(chunked_rec_path)
    ingest_data(weaviate_client, recs_data)

    # Ingest Rationales
    rats_data = load_json(chunked_rat_path)
    ingest_data(weaviate_client, rats_data)

    weaviate_client.close()
