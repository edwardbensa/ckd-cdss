"""
Ingestion script: Loads chunked JSONs into Weaviate.
"""
import weaviate
import weaviate.classes.config as wc
from loguru import logger
from src.rag.etl.utils import load_json
from src.config import INTERIM_DATA_DIR, WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT

COLLECTION_NAME = "NiceGuidelines"

def create_collection(client):
    """
    Creates a single collection that can store 
    Recommendations, Rationales, and Tables.
    """
    # Check if collection exists
    if client.collections.exists(COLLECTION_NAME):
        logger.info(f"{COLLECTION_NAME} collection already exists. Skipping creation.")
        return False

    # Create unified schema
    client.collections.create(
        name=COLLECTION_NAME,
        description="NICE NG203 guidelines: Recommendations, Rationales, and Tables.",

        # MiniLM config
        vectorizer_config=wc.Configure.Vectorizer.text2vec_transformers(),
        generative_config=wc.Configure.Generative.openai(),

        properties=[
            wc.Property(
                name="full_context",
                data_type=wc.DataType.TEXT,
                vectorize_property_name=False,
                tokenization=wc.Tokenization.WORD
            ),

            # Metadata
            wc.Property(name="text", data_type=wc.DataType.TEXT,
                        skip_vectorization=True),
            wc.Property(name="full_text", data_type=wc.DataType.TEXT,
                        skip_vectorization=True),
            wc.Property(name="section", data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.FIELD, skip_vectorization=True),
            wc.Property(name="subsection", data_type=wc.DataType.TEXT,
                        tokenization=wc.Tokenization.WORD, skip_vectorization=True),

            # Type discriminator
            wc.Property(
                name="type", 
                data_type=wc.DataType.TEXT, 
                tokenization=wc.Tokenization.FIELD,
                skip_vectorization=True
            ),

            # Flexible ID Storage
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
    logger.success(f"Created unified {COLLECTION_NAME} collection.")
    return True

def ingest_data(client, data_list):
    """Batched ingestion of data into Weaviate."""
    collection = client.collections.get(COLLECTION_NAME)

    with collection.batch.dynamic() as batch:
        for item in data_list:
            properties = {
                "full_context": item.get("full_context"),
                "full_text": item.get("full_text"),
                "text": item.get("text"),
                "section": item.get("section"),
                "subsection": item.get("subsection"),
                "type": item.get("type"),
                "guideline_id": item.get("guideline_id"),
                "source_url": item.get("source_url"),

                # Conditional IDs
                "rec_id": item.get("rec_id"),
                "rationale_id": item.get("rationale_id"),
                "table_id": item.get("table_id"),
            }

            properties = {k: v for k, v in properties.items() if v is not None}

            batch.add_object(properties=properties)

    if len(collection.batch.failed_objects) > 0:
        logger.error(f"Failed to import {len(collection.batch.failed_objects)} objects.")
        for fail in collection.batch.failed_objects[:3]:
            logger.error(f"Error: {fail.message}")
    else:
        logger.success(f"Successfully ingested {len(data_list)} objects.")

def check_data_exists(client):
    """Check if data already exists in the collection."""
    if not client.collections.exists(COLLECTION_NAME):
        return False
    
    collection = client.collections.get(COLLECTION_NAME)
    result = collection.aggregate.over_all(total_count=True)
    count = result.total_count
    
    logger.info(f"Found {count} existing objects in {COLLECTION_NAME}")
    return count > 0

if __name__ == "__main__":
    # Connect to Weaviate - use environment variables in Docker
    weaviate_client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST, #type: ignore
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST, #type: ignore
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=False
    )

    try:
        # Check if data already exists
        if check_data_exists(weaviate_client):
            logger.info("Data already exists in Weaviate. Skipping ingestion.")
            weaviate_client.close()
            exit(0)

        # Define paths
        CHUNKED_GUIDELINES_DIR = INTERIM_DATA_DIR / "nice_guidelines"
        chunked_rec_path = CHUNKED_GUIDELINES_DIR / "nice_ng203_recommendations_chunked.json"
        chunked_rat_path = CHUNKED_GUIDELINES_DIR / "nice_ng203_rationale-and-impact_chunked.json"

        # Setup DB
        created = create_collection(weaviate_client)
        
        if created:
            # Ingest recommendations
            logger.info(f"Loading recommendations from {chunked_rec_path}")
            recs_data = load_json(chunked_rec_path)
            ingest_data(weaviate_client, recs_data)

            # Ingest rationales
            logger.info(f"Loading rationales from {chunked_rat_path}")
            rats_data = load_json(chunked_rat_path)
            ingest_data(weaviate_client, rats_data)
            
            logger.success("Data ingestion complete!")
        else:
            logger.info("Collection already exists. Skipping data ingestion.")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise
    finally:
        weaviate_client.close()
