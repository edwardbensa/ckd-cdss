"""
Indexing script supporting BM25
"""
import weaviate
import weaviate.classes.config as wc
from loguru import logger
from src.rag.etl.load import COLLECTION_NAME

# Connect to Weaviate
weaviate_client = weaviate.connect_to_local()

def rebuild_indexes():
    collection = weaviate_client.collections.get(COLLECTION_NAME)

    logger.info("Rebuilding BM25 and inverted indexes…")
    collection.config.update(
        inverted_index_config=wc.Configure.inverted_index(
            cleanup_interval_seconds=60
        )
    )
    logger.success("Index rebuild triggered.")
