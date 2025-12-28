'''Configuration module for project paths and logging setup.'''

# Imports
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
#logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

DIPSTICK_IMAGES_DIR = PROCESSED_DATA_DIR / "dipstick_imgs"
RAW_TABLES_DIR = RAW_DATA_DIR / "tabular_data"
TABLES_DIR = PROCESSED_DATA_DIR / "tabular_data"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# Weaviate
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST") # type:ignore
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT")) # type:ignore
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT")) # type:ignore


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
