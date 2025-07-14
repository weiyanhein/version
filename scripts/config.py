import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve configuration values
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Basic validation to ensure all required variables are set
if not all([MONGO_URI, DB_NAME, COLLECTION_NAME, CHROMA_PERSIST_DIR, OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL]):
    raise ValueError("One or more environment variables are not set. Please check your .env file.")

