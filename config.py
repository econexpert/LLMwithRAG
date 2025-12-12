import os

MONGO_URI = os.getenv("MONGO_URI")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
