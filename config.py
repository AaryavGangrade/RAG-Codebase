import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Using the latest top-tier embedding and LLM models from Google GenAI
    EMBEDDING_MODEL = "models/gemini-embedding-2"
    LLM_MODEL = "models/gemini-2.5-flash"
    VECTOR_DB_PATH = "faiss_index.bin"
    BM25_PATH = "bm25_index.pkl"
    GRAPH_PATH = "code_graph.pkl"
    METADATA_PATH = "metadata.json"
