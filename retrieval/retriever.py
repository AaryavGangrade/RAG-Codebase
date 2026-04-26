import faiss
import numpy as np
from typing import List, Dict
from utils.helpers import save_json, load_json
from config import Config
from embeddings.embedder import Embedder

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.dimension = 1536
        self.index = None
        self.metadata = []

    def build_index(self, chunks: List[Dict]):
        if not chunks:
            return
            
        embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        self.dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        self.metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
        
        faiss.write_index(self.index, Config.VECTOR_DB_PATH)
        save_json(self.metadata, Config.METADATA_PATH)

    def load_index(self):
        self.index = faiss.read_index(Config.VECTOR_DB_PATH)
        self.metadata = load_json(Config.METADATA_PATH)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.load_index()
            
        query_emb = np.array(self.embedder.get_embeddings([query]), dtype=np.float32)
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(distances[0][i])
                results.append(item)
        return results
