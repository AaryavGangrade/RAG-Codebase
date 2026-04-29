import faiss
import numpy as np
import pickle
import networkx as nx
from typing import List, Dict
from utils.helpers import save_json, load_json
from config import Config
from embeddings.embedder import Embedder
from rank_bm25 import BM25Okapi

class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.dimension = 1536
        self.index = None
        self.bm25 = None
        self.graph = None
        self.metadata = []

    def build_index(self, chunks: List[Dict]):
        if not chunks:
            return
            
        embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        self.dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        self.metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]
        
        # Build BM25 index
        tokenized_corpus = [c["chunk_text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        faiss.write_index(self.index, Config.VECTOR_DB_PATH)
        save_json(self.metadata, Config.METADATA_PATH)
        with open(Config.BM25_PATH, 'wb') as f:
            pickle.dump(self.bm25, f)

    def load_index(self):
        self.index = faiss.read_index(Config.VECTOR_DB_PATH)
        self.metadata = load_json(Config.METADATA_PATH)
        with open(Config.BM25_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
        try:
            with open(Config.GRAPH_PATH, 'rb') as f:
                self.graph = pickle.load(f)
        except FileNotFoundError:
            self.graph = nx.DiGraph()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.load_index()
            
        # FAISS Retrieval
        query_emb = np.array(self.embedder.get_embeddings([query]), dtype=np.float32)
        distances, faiss_indices = self.index.search(query_emb, top_k * 2)
        
        # BM25 Retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, idx in enumerate(faiss_indices[0]):
            if idx != -1:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
                
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
            
        # Get top-k combined
        best_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        results = []
        for idx in best_indices:
            if idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = rrf_scores[idx]
                results.append(item)
                
        # Graph Expansion: add 1-hop neighbors
        expanded_results = []
        for res in results:
            expanded_results.append(res)
            node_name = res["name"]
            if self.graph and node_name in self.graph:
                neighbors = list(self.graph.neighbors(node_name))
                for neighbor in neighbors:
                    # Find the neighbor in metadata
                    neighbor_item = next((m for m in self.metadata if m["name"] == neighbor), None)
                    if neighbor_item and neighbor_item not in expanded_results:
                        neighbor_item = neighbor_item.copy()
                        neighbor_item["score"] = res["score"] * 0.5 # lower score for expanded context
                        expanded_results.append(neighbor_item)
                        
        return expanded_results
