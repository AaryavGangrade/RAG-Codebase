from typing import List, Dict
import google.generativeai as genai
from config import Config

class Embedder:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = Config.EMBEDDING_MODEL

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # embed_content accepts a list of strings and returns a dictionary with 'embedding' key
        result = genai.embed_content(
            model=self.model,
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
        
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self.get_embeddings(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb
        return chunks
