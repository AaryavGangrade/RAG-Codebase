import google.generativeai as genai
from config import Config
from typing import List, Dict

class LLMResponder:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model_name = Config.LLM_MODEL
        self.system_prompt = """You are a senior software engineer and codebase intelligence assistant. 
You are given bits of parsed code (chunks) from a codebase and a user query.
Use the provided code context to answer the user's query precisely and accurately.
Always reference specific file names, classes, or functions in your explanation.
If the context does not contain the answer, say "I don't have enough context in the provided codebase to answer this."
"""
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt
        )

    def generate_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        context_str = "\n\n".join([f"--- CHUNK {i+1} ---\n{c['chunk_text']}" for i, c in enumerate(retrieved_chunks)])
        
        prompt = f"Context:\n{context_str}\n\nQuery:\n{query}"
        
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        return response.text
