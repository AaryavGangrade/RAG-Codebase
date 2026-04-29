import google.generativeai as genai
from config import Config
from typing import List, Dict
import json

class Reranker:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        if not results:
            return results
            
        chunks_str = ""
        for i, res in enumerate(results):
            code_snippet = res.get("code", "")[:500] # Truncate to save tokens
            chunks_str += f"--- CHUNK {i} ---\nName: {res.get('name')}\nCode:\n{code_snippet}\n\n"
            
        prompt = f"""Given the user query: "{query}"
        
Evaluate the relevance of the following code chunks to answering the query.
Assign a score from 0 to 10 for each chunk based on its usefulness.
Return ONLY a JSON array of numbers in the exact order of the chunks, e.g., [8, 2, 5]. Do not include any other text or markdown formatting.

{chunks_str}
"""
        try:
            response = self.model.generate_content(prompt, generation_config={"temperature": 0.1})
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
                
            scores = json.loads(text)
            
            if isinstance(scores, list) and len(scores) == len(results):
                for i, res in enumerate(results):
                    res["score"] = float(scores[i])
        except Exception as e:
            print(f"Neural reranking failed: {e}. Falling back to default scores.")
            
        return sorted(results, key=lambda x: x["score"], reverse=True)
