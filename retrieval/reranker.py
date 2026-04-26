from typing import List, Dict

class Reranker:
    def __init__(self):
        pass

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        query_terms = query.lower().split()
        for res in results:
            boost = 0.0
            name = res.get("name", "").lower()
            if any(term in name for term in query_terms):
                boost -= 0.1

            res["score"] += boost
            
        return sorted(results, key=lambda x: x["score"])
