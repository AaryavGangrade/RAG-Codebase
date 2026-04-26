from typing import List, Dict

class CodeChunker:
    def __init__(self, min_lines: int = 5):
        self.min_lines = min_lines

    def chunk(self, parsed_items: List[Dict]) -> List[Dict]:
        chunks = []
        for item in parsed_items:
            line_count = len(item["code"].splitlines())
            
            if item["type"] == "function" and line_count < self.min_lines:
                continue
                
            chunk_text = f"File: {item['file']}\nType: {item['type']}\nName: {item['name']}\nDocstring: {item['docstring']}\nCode:\n{item['code']}"
            item["chunk_text"] = chunk_text
            chunks.append(item)
            
        return chunks
