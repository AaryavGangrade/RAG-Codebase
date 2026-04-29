from typing import List, Dict
import networkx as nx
import pickle
from config import Config

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

    def build_graph(self, chunks: List[Dict]) -> nx.DiGraph:
        G = nx.DiGraph()
        
        # Add nodes
        for i, chunk in enumerate(chunks):
            # We use the index as the node ID to perfectly match the FAISS indices, or we can use the name
            # Let's use the chunk name/file as identifier
            node_id = chunk['name']
            G.add_node(node_id, index=i, type=chunk['type'], file=chunk['file'])
            
        # Add edges based on calls
        chunk_names = {c['name'] for c in chunks}
        for chunk in chunks:
            for call in chunk.get('calls', []):
                if call in chunk_names and call != chunk['name']:
                    G.add_edge(chunk['name'], call, type='calls')
                    
        # Save graph
        with open(Config.GRAPH_PATH, 'wb') as f:
            pickle.dump(G, f)
            
        return G
