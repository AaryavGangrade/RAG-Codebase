import argparse
import os
from ingest.parser import CodebaseParser
from ingest.chunker import CodeChunker
from embeddings.embedder import Embedder
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.prompt import LLMResponder
from config import Config

def index_repo(repo_path: str):
    print(f"Indexing repository at: {repo_path}")
    
    parser = CodebaseParser(repo_path)
    parsed_items = parser.parse_codebase()
    print(f"Extracted {len(parsed_items)} logical components (functions, classes, modules).")

    chunker = CodeChunker(min_lines=5)
    chunks = chunker.chunk(parsed_items)
    print(f"Formed {len(chunks)} chunks after filtering trivial components.")

    embedder = Embedder()
    print("Generating embeddings (this may take a while)...")
    embedded_chunks = embedder.embed_chunks(chunks)
    
    retriever = Retriever()
    retriever.build_index(embedded_chunks)
    print(f"Successfully built and saved FAISS index with {len(embedded_chunks)} items.")

def query_repo(query: str):
    print(f"Query: {query}")
    
    if not os.path.exists(Config.VECTOR_DB_PATH):
        print("Error: Index not found. Please index a repository first using --index <path>")
        return

    retriever = Retriever()
    raw_results = retriever.search(query, top_k=5)
    
    reranker = Reranker()
    best_results = reranker.rerank(query, raw_results)
    
    print(f"\nRetrieved {len(best_results)} relevant chunks.")
    for res in best_results:
        print(f" - [{res['type']}] {res['name']} (File: {res.get('file', 'N/A')})")

    llm = LLMResponder()
    print("\nGenerating answer...")
    answer = llm.generate_response(query, best_results)
    
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(answer)
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codebase Intelligence RAG System")
    parser.add_argument("--index", type=str, help="Path to the python repository to index.")
    parser.add_argument("--query", type=str, help="Question to ask the codebase.")
    
    args = parser.parse_args()
    
    if args.index:
        index_repo(args.index)
    elif args.query:
        query_repo(args.query)
    else:
        print("No args provided. Running DEMO mode on current directory.")
        if not Config.GEMINI_API_KEY:
            print("WARNING: GEMINI_API_KEY environment variable is not set. Demo will fail when embedding.")
        else:
            index_repo(".")
            query_repo("Explain the Retriever class and how it searches?")
