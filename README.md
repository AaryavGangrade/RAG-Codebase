# Codebase Intelligence System using Agentic Hybrid-Graph RAG

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-green.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph-blue.svg)
![RAG](https://img.shields.io/badge/Architecture-Agentic%20RAG-purple.svg)

A production-ready **Agentic Retrieval-Augmented Generation (RAG)** system designed specifically for deep analysis and understanding of Python codebases. 

Unlike standard text-based RAGs, this tool is **AST-aware and Graph-augmented**. It parses your code, maps out execution dependencies, and allows a Gemini-powered Agent to autonomously search, read, and traverse your codebase to answer complex developer queries.

---

## Features

- **Agentic RAG Loop**: The LLM isn't just a passive reader. It acts as an autonomous agent equipped with a `search_codebase` tool, allowing it to perform multi-step retrieval to trace complex logic before answering.
- **Hybrid Retrieval (Dense + Sparse)**: Combines Meta's `FAISS` for semantic vector search with `BM25` for exact-keyword matching. Results are merged using Reciprocal Rank Fusion (RRF) for unparalleled accuracy.
- **Code Graph Expansion**: Extracts `ast.Call` nodes to build a `networkx` dependency graph. When a function is retrieved, the system automatically pulls its 1-hop callers and callees into the context window.
- **Neural Reranking Pipeline**: Uses Gemini as an LLM-judge to rapidly score and rerank the top retrieved chunks (0-10) based on deep semantic relevance to the query.
- **AST-Aware Parsing & Smart Chunking**: Uses Python's built-in `ast` module to accurately identify logical code components (filtering out trivial functions under 5 lines) to maintain a high-quality vector space.

---

## Architecture Stack

This repository is highly modularised for production environments:

- `ingest/`: Abstract Syntax Tree parsing, dependency extraction, and `networkx` graph building.
- `embeddings/`: Interfaces with Google AI to vectorize code definitions.
- `retrieval/`: Manages the FAISS semantic index, BM25 sparse index, and neural reranking heuristics.
- `llm/`: Contains the `CodebaseAgent` that orchestrates tool calling and multi-turn reasoning.
- `app.py`: Streamlit frontend with direct ZIP upload and GitHub URL cloning capabilities.

---

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your API Key**:
   Create a `.env` file in the root directory with your [Google AI Studio](https://aistudio.google.com/) key:
   ```env
   GEMINI_API_KEY="your-api-key-here"
   ```

---

## Usage Guide

### 1. Launch the Web Interface (Recommended)
This system comes with an interactive Graphical UI where you can supply paths, upload ZIP files, or paste GitHub URLs and effortlessly chat with the AI.
```bash
streamlit run app.py
```

### 2. Standard CLI Usage (Alternative)

**Indexing:**
```bash
python main.py --index .
```

**Querying:**
```bash
python main.py --query "Where is the retriever module implemented?"
```
