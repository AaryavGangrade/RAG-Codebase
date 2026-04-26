# 🧠 Codebase Intelligence System using RAG

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-green.svg)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple.svg)

A production-ready **Retrieval-Augmented Generation (RAG)** system designed specifically for analyzing and understanding Python codebases. 

Unlike standard text-based RAGs, this tool is **AST-aware**. It intelligently parses `functions`, `classes`, and `docstrings` preserving their boundaries instead of blindly splitting raw text. It leverages the latest **Google Gemini 2.5** intelligence to answer complex developer queries with precise file and code references.

---

## ✨ Features

- **🧠 AST-Aware Parsing**: Uses Python's built-in `ast` module to accurately identify logical code components and extract relationships (imports, line numbers).
- **🔪 Smart Chunking**: Filters out trivial boilerplate (functions under 5 lines) to maintain a high-quality vector space.
- **⚡ High-Performance Vector Retrieval**: Uses Meta's `FAISS` library for ultra-fast L2 similarity search over embedded code chunks.
- **📈 Custom Reranking**: Applies a custom heuristic reranker based on keyword and metadata boosting to improve top-k accuracy.
- **🤖 Grounded AI Responses**: Seamlessly integrates with Google's Gemini generation pipeline to provide highly accurate, codebase-specific explanations.

---

## 🏗️ Architecture Stack

This repository is highly modularised for production environments:

- `ingest/`: Abstract Syntax Tree parsing and smart chunking logic.
- `embeddings/`: Interfaces with Google AI to vectorize code definitions.
- `retrieval/`: Manages the FAISS index database and scoring heuristics.
- `llm/`: Prompts and generation parameters for context-aware Q&A.

---

## 🚀 Quick Setup

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

## 💻 Usage Guide

### 1. Indexing the Codebase
Analyze and ingest all Python files in your target directory structure. This step generates a local `faiss_index.bin` database alongside chunk mapping metadata.
```bash
python main.py --index .
```

### 2. Querying the Codebase
Ask natural language questions to interface with the codebase architecture directly:
```bash
python main.py --query "Where is the retriever module implemented?"
python main.py --query "Explain how the embedder gets its embeddings?"
python main.py --query "Can you explain me the various subfolders created and what their uses are?"
```
