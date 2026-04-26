# Codebase Intelligence System using RAG

A production-quality Retrieval-Augmented Generation (RAG) system for Python codebases. This tool uses AST to parse functions, classes, and docstrings intelligently, creates embeddings using OpenAI, indexes them with FAISS for semantic search, and grounds LLM responses answering developer queries about the repository.

## Architecture

1. **Ingestion (`ingest/`)**: Uses Python's built-in `ast` module to accurately identify and extract boundaries of variables, classes, imports, and functions. Drops trivial functions (under 5 lines).
2. **Embedding (`embeddings/`)**: Uses OpenAI's `text-embedding-ada-002` to vectorise the logical chunks.
3. **Retrieval (`retrieval/`)**: FAISS is used for high-performance L2 distance similarity searching. A basic reranker is included to adjust scores.
4. **LLM (`llm/`)**: Grounds requests in GPT-4, utilizing structured context templates to get accurate answers and references.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your environment variables:
   Create a `.env` file in the root directory (or simply export it) with your OpenAI API Key:
   ```
   OPENAI_API_KEY="your-api-key-here"
   ```

## Usage

### 1. Indexing a Codebase
Point the CLI to the root of a Python repository. It will find all `.py` files, parse them, extract chunks, configure embeddings, and create a FAISS index.
```bash
python main.py --index path/to/python/repo
```

### 2. Querying the Codebase
Ask natural language questions about the indexed repository.
```bash
python main.py --query "Where is the authentication logic implemented?"
python main.py --query "Explain how the embedder gets its embeddings?"
python main.py --query "Suggest refactoring improvements for main.py"
```

## Advanced Features Implemented
- Intelligent chunking that understands function/class boundaries instead of raw text splitting.
- Metadata filtering & extraction (line numbers, imports, docstrings mapping).
- Reranking abstraction to enhance search metrics.
