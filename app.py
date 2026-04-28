import streamlit as st
import os
import tempfile
import zipfile
import subprocess
import shutil
from config import Config
from ingest.parser import CodebaseParser
from ingest.chunker import CodeChunker
from embeddings.embedder import Embedder
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from llm.prompt import LLMResponder

st.set_page_config(page_title="RAG Codebase Intelligence", layout="centered", page_icon="🧠")

st.title("🧠 AST-Aware Codebase Intelligence")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None

# Helper function to process codebase
def process_codebase(source_dir, repo_name):
    with st.spinner(f"Ingesting code from {repo_name}..."):
        # Pipeline
        parser = CodebaseParser(source_dir)
        parsed_items = parser.parse_codebase()
        
        if not parsed_items:
            st.error(f"Failed to find any Python files or parseable code in {repo_name}. Check if the repository contains valid .py files.")
            return

        chunker = CodeChunker(min_lines=5)
        chunks = chunker.chunk(parsed_items)
        
        if not chunks:
            st.error(f"Found {len(parsed_items)} items in {repo_name}, but none were long enough to chunk (e.g. functions < 5 lines).")
            return

        embedder = Embedder()
        embedded_chunks = embedder.embed_chunks(chunks)
        
        retriever = Retriever()
        retriever.build_index(embedded_chunks)
        
    st.success(f"Indexed {len(embedded_chunks)} chunks from {repo_name}!")

# UI for loading codebase
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>📎 Load Codebase</h4>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Upload ZIP", "🌐 GitHub URL"])

with tab1:
    uploaded_zip = st.file_uploader("Upload Codebase ZIP", type=["zip"])
    if uploaded_zip and st.button("Process ZIP", use_container_width=True):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        
        with st.spinner("Extracting ZIP..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        # Determine the root directory (handle cases where ZIP contains a single root folder)
        extracted_items = os.listdir(temp_dir)
        extracted_items.remove("uploaded.zip")
        
        source_dir = temp_dir
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
            source_dir = os.path.join(temp_dir, extracted_items[0])

        st.session_state.indexed_repo = uploaded_zip.name
        process_codebase(source_dir, uploaded_zip.name)

with tab2:
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")
    if repo_url and st.button("Clone & Process", use_container_width=True):
        if not repo_url.startswith("http"):
            repo_url = "https://" + repo_url
        
        temp_dir = tempfile.mkdtemp()
        with st.spinner(f"Cloning {repo_url}..."):
            try:
                subprocess.run(["git", "clone", repo_url, temp_dir], check=True, capture_output=True, text=True)
                repo_name = repo_url.split("/")[-1].replace(".git", "")
                st.session_state.indexed_repo = repo_name
                process_codebase(temp_dir, repo_name)
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to clone repository. Git error: {e.stderr}")
            except FileNotFoundError:
                st.error("Git is not installed or not found in the system PATH. Please install Git to use this feature.")
            except Exception as e:
                st.error(f"An unexpected error occurred during processing: {str(e)}")

# Show what is currently active
if st.session_state.indexed_repo:
    st.markdown(f"<p style='text-align: center; color: lightgreen;'>✓ Active Codebase: <b>{st.session_state.indexed_repo}</b></p>", unsafe_allow_html=True)

st.divider()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query input area
if prompt := st.chat_input("Ask a question about the indexed codebase..."):
    
    # 🚨 Block query if no folder is selected yet
    if not st.session_state.indexed_repo or not os.path.exists(Config.VECTOR_DB_PATH):
        st.error("⚠️ You must select and index a folder before asking questions!")
    else:
        # Append and display user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching the latent space and generating insights..."):
                retriever = Retriever()
                raw_results = retriever.search(prompt, top_k=5)
                
                reranker = Reranker()
                best_results = reranker.rerank(prompt, raw_results)
                
                llm = LLMResponder()
                response_text = llm.generate_response(prompt, best_results)
            
            st.markdown(response_text)
            
            # Show chunks retrieved in a neat collapsible dropdown
            with st.expander("View Retrieved Context & Files"):
                for res in best_results:
                    st.markdown(f"📄 **[{res['type'].upper()}] {res['name']}** - `{res.get('file', 'N/A')}`")
                    
        # Append assistant role
        st.session_state.messages.append({"role": "assistant", "content": response_text})

