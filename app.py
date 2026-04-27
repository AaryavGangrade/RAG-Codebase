import streamlit as st
import os
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

# Move the attach folder to the middle of the screen using columns
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h4 style='text-align: center;'>📎 Load Codebase</h4>", unsafe_allow_html=True)
    
    # The Browse Button
    if st.button("Browse local folders...", use_container_width=True):
        # We spawn a tiny child process for Tkinter to completely avoid Streamlit's multi-threading crash
        import subprocess
        import sys
        
        picker_code = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
folder = filedialog.askdirectory(master=root)
root.destroy()
print(folder)
"""
        try:
            selected_dir = subprocess.check_output([sys.executable, "-c", picker_code]).decode("utf-8").strip()
        except Exception:
            selected_dir = ""
        
        if selected_dir:
            st.session_state.indexed_repo = selected_dir
            with st.spinner(f"Ingesting code from {selected_dir}..."):
                # Pipeline
                parser = CodebaseParser(selected_dir)
                parsed_items = parser.parse_codebase()
                
                chunker = CodeChunker(min_lines=5)
                chunks = chunker.chunk(parsed_items)
                
                embedder = Embedder()
                embedded_chunks = embedder.embed_chunks(chunks)
                
                retriever = Retriever()
                retriever.build_index(embedded_chunks)
                
            st.success(f"Indexed {len(embedded_chunks)} chunks from {os.path.basename(selected_dir)}!")

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
