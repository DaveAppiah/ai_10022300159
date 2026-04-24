# app.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Streamlit UI for RAG Chat Assistant
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag_pipeline import RAGPipeline

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Academic City RAG Assistant",
    page_icon="🎓",
    layout="wide",
)

# ── Title & Description ───────────────────────────────────────────────────────
st.title("🎓 Academic City RAG Assistant")
st.markdown("""
This is a Retrieval-Augmented Generation (RAG) chat assistant for Academic City University.
Ask questions about the Ghana Election Results and Ghana 2025 Budget Statement.
""")

# ── Initialize Session State ──────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.is_ready = False
    st.session_state.build_started = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Auto-build on first load ───────────────────────────────────────────────────
if not st.session_state.is_ready and not st.session_state.build_started:
    st.session_state.build_started = True
    with st.spinner("Building index on first load... This may take a minute."):
        try:
            pipeline = RAGPipeline()
            pipeline.build(chunking_strategy="fixed")
            st.session_state.pipeline = pipeline
            st.session_state.is_ready = True
            st.success("✅ Index built automatically!")
        except Exception as e:
            st.error(f"❌ Error building index: {str(e)}")

# ── Build Pipeline Button (manual rebuild) ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Build status indicator
    if st.session_state.is_ready:
        st.success("✅ Index Ready")
    else:
        st.warning("⏳ Building...")
    
    if st.button("🔨 Rebuild Index", use_container_width=True):
        with st.spinner("Building index... This may take a minute."):
            try:
                pipeline = RAGPipeline()
                pipeline.build(chunking_strategy="fixed")
                st.session_state.pipeline = pipeline
                st.session_state.is_ready = True
                st.success("✅ Index built successfully!")
            except Exception as e:
                st.error(f"❌ Error building index: {str(e)}")
    
    st.divider()
    
    # Advanced Options
    st.subheader("Advanced Options")
    prompt_version = st.slider("Prompt Version", 1, 3, 2)
    top_k = st.slider("Top-K Retrieval", 1, 10, 5)
    use_expansion = st.checkbox("Query Expansion", value=True)
    pure_llm_mode = st.checkbox("Pure LLM Mode (No Retrieval)", value=False)

# ── Main Chat Interface ───────────────────────────────────────────────────────
if not st.session_state.is_ready:
    if st.session_state.build_started and not st.session_state.is_ready:
        st.warning("⏳ Index is building... Please wait until completion.")
    else:
        st.info("👋 Starting up... Building index on first load. Please wait.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_query = st.chat_input("Ask a question...")
    
    if user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Get response from pipeline
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.pipeline.query(
                    user_query,
                    top_k=top_k,
                    prompt_version=prompt_version,
                    use_expansion=use_expansion,
                    pure_llm_mode=pure_llm_mode,
                )
                
                if "error" in result:
                    response = f"❌ Error: {result['error']}"
                else:
                    response = result.get("answer", "No response generated.")
                
                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Show retrieval details and logs
                with st.expander("📋 Pipeline Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Chunks Retrieved", len(result.get("retrieved_chunks", [])))
                        st.metric("Low Confidence", "Yes" if result.get("low_confidence") else "No")
                    with col2:
                        st.metric("Pure LLM Mode", "On" if result.get("pure_llm_mode") else "Off")
                        st.metric("Prompt Version", result.get("prompt_version", "N/A"))
                    
                    st.subheader("Expanded Query")
                    st.text(result.get("expanded_query", user_query))
                    
                    if result.get("retrieved_chunks"):
                        st.subheader("Retrieved Chunks")
                        for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                            with st.expander(f"Chunk {i} (Score: {chunk.get('final_score', 0):.2f})"):
                                st.text(chunk.get("text", "")[:500] + "...")
                    
                    st.subheader("Full Pipeline Log")
                    st.text_area("Logs", result.get("pipeline_log", ""), height=200, disabled=True)
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
