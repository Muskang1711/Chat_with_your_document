import streamlit as st
import requests
import json
import time
from typing import Dict, Any
import os

# Page configuration
st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
    }
    
    .context-card {
        background-color: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .metric-card {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .answer-box {
        background-color: #e6fffa;
        border: 2px solid #38b2ac;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #fff5f5;
        border: 2px solid #fc8181;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
    return None


def ask_question(question: str, max_chunks: int = 3) -> Dict[str, Any]:
    """Send question to API"""
    try:
        payload = {
            "question": question,
            "max_chunks": max_chunks,
            "include_metadata": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API error: {response.status_code}",
                "detail": response.text
            }
    
    except requests.exceptions.Timeout:
        return {"error": "Request timeout. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def reload_document():
    """Reload document in API"""
    try:
        response = requests.post(f"{API_BASE_URL}/reload", timeout=30)
        return response.status_code == 200
    except:
        return False


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Chat</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about your document using AI")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.warning(f"Please ensure the API is running at {API_BASE_URL}")
            st.code("python main.py", language="bash")
            st.stop()
        
        # Stats
        st.markdown("### 📊 System Statistics")
        stats = get_stats()
        
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
            with col2:
                st.metric("Embedding Dim", stats.get("model_info", {}).get("embedding_dimension", 0))
            
            # Model info
            with st.expander("🤖 Model Information"):
                st.json(stats.get("model_info", {}))
            
            # Vector store info
            with st.expander("💾 Vector Store"):
                st.json(stats.get("vector_store", {}))
        
        # Settings
        st.markdown("### 🎛️ Query Settings")
        max_chunks = st.slider(
            "Max Context Chunks",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant chunks to retrieve"
        )
        
        # Actions
        st.markdown("### 🔧 Actions")
        if st.button("🔄 Reload Document", use_container_width=True):
            with st.spinner("Reloading document..."):
                if reload_document():
                    st.success("Document reloaded successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to reload document")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Question")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="What is the Transformer architecture?",
            height=100,
            key="question_input"
        )
        
        # Submit button
        submit_btn = st.button(
            "🚀 Get Answer",
            type="primary",
            use_container_width=True
        )
        
        # Sample questions
        st.markdown("#### 💡 Sample Questions")
        sample_questions = [
            "What is the Transformer architecture?",
            "How does self-attention work?",
            "What are the key components of the model?",
            "What datasets were used for training?",
            "What are the performance results?"
        ]
        
        for sample in sample_questions:
            if st.button(f"→ {sample}", key=f"sample_{sample[:20]}"):
                st.session_state.question_input = sample
                st.experimental_rerun()
    
    with col2:
        st.markdown("### 📝 Answer & Context")
        
        # Process question
        if submit_btn and question:
            with st.spinner("🔍 Searching and generating answer..."):
                start_time = time.time()
                result = ask_question(question, max_chunks)
                processing_time = time.time() - start_time
            
            if "error" in result:
                st.markdown(
                    f'<div class="error-box">❌ {result["error"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                # Display answer
                st.markdown("#### 🎯 Answer")
                st.markdown(
                    f'<div class="answer-box">{result.get("answer", "No answer generated")}</div>',
                    unsafe_allow_html=True
                )
                
                # Display metrics
                if "metadata" in result:
                    st.markdown("#### 📊 Metrics")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric(
                            "Processing Time",
                            f"{processing_time:.2f}s"
                        )
                    with col_m2:
                        st.metric(
                            "Tokens Used",
                            result["metadata"].get("total_tokens", 0)
                        )
                    with col_m3:
                        st.metric(
                            "Contexts Used",
                            result["metadata"].get("num_contexts", 0)
                        )
                
                # Display context
                st.markdown("#### 📚 Retrieved Context")
                
                contexts = result.get("context", [])
                
                if contexts:
                    # Tabs for each context
                    tabs = st.tabs([f"Context {i+1}" for i in range(len(contexts))])
                    
                    for i, (tab, ctx) in enumerate(zip(tabs, contexts)):
                        with tab:
                            # Context metadata
                            col_c1, col_c2, col_c3 = st.columns(3)
                            with col_c1:
                                st.metric("Page", ctx.get("page", "N/A"))
                            with col_c2:
                                st.metric("Section", ctx.get("section", "N/A"))
                            with col_c3:
                                st.metric(
                                    "Relevance",
                                    f"{ctx.get('relevance_score', 0):.3f}"
                                )
                            
                            # Context content
                            st.markdown("**Content:**")
                            st.text_area(
                                f"Context {i+1} Content",
                                value=ctx.get("content", ""),
                                height=200,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                            
                            # Chunk ID
                            st.caption(f"Chunk ID: {ctx.get('chunk_id', 'N/A')}")
                else:
                    st.info("No relevant context found")
        
        elif not question and submit_btn:
            st.warning("Please enter a question")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray;">
            RAG Document Chat System | Built with FastAPI & Streamlit | 
            Using ChromaDB, HuggingFace Embeddings & Gemini LLM
        </div>
        """,
        unsafe_allow_html=True
    )


# Session state for chat history (optional enhancement)
def init_session_state():
    """Initialize session state variables"""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0


# Alternative Chat Interface (optional)
def chat_interface():
    """Alternative chat-style interface"""
    st.title("💬 Chat with Document")
    
    # Initialize session
    init_session_state()
    
    # Chat history
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
    
    # Input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(prompt, max_chunks=3)
                
                if "error" in response:
                    answer = f"Error: {response['error']}"
                else:
                    answer = response.get("answer", "No answer generated")
                
                st.write(answer)
                
                # Show context in expander
                if "context" in response and response["context"]:
                    with st.expander(f"📚 View {len(response['context'])} source contexts"):
                        for i, ctx in enumerate(response["context"]):
                            st.markdown(f"**Context {i+1}** (Page {ctx.get('page', 'N/A')})")
                            st.text(ctx.get("content", "")[:300] + "...")
                            st.caption(f"Relevance: {ctx.get('relevance_score', 0):.3f}")
        
        # Add assistant message
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.question_count += 1


if __name__ == "__main__":
    # Choose interface style
    interface_mode = st.sidebar.radio(
        "Interface Mode",
        ["Dashboard", "Chat"],
        index=0
    )
    
    if interface_mode == "Dashboard":
        main()
    else:
        chat_interface()