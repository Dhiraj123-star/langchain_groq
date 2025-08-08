"""
Main Streamlit Application for Langchain + Groq Conversational Assistant with RAG
Modern implementation with Document Q&A capabilities
"""

import os
import sys
import time
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Import our custom handlers
from chat_handler import ChatHandler, create_chat_handler
from rag_handler import RAGHandler, create_rag_handler

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_handler" not in st.session_state:
        st.session_state.chat_handler = None
    
    if "rag_handler" not in st.session_state:
        st.session_state.rag_handler = None
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit_session"
    
    if "last_settings" not in st.session_state:
        st.session_state.last_settings = {}
    
    if "mode" not in st.session_state:
        st.session_state.mode = "chat"  # "chat" or "rag"
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="LangChain + Groq AI Assistant",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode Selection
        st.subheader("🎯 Mode Selection")
        mode = st.radio(
            "Choose mode:",
            ["💬 Chat", "📄 Document Q&A"],
            index=0 if st.session_state.mode == "chat" else 1,
            help="Switch between regular chat and document-based Q&A"
        )
        
        # Update session state based on selection
        if mode == "💬 Chat":
            st.session_state.mode = "chat"
        else:
            st.session_state.mode = "rag"
        
        st.divider()
        
        # API Key Management
        st.subheader("🔑 API Keys")
        
        # Groq API Key
        env_groq_key = os.getenv("GROQ_API_KEY")
        if env_groq_key:
            st.success("✅ Groq API Key loaded from environment")
            groq_api_key = env_groq_key
            st.text_input(
                "Groq API Key", 
                value="***" + env_groq_key[-4:] if len(env_groq_key) > 4 else "***",
                disabled=True,
                help="API key loaded from .env file"
            )
        else:
            st.warning("⚠️ No Groq API key found in environment")
            groq_api_key = st.text_input(
                "Groq API Key", 
                value="",
                type="password",
                help="Enter your Groq API key from https://console.groq.com/"
            )
        
        # OpenAI API Key (for embeddings)
        env_openai_key = os.getenv("OPENAI_API_KEY")
        if env_openai_key:
            st.success("✅ OpenAI API Key loaded from environment")
            openai_api_key = env_openai_key
            st.text_input(
                "OpenAI API Key", 
                value="***" + env_openai_key[-4:] if len(env_openai_key) > 4 else "***",
                disabled=True,
                help="API key loaded from .env file (for embeddings)"
            )
        else:
            st.warning("⚠️ No OpenAI API key found in environment")
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                value="",
                type="password",
                help="Enter your OpenAI API key (required for embeddings in RAG mode)"
            )
        
        # Check required API keys
        if not groq_api_key:
            st.error("🚫 Please provide a Groq API key to continue")
            st.stop()
        
        if st.session_state.mode == "rag" and not openai_api_key:
            st.error("🚫 OpenAI API key is required for Document Q&A mode")
            st.info("💡 OpenAI embeddings provide better quality for document search")
            st.stop()
        
        # Model Selection
        st.subheader("🤖 Model Configuration")
        
        model_options = [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b", 
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            "gemma2-9b-it"
        ]
        
        selected_model = st.selectbox(
            "Generation Model (Groq)",
            model_options,
            index=0,
            help="Choose the Groq model for text generation"
        )
        
        # Embedding model selection (only for RAG mode)
        if st.session_state.mode == "rag":
            embedding_options = [
                "text-embedding-3-small",
                "text-embedding-3-large", 
                "text-embedding-ada-002"
            ]
            
            selected_embedding = st.selectbox(
                "Embedding Model (OpenAI)",
                embedding_options,
                index=0,
                help="Choose OpenAI embedding model for document search"
            )
        else:
            selected_embedding = "text-embedding-3-small"
        
        # Chat-specific settings
        if st.session_state.mode == "chat":
            # Streaming Toggle
            enable_streaming = st.checkbox(
                "🌊 Enable Streaming",
                value=True,
                help="Stream responses in real-time for better user experience"
            )
        else:
            enable_streaming = False
        
        # Temperature Slider
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses (0.0 = focused, 1.0 = creative)"
        )
        
        # Mode-specific sections
        if st.session_state.mode == "rag":
            render_rag_sidebar()
        
        # Action Buttons
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_conversation()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                reset_application()
        
        # Statistics
        render_sidebar_stats()
        
        # Setup Instructions
        render_setup_instructions()
    
    return {
        "groq_api_key": groq_api_key,
        "openai_api_key": openai_api_key,
        "model": selected_model,
        "embedding_model": selected_embedding,
        "streaming": enable_streaming,
        "temperature": temperature
    }

def render_rag_sidebar():
    """Render RAG-specific sidebar content"""
    st.subheader("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt', 'docx', 'csv'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, DOCX, CSV"
    )
    
    if uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        if uploaded_files:
            process_uploaded_files(uploaded_files)
    
    # Document management
    if st.session_state.rag_handler:
        stats = st.session_state.rag_handler.get_document_stats()
        if stats["total_chunks"] > 0:
            st.success(f"📚 {stats['total_chunks']} chunks from {len(stats['sources'])} documents")
            
            with st.expander("📋 Loaded Documents"):
                for source in stats['sources']:
                    st.write(f"• {source}")
            
            if st.button("🗑️ Clear Documents", use_container_width=True):
                st.session_state.rag_handler.clear_documents()
                st.session_state.uploaded_files = []
                st.rerun()

def render_sidebar_stats():
    """Render sidebar statistics"""
    if st.session_state.messages:
        st.divider()
        st.subheader("📊 Chat Stats")
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        ai_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.metric("Messages", f"{user_msgs + ai_msgs}")
        st.metric("Exchanges", f"{min(user_msgs, ai_msgs)}")

def render_setup_instructions():
    """Render setup instructions"""
    with st.expander("📋 Setup Instructions"):
        st.markdown("""
        **Environment Setup:**
        1. Create a `.env` file in your project directory
        2. Add: `GROQ_API_KEY=your_groq_api_key_here`
        3. Add: `OPENAI_API_KEY=your_openai_api_key_here`
        4. Install dependencies: `pip install langchain-openai`
        5. Restart the application
        
        **Get API Keys:**
        - **Groq**: Visit [Groq Console](https://console.groq.com/)
        - **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/)
        
        **RAG Features:**
        - 📄 Upload PDF, TXT, DOCX, CSV files
        - 🔍 Ask questions about document content
        - 🎯 High-quality OpenAI embeddings for better search
        - 🧠 Multiple embedding model options
        - 📚 Source citations with document references
        
        **Chat Features:**
        - 🌊 Real-time streaming responses
        - 🧠 Conversation memory
        - 🚀 Lightning-fast inference
        - 🔧 Multiple model options
        """)

def process_uploaded_files(uploaded_files):
    """Process uploaded files for RAG"""
    if not st.session_state.rag_handler:
        return
    
    with st.spinner("📄 Processing documents..."):
        results = st.session_state.rag_handler.load_documents(uploaded_files)
        
        if results["success"]:
            st.success(f"✅ Processed {len(results['success'])} documents ({results['total_chunks']} chunks)")
            
            # Show successful files
            for file in results["success"]:
                st.write(f"✅ {file}")
        
        if results["errors"]:
            st.error("❌ Some files failed to process:")
            for error in results["errors"]:
                st.write(f"❌ {error}")

def initialize_handlers(config: Dict[str, Any]):
    """Initialize chat and RAG handlers"""
    current_settings = {
        "model": config["model"],
        "embedding_model": config.get("embedding_model", "text-embedding-3-small"),
        "temperature": config["temperature"],
        "streaming": config["streaming"] if st.session_state.mode == "chat" else False
    }
    
    try:
        # Initialize chat handler
        if (st.session_state.chat_handler is None or 
            st.session_state.last_settings.get("model") != current_settings["model"] or
            st.session_state.last_settings.get("temperature") != current_settings["temperature"] or
            st.session_state.last_settings.get("streaming") != current_settings["streaming"]):
            
            if st.session_state.chat_handler is None:
                st.session_state.chat_handler = create_chat_handler(
                    api_key=config["groq_api_key"],  # Use groq_api_key here
                    model_name=config["model"],
                    temperature=config["temperature"],
                    streaming=config["streaming"] if st.session_state.mode == "chat" else False
                )
            else:
                st.session_state.chat_handler.update_settings(
                    model_name=config["model"],
                    temperature=config["temperature"],
                    streaming=config["streaming"] if st.session_state.mode == "chat" else False
                )
        
        # Initialize RAG handler
        if st.session_state.rag_handler is None:
            st.session_state.rag_handler = create_rag_handler(
                groq_api_key=config["groq_api_key"],
                openai_api_key=config["openai_api_key"],
                model_name=config["model"],
                temperature=config["temperature"],
                embedding_model=config["embedding_model"]
            )
        else:
            # Update RAG handler settings
            st.session_state.rag_handler.update_model_settings(
                model_name=config["model"],
                temperature=config["temperature"],
                embedding_model=config["embedding_model"]
            )
        
        st.session_state.last_settings = current_settings
        
    except Exception as e:
        st.error(f"❌ Failed to initialize handlers: {str(e)}")
        st.info("💡 Please check your API key and internet connection")
        st.stop()

def display_chat_history():
    """Display the conversation history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("sources"):
                # Display RAG answer with sources
                st.markdown(message["content"])
                
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}: {source['source']}**")
                        st.write(source["content"])
                        if source["page"] != "N/A":
                            st.write(f"*Page: {source['page']}*")
                        st.divider()
            else:
                st.markdown(message["content"])

def handle_user_input(config: Dict[str, Any]):
    """Handle user input and generate response"""
    # Different placeholders for different modes
    placeholder_text = (
        "Ask me anything..." if st.session_state.mode == "chat" 
        else "Ask a question about your documents..."
    )
    
    if prompt := st.chat_input(placeholder_text):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response based on mode
        with st.chat_message("assistant"):
            start_time = time.time()
            
            try:
                if st.session_state.mode == "chat":
                    handle_chat_response(prompt, config, start_time)
                else:
                    handle_rag_response(prompt, start_time)
                    
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.info("💡 Please check your settings and try again")

def handle_chat_response(prompt: str, config: Dict[str, Any], start_time: float):
    """Handle chat mode response"""
    chat_handler = st.session_state.chat_handler
    
    if config["streaming"]:
        # Streaming response
        message_placeholder = st.empty()
        full_response = ""
        
        for chunk in chat_handler.stream_response(prompt, st.session_state.session_id):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        inference_time = time.time() - start_time
        st.caption(f"🌊 Response streamed in {inference_time:.2f}s")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response
        })
    else:
        # Regular response
        with st.spinner("🤔 Thinking..."):
            response = chat_handler.get_response(prompt, st.session_state.session_id)
            
            inference_time = time.time() - start_time
            st.markdown(response)
            st.caption(f"⚡ Response generated in {inference_time:.2f}s")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })

def handle_rag_response(prompt: str, start_time: float):
    """Handle RAG mode response"""
    rag_handler = st.session_state.rag_handler
    
    with st.spinner("🔍 Searching documents..."):
        result = rag_handler.ask_question(prompt)
        
        inference_time = time.time() - start_time
        
        if result["error"]:
            st.error(result["answer"])
        else:
            st.markdown(result["answer"])
            
            # Display sources
            if result["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(result["sources"], 1):
                        st.write(f"**Source {i}: {source['source']}**")
                        st.write(source["content"])
                        if source["page"] != "N/A":
                            st.write(f"*Page: {source['page']}*")
                        st.divider()
            
            st.caption(f"🔍 Answer retrieved in {inference_time:.2f}s")
            
            # Add to session state with sources
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["answer"],
                "sources": result["sources"]
            })

def clear_conversation():
    """Clear the conversation history"""
    st.session_state.messages = []
    if st.session_state.chat_handler:
        st.session_state.chat_handler.clear_history(st.session_state.session_id)
    st.rerun()

def reset_application():
    """Reset the entire application state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def render_header():
    """Render the main header"""
    st.title("🚀 LangChain + Groq AI Assistant")
    
    # Mode indicator
    mode_text = "💬 Chat Mode" if st.session_state.mode == "chat" else "📄 Document Q&A Mode"
    st.markdown(f"*{mode_text} - Powered by Groq's lightning-fast inference*")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", "🟢 Online", help="Application is running")
    with col2:
        if st.session_state.chat_handler:
            st.metric("Model", st.session_state.chat_handler.model_name, help="Current AI model")
    with col3:
        if st.session_state.mode == "chat" and st.session_state.chat_handler:
            stream_status = "🌊 Enabled" if st.session_state.chat_handler.streaming else "📝 Disabled"
            st.metric("Streaming", stream_status, help="Response streaming status")
        elif st.session_state.mode == "rag" and st.session_state.rag_handler:
            stats = st.session_state.rag_handler.get_document_stats()
            st.metric("Documents", f"📚 {stats['total_chunks']} chunks", help="Loaded document chunks")
    with col4:
        st.metric("Mode", mode_text.split()[0], help="Current operation mode")

def main():
    """Main application function"""
    # Setup
    setup_page_config()
    initialize_session_state()
    
    # Render UI
    render_header()
    
    # Get configuration from sidebar
    config = render_sidebar()
    
    # Initialize handlers
    initialize_handlers(config)
    
    # Show welcome message based on mode
    if not st.session_state.messages:
        if st.session_state.mode == "chat":
            welcome_msg = f"Hello! I'm your AI assistant powered by **{config['model']}** on Groq. How can I help you today?"
        else:
            welcome_msg = f"Hello! I'm ready to answer questions about your documents using **{config['model']}**. Please upload documents in the sidebar to get started!"
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Display conversation
    display_chat_history()
    
    # Handle user input
    handle_user_input(config)

if __name__ == "__main__":
    main()