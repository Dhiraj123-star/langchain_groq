"""
Main Streamlit Application for Langchain + Groq Conversational Assistant
Modern implementation without deprecation warnings
"""

import os
import sys
import time
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Import our custom chat handler
from chat_handler import ChatHandler, create_chat_handler

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_handler" not in st.session_state:
        st.session_state.chat_handler = None
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit_session"
    
    if "last_settings" not in st.session_state:
        st.session_state.last_settings = {}

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Langchain + Groq Assistant",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar() -> Dict[str, Any]:
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Management
        env_api_key = os.getenv("GROQ_API_KEY")
        
        if env_api_key:
            st.success("✅ API Key loaded from environment")
            api_key = env_api_key
            st.text_input(
                "Groq API Key", 
                value="***" + env_api_key[-4:] if len(env_api_key) > 4 else "***",
                disabled=True,
                help="API key loaded from .env file"
            )
        else:
            st.warning("⚠️ No API key found in environment")
            api_key = st.text_input(
                "Groq API Key", 
                value="",
                type="password",
                help="Enter your Groq API key from https://console.groq.com/ or add GROQ_API_KEY to your .env file"
            )
        
        if not api_key:
            st.error("🚫 Please provide a Groq API key to continue")
            st.stop()
        
        # Model Selection
        model_options = [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b", 
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            "gemma2-9b-it"
        ]
        
        selected_model = st.selectbox(
            "🤖 Model",
            model_options,
            index=0,
            help="Choose the Groq model for inference"
        )
        
        # Streaming Toggle
        enable_streaming = st.checkbox(
            "🌊 Enable Streaming",
            value=True,
            help="Stream responses in real-time for better user experience"
        )
        
        # Temperature Slider
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses (0.0 = focused, 1.0 = creative)"
        )
        
        # Action Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_conversation()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                reset_application()
        
        # Chat Statistics
        if st.session_state.messages:
            st.divider()
            st.subheader("📊 Chat Stats")
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            ai_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.metric("Messages", f"{user_msgs + ai_msgs}")
            st.metric("Exchanges", f"{min(user_msgs, ai_msgs)}")
        
        # Setup Instructions
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            **Environment Setup:**
            1. Create a `.env` file in your project directory
            2. Add: `GROQ_API_KEY=your_actual_api_key_here`
            3. Restart the application
            
            **Get API Key:**
            - Visit [Groq Console](https://console.groq.com/)
            - Create account and generate API key
            
            **Features:**
            - 🌊 Real-time streaming responses
            - 🧠 Conversation memory
            - 🚀 Lightning-fast inference
            - 🔧 Multiple model options
            """)
    
    return {
        "api_key": api_key,
        "model": selected_model,
        "streaming": enable_streaming,
        "temperature": temperature
    }

def initialize_chat_handler(config: Dict[str, Any]) -> ChatHandler:
    """Initialize or update chat handler based on configuration"""
    current_settings = {
        "model": config["model"],
        "temperature": config["temperature"],
        "streaming": config["streaming"]
    }
    
    # Check if we need to create new handler or update existing one
    if (st.session_state.chat_handler is None or 
        st.session_state.last_settings != current_settings):
        
        try:
            if st.session_state.chat_handler is None:
                # Create new handler
                st.session_state.chat_handler = create_chat_handler(
                    api_key=config["api_key"],
                    model_name=config["model"],
                    temperature=config["temperature"],
                    streaming=config["streaming"]
                )
                
                # Add welcome message
                if not st.session_state.messages:
                    welcome_msg = f"Hello! I'm your AI assistant powered by **{config['model']}** on Groq."
                    if config["streaming"]:
                        welcome_msg += " 🌊 Streaming is enabled for real-time responses!"
                    else:
                        welcome_msg += " How can I help you today?"
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": welcome_msg
                    })
            else:
                # Update existing handler
                st.session_state.chat_handler.update_settings(
                    model_name=config["model"],
                    temperature=config["temperature"],
                    streaming=config["streaming"]
                )
            
            st.session_state.last_settings = current_settings
            
        except Exception as e:
            st.error(f"❌ Failed to initialize chat handler: {str(e)}")
            st.info("💡 Please check your API key and internet connection")
            st.stop()
    
    return st.session_state.chat_handler

def display_chat_history():
    """Display the conversation history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(chat_handler: ChatHandler, config: Dict[str, Any]):
    """Handle user input and generate AI response"""
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            start_time = time.time()
            
            try:
                if config["streaming"]:
                    # Streaming response
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in chat_handler.stream_response(prompt, st.session_state.session_id):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")  # Cursor effect
                    
                    # Remove cursor and show final response
                    message_placeholder.markdown(full_response)
                    
                    inference_time = time.time() - start_time
                    st.caption(f"🌊 Response streamed in {inference_time:.2f}s")
                    
                    # Add to session state
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
                        
                        # Add to session state
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.info("💡 Please check your API key and try again")

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
    st.title("🚀 LangChain + Groq Conversational Assistant")
    st.markdown("*Powered by Groq's lightning-fast inference and modern LangChain implementation*")
    
    # Add status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", "🟢 Online", help="Application is running")
    with col2:
        if st.session_state.chat_handler:
            st.metric("Model", st.session_state.chat_handler.model_name, help="Current AI model")
    with col3:
        if st.session_state.chat_handler:
            stream_status = "🌊 Enabled" if st.session_state.chat_handler.streaming else "📝 Disabled"
            st.metric("Streaming", stream_status, help="Response streaming status")

def main():
    """Main application function"""
    # Setup
    setup_page_config()
    initialize_session_state()
    
    # Render UI
    render_header()
    
    # Get configuration from sidebar
    config = render_sidebar()
    
    # Initialize chat handler
    chat_handler = initialize_chat_handler(config)
    
    # Display conversation
    display_chat_history()
    
    # Handle user input
    handle_user_input(chat_handler, config)

def run_cli():
    """CLI version of the application"""
    print("🚀 LangChain + Groq CLI Assistant")
    print("=" * 50)
    print("Commands: 'quit/exit/bye' to exit, 'clear' to clear history")
    print("-" * 50)
    
    try:
        # Initialize chat handler
        handler = create_chat_handler(streaming=False)
        
        print(f"✅ Successfully connected to Groq API")
        print(f"📡 Using model: {handler.model_name}")
        
        session_id = "cli_session"
        
        while True:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if user_input.lower() == 'clear':
                handler.clear_history(session_id)
                print("🗑️ Conversation history cleared!")
                continue
                
            if not user_input.strip():
                continue
                
            try:
                start_time = time.time()
                response = handler.get_response(user_input, session_id)
                inference_time = time.time() - start_time
                
                print(f"\n🤖 Assistant: {response}")
                print(f"⚡ ({inference_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Please check your connection and API key")
                
    except Exception as e:
        print(f"❌ Error initializing: {str(e)}")
        print("💡 Please check your GROQ_API_KEY in .env file")
        exit(1)

if __name__ == "__main__":
    # Check for CLI mode
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        # Run Streamlit app
        main()