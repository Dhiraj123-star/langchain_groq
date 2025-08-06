"""
Simple Langchain + Groq API Conversational Assistant
A basic chatbot that uses Groq's fast inference with Langchain's conversation management
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import streamlit as st
from typing import List
import time

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b" 

def setup_groq_chat(api_key=None, model_name=None, temperature=0.7):
    """Initialize the Groq chat model"""
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
        
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables or provided as parameter")
    
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name or DEFAULT_MODEL_NAME,
        temperature=temperature,
        max_tokens=1024
    )

def create_conversation_chain(llm):
    """Create a conversation chain with memory"""
    
    # Custom prompt template
    template = """
    You are a helpful and friendly AI assistant. You provide clear, informative responses 
    and maintain context throughout the conversation.
    
    Current conversation:
    {history}
    
    Human: {input}
    Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Memory to store conversation history
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="history"
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    return conversation

def main():
    st.set_page_config(
        page_title="Langchain + Groq Assistant",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Langchain + Groq Conversational Assistant")
    st.markdown("*Powered by Groq's lightning-fast inference and Langchain's conversation management*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input - check environment first
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
        
        # Model selection
        model_options = [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b", 
            "moonshotai/kimi-k2-instruct",
            "gemma2-9b-it"
        ]
        
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=0,
            help="Choose the Groq model for inference"
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in responses"
        )
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation = None
            st.rerun()
        
        # Environment setup instructions
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            **For .env file setup:**
            1. Create a `.env` file in your project directory
            2. Add: `GROQ_API_KEY=your_actual_api_key_here`
            3. Restart the application
            
            **Get API Key:**
            - Visit [Groq Console](https://console.groq.com/)
            - Create account and generate API key
            """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize conversation chain when needed
    if "conversation" not in st.session_state or st.session_state.get("last_model") != selected_model:
        try:
            # Initialize Groq chat
            llm = setup_groq_chat(api_key, selected_model, temperature)
            
            # Create conversation chain
            st.session_state.conversation = create_conversation_chain(llm)
            st.session_state.last_model = selected_model
            
            if not st.session_state.messages:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Hello! I'm your AI assistant powered by {selected_model} on Groq. How can I help you today?"
                })
                
        except Exception as e:
            st.error(f"❌ Failed to initialize: {str(e)}")
            st.info("💡 Please check your API key and internet connection")
            st.stop()
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                try:
                    # Get response from conversation chain
                    response = st.session_state.conversation.predict(input=prompt)
                    
                    inference_time = time.time() - start_time
                    
                    st.markdown(response)
                    
                    # Show inference time
                    st.caption(f"⚡ Response generated in {inference_time:.2f}s")
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.info("💡 Please check your API key and try again")

if __name__ == "__main__":
    # For running without Streamlit (command line version)
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--cli":
        print("🚀 Langchain + Groq CLI Assistant")
        print("Type 'quit' to exit\n")
        
        try:
            # Initialize
            llm = setup_groq_chat()
            conversation = create_conversation_chain(llm)
            
            print(f"✅ Successfully connected to Groq API")
            print(f"📡 Using model: {DEFAULT_MODEL_NAME}")
            
        except Exception as e:
            print(f"❌ Error initializing: {str(e)}")
            print("💡 Please check your GROQ_API_KEY in .env file")
            exit(1)
        
        while True:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            try:
                start_time = time.time()
                response = conversation.predict(input=user_input)
                inference_time = time.time() - start_time
                
                print(f"\n🤖 Assistant: {response}")
                print(f"⚡ ({inference_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Please check your connection and API key")
    else:
        # Run Streamlit app
        main()


