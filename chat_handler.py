"""
Chat Handler Module for Langchain + Groq API
Handles all backend logic for chat functionality with modern LangChain implementation
"""

import os
import time
from typing import Generator, List, Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_MODEL_NAME = "gemma2-9b-it"

class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """In-memory chat message history for session management"""
    
    def __init__(self):
        self.messages: List = []
    
    def add_message(self, message) -> None:
        """Add a message to the store"""
        self.messages.append(message)
    
    def clear(self) -> None:
        """Clear all messages"""
        self.messages = []

class ChatHandler:
    """Main chat handler class with modern LangChain implementation"""
    
    def __init__(self, api_key: str = None, model_name: str = None, 
                temperature: float = 0.7, streaming: bool = False):
        """
        Initialize the chat handler
        
        Args:
            api_key: Groq API key
            model_name: Model to use
            temperature: Response temperature
            streaming: Enable streaming responses
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.temperature = temperature
        self.streaming = streaming
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables or provided as parameter")
        
        # Initialize components
        self._setup_llm()
        self._setup_chat_chain()
        
    def _setup_llm(self):
        """Setup the Groq LLM"""
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            streaming=self.streaming
        )
    
    def _setup_chat_chain(self):
        """Setup the modern chat chain with message history"""
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly AI assistant. You provide clear, informative responses and maintain context throughout the conversation."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm
        
        # Store for session histories
        self.session_histories = {}
        
        # Create chain with message history
        self.chat_chain = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = InMemoryChatMessageHistory()
        return self.session_histories[session_id]
    
    def get_response(self, user_input: str, session_id: str = "default") -> str:
        """
        Get a regular (non-streaming) response
        
        Args:
            user_input: User's input message
            session_id: Session identifier for conversation history
            
        Returns:
            AI response string
        """
        try:
            response = self.chat_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def stream_response(self, user_input: str, session_id: str = "default") -> Generator[str, None, None]:
        """
        Get a streaming response
        
        Args:
            user_input: User's input message
            session_id: Session identifier for conversation history
            
        Yields:
            Chunks of the AI response
        """
        try:
            if self.streaming:
                # True streaming from Groq API
                for chunk in self.chat_chain.stream(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                ):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
            else:
                # Fallback: Simulate streaming for better UX
                full_response = self.get_response(user_input, session_id)
                
                # Stream by characters for smoother effect
                for char in full_response:
                    yield char
                    time.sleep(0.01)  # Small delay for visual effect
                    
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session"""
        if session_id in self.session_histories:
            self.session_histories[session_id].clear()
    
    def get_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if session_id in self.session_histories:
            history = self.session_histories[session_id].messages
            return [
                {
                    "role": "human" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content
                }
                for msg in history
            ]
        return []
    
    def update_settings(self, model_name: str = None, temperature: float = None, 
                    streaming: bool = None):
        """Update chat settings and reinitialize if needed"""
        updated = False
        
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            updated = True
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            updated = True
            
        if streaming is not None and streaming != self.streaming:
            self.streaming = streaming
            updated = True
        
        if updated:
            self._setup_llm()
            self._setup_chat_chain()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b", 
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            "gemma2-9b-it"
        ]

def create_chat_handler(api_key: str = None, model_name: str = None, 
                    temperature: float = 0.7, streaming: bool = False) -> ChatHandler:
    """
    Factory function to create a chat handler
    
    Args:
        api_key: Groq API key
        model_name: Model to use
        temperature: Response temperature
        streaming: Enable streaming responses
        
    Returns:
        Initialized ChatHandler instance
    """
    return ChatHandler(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        streaming=streaming
    )

def test_chat_handler():
    """Test function for the chat handler (for CLI usage)"""
    try:
        # Initialize chat handler
        handler = create_chat_handler(streaming=False)
        
        print(f"✅ Successfully connected to Groq API")
        print(f"📡 Using model: {handler.model_name}")
        print("Type 'quit' to exit\n")
        
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

if __name__ == "__main__":
    # Run CLI test when called directly
    test_chat_handler()