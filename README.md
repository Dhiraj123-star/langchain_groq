
# 🚀 LangChain + Groq Conversational Assistant with RAG

A powerful, lightning-fast AI chatbot that combines **LangChain's** conversation management with **Groq's** ultra-fast inference capabilities and **OpenAI's** high-quality embeddings for document Q\&A. Built with **Streamlit** for an intuitive web interface featuring both conversational chat and intelligent document analysis with **real-time streaming responses**.
Now fully containerized using **Docker** and orchestrated with **Docker Compose** for easy deployment and scaling.

## ✨ Key Features

### 🧠 **Advanced AI Capabilities**

* **Dual Mode Operation**: Switch between conversational chat and document Q\&A (RAG)
* **Multiple Model Support**: Choose from Llama-3.1, GPT-OSS, Qwen3, and Gemma2 models
* **Real-time Streaming**: Live response generation with visual streaming effects (chat mode)
* **Conversation Memory**: Maintains context throughout conversations using modern LangChain implementation
* **Document Intelligence**: Upload and chat with PDFs, DOCX, TXT, and CSV files
* **Smart Prompting**: Custom prompt templates for consistent, helpful responses
* **Lightning-fast Inference**: Sub-second response times powered by Groq's optimized infrastructure

### 📄 **Document Q\&A (RAG) Features**

* **Multi-format Support**: Upload PDF, TXT, DOCX, and CSV documents
* **High-Quality Embeddings**: Powered by OpenAI's text-embedding models for superior semantic search
* **Source Citations**: Get precise references with page numbers and content snippets
* **Document Statistics**: Real-time metrics on loaded documents and chunks
* **Smart Chunking**: Intelligent text splitting with overlap for better context retention
* **Vector Search**: FAISS-powered similarity search for relevant document retrieval
* **Multiple Embedding Models**: Choose from text-embedding-3-small, 3-large, or ada-002

### 🌊 **Streaming Experience**

* **Live Response Generation**: Watch responses appear in real-time as they're generated (chat mode)
* **Visual Cursor Effect**: Smooth typing animation during streaming
* **Streaming Toggle**: Switch between streaming and instant response modes
* **Performance Metrics**: Track streaming vs. regular response times
* **Fallback Support**: Graceful degradation when streaming isn't available

### 🐳 **Containerized Deployment**

* **Dockerfile** optimized with **Python 3.11 slim** base image for lightweight builds
* **System dependencies** pre-installed for LangChain and document parsing
* **Non-root user execution** for better security
* **Healthchecks** for automatic container monitoring
* **Environment variables** managed via `.env` for API keys and Streamlit config
* **Persistent storage** for data and logs via Docker volumes

### 🖥️ **Dual Interface Options**

* **Web Interface**: Beautiful, responsive Streamlit UI with dual-mode support
* **Mode Switching**: Seamlessly toggle between chat and document Q\&A modes
* **Interactive Controls**: Adjustable temperature, model switching, embedding selection
* **Real-time Metrics**: Response generation and streaming time tracking
* **Session Management**: Multiple conversation sessions with isolated history
* **Document Management**: Upload, view, and clear documents with visual feedback

### 🔐 **Secure Configuration**

* **Dual API Support**: Secure management of both Groq and OpenAI API keys
* **Environment Variables**: Secure API key management via `.env` files
* **Flexible Setup**: Works with environment variables or manual input
* **Visual Feedback**: Clear status indicators for API key validation and mode status
* **Setup Guidance**: Built-in instructions for easy configuration

### ⚙️ **Developer & DevOps Friendly**

* **Modern Architecture**: Built with latest LangChain patterns and RunnableWithMessageHistory
* **Modular Design**: Separate handlers for chat and RAG functionality
* **Hot Reloading** in development mode with Docker volume mounting
* **Docker Compose** for orchestration, networking, and resource limits
* **Comprehensive Error Handling**: Helpful error messages and graceful fallbacks
* **Production Ready**: Logging, validation, and health checks included

## 🎯 Core Functionalities

| Feature                         | Description                                                             | Benefits                                             |
| ------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------- |
| **💬 Chat Mode**                | Real-time conversational AI with streaming                              | Natural conversations with lightning-fast responses  |
| **📄 Document Q\&A Mode**       | Upload documents and ask questions                                      | Extract insights from your files with AI precision   |
| **🌊 Real-time Streaming**      | Live response generation with visual effects                            | Enhanced user engagement and immediate feedback      |
| **🤖 Multi-Model Chat**         | Support for 5+ different AI models                                      | Choose the best model for your specific use case     |
| **🔍 Smart Document Search**    | FAISS vector search with OpenAI embeddings                              | Find relevant information with superior accuracy     |
| **📚 Source Citations**         | Document references with page numbers                                   | Verify AI responses with original sources            |
| **🧠 Session Memory**           | Persistent conversation context with isolation                          | Natural, flowing conversations across sessions       |
| **⚡ Dual Response Modes**       | Toggle between streaming and instant responses                          | Optimize for speed or experience preference          |
| **🐳 Containerized Deployment** | Docker + Docker Compose setup with volumes, networks, and health checks | Easy deployment, scalability, and persistent storage |
| **🔐 Secure Config**            | Environment-based API management                                        | Enhanced security with visual validation             |
| **📊 Live Metrics**             | Real-time performance tracking                                          | Monitor response times and document statistics       |

## 🚀 Quick Start (Docker)

1. **Clone the repository**

   ```
   git clone <your-repo-url>
   cd langchain-groq-assistant
   ```

2. **Create a `.env` file** in your project root:

   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Build and run using Docker Compose**

   ```
   docker-compose up --build
   ```

4. **Access the app** at:

   ```
   http://localhost:8501
   ```

**Persistent storage**:

* `./data` → `/app/data` for documents
* `./logs` → `/app/logs` for logs

