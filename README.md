# 🚀 LangChain + Groq Conversational Assistant with RAG

A powerful, lightning-fast AI chatbot that combines **LangChain's** conversation management with **Groq's** ultra-fast inference capabilities and **OpenAI's** high-quality embeddings for document Q&A. Built with **Streamlit** for an intuitive web interface featuring both conversational chat and intelligent document analysis with **real-time streaming responses**.

## ✨ Key Features

### 🧠 **Advanced AI Capabilities**
- **Dual Mode Operation**: Switch between conversational chat and document Q&A (RAG)
- **Multiple Model Support**: Choose from Llama-3.1, GPT-OSS, Qwen3, and Gemma2 models
- **Real-time Streaming**: Live response generation with visual streaming effects (chat mode)
- **Conversation Memory**: Maintains context throughout conversations using modern LangChain implementation
- **Document Intelligence**: Upload and chat with PDFs, DOCX, TXT, and CSV files
- **Smart Prompting**: Custom prompt templates for consistent, helpful responses
- **Lightning-fast Inference**: Sub-second response times powered by Groq's optimized infrastructure

### 📄 **Document Q&A (RAG) Features**
- **Multi-format Support**: Upload PDF, TXT, DOCX, and CSV documents
- **High-Quality Embeddings**: Powered by OpenAI's text-embedding models for superior semantic search
- **Source Citations**: Get precise references with page numbers and content snippets
- **Document Statistics**: Real-time metrics on loaded documents and chunks
- **Smart Chunking**: Intelligent text splitting with overlap for better context retention
- **Vector Search**: FAISS-powered similarity search for relevant document retrieval
- **Multiple Embedding Models**: Choose from text-embedding-3-small, 3-large, or ada-002

### 🌊 **Streaming Experience**
- **Live Response Generation**: Watch responses appear in real-time as they're generated (chat mode)
- **Visual Cursor Effect**: Smooth typing animation during streaming
- **Streaming Toggle**: Switch between streaming and instant response modes
- **Performance Metrics**: Track streaming vs. regular response times
- **Fallback Support**: Graceful degradation when streaming isn't available

### 🖥️ **Dual Interface Options**
- **Web Interface**: Beautiful, responsive Streamlit UI with dual-mode support
- **Mode Switching**: Seamlessly toggle between chat and document Q&A modes
- **Interactive Controls**: Adjustable temperature, model switching, embedding selection
- **Real-time Metrics**: Response generation and streaming time tracking
- **Session Management**: Multiple conversation sessions with isolated history
- **Document Management**: Upload, view, and clear documents with visual feedback

### 🔐 **Secure Configuration**
- **Dual API Support**: Secure management of both Groq and OpenAI API keys
- **Environment Variables**: Secure API key management via `.env` files
- **Flexible Setup**: Works with environment variables or manual input
- **Visual Feedback**: Clear status indicators for API key validation and mode status
- **Setup Guidance**: Built-in instructions for easy configuration

### ⚙️ **Developer-Friendly**
- **Modern Architecture**: Built with latest LangChain patterns and RunnableWithMessageHistory
- **Modular Design**: Separate handlers for chat and RAG functionality
- **Session-based Memory**: In-memory chat history with proper session isolation
- **Dynamic Updates**: Hot-swap models and settings without restart
- **Comprehensive Error Handling**: Helpful error messages and graceful fallbacks
- **Production Ready**: Proper logging, validation, and edge case handling

## 🎯 Core Functionalities

| Feature | Description | Benefits |
|---------|-------------|----------|
| **💬 Chat Mode** | Real-time conversational AI with streaming | Natural conversations with lightning-fast responses |
| **📄 Document Q&A Mode** | Upload documents and ask questions | Extract insights from your files with AI precision |
| **🌊 Real-time Streaming** | Live response generation with visual effects | Enhanced user engagement and immediate feedback |
| **🤖 Multi-Model Chat** | Support for 5+ different AI models | Choose the best model for your specific use case |
| **🔍 Smart Document Search** | FAISS vector search with OpenAI embeddings | Find relevant information with superior accuracy |
| **📚 Source Citations** | Document references with page numbers | Verify AI responses with original sources |
| **🧠 Session Memory** | Persistent conversation context with isolation | Natural, flowing conversations across sessions |
| **⚡ Dual Response Modes** | Toggle between streaming and instant responses | Optimize for speed or experience preference |
| **🔐 Secure Config** | Environment-based API management | Enhanced security with visual validation |
| **📊 Live Metrics** | Real-time performance tracking | Monitor response times and document statistics |

## 🚀 Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone <your-repo-url>
cd langchain-groq-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain-groq streamlit langchain langchain-community langchain-openai python-dotenv faiss-cpu pypdf docx2txt
```

### 2. **Environment Setup**

Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
OPENAI_API_KEY=your_actual_openai_api_key_here
```

**Get Your API Keys:**
1. **Groq API Key**: Visit [Groq Console](https://console.groq.com/) - Required for AI generation
2. **OpenAI API Key**: Visit [OpenAI Platform](https://platform.openai.com/) - Required for document embeddings

### 3. **Run the Application**

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 💻 Usage Examples

### Chat Mode Features

1. **🌊 Streaming Toggle**: Enable/disable real-time response streaming
2. **🤖 Model Selection**: Choose from 5 different AI models in the sidebar
3. **🌡️ Temperature Control**: Adjust response creativity (0.0 = focused, 1.0 = creative)
4. **🔄 Session Management**: Clear history, reset app, or continue conversations
5. **📊 Live Metrics**: Watch streaming performance and chat statistics

### Document Q&A Mode Features

1. **📄 Document Upload**: Drag and drop PDF, TXT, DOCX, or CSV files
2. **🔍 Smart Search**: Ask questions and get AI-powered answers from your documents
3. **📚 Source Citations**: View exact references with page numbers and snippets
4. **📊 Document Stats**: See loaded documents and chunk counts
5. **🧠 Embedding Models**: Choose from OpenAI's embedding models for better search quality
6. **🗑️ Document Management**: Clear documents or reset the system

### Example Document Q&A Workflow

```
1. Upload documents (PDF, DOCX, TXT, CSV)
2. Wait for processing confirmation
3. Ask questions like:
   - "What are the main topics covered in this document?"
   - "Summarize the key findings from the research paper"
   - "What does the contract say about payment terms?"
4. Get AI answers with source citations
5. Click on sources to see exact document references
```

## 🔧 Configuration Options

### Available Models (Both Modes)

| Model | Description | Best For | Chat Streaming | RAG Support |
|-------|-------------|----------|----------------|-------------|
| `llama-3.1-8b-instant` | Fast, efficient Llama model | General conversations | ✅ Full | ✅ |
| `openai/gpt-oss-120b` | Large powerful model | Complex reasoning | ✅ Full | ✅ |
| `openai/gpt-oss-20b` | Balanced performance model | Most tasks | ✅ Full | ✅ |
| `qwen/qwen3-32b` | Advanced reasoning model | Technical discussions | ✅ Full | ✅ |
| `gemma2-9b-it` | Google's instruction-tuned model | Task completion | ✅ Full | ✅ |

### OpenAI Embedding Models

| Model | Description | Best For | Performance |
|-------|-------------|----------|-------------|
| `text-embedding-3-small` | Fast, cost-effective | General documents | ⚡ Fast |
| `text-embedding-3-large` | High performance | Complex documents | 🎯 Best Quality |
| `text-embedding-ada-002` | Legacy model | Compatibility | 📊 Balanced |

### Supported Document Formats

- **PDF**: Research papers, reports, manuals
- **DOCX/DOC**: Word documents, contracts, proposals  
- **TXT**: Plain text files, code, notes
- **CSV**: Data files, spreadsheets, logs

## 🛠️ Advanced Features

### RAG (Retrieval-Augmented Generation)
- **Document Processing**: Intelligent text splitting with configurable chunk sizes
- **Vector Storage**: FAISS-based similarity search for fast retrieval
- **Source Attribution**: Exact citations with page numbers and content snippets
- **Multi-document Support**: Ask questions across multiple uploaded files
- **Smart Retrieval**: Top-k similarity search with configurable parameters

### Real-time Streaming (Chat Mode)
- **Live Generation**: Responses appear character by character as they're generated
- **Visual Effects**: Smooth cursor animation and typing indicators
- **Performance Metrics**: Track streaming speed and total response time
- **Fallback Support**: Graceful degradation to character-by-character simulation

### Session Management
- **Memory Isolation**: Each session maintains separate conversation history
- **Dynamic Updates**: Change models and settings without losing context
- **History Management**: Clear individual sessions or reset entire application
- **Document Persistence**: Uploaded documents remain available across conversations

### Modern Architecture
- **RunnableWithMessageHistory**: Latest LangChain conversation patterns
- **Modular Handlers**: Separate ChatHandler and RAGHandler classes
- **In-memory Storage**: Fast, efficient session-based chat history
- **Hot Configuration**: Update settings without restarting the application

## 🚦 Troubleshooting

### Common Issues

**❌ "GROQ_API_KEY not found"**
- Solution: Create `.env` file with your Groq API key

**❌ "OpenAI API key is required for Document Q&A mode"**
- Solution: Add OPENAI_API_KEY to your `.env` file
- Note: OpenAI embeddings provide superior document search quality

**❌ "'RAGHandler' object has no attribute 'api_key'"**
- Solution: Updated code fixes this initialization error

**❌ "Failed to load document"**
- Check file format is supported (PDF, TXT, DOCX, CSV)
- Ensure file is not corrupted or password-protected
- Try with a smaller file to test

**❌ "Streaming not working"**
- Streaming only works in Chat mode, not Document Q&A mode
- Check internet connection stability
- Try toggling streaming mode off/on

### Performance Tips

- **For fastest responses**: Use `llama-3.1-8b-instant` with streaming disabled
- **For best chat experience**: Use any model with streaming enabled
- **For document Q&A**: Use `text-embedding-3-large` for best search quality
- **For cost optimization**: Use `text-embedding-3-small` for general documents
- **For complex documents**: Use `openai/gpt-oss-120b` or `qwen/qwen3-32b` with `text-embedding-3-large`

### Document Upload Tips

- **Optimal file sizes**: Keep documents under 10MB for better performance
- **Multiple documents**: Upload related documents together for cross-reference queries
- **Clear documents**: Use "Clear Documents" button to reset and upload new files
- **Supported formats**: Ensure files are in PDF, TXT, DOCX, or CSV format

## 🔄 Recent Updates

### Version 2.0 - RAG Implementation
- ✅ Added Document Q&A mode with RAG functionality
- ✅ Integrated OpenAI embeddings for superior semantic search
- ✅ Implemented FAISS vector store for fast similarity search
- ✅ Added support for PDF, DOCX, TXT, and CSV documents
- ✅ Built source citation system with page references
- ✅ Created modular architecture with separate handlers
- ✅ Added document statistics and management features
- ✅ Implemented mode switching between chat and RAG

**⚡ Built with ❤️ using LangChain + Groq + OpenAI + Streamlit**

*Experience lightning-fast AI conversations and intelligent document analysis - now with RAG capabilities!*