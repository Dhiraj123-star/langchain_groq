# 🚀 LangChain + Groq Conversational Assistant

A powerful, lightning-fast AI chatbot that combines **LangChain's** conversation management with **Groq's** ultra-fast inference capabilities. Built with **Streamlit** for an intuitive web interface and featuring both web and CLI modes with **real-time streaming responses**.

## ✨ Key Features

### 🧠 **Advanced AI Capabilities**
- **Multiple Model Support**: Choose from Llama-3.1, GPT-OSS, Qwen3, and Gemma2 models
- **Real-time Streaming**: Live response generation with visual streaming effects
- **Conversation Memory**: Maintains context throughout conversations using modern LangChain implementation
- **Smart Prompting**: Custom prompt templates for consistent, helpful responses
- **Lightning-fast Inference**: Sub-second response times powered by Groq's optimized infrastructure

### 🌊 **Streaming Experience**
- **Live Response Generation**: Watch responses appear in real-time as they're generated
- **Visual Cursor Effect**: Smooth typing animation during streaming
- **Streaming Toggle**: Switch between streaming and instant response modes
- **Performance Metrics**: Track streaming vs. regular response times
- **Fallback Support**: Graceful degradation when streaming isn't available

### 🖥️ **Dual Interface Options**
- **Web Interface**: Beautiful, responsive Streamlit UI with real-time streaming chat
- **CLI Mode**: Terminal-based interface with streaming support for developers
- **Interactive Controls**: Adjustable temperature, model switching, streaming toggle
- **Real-time Metrics**: Response generation and streaming time tracking
- **Session Management**: Multiple conversation sessions with isolated history

### 🔐 **Secure Configuration**
- **Environment Variables**: Secure API key management via `.env` files
- **Flexible Setup**: Works with environment variables or manual input
- **Visual Feedback**: Clear status indicators for API key validation and streaming status
- **Setup Guidance**: Built-in instructions for easy configuration

### ⚙️ **Developer-Friendly**
- **Modern Architecture**: Built with latest LangChain patterns and RunnableWithMessageHistory
- **Session-based Memory**: In-memory chat history with proper session isolation
- **Dynamic Updates**: Hot-swap models and settings without restart
- **Comprehensive Error Handling**: Helpful error messages and graceful fallbacks
- **Production Ready**: Proper logging, validation, and edge case handling

## 🎯 Core Functionalities

| Feature | Description | Benefits |
|---------|-------------|----------|
| **🌊 Real-time Streaming** | Live response generation with visual effects | Enhanced user engagement and immediate feedback |
| **🤖 Multi-Model Chat** | Support for 5+ different AI models | Choose the best model for your specific use case |
| **🧠 Session Memory** | Persistent conversation context with isolation | Natural, flowing conversations across sessions |
| **⚡ Dual Response Modes** | Toggle between streaming and instant responses | Optimize for speed or experience preference |
| **🖥️ Dual Interfaces** | Web UI + CLI with streaming support | Flexibility for different user workflows |
| **🔐 Secure Config** | Environment-based API management | Enhanced security with visual validation |
| **📊 Live Metrics** | Real-time performance tracking | Monitor streaming vs regular response times |
| **🔄 Dynamic Controls** | Live parameter adjustment without restart | Fine-tune responses and streaming on the fly |

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
pip install langchain-groq streamlit langchain langchain-community python-dotenv
```

### 2. **Environment Setup**

Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

**Get Your Groq API Key:**
1. Visit [Groq Console](https://console.groq.com/)
2. Create a free account
3. Generate your API key
4. Add it to your `.env` file

### 3. **Run the Application**

**Web Interface with Streaming (Recommended):**
```bash
streamlit run app.py
```

**CLI Mode with Streaming:**
```bash
python app.py --cli
```

## 💻 Usage Examples

### Web Interface Features

1. **🌊 Streaming Toggle**: Enable/disable real-time response streaming
2. **🤖 Model Selection**: Choose from 5 different AI models in the sidebar
3. **🌡️ Temperature Control**: Adjust response creativity (0.0 = focused, 1.0 = creative)
4. **🔄 Session Management**: Clear history, reset app, or continue conversations
5. **📊 Live Metrics**: Watch streaming performance and chat statistics
6. **⚡ Real-time Chat**: Instant or streaming responses with visual effects

### CLI Commands

```bash
# Start CLI mode with streaming support
python app.py --cli

# Example streaming conversation
👤 You: Explain quantum computing
🤖 Assistant: [Streams response in real-time...]
🌊 Response streamed in 1.23s

# Commands
quit / exit / bye  # Exit application
clear             # Clear conversation history
```

## 🔧 Configuration Options

### Available Models

| Model | Description | Best For | Streaming Support |
|-------|-------------|----------|-------------------|
| `llama-3.1-8b-instant` | Fast, efficient Llama model | General conversations | ✅ Full |
| `openai/gpt-oss-120b` | Large powerful model | Complex reasoning | ✅ Full |
| `openai/gpt-oss-20b` | Balanced performance model | Most tasks | ✅ Full |
| `qwen/qwen3-32b` | Advanced reasoning model | Technical discussions | ✅ Full |
| `gemma2-9b-it` | Google's instruction-tuned model | Task completion | ✅ Full |

### Streaming Parameters

- **🌊 Streaming Mode**: Real-time response generation with visual cursor
- **📝 Regular Mode**: Instant full responses with thinking indicator
- **🌡️ Temperature**: `0.0-1.0` - Controls response randomness
- **💾 Session Management**: Isolated conversation histories
- **⚡ Performance Tracking**: Streaming vs regular response time metrics

## 🛠️ Advanced Features

### Real-time Streaming
- **Live Generation**: Responses appear character by character as they're generated
- **Visual Effects**: Smooth cursor animation and typing indicators
- **Performance Metrics**: Track streaming speed and total response time
- **Fallback Support**: Graceful degradation to character-by-character simulation

### Session Management
- **Memory Isolation**: Each session maintains separate conversation history
- **Dynamic Updates**: Change models and settings without losing context
- **History Management**: Clear individual sessions or reset entire application

### Modern Architecture
- **RunnableWithMessageHistory**: Latest LangChain conversation patterns
- **In-memory Storage**: Fast, efficient session-based chat history
- **Hot Configuration**: Update settings without restarting the application

## 🚦 Troubleshooting

### Common Issues

**❌ "GROQ_API_KEY not found"**
- Solution: Create `.env` file with your API key

**❌ "Streaming not working"**
- Check internet connection stability
- Try toggling streaming mode off/on
- Verify model supports streaming

**❌ "Failed to initialize chat handler"**
- Check internet connection
- Verify API key is valid
- Ensure sufficient API credits

**❌ "Session history issues"**
- Try clearing conversation history
- Reset application if problems persist

### Performance Tips

- **For fastest responses**: Use `llama-3.1-8b-instant` with streaming disabled
- **For best experience**: Use any model with streaming enabled
- **For complex tasks**: Use `openai/gpt-oss-120b` or `qwen/qwen3-32b`
- **For balanced performance**: Use `gemma2-9b-it` with streaming enabled

**⚡ Built with ❤️ using LangChain + Groq + Streamlit**

*Experience lightning-fast AI conversations with real-time streaming - no complex setup required!*