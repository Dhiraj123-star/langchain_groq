# 🚀 LangChain + Groq Conversational Assistant

A powerful, lightning-fast AI chatbot that combines **LangChain's** conversation management with **Groq's** ultra-fast inference capabilities. Built with **Streamlit** for an intuitive web interface and featuring both web and CLI modes.


## ✨ Key Features

### 🧠 **Advanced AI Capabilities**
- **Multiple Model Support**: Choose from Llama-3.1, GPT-OSS, Kimi-K2, and Gemma2 models
- **Conversation Memory**: Maintains context throughout conversations using LangChain's memory system
- **Smart Prompting**: Custom prompt templates for consistent, helpful responses
- **Fast Inference**: Sub-second response times powered by Groq's optimized infrastructure

### 🖥️ **Dual Interface Options**
- **Web Interface**: Beautiful, responsive Streamlit UI with real-time chat
- **CLI Mode**: Terminal-based interface for developers and power users
- **Real-time Metrics**: Response generation time tracking
- **Interactive Controls**: Adjustable temperature, model switching, conversation clearing

### 🔐 **Secure Configuration**
- **Environment Variables**: Secure API key management via `.env` files
- **Flexible Setup**: Works with environment variables or manual input
- **Visual Feedback**: Clear status indicators for API key validation
- **Setup Guidance**: Built-in instructions for easy configuration

### ⚙️ **Developer-Friendly**
- **Clean Architecture**: Modular, well-documented code structure
- **Error Handling**: Comprehensive error management with helpful messages
- **Extensible Design**: Easy to add new features and models
- **Production Ready**: Proper logging, validation, and edge case handling

## 🎯 Core Functionalities

| Feature | Description | Benefits |
|---------|-------------|----------|
| **Multi-Model Chat** | Support for 4+ different AI models | Choose the best model for your use case |
| **Persistent Memory** | Conversation context retention | Natural, flowing conversations |
| **Real-time UI** | Instant response rendering | Smooth user experience |
| **Dual Modes** | Web + CLI interfaces | Flexibility for different workflows |
| **Secure Config** | Environment-based API management | Enhanced security practices |
| **Performance Metrics** | Response time tracking | Monitor and optimize performance |
| **Dynamic Controls** | Live parameter adjustment | Fine-tune responses on the fly |

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
pip install langchain-groq streamlit langchain python-dotenv
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

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**CLI Mode:**
```bash
python app.py --cli
```

## 💻 Usage Examples

### Web Interface Features

1. **Model Selection**: Choose from multiple AI models in the sidebar
2. **Temperature Control**: Adjust response creativity (0.0 = focused, 1.0 = creative)
3. **Conversation Management**: Clear history or continue previous conversations
4. **Real-time Chat**: Instant responses with typing indicators
5. **Performance Metrics**: See response generation times

### CLI Commands

```bash
# Start CLI mode
python app.py --cli

# Example conversation
👤 You: What is machine learning?
🤖 Assistant: Machine learning is a subset of artificial intelligence...
⚡ (0.85s)

# Exit commands
quit / exit / bye
```

## 🏗️ Project Structure

```
langchain-groq-assistant/
├── app.py                 # Main application file
├── .env                   # Environment variables (create this)
├── .env.example          # Example environment file
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration Options

### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `llama-3.1-8b-instant` | Fast, efficient Llama model | General conversations |
| `openai/gpt-oss-20b` | Powerful open-source model | Complex reasoning |
| `moonshotai/kimi-k2-instruct` | Instruction-tuned model | Task-oriented chat |
| `gemma2-9b-it` | Google's Gemma model | Balanced performance |

### Parameters

- **Temperature**: `0.0-1.0` - Controls response randomness
- **Max Tokens**: `1024` - Maximum response length
- **Memory**: Conversation buffer with full context retention

## 🛠️ Advanced Features

### Conversation Memory
```python
# The app maintains conversation context using LangChain's ConversationBufferMemory
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history"
)
```

### Custom Prompting
```python
# Customizable system prompt for consistent behavior
template = """
You are a helpful and friendly AI assistant. You provide clear, informative responses 
and maintain context throughout the conversation.
"""
```

### Error Handling
- Automatic API key validation
- Network error recovery
- User-friendly error messages
- Graceful degradation

## 🚦 Troubleshooting

### Common Issues

**❌ "GROQ_API_KEY not found"**
- Solution: Create `.env` file with your API key

**❌ "Failed to initialize"**
- Check internet connection
- Verify API key is valid
- Ensure sufficient API credits

**❌ "Model not available"**
- Try a different model from the dropdown
- Check Groq service status

### Debug Mode

Enable verbose logging by setting `verbose=True` in the conversation chain.


**⚡ Built with ❤️ using LangChain + Groq + Streamlit**

*Start chatting with AI in seconds - no complex setup required!*