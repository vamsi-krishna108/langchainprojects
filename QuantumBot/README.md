# ⚛️ QuantumBot — RAG + LangGraph Agent + Memory

A quantum computing chatbot powered by LangChain, LangGraph, Groq LLM, 
and Retrieval Augmented Generation (RAG).

## 🚀 Live Demo
[QuantumBot on Streamlit Cloud](https://langchainprojects-fqthc6ddbmups2oxsnvjpk.streamlit.app/)

## 🧠 Features

- **RAG** — searches IBM Quantum documentation for accurate answers
- **LangGraph Agent** — intelligently decides which tool to use
- **3 Knowledge Sources** — IBM Quantum Docs + ArXiv + Wikipedia
- **4 Memory Types** — Buffer, Summary, Buffer Window, Entity
- **Streaming Responses** — word by word like ChatGPT
- **Chat History** — full conversation displayed on screen

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (qwen/qwen3-32b) |
| Agent | LangGraph ReAct Agent |
| Vector Store | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| RAG Source | IBM Quantum Docs |
| External Tools | ArXiv + Wikipedia |
| UI | Streamlit |

## 📁 Project Structure
```
QuantumBot/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── .gitignore          # Git ignore rules
```

## ⚙️ Setup Locally

**1. Clone the repo**
```bash
git clone https://github.com/vamsi-krishna108/langchainprojects.git
cd langchainprojects/QuantumBot
```

**2. Create conda environment**
```bash
conda create -n langchain_env python=3.11 -y
conda activate langchain_env
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add API keys**
```bash
cp .env.example .env
# Edit .env and add your keys
```

**.env file:**
```
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

**5. Run**
```bash
streamlit run app.py
```

## 🔑 Get API Keys

| Key | Link |
|---|---|
| Groq API Key | https://console.groq.com |
| LangChain API Key | https://smith.langchain.com |

## 🧩 How It Works
```
User Question
      ↓
LangGraph Agent thinks (ReAct loop)
      ↓
Picks best tool:
  "What is a qubit?"      → IBM Quantum Docs
  "Latest research?"      → ArXiv
  "Simple definition?"    → Wikipedia
      ↓
Retrieves relevant context
      ↓
Groq LLM generates answer
      ↓
Streams response word by word
      ↓
Saves to memory + chat history
```

## 💾 Memory Types

| Type | Description | Best For |
|---|---|---|
| Buffer | Stores full conversation | Short chats |
| Summary | Compresses old messages | Long chats |
| Buffer Window | Keeps last 3 exchanges | Medium chats |
| Entity | Tracks specific topics | Fact tracking |

## 🚀 Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Add secrets in Settings → Secrets:
```toml
GROQ_API_KEY = "your_key_here"
LANGCHAIN_API_KEY = "your_key_here"
```
5. Deploy!

## 📚 What I Learned Building This

- RAG pipeline with FAISS vector store
- LangGraph ReAct agent with multiple tools
- LangChain memory types and differences
- Streamlit session state management
- Streaming responses
- Deploying to Streamlit Cloud
- Managing API keys securely

## 🤝 Connect

Built by [Vamsi Krishna](https://github.com/vamsi-krishna108)