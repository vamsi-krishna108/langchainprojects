# LangChaninProjects

# Local-First AI Chatbot: Groq + Ollama2 + Streamlit + LangChain

Interactive chatbot supporting **dual LLM backends**—fast cloud inference (Groq) and fully local/private (Ollama Llama2). Perfect for ML demos, interviews, and portfolio showcase.

## ✨ Features
- **Dual-mode**: Toggle between Groq (cloud, ultra-fast) and Ollama Llama2 (local, offline)
- **LangChain**: Clean prompt chaining + parsing pipeline
- **Streamlit**: Production-ready UI deployed to cloud
- **Hybrid workflow**: Local dev → Cloud sharing

## 🛠️ Quick Start (Local)

### Prerequisites
```bash
pip install -r requirements.txt
```

### Local Ollama Setup
```bash
# Install Ollama (Windows: ollama.com/download)
ollama pull llama2  # ~4GB, runs on CPU/GPU
```

### Run
```bash
# Local with Ollama
streamlit run app.py --server.headless true
```

**Live URL**: `http://localhost:8501`

## ☁️ Cloud Deployment (Groq)
1. Push to GitHub (`.env` excluded via `.gitignore`)
2. Deploy: [share.streamlit.io](https://share.streamlit.io)
3. Auto-uses Groq (cloud) for public demos

## 📁 Project Structure
```
.
├── app.py              # Main Streamlit + LangChain app
├── requirements.txt    # Dependencies
├── .env.example        # API keys template
├── README.md           # This file
└── .gitignore          # Excludes secrets
```

## 🔧 Configuration

### `.env` (Local Only)
```env
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional tracing
```

### Model Toggle (groq_api.py)
```python
# Switch easily:
llm = ChatGroq(model="llama3-8b-8192")     # Cloud (deploy-ready)
# llm = ChatOllama(model="llama2")         # Local (offline)
```

## 🚀 Performance
| Backend | Speed | Cost | Privacy | Use Case |
|---------|-------|------|---------|----------|
| **Groq** | ⚡ 200+ tokens/sec | Free tier | Cloud | Demos, LinkedIn |
| **Ollama2** | 🐌 10-30 tokens/sec | Free | 100% Local | Dev, Offline |

## 📱 Screenshots
*(Add your app screenshots here)*

## 🔗 Live Demo
**[Try Live →](https://your-app.streamlit.app)**  


## 💼 For Interviews/Portfolio
- **Local**: Shows Ollama + LangChain skills (offline capable)
- **Cloud**: Production deployment + GitHub workflow
- **Dual**: Backend flexibility awareness

## 🛠️ Tech Stack
```
Frontend: Streamlit
Backend: LangChain + (Groq/Ollama)
Models: Llama3-8B (Groq), Llama2-7B (Ollama)
Environment: Python 3.10+
```

***

**Built by Jonnagiri Vamsi Krishna** | #AI #MachineLearning #Streamlit #Ollama #LangChain

***

