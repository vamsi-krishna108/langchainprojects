```markdown
# 🚀 Groq + LangChain + Streamlit Chatbot

An interactive AI **chatbot** built with **Groq**, **LangChain**, and **Streamlit**.

🔗 **Live App:** https://langchainprojects-arza4brrnaazjflpuw5j5l.streamlit.app/

This project showcases a production-style LLM chatbot with:
- Fast cloud inference using **Groq**
- Clean **Streamlit** UI
- **LangChain** prompt pipeline for structured responses

---

## ✨ Features

- **Conversational Chat UI**  
  Simple textbox + response area for natural Q&A.

- **Groq LLM Backend**  
  Uses `ChatGroq` with a high-quality open-source model (e.g. Qwen / Llama family) for low-latency responses.

- **LangChain Integration**  
  Uses `ChatPromptTemplate` and `StrOutputParser` for prompt formatting and output parsing. 

- **Secure API Key Handling**  
  - Local: `.env` via `python-dotenv`  
  - Cloud: `st.secrets["GROQ_API_KEY"]` in Streamlit Secrets  
  Same code runs both locally and on Streamlit Cloud.

---

## 🏗️ Tech Stack

- **Language:** Python 3.10+
- **Frontend:** Streamlit
- **LLM Orchestration:** LangChain
- **Model Provider:** Groq (`langchain_groq`)
- **Config:** `.env` (local), Streamlit Secrets (cloud) 

---

## 📂 Project Structure

```text
Groq_api/
├── groq_api.py        # Main Streamlit + Groq chatbot
├── llama2.py          # Local-only Ollama Llama2 version (offline)
├── requirements.txt   # Dependencies
├── .gitignore         # Excludes .env and secrets
├── LICENSE
└── README.md
```

- `groq_api.py`: Cloud-ready Groq chatbot (deployed on Streamlit Cloud). 
- `llama2.py`: Optional local-only Ollama Llama2 chatbot for offline use. 

---

## ⚙️ Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/vamsi-krishna108/langchainprojects.git
   cd langchainprojects/Groq_api
   ```

2. **Create & activate virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   # or
   source venv/bin/activate   # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` includes Streamlit, LangChain core, and Groq integration.

4. **Create `.env` file** (local only)
   ```bash
   GROQ_API_KEY=your_groq_key_here
   LANGCHAIN_API_KEY=your_langsmith_key_here  # optional
   ```

5. **Run the app**
   ```bash
   streamlit run groq_api.py
   ```

6. Open in browser:
   - `http://localhost:8501`

---

## 🌐 Streamlit Cloud Deployment (Groq)

1. Push this folder to **GitHub** (your repo already set up at  
   `https://github.com/vamsi-krishna108/langchainprojects/tree/main/Groq_api`). 

2. On **Streamlit Cloud**:
   - Create a new app from this repo.
   - Set **Main file path** to:
     ```text
     Groq_api/groq_api.py
     ```

3. In **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_key_here"
   LANGCHAIN_API_KEY = "your_langsmith_key_here"
   ```

4. Deploy – the app will be available at:
   ```text
   https://langchainprojects-arza4brrnaazjflpuw5j5l.streamlit.app/
   ```

---

## 🧠 How It Works

1. **User Input**  
   User enters a question into the Streamlit input box. [file:2]

2. **Prompt Construction (LangChain)**  
   `ChatPromptTemplate` builds a system + user message:
   - System: “You are a helpful assistant. Respond directly without thinking steps.”
   - User: “Question: {question}” 

3. **LLM Call (Groq)**  
   `ChatGroq(model="qwen/qwen3-32b")` (or similar) generates a response using the Groq API. [file:2]

4. **Output Parsing**  
   `StrOutputParser` converts the response into a plain string.

5. **Display in UI**  
   Response is displayed via Streamlit components (spinner + success/output box). [file:2]

---

Then reference them in README:
```markdown

This project demonstrates:

- Building a real **LLM chatbot** with **Groq + LangChain + Streamlit**.
- Handling **API keys** safely with `.env` (local) and Streamlit Secrets (cloud).
- Deploying to **Streamlit Cloud** and sharing a live demo link.
- Optional: maintaining a **local-only** Ollama Llama2 version (`llama2.py`) to show offline capability.

**Built by Jonnagiri Vamsi Krishna**  
`#AI #MachineLearning #Streamlit #Groq #LangChain #Python`
```
