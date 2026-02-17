"""
groq_api.py - Perfect Groq + LangChain + Streamlit Chatbot (Cloud Deploy)
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment (optional tracing)
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(page_title="Groq Chatbot", layout="wide")

### Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond directly without thinking steps or showing internal reasoning."),
    ("user", "Question: {question}")
])

### UI
st.title("🧠 Groq Chatbot")
st.caption("Fast cloud AI - Powered by Groq!")
st.markdown("---")

input_text = st.text_input("💭 Ask anything:", key="input", 
                           placeholder="Explain machine learning...")

# Cloud secrets (safe)
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
except:
    pass  # Local fallback

# Check API key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY missing! Add to `.env` (local) or Secrets tab (cloud).")
    st.stop()

### Groq LLM
@st.cache_resource
def load_llm():
    return ChatGroq(model="qwen/qwen3-32b")

llm = load_llm()
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

### Run
if input_text:
    with st.spinner("Groq thinking..."):
        try:
            with st.chat_message("user"):
                st.write(input_text)
            with st.chat_message("assistant"):
                response = chain.invoke({"question": input_text})
                st.write(response)
        except Exception as e:
            st.error(f"❌ Groq error: {str(e)}")
            st.info("💡 Add GROQ_API_KEY to Secrets tab (cloud) or .env (local)")

st.markdown("---")
st.caption("☁️ Cloud deploy | 👨‍💻 Built by Vamsi Krishna |")
