"""
locallama2.py - Perfect Ollama Llama2 + LangChain + Streamlit Chatbot (Local Only)
"""

from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment (optional tracing)
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.set_page_config(page_title="Ollama Llama2 Chatbot", layout="wide")

### Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond directly without thinking steps or showing internal reasoning."),
    ("user", "Question: {question}")
])

### UI
st.title("🤖 Ollama Llama2 Chatbot")
st.caption("Fully local AI - No internet, no API keys needed!")
st.markdown("---")

input_text = st.text_input("💭 Ask anything:", key="input", 
                          placeholder="Explain machine learning...")

### Ollama LLM (Llama2 only)
@st.cache_resource
def load_llm():
    return ChatOllama(model="llama2")

llm = load_llm()
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

### Run
if input_text:
    with st.spinner("Llama2 thinking..."):
        try:
            with st.chat_message("user"):
                st.write(input_text)
            with st.chat_message("assistant"):
                response = chain.invoke({"question": input_text})
                st.write(response)
        except Exception as e:
            st.error(f"❌ Ollama error: {str(e)}")
            st.info("💡 Run `ollama pull llama2` and `ollama run llama2` first")

st.markdown("---")
st.caption("🖥️ Local only | 👨‍💻 Built by Vamsi Krishna | 🚀 Companion: groq_api.py")

