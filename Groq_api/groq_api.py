from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()  # Local .env

# Cloud secrets (safe)
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
except:
    pass  # Local fallback

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Check API key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY missing! Add to `.env` (local) or Secrets tab (cloud).")
    st.stop()

### Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond directly without thinking steps."),
    ("user", "Question: {question}")
])

st.title("🧠 LangChain Chatbot: Groq + Streamlit")
input_text = st.text_input("Ask anything:", key="input")

### LLM Chain
llm=ChatGroq(model="qwen/qwen3-32b") 
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    with st.spinner("Groq thinking..."):
        try:
            response = chain.invoke({"question": input_text})
            st.success(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
