import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
except:
    groq_api_key = os.getenv("GROQ_API_KEY")

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="QuantumBot", page_icon="⚛️")
st.title("⚛️ QuantumBot")
st.caption("RAG + LangGraph Agent + Real Memory")

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
memory_type = st.sidebar.selectbox(
    "Memory Type",
    ["Buffer", "Summary", "Buffer Window", "Entity"]
)
st.sidebar.info(f"Memory: {memory_type}")

# ── LLM ────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="qwen/qwen3-32b",
    temperature=0
)

# ── Build Vector Store ─────────────────────────────────────────
if "vectors" not in st.session_state:
    with st.spinner("📚 Loading IBM Quantum Docs... ⏳"):
        urls = ["https://quantum.cloud.ibm.com/docs/en/guides"]
        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load {url}: {e}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
    st.success("✅ Docs loaded!")

# ── Tools ──────────────────────────────────────────────────────
retriever_tool = create_retriever_tool(
    st.session_state.vectors.as_retriever(search_kwargs={"k": 3}),
    name="quantum_notes",
    description="Search IBM Quantum docs for qubits, gates, circuits, Qiskit."
)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300))
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300))
tools = [retriever_tool, arxiv, wiki]

# ── LangGraph Agent (works on Python 3.13) ─────────────────────
# create_react_agent from langgraph handles memory natively
# via messages list — no separate memory object needed
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_react_agent(llm, tools)

# ── Memory Setup (for display + context) ──────────────────────
# ── Memory Setup (simple dict — no langchain.memory needed) ────
if "lc_memory" not in st.session_state:
    st.session_state.lc_memory = {"buffer": "", "entity_store": {}}

def save_memory(user_input, bot_output):
    if memory_type == "Buffer":
        st.session_state.lc_memory["buffer"] += f"User: {user_input}\nBot: {bot_output}\n"

    elif memory_type == "Summary":
        # Keep only last summary — compressed
        st.session_state.lc_memory["buffer"] = f"Summary so far: User asked about {user_input[:50]}..."

    elif memory_type == "Buffer Window":
        # Keep only last 3 exchanges
        lines = st.session_state.lc_memory["buffer"].split("\n")
        lines = [l for l in lines if l]  # remove empty
        lines.append(f"User: {user_input}")
        lines.append(f"Bot: {bot_output}")
        st.session_state.lc_memory["buffer"] = "\n".join(lines[-6:])  # last 3 pairs

    elif memory_type == "Entity":
        # Extract simple entity (just store user name/topic mentions)
        st.session_state.lc_memory["entity_store"][f"topic_{len(st.session_state.chat_history)}"] = user_input

# ── Chat History (display) ─────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── LangGraph uses messages list as real memory ────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are QuantumBot, expert quantum computing assistant.")
    ]

# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── User Input ─────────────────────────────────────────────────
user_prompt = st.chat_input("Ask about qubits, gates, circuits, research...")

if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)

    # Add user message to real LangGraph memory
    st.session_state.messages.append(HumanMessage(content=user_prompt))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        with st.spinner("🔍 Thinking and searching..."):
            # LangGraph agent receives full message history = real memory!
            response = st.session_state.agent_executor.invoke({
                "messages": st.session_state.messages
            })
            full_response = response["messages"][-1].content

        # Stream word by word
        displayed = ""
        for word in full_response.split():
            displayed += word + " "
            response_placeholder.write(displayed + "▌")
            time.sleep(0.03)
        response_placeholder.write(full_response)

    # Add assistant response to LangGraph memory
    st.session_state.messages.append(AIMessage(content=full_response))

    # Save to langchain memory (for memory state display)
    # ✅ Replace with
    save_memory(user_prompt, full_response)

    # Save to display history
    st.session_state.chat_history.append({"role": "user",      "content": user_prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    with st.expander("🧠 Agent Thought Process"):
        st.json({"input": user_prompt, "output": full_response})

    
    # ✅ Replace with
    with st.expander("💾 Current Memory State"):
        if memory_type == "Entity":
            st.json(st.session_state.lc_memory["entity_store"])
        else:
            st.text(st.session_state.lc_memory["buffer"])