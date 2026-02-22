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
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
# Works both locally and on Streamlit Cloud
try:
    # Streamlit Cloud
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
except:
    # Local .env file
    groq_api_key = os.getenv("GROQ_API_KEY")


# ── Page Config ────────────────────────────────────────────────
st.set_page_config(page_title="QuantumBot", page_icon="⚛️")
st.title("⚛️ QuantumBot")
st.caption("RAG + Custom Agent + Summary Buffer Memory")

# ── Sidebar — Memory Type Selector ────────────────────────────
st.sidebar.title("⚙️ Settings")
memory_type = st.sidebar.selectbox(
    "Memory Type",
    ["Buffer", "Summary", "Buffer Window", "Entity"]
)
agent_type = st.sidebar.selectbox(
    "Agent Type",
    ["Custom ReAct", "Zero Shot", "Conversational"]
)
st.sidebar.info(f"Memory: {memory_type} | Agent: {agent_type}")

# ── LLM ────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model="qwen/qwen3-32b",
    temperature=0,
    model_kwargs={"reasoning_format": "hidden"}
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
        # ✅ Replace with HuggingFace embeddings (free, works on cloud)

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

# ── Memory Setup ───────────────────────────────────────────────
if "lc_memory" not in st.session_state:
    if memory_type == "Buffer":
        from langchain.memory import ConversationBufferMemory
        st.session_state.lc_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False
        )
    elif memory_type == "Summary":
        from langchain.memory import ConversationSummaryMemory
        st.session_state.lc_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history"
        )
    elif memory_type == "Buffer Window":
        from langchain.memory import ConversationBufferWindowMemory
        st.session_state.lc_memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history"
        )
    elif memory_type == "Entity":
        from langchain.memory import ConversationEntityMemory
        st.session_state.lc_memory = ConversationEntityMemory(
            llm=llm,
            memory_key="chat_history"
        )

# ── Agent Setup ────────────────────────────────────────────────
if agent_type == "Custom ReAct":
    react_prompt = PromptTemplate.from_template("""
You are QuantumBot, expert quantum computing assistant.

Previous Conversation:
{chat_history}

Tools available: {tools}

Format STRICTLY:
Question: {input}
Thought: what to do
Action: one of [{tool_names}]
Action Input: search query
Observation: result
Thought: I know the answer
Final Answer: clear answer

Begin!
Question: {input}
Thought:{agent_scratchpad}
""")
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

elif agent_type == "Zero Shot":
    from langchain.agents import initialize_agent, AgentType
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

elif agent_type == "Conversational":
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
    conv_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=conv_memory,
        verbose=True,
        handle_parsing_errors=True
    )

# ── Chat History (display) ─────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── User Input ─────────────────────────────────────────────────
user_prompt = st.chat_input("Ask about qubits, gates, circuits, research...")

if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)

    # Get real memory string for prompt
    memory_string = st.session_state.lc_memory.buffer if hasattr(
        st.session_state.lc_memory, "buffer"
    ) else "No previous conversation."

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        with st.spinner("🔍 Thinking..."):
            if agent_type == "Custom ReAct":
                response = agent_executor.invoke({
                    "input": user_prompt,
                    "chat_history": memory_string
                })
            else:
                response = agent_executor.invoke({"input": user_prompt})

            full_response = response["output"]

        # Stream word by word
        displayed = ""
        for word in full_response.split():
            displayed += word + " "
            response_placeholder.write(displayed + "▌")
            time.sleep(0.03)
        response_placeholder.write(full_response)

    # Save to real LLM memory
    st.session_state.lc_memory.save_context(
        {"input": user_prompt},
        {"output": full_response}
    )

    # Save to display history
    st.session_state.chat_history.append({"role": "user",      "content": user_prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    with st.expander("🧠 Agent Thought Process"):
        st.json(response)

    with st.expander("💾 Current Memory State"):
        if hasattr(st.session_state.lc_memory, "buffer"):
            st.text(st.session_state.lc_memory.buffer)
        elif hasattr(st.session_state.lc_memory, "entity_store"):
            st.json(st.session_state.lc_memory.entity_store)
