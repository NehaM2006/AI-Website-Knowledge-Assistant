import requests
from bs4 import BeautifulSoup
import streamlit as st
import faiss
import numpy as np

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="WebGPT AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    padding-top:20px;
}

.stButton>button{
    width:100%;
    height:45px;
    border-radius:10px;
    font-weight:bold;
}

.stTextInput>div>div>input{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents" not in st.session_state:
    st.session_state.documents = 0

if "last_url" not in st.session_state:
    st.session_state.last_url = ""

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

llm = load_llm()

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# --------------------------------------------------
# VECTOR DATABASE
# --------------------------------------------------

embedding_dimension = 384

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(embedding_dimension)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = {}


# --------------------------------------------------
# SCRAPE WEBSITE
# --------------------------------------------------

def scrape_website(url):
    """
    Scrapes paragraph text from a website.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        text = " ".join(
            p.get_text(strip=True)
            for p in paragraphs
        )

        if not text:
            return None

        return text[:10000]

    except Exception as e:
        st.error(f"Error: {e}")
        return None


# --------------------------------------------------
# STORE DATA IN VECTOR DATABASE
# --------------------------------------------------

def store_in_faiss(text, url):

    index = st.session_state.faiss_index
    vector_store = st.session_state.vector_store

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    if len(chunks) == 0:
        return "No content found."

    vectors = embeddings.embed_documents(chunks)

    vectors = np.array(
        vectors,
        dtype=np.float32
    )

    start_index = len(vector_store)

    index.add(vectors)

    for i, chunk in enumerate(chunks):

        vector_store[start_index + i] = {

            "url": url,

            "content": chunk

        }

    return f"✅ Successfully indexed {len(chunks)} chunks."


# --------------------------------------------------
# RETRIEVE ANSWER
# --------------------------------------------------

def retrieve_and_answer(query):

    index = st.session_state.faiss_index
    vector_store = st.session_state.vector_store

    if index.ntotal == 0:

        return "Please index a website first."

    query_vector = embeddings.embed_query(query)

    query_vector = np.array(
        query_vector,
        dtype=np.float32
    ).reshape(1, -1)

    distances, indices = index.search(
        query_vector,
        k=4
    )

    context = ""

    for idx in indices[0]:

        if idx in vector_store:

            context += (
                vector_store[idx]["content"]
                + "\n\n"
            )

    if context == "":

        return "No relevant information found."

    prompt = f"""
You are an AI assistant.

Answer ONLY using the information below.

Context:

{context}

Question:

{query}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer
# ==================================================
# HEADER
# ==================================================

st.title("🤖 WebGPT AI")

st.markdown(
    """
Ask questions from any website using **RAG + FAISS + Ollama**.
"""
)

st.divider()

# ==================================================
# DASHBOARD
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Indexed Websites",
        st.session_state.documents
    )

with col2:
    st.metric(
        "AI Model",
        "Mistral"
    )

with col3:
    st.metric(
        "Vector DB",
        "FAISS"
    )

st.divider()

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🤖 WebGPT AI")

    st.markdown("---")

    st.subheader("📌 About")

    st.write(
        """
An AI-powered Website Knowledge Assistant
built using Retrieval-Augmented Generation.
        """
    )

    st.markdown("---")

    st.subheader("🛠 Tech Stack")

    st.write("""
✅ Python

✅ Streamlit

✅ Ollama

✅ LangChain

✅ FAISS

✅ BeautifulSoup

✅ HuggingFace
""")

    st.markdown("---")

    st.success("🟢 AI Model Ready")

    st.info(
        f"Indexed Websites : {st.session_state.documents}"
    )

    if st.session_state.last_url:

        st.write("### 🌐 Last Indexed Website")

        st.write(st.session_state.last_url)

    st.markdown("---")

    st.caption("Made by Neha Malhotra")

# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs(
    [
        "🌐 Website Indexing",
        "💬 AI Chat"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Index a Website")

    url = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com"
    )

    if st.button("🚀 Index Website"):

        if url == "":

            st.warning("Please enter a website URL.")

        else:

            with st.spinner("🌍 Scraping website..."):

                content = scrape_website(url)

            if content:

                with st.spinner("🧠 Creating embeddings..."):

                    message = store_in_faiss(
                        content,
                        url
                    )

                st.session_state.documents += 1

                st.session_state.last_url = url

                st.success(message)

            else:

                st.error(
                    "Unable to scrape website."
                )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("Ask Questions")

    query = st.text_input(
        "Ask anything about the indexed website..."
    )

    if st.button("Generate Answer"):

        if query == "":

            st.warning("Please ask a question.")

        else:

            with st.spinner("🤖 Thinking..."):

                answer = retrieve_and_answer(query)

            st.session_state.chat_history.append(
                (
                    "You",
                    query
                )
            )

            st.session_state.chat_history.append(
                (
                    "Assistant",
                    answer
                )
            )
# ==================================================
# CHAT HISTORY
# ==================================================

st.divider()

st.subheader("💬 Conversation")

if len(st.session_state.chat_history) == 0:

    st.info("No conversation yet. Ask your first question!")

else:

    for sender, message in st.session_state.chat_history:

        if sender == "You":

            with st.chat_message("user"):
                st.write(message)

        else:

            with st.chat_message("assistant"):
                st.write(message)

# ==================================================
# CLEAR CHAT
# ==================================================

col1, col2 = st.columns([1, 5])

with col1:

    if st.button("🗑 Clear Chat"):

        st.session_state.chat_history = []

        st.success("Chat history cleared!")

        st.rerun()

# ==================================================
# VIEW DATABASE
# ==================================================

with st.expander("📚 View Indexed Knowledge"):

    if len(st.session_state.vector_store) == 0:

        st.info("No website indexed yet.")

    else:

        for idx, data in st.session_state.vector_store.items():

            st.markdown(f"### Chunk {idx+1}")

            st.caption(data["url"])

            st.write(data["content"][:350] + "...")

            st.divider()

# ==================================================
# PROJECT STATISTICS
# ==================================================

st.divider()

st.subheader("📊 Project Statistics")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Chunks Stored",
        len(st.session_state.vector_store)
    )

with c2:

    st.metric(
        "Questions Asked",
        len(st.session_state.chat_history)//2
    )

with c3:

    st.metric(
        "Embedding Model",
        "MiniLM"
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
"""
<center>

### 🤖 WebGPT AI

AI-Powered Website Question Answering System

**Python • Streamlit • LangChain • Ollama • FAISS • HuggingFace**

Developed by **Neha Malhotra**

</center>
""",
unsafe_allow_html=True
)