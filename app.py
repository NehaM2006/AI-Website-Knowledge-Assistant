import os
import requests
import streamlit as st

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="WebGPT AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

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

.stChatMessage{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "documents" not in st.session_state:
    st.session_state.documents = 0

if "last_url" not in st.session_state:
    st.session_state.last_url = ""

# Stores LangChain FAISS database
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Stores chunks for displaying later
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ==========================================================
# LOAD GROQ MODEL
# ==========================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

llm = load_llm()

# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()


# --------------------------------------------------
# SCRAPE WEBSITE
# --------------------------------------------------

def scrape_website(url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
        )

        if len(text.strip()) == 0:
            return None

        # Remove extra spaces
        text = " ".join(text.split())

        return text

    except Exception as e:

        st.error(f"Error : {e}")

        return None

# ==========================================================
# CREATE VECTOR DATABASE
# ==========================================================

def create_vector_database(text, url):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    if len(chunks) == 0:
        return "No content found."

    metadata = []

    for i in range(len(chunks)):
        metadata.append({
            "url": url,
            "chunk": i + 1
        })

    # Save chunks for displaying later
    st.session_state.chunks = []

    for i, chunk in enumerate(chunks):

        st.session_state.chunks.append({
            "id": i + 1,
            "url": url,
            "content": chunk
        })

    # Create LangChain FAISS Vector Database
    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata
    )

    st.session_state.vector_db = vector_db

    return f"✅ Successfully indexed {len(chunks)} chunks."



# ==========================================================
# RAG QUESTION ANSWERING
# ==========================================================

def retrieve_and_answer(query):

    vector_db = st.session_state.vector_db

    if vector_db is None:
        return None, []

    # Use MMR Retrieval
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    docs = retriever.invoke(query)

    if len(docs) == 0:
        return "No relevant information found.", []

    context = ""

    for i, doc in enumerate(docs):

        context += f"""
Chunk {i+1}

{doc.page_content}

--------------------------------------

"""

    prompt = f"""
You are WebGPT AI.

Answer ONLY using the context provided below.

If the answer cannot be found in the context,
reply with:

"I couldn't find this information on the indexed website."

Do NOT make up information.

Context:

{context}

Question:

{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs
# ==================================================
# HEADER
# ==================================================

st.title("🤖 WebGPT AI")

st.markdown(
    """
Ask questions from any website using **RAG + FAISS + Groq**.
"""
)

st.divider()

# ==================================================
# DASHBOARD
# ==================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌐 Websites",
        st.session_state.documents
    )

with col2:
    st.metric(
        "📄 Chunks",
        len(st.session_state.chunks)
    )

with col3:
    st.metric(
        "🤖 Model",
        "Llama 3.3"
    )

with col4:
    st.metric(
        "🧠 Vector DB",
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

    st.write("""
    WebGPT AI is a Retrieval-Augmented Generation (RAG)
    application that indexes website content using
    FAISS Vector Search and answers questions
    using Groq's Llama 3.3 model.
    """)

    st.markdown("---")

    st.subheader("🛠 Tech Stack")

    st.markdown("""
    - Python
    - Streamlit
    - LangChain
    - FAISS
    - HuggingFace Embeddings
    - Groq (Llama 3.3)
    - BeautifulSoup
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
                st.toast("Website Scraped Successfully!")

            if content:

                with st.spinner("🧠 Creating embeddings..."):

                   message = create_vector_database(
                        content,
                        url
                    )
                   st.toast("Embeddings Created!")

                st.session_state.documents += 1

                st.session_state.last_url = url

                st.success(message)
                st.balloons()
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

            with st.spinner("🧠 Retrieving Relevant Knowledge..."):
                answer, retrieved_docs = retrieve_and_answer(query)
                st.success("Answer Generated Successfully!")

            st.session_state.chat_history.append(
                (
                    "You",
                    query
                )
            )

            assistant_response = answer

            if retrieved_docs:

                assistant_response += "\n\n📚 Source Chunks Used:\n"

                for i, doc in enumerate(retrieved_docs):

                    assistant_response += (
                        f"\nChunk {i+1}"
                        f"\nSource: {doc.metadata['url']}"
                        f"\n{'-'*40}\n"
                        f"{doc.page_content[:250]}...\n"
                    )

            st.session_state.chat_history.append(
                (
                    "Assistant",
                    assistant_response
                )
            )
# ==================================================
# CHAT HISTORY
# ==================================================

st.divider()

st.header("💬 AI Conversation")

if len(st.session_state.chat_history) == 0:

    st.info("No conversation yet. Ask your first question!")

else:

    for sender, message in st.session_state.chat_history:

        if sender == "You":

            with st.chat_message("user"):
                st.write(message)

        else:

            with st.chat_message("assistant"):
                st.markdown(message)
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

with st.expander("📚 View Indexed Knowledge", expanded=False):

    if len(st.session_state.chunks) == 0:

        st.info("No website indexed.")

    else:

        for chunk in st.session_state.chunks:

            with st.container():

                st.markdown(
                    f"### 📄 Chunk {chunk['id']}"
                )

                st.caption(chunk["url"])

                st.info(chunk["content"])

                st.divider()

# ==================================================
# PROJECT STATISTICS
# ==================================================

st.divider()

st.header("📊 Dashboard Statistics")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Chunks",
        len(st.session_state.chunks)
    )

with c2:
    st.metric(
        "Questions",
        len(st.session_state.chat_history)//2
    )

with c3:
    st.metric(
        "Embedding",
        "MiniLM"
    )

with c4:
    st.metric(
        "Retriever",
        "MMR"
    )
# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown(
"""
<center>

## 🤖 WebGPT AI

AI-Powered Website Knowledge Assistant

**Python • Streamlit • LangChain • FAISS • HuggingFace • Groq**

Made with ❤️ by **Neha Malhotra**

</center>
""",
unsafe_allow_html=True
)