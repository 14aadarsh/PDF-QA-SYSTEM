import streamlit as st
import os
from pdf_loader import load_and_split_pdf
from vector_store import create_vector_store
from qa_chain import create_qa_chain, get_answer

# ── API Key ───────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY", "")

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📄",
    layout="centered"
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Q&A System")
    st.markdown("**How to use:**")
    st.markdown("1. Upload a PDF file")
    st.markdown("2. Ask any question")
    st.markdown("3. Get instant answers!")
    st.divider()
    st.markdown("**Powered by:**")
    st.markdown("🦙 LLaMA 3 via Groq")
    st.markdown("🔗 LangChain")
    st.markdown("📊 FAISS Vector Store")
    st.markdown("🤗 HuggingFace Embeddings")

# ── Title ─────────────────────────────────────────────────
st.title("📄 PDF Question Answering System")
st.markdown(
    "Upload any PDF and ask questions — "
    "powered by **RAG + LLaMA 3**"
)
st.divider()

# ── Session State ─────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── PDF Upload ────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    help="Upload any PDF file to get started"
)

if uploaded_file:
    if not api_key:
        st.error("❌ API Key not found! Please check Secrets.")
    elif (
        not st.session_state.pdf_loaded or
        uploaded_file.name != st.session_state.pdf_name
    ):
        with st.spinner("Reading and processing PDF..."):
            try:
                chunks = load_and_split_pdf(uploaded_file)
                vector_store = create_vector_store(chunks)
                st.session_state.qa_chain = create_qa_chain(
                    vector_store, api_key
                )
                st.session_state.pdf_loaded = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []
                st.success(
                    f"✅ PDF processed: **{uploaded_file.name}**"
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if st.session_state.pdf_loaded:
    st.success(f"✅ Loaded: **{st.session_state.pdf_name}**")
    if st.button("🔄 Upload New PDF"):
        st.session_state.pdf_loaded = False
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.rerun()

st.divider()

# ── Chat ──────────────────────────────────────────────────
if st.session_state.pdf_loaded and st.session_state.qa_chain:
    st.subheader("💬 Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("pages"):
                st.caption(f"📌 Source Pages: {msg['pages']}")

    question = st.chat_input("Ask anything about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, pages = get_answer(
                        st.session_state.qa_chain,
                        question
                    )
                    st.write(answer)
                    if pages:
                        st.caption(f"📌 Source Pages: {pages}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "pages": pages
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
else:
    st.info("👆 Please upload a PDF to get started!")
