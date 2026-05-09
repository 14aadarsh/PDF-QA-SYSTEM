import streamlit as st
from pdf_loader import load_and_split_pdf
from vector_store import create_vector_store
from qa_chain import create_qa_chain, get_answer

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📄",
    layout="centered"
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk-..."
    )
    st.caption("🔒 Key safe hai — sirf is session mein use hogi")
    st.divider()
    st.markdown("### Kaise use karein:")
    st.markdown("1️⃣ Groq API Key daalo")
    st.markdown("2️⃣ PDF upload karo")
    st.markdown("3️⃣ Question poochho")
    st.markdown("4️⃣ Answer pao ✅")
    st.divider()
    st.markdown("🔗 Key yahan se lo: [console.groq.com](https://console.groq.com)")

# ── Title ─────────────────────────────────────────────────
st.title("📄 PDF Question Answering System")
st.markdown("Apni PDF upload karo aur kuch bhi poochho!")
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
    "📂 PDF Upload karo",
    type=["pdf"]
)

if uploaded_file:
    if not api_key:
        st.warning("⚠️ Pehle sidebar mein Groq API Key daalo!")
    elif (
        not st.session_state.pdf_loaded or
        uploaded_file.name != st.session_state.pdf_name
    ):
        with st.spinner("📖 PDF padh raha hoon..."):
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
                    f"✅ PDF ready: **{uploaded_file.name}**"
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if st.session_state.pdf_loaded:
    st.success(f"✅ Loaded: **{st.session_state.pdf_name}**")
    if st.button("🔄 Nayi PDF Upload karo"):
        st.session_state.pdf_loaded = False
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.rerun()

st.divider()

# ── Chat ──────────────────────────────────────────────────
if st.session_state.pdf_loaded and st.session_state.qa_chain:
    st.subheader("💬 Questions Poochho")

    # Purane messages dikhao
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("pages"):
                st.caption(f"📌 Source Pages: {msg['pages']}")

    # Naya question
    question = st.chat_input("Apna question likho...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon..."):
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
    st.info("👆 Pehle PDF upload karo!")
