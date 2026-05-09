PDF Question Answering System : A RAG-based application that allows users to upload any PDF and ask questions about its content using LLM.

LIVE DEMO : https://pdf-app-system-tqfa6zuhzk4muu3wujiemn.streamlit.app/

About The Project :
This project uses **Retrieval Augmented Generation (RAG)** architecture to answer questions from PDF documents. 
Instead of sending the entire PDF to the LLM, it intelligently retrieves only the relevant sections and generates accurate answers.

#TECH STACK
| Technology | Purpose |
|---|---|
| **LangChain** | LLM Pipeline & Orchestration |
| **Groq API (LLaMA 3)** | Language Model for Answer Generation |
| **FAISS** | Vector Store for Similarity Search |
| **HuggingFace Embeddings** | Converting text to vectors |
| **Streamlit** | Web Interface |
| **PyPDF** | PDF Loading & Parsing |

**How It Works**
User uploads PDF
      ↓
PDF is split into chunks
      ↓
Chunks converted to embeddings (HuggingFace)
      ↓
Embeddings stored in FAISS Vector Store
      ↓
User asks a question
      ↓
Relevant chunks retrieved (Similarity Search)
      ↓
LLaMA 3 generates answer from relevant chunks
      ↓
Answer displayed with Source Page Numbers


***Project Structure**
pdf-qa-project/
├── Pdf_Loader.py        # PDF loading and text splitting
├── vector_store.py      # Embeddings and FAISS vector store
├── qa_chain.py          # LLM chain and answer generation
├── Streamlit_app.py     # Web interface
└── requirements.txt     # Project dependencies

## Features
- Upload any PDF file
- Intelligent search using FAISS similarity search
- Accurate answers using LLaMA 3 model
- Source page numbers shown with every answer
- Clean chat interface
- API key is session-based and secure
