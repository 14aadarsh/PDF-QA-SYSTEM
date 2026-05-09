from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, api_key=None):
    
    # HuggingFace embeddings - bilkul FREE
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store