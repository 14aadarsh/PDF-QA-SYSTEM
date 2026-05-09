from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def create_qa_chain(vector_store, api_key):

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

    prompt = PromptTemplate(
        template="""
You are a helpful PDF assistant.
Answer the user's question based ONLY on the context below.
If the answer is not in the context, say:
"This information is not available in the document."
Be clear and concise.

Context: {context}
Question: {question}
Answer:
""",
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def get_answer(qa_chain, question):
    chain = qa_chain["chain"]
    retriever = qa_chain["retriever"]

    answer = chain.invoke(question)

    source_pages = []
    docs = retriever.invoke(question)
    for doc in docs:
        page = doc.metadata.get("page", None)
        if page is not None and (page + 1) not in source_pages:
            source_pages.append(page + 1)

    return answer, source_pages