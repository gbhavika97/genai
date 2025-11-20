import os
import streamlit as st
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

st.title("📄 Gemma RAG – Document Q&A")

# Validate API keys
if not groq_api_key:
    st.error("❌ Missing GROQ_API_KEY in .env")
if not google_api_key:
    st.error("❌ Missing GOOGLE_API_KEY in .env")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="gemma-1.5-chat")

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question strictly using the provided context.
If the answer is not in the context, say "Information not found."

<context>
{context}
</context>

Question: {input}
""")

# PDF Upload
uploaded = st.file_uploader("📌 Upload PDF for Question Answering", type="pdf")

def create_vector_store():
    if uploaded is None:
        st.warning("⚠ Upload a PDF first!")
        return

    if "vectors" not in st.session_state:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded.read())

        docs = PyPDFLoader("uploaded.pdf").load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = FAISS.from_documents(split_docs, st.session_state.embeddings)

        st.success("🎉 Vector store created successfully!")

# Button to build vector store
if st.button("🔄 Create Vector Store"):
    create_vector_store()

# Question box
query = st.text_input("🔍 Ask a question from the document")

# Retrieval QA
if query and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.time()
    response = retrieval_chain.invoke({"input": query})
    end = time.time()

    st.subheader("🧠 Answer")
    st.write(response["answer"])
    st.write(f"⏱ Time taken: {round(end - start, 2)} seconds")

    with st.expander("📎 Retrieved Context"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---")


