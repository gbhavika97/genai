import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------------------
# ENV + LLM SETUP
# -------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY in environment variables.")
    st.stop()

st.set_page_config(page_title="RAG Chat — Research PDFs", page_icon="📚")
st.title("📚 RAG Chat with Research Papers (Groq + LangChain)")

# Use a currently supported Groq model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="groq/compound",   # or "llama3-70b-versatile"
    temperature=0.1,
    max_tokens=2048,
)

# Prompt used ONLY when we already have context
prompt_with_context = ChatPromptTemplate.from_template(
    """
You are a helpful RAG assistant.
Answer the user's question using ONLY the provided context and chat history.

- Use the context as your knowledge base.
- If the context gives partial clues, still try to give a helpful answer.
- Do NOT say "Information not found" unless you are explicitly told to.
- Do NOT mention that you are an AI model.
- Do NOT repeat the context verbatim; summarize and answer.

Chat History:
{history}

Context:
{context}

Question:
{input}

Answer (be concise and grounded in the context):
"""
)

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"user":..., "assistant":...}

# -------------------------------------------------------------------
# BUILD VECTOR STORE FROM UPLOADED PDFs
# -------------------------------------------------------------------
def build_vector_store(pdf_files):
    if not pdf_files:
        st.warning("⚠ Please upload at least one PDF.")
        return

    os.makedirs("uploaded_pdfs", exist_ok=True)
    docs = []

    for f in pdf_files:
        path = os.path.join("uploaded_pdfs", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())

        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    st.session_state.vectors = FAISS.from_documents(chunks, embeddings)

    st.success(f"📚 Vector store created from {len(pdf_files)} PDF(s).")

# -------------------------------------------------------------------
# SIDEBAR: PDF UPLOAD + BUILD
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded_pdfs = st.file_uploader(
        "Upload one or more research PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("🔄 Build Knowledge Base"):
        build_vector_store(uploaded_pdfs)

    if st.session_state.vectors is not None:
        st.success("✅ Knowledge base ready")
    else:
        st.info("Upload PDFs and click 'Build Knowledge Base'")

# -------------------------------------------------------------------
# CHAT UI
# -------------------------------------------------------------------
st.subheader("💬 Chat with your documents")

# Show previous conversation
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])

user_question = st.chat_input("Ask a question about the uploaded papers...")

if user_question:
    if st.session_state.vectors is None:
        st.warning("⚠ Please build the knowledge base first.")
    else:
        # Build history string (last 10 turns)
        history_text = "\n".join(
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in st.session_state.history[-10:]
        ) or "No previous conversation."

        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(user_question)

        # If NO docs retrieved -> we handle "Information not found" ourselves
        if not docs:
            answer = "Information not found."
            with st.chat_message("user"):
                st.write(user_question)
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.history.append(
                {"user": user_question, "assistant": answer}
            )
        else:
            # Build context string from retrieved documents
            context_text = "\n\n".join(d.page_content for d in docs)

            messages = prompt_with_context.format_messages(
                history=history_text,
                context=context_text,
                input=user_question,
            )

            with st.chat_message("user"):
                st.write(user_question)

            start = time.time()
            llm_resp = llm.invoke(messages)
            elapsed = round(time.time() - start, 2)

            answer = (
                llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)
            )

            with st.chat_message("assistant"):
                st.write(answer)
                st.caption(f"⏱ {elapsed} seconds")

            st.session_state.history.append(
                {"user": user_question, "assistant": answer}
            )

            # Show context chunks only when we actually had context
            with st.expander("📎 Retrieved context"):
                for i, d in enumerate(docs, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(d.page_content)
                    st.write("---")



















