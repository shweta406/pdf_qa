import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

# API keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# UI
st.title("Conversational PDF RAG")
st.write("Upload PDFs and chat with them")

if not api_key:
    st.error("GROQ_API_KEY missing")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# Session
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Chat history function
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# Cache vectorstore (important for performance)
@st.cache_resource
def create_vectorstore(splits):
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )


if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        temppdf = f"./temp_{uploaded_file.name}"   # FIXED overwrite bug
        with open(temppdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    if not documents:
        st.error("No valid content found in PDFs")
        st.stop()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Vector DB
    vectorstore = create_vectorstore(splits)
    retriever = vectorstore.as_retriever()

    # Contextual retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase question based on chat history if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using context. If unsure, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Input
    user_input = st.text_input("Ask something:")

    if user_input:
        session_history = get_session_history(session_id)

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        # Show answer
        st.write("### Assistant")
        st.write(response["answer"])

        # Clean chat history (FIXED your issue)
        st.write("### Chat History")
        for msg in session_history.messages:
            role = "User" if msg.type == "human" else "AI"
            st.write(f"**{role}:** {msg.content}")
