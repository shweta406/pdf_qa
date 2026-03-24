import streamlit as st
import os
from dotenv import load_dotenv

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

# -------------------- CONFIG --------------------
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY missing")
    st.stop()

# -------------------- MODELS --------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# -------------------- UI --------------------
st.set_page_config(page_title="Conversational RAG", layout="wide")

st.title("Conversational PDF RAG")
st.write("Upload PDFs and chat with them")

session_id = st.text_input("Session ID", value="default_session")

# -------------------- SESSION STORE --------------------
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

@st.cache_resource
def create_vectorstore(splits):
    return Chroma.from_documents(documents=splits, embedding=embeddings)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        temp_path = "./temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    vectorstore = create_vectorstore(splits)
    retriever = vectorstore.as_retriever()

    # -------------------- PROMPTS --------------------
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question so it is standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using context. Max 3 sentences.\n\n{context}"),
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

    # -------------------- CHAT DISPLAY --------------------
    if session_id in st.session_state.store:
        for msg in st.session_state.store[session_id].messages:
            if msg.type == "human":
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif msg.type == "ai":
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

    # -------------------- CHAT INPUT --------------------
    user_input = st.chat_input("Ask your question...")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get response
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
