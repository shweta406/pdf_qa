import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from chromadb.config import Settings
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

# Configure tokens from environment or Streamlit secrets
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up streamlit
st.title("conversational rag with pdf uploads and chat history")
st.write("upload pdf's and chat with content")

# ensure Groq key is available
if not api_key:
    st.error("GROQ_API_KEY is not set. Add it to your .env or Streamlit secrets.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

##chat interface
session_id = st.text_input("session_id", value="default_session")

## statefully manage chat history
if "store" not in st.session_state:
    st.session_state.store = {}


uploaded_files = st.file_uploader("choose a pdf file", type="pdf", accept_multiple_files=True)
##process uploaded pdf's

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = "./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    if not documents:
        st.error("No pages were loaded from the uploaded PDFs.")
        st.stop()

    # split and create embeddings for all documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        client_settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=False,
        ),
        collection_name=f"session_{session_id}",
    )
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. do not answer the question,"
        "just formulate it if needed and otherwise return it as it is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # answer question
    system_prompt = (
        "you are an assistant for question-answering tasks."
        "use the following pieces of retrieved context to answer"
        "the question. if you don't know the answer, say that you"
        "don't know. use three sentences maximum and keep the answer concise"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    user_input = st.text_input("your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
        st.write(st.session_state.store)
        st.write("assistant:", response["answer"])
        st.write("chat history", session_history.messages)
