# Conversational RAG System with PDF Uploads  
Streamlit + LangChain + ChromaDB + Groq (LLaMA 3.1)

---

## 1. Introduction

This project implements a **Conversational Retrieval-Augmented Generation (RAG) system** that enables users to upload PDF documents and interact with them through natural language queries.

Unlike traditional Q&A systems, this application:
- Maintains session-based conversational memory  
- Reformulates follow-up queries into context-aware standalone questions  
- Retrieves relevant document chunks using vector search  
- Generates concise answers using an LLM  

The system is designed for interactive document understanding.

---

## 2. Key Features

### Core Functionality
- Multi-PDF upload and processing  
- Semantic search using embeddings  
- Conversational memory with session tracking  
- Context-aware question reformulation  
- Retrieval-Augmented Generation pipeline  

### System Capabilities
- Handles follow-up questions intelligently  
- Reduces hallucination using retrieved context  
- Generates concise answers (≤ 3 sentences)  
- Supports multiple user sessions  

---

## 3. System Architecture

```
User Input (Query)
        │
        ▼
Chat History + Query
        │
        ▼
History-Aware Question Reformulation
        │
        ▼
Retriever (Chroma Vector DB)
        │
        ▼
Relevant Document Chunks
        │
        ▼
LLM (Groq - LLaMA 3.1)
        │
        ▼
Final Answer
```

---

## 4. Tech Stack

| Component         | Technology Used |
|------------------|----------------|
| Frontend UI      | Streamlit      |
| LLM              | Groq (LLaMA 3.1-8B-Instant) |
| Framework        | LangChain      |
| Vector Database  | ChromaDB       |
| Embeddings       | HuggingFace (all-MiniLM-L6-v2) |
| Document Loader  | PyPDFLoader    |
| Text Splitting   | RecursiveCharacterTextSplitter |

---

## 5. Installation Guide

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd <your-repo>
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Environment Configuration

Create a `.env` file:

```bash
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

---

## 7. Running the Application

```bash
streamlit run app.py
```

---

## 8. Detailed Code Explanation

### Embedding Initialization
```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

### PDF Loading
```python
loader = PyPDFLoader(temppdf)
docs = loader.load()
```

### Text Splitting
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=200
)
```

### Vector Store Creation
```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
```

### Retriever
```python
retriever = vectorstore.as_retriever()
```

### History-Aware Retrieval
```python
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

### QA Chain
```python
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
```

### RAG Pipeline
```python
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)
```

### Chat History Management
```python
st.session_state.store = {}
```

```python
ChatMessageHistory()
```

### Stateful Execution
```python
RunnableWithMessageHistory(...)
```

---

## 9. Example Usage

### Input
```
What is the main topic of the document?
```

### Follow-up
```
Explain it in simpler terms
```

---

## 10. Limitations

- Uses fixed filename (`temp.pdf`) → risk of overwrite  
- No persistence for vector database  
- Large chunk size reduces retrieval accuracy  
- Limited error handling  
- In-memory session storage  

---

## 11. Improvements

- Use unique temp filenames  
- Persist ChromaDB  
- Optimize chunk size (1000–1500 recommended)  
- Add metadata filtering  
- Implement streaming responses  
- Add logging and error handling  

---

## 12. Deployment Options

- Streamlit Cloud  
- Hugging Face Spaces  
- AWS / GCP  

---


## 14. License

Add your preferred license (e.g., MIT License).
