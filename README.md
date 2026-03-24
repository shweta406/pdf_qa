Conversational RAG with PDF Uploads (Streamlit + LangChain + Groq)
Overview

This project implements a Conversational Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content. The system maintains session-based chat history and generates context-aware responses using a large language model.

The application integrates LangChain for orchestration, ChromaDB for vector storage, HuggingFace embeddings for semantic search, and Groq (LLaMA 3.1) for inference, all served through a Streamlit interface.

Features
Upload and process multiple PDF files
Context-aware question reformulation using chat history
Session-based conversational memory
Semantic document retrieval using embeddings
Fast response generation using Groq LLM
Simple web interface using Streamlit
Tech Stack
Python
Streamlit
LangChain
ChromaDB
HuggingFace Embeddings (all-MiniLM-L6-v2)
Groq API (LLaMA 3.1)
Project Structure
.
├── app.py              # Main Streamlit application
├── .env                # Environment variables (API keys)
├── requirements.txt    # Project dependencies
└── README.md
Environment Variables

Create a .env file in the root directory and add the following:

HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key

You can also configure these values using Streamlit secrets.

Installation
git clone <your-repo-url>
cd <your-repo>

python -m venv venv
source venv/bin/activate       # For Windows: venv\Scripts\activate

pip install -r requirements.txt
Running the Application
streamlit run app.py
How It Works
1. Document Processing

Uploaded PDF files are loaded using PyPDFLoader and split into smaller chunks using RecursiveCharacterTextSplitter.

2. Embedding and Storage

The text chunks are converted into vector embeddings using HuggingFace embeddings and stored in a Chroma vector database.

3. Retrieval

User queries are reformulated into standalone questions using chat history. Relevant document chunks are retrieved from the vector store.

4. Response Generation

The retrieved context and user query are passed to the Groq LLM, which generates concise answers limited to three sentences.

5. Chat History

Session-based chat history is maintained using ChatMessageHistory and RunnableWithMessageHistory.

Example Workflow
Upload one or more PDF files
Enter a session ID
Ask a question related to the document
Ask follow-up questions that rely on previous context
Limitations
Temporary file handling uses a fixed filename (temp.pdf), which can cause conflicts
Vector database is not persisted across sessions
Performance may degrade with large PDF files
Limited error handling for invalid or corrupted files
Lightweight embedding model may reduce retrieval accuracy
Suggested Improvements
Use unique filenames for uploaded files to avoid overwriting
Persist ChromaDB for reuse across sessions
Optimize chunk size and overlap strategy
Add better error handling and validation
Implement streaming responses for improved UX
Add authentication for multi-user environments
Conclusion

This project demonstrates a functional implementation of a conversational RAG pipeline suitable for learning and prototyping. It is not optimized for production use and requires additional enhancements for scalability, reliability, and performance.
