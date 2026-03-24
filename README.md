# PDF Q&A (Conversational RAG)

Upload PDFs, embed them with Hugging Face sentence-transformers, and chat via Groq's `llama-3.1-8b-instant` with chat history aware retrieval.

## Quickstart
1) Python 3.10+ recommended. Create a venv.
2) `pip install -r requirements.txt`
3) Set env var `HF_TOKEN` (Hugging Face access token with model read). You can place it in a local `.env`.
4) Run: `streamlit run app.py`
5) In the UI, enter your Groq API key, upload one or more PDFs, then ask questions.

## Deployment (Streamlit Community Cloud)
1) Push this repo to GitHub (done once locally), then create a new Streamlit app from the repo.
2) In Streamlit app settings, set **Secrets** or **Environment variables**:
   - `HF_TOKEN`: your Hugging Face token
3) Set **Main file path** to `app.py`.
4) Deploy. At runtime, enter your Groq API key in the textbox when prompted.

## Deployment (Render free web service)
1) In Render, create a new **Web Service** from this GitHub repo.
2) Build command: `pip install -r requirements.txt`
3) Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4) Add environment variables: `HF_TOKEN` (required). Users will still paste Groq key in the UI.
5) Deploy; Render exposes a public URL.

## Notes
- `.env`, `venv/`, and Streamlit/IDE artifacts are git-ignored.
- Groq key is requested at runtime and never stored server-side.
- Vector store is kept in-memory per session using Chroma's ephemeral store.
