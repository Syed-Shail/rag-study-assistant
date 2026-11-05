# RAG Study Assistant

A document based study assistant built around the project in the resume: Python, LangChain, ChromaDB, Ollama, and Streamlit. Upload PDF, DOCX, or TXT study material, index its text, ask questions in natural language, and inspect the passages used to answer. Chat and embedding models run through your local Ollama service; the app does not require an OpenAI API key.

## What it does

- Extracts text from PDF, DOCX, and TXT files and keeps page numbers where available.
- Splits documents into overlapping chunks, creates local Ollama embeddings, and stores text plus metadata in persistent ChromaDB.
- Retrieves the closest passages for each question and asks the chat model to answer only from retrieved material.
- Uses concise few-shot examples to demonstrate grounded answers, citation style, and when to abstain.
- Includes inline filename/page citations and an expandable view of the retrieved passages.
- Keeps chat history for follow-up questions, offers a new-chat control, and replaces old chunks when the same filename is re-uploaded.
- Rejects unsupported file types, empty/scanned documents, and files over the configured size limit.
- Embeds and stores chunks in bounded batches, including documents spanning 1,000+ pages.

## Run locally

Python 3.11 is recommended. Ollama must be installed and running.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Download the default chat and embedding models (skip a command if that model is already installed):

```bash
ollama pull phi3:latest
ollama pull nomic-embed-text
```

Confirm Ollama can see the models with `ollama list`, then run:

```bash
streamlit run app/main.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Model downloads need internet access, but after the models are present, document indexing and questions run locally through Ollama.

## Run with Docker

```bash
cp .env.example .env
ollama pull phi3:latest
ollama pull nomic-embed-text
docker compose up --build
```

Open `http://localhost:8501`. Chroma data persists in `data/chroma/` on the host. Docker Desktop uses `host.docker.internal` to reach Ollama on your laptop; if the container cannot connect, run the app locally or configure Ollama to accept connections from Docker and set `OLLAMA_BASE_URL` in `.env` accordingly.

## Run automated tests

With the project virtual environment active:

```bash
python3 -m unittest discover -s tests -v
```

The automated suite uses a temporary Chroma database and mocked model calls, so it does not require downloading models or call Ollama. It covers PDF/DOCX/TXT parsing, 1,024-page indexing and batching, Chroma persistence and replacement, failure recovery, Nomic prefixes, semantic retrieval, abstention, citations, chat history, the full mocked RAG flow, and Streamlit startup.

## Configuration

See `.env.example`. `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`, Chroma directory, collection name, and upload size limit can be configured there. `RETRIEVAL_K` is limited to 1-10. If you already created `.env` from the previous OpenAI version, change `CHROMA_COLLECTION` to `study_materials_ollama`; the old and new embedding models produce incompatible vectors.

## Accuracy and limitations

Answers are constrained to retrieved passages, use temperature zero, and cite source filename and page. Retrieval is semantic cosine search. Nomic embeddings receive the model's recommended query/document prefixes. The similarity cutoff avoids answering from obviously unrelated passages, but it cannot guarantee correctness; check the displayed passages for important work. PDF pages that are image scans need OCR before ingestion. DOCX files do not have reliable page boundaries, so citations point to page 1. Larger Ollama models can improve answer quality but need more RAM and disk space.

The local Chroma folder contains extracted study content. Keep it private and remove it when you no longer need the indexed material. Do not commit `.env` or `data/chroma/`.
