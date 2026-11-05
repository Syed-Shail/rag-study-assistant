"""Streamlit user interface for the RAG Study Assistant."""

import sys
from pathlib import Path

# Streamlit can put the script directory (app/) on sys.path without its parent.
# Add the project root so imports such as `from app.config ...` work reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.config import get_settings
from app.ingestion import SUPPORTED_TYPES, get_collection, ingest_document
from app.qa import answer_question

st.set_page_config(page_title="RAG Study Assistant", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1050px; padding-top: 2rem;}
      [data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.18); border-radius: 14px; padding: .8rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 RAG Study Assistant")
st.caption("Ask questions about your notes and course documents. Answers are grounded in your sources and include citations.")

try:
    settings = get_settings()
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.info("Copy `.env.example` to `.env`, make sure Ollama is running, then restart the app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Study library")
    uploads = st.file_uploader(
        "Add study material",
        type=sorted(SUPPORTED_TYPES),
        accept_multiple_files=True,
        help=f"PDF, DOCX, and TXT files up to {settings.max_file_mb} MB each.",
    )
    if st.button("Index selected files", type="primary", disabled=not uploads, use_container_width=True):
        for upload in uploads or []:
            if upload.size > settings.max_file_mb * 1024 * 1024:
                st.error(f"{upload.name} exceeds the {settings.max_file_mb} MB limit.")
                continue
            try:
                with st.spinner(f"Reading and indexing {upload.name}…"):
                    result = ingest_document(upload.name, upload.getvalue(), settings)
                st.success(f"Indexed {result.filename}: {result.chunks_added} chunks from {result.pages_read} page(s).")
            except Exception as exc:
                st.error(f"Could not index {upload.name}: {exc}")

    try:
        collection = get_collection(settings)
        records = collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in records.get("metadatas", []) or []:
            name = str(metadata.get("source", "Unknown"))
            counts[name] = counts.get(name, 0) + 1
        if counts:
            st.subheader("Indexed documents")
            for name, count in sorted(counts.items()):
                st.caption(f"{name} · {count} chunks")
        else:
            st.caption("No documents indexed yet.")
        if counts and st.button("Clear study library", use_container_width=True):
            collection.delete(where={"source": {"$in": list(counts)}})
            st.session_state.messages = []
            st.rerun()
    except Exception as exc:
        st.warning(f"Could not read the local study library: {exc}")

    st.divider()
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Your indexed text stays in this project folder. Chat and embeddings use the Ollama service at the configured URL.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved source passages"):
                for source in message["sources"]:
                    st.markdown(f"**{source['filename']} · page {source['page']}**")
                    st.caption(source["text"])

if not st.session_state.messages:
    st.markdown("#### Start with a question")
    st.markdown("Try: *Explain the key ideas in my lecture notes* or *What is the difference between X and Y?*")

question = st.chat_input("Ask about your uploaded study material…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your study material…"):
                result = answer_question(question, st.session_state.messages[:-1], settings)
            st.markdown(result.text)
            if result.sources:
                with st.expander("Retrieved source passages"):
                    for source in result.sources:
                        st.markdown(f"**{source.filename} · page {source.page}**")
                        st.caption(source.text)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.text, "sources": [source.__dict__ for source in result.sources]}
            )
        except Exception as exc:
            error = f"I couldn’t complete that request: {exc}"
            st.error(error)
            st.info(
                f"Check that Ollama is running at `{settings.ollama_base_url}` and that both configured models are installed. "
                f"You can install them with `ollama pull {settings.chat_model}` and `ollama pull {settings.embedding_model}`."
            )
            st.session_state.messages.append({"role": "assistant", "content": error, "sources": []})
