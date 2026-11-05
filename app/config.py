"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "phi3:latest"
    embedding_model: str = "nomic-embed-text"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    collection_name: str = "study_materials_ollama"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    max_file_mb: int = 25


def get_settings() -> Settings:
    """Return validated settings, supporting a local .env file via python-dotenv."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT_DIR / ".env")
    except ImportError:
        pass
    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    if chunk_size < 100 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_SIZE must be at least 100 and CHUNK_OVERLAP must be smaller.")
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(ROOT_DIR / "data" / "chroma")))
    if not chroma_dir.is_absolute():
        chroma_dir = ROOT_DIR / chroma_dir
    collection_name = os.getenv("CHROMA_COLLECTION", "study_materials_ollama")
    # The previous project version used this default for OpenAI vectors.
    if collection_name == "study_materials":
        collection_name = "study_materials_ollama"
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "phi3:latest"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=max(1, min(10, int(os.getenv("RETRIEVAL_K", "5")))),
        max_file_mb=max(1, int(os.getenv("MAX_FILE_MB", "25"))),
    )
