"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

PROVIDERS = ("openai", "ollama")
DEFAULT_MODELS = {
    "openai": {"chat": "gpt-4o-mini", "embedding": "text-embedding-3-small"},
    "ollama": {"chat": "phi3:latest", "embedding": "nomic-embed-text"},
}
# Largest cosine distance at which the closest passage still counts as evidence. Each embedding
# model spreads distances differently, so calibrate with `python -m eval.run_eval` after changing models.
# nomic-embed-text: calibrated on the eval development split. OpenAI: conservative default.
DEFAULT_CUTOFF = {"openai": 0.72, "ollama": 0.29}
RETRIEVAL_MODES = ("hybrid", "dense", "keyword")


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "phi3:latest"
    embedding_model: str = "nomic-embed-text"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    collection_name: str = "study_materials_ollama"
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 5
    max_file_mb: int = 25
    relevance_cutoff: float = 0.72
    retrieval_mode: str = "hybrid"
    repair_citations: bool = True


def resolve_provider(value: str | None, has_openai_key: bool) -> str:
    """Pick the model provider. "auto" uses OpenAI when a key is configured, otherwise local Ollama."""
    choice = (value or "auto").strip().lower()
    if choice == "auto":
        return "openai" if has_openai_key else "ollama"
    if choice not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of auto, {', '.join(PROVIDERS)}; got {value!r}.")
    if choice == "openai" and not has_openai_key:
        raise ValueError("LLM_PROVIDER=openai needs OPENAI_API_KEY to be set.")
    return choice


def get_settings() -> Settings:
    """Return validated settings, supporting a local .env file via python-dotenv."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT_DIR / ".env")
    except ImportError:
        pass
    provider = resolve_provider(os.getenv("LLM_PROVIDER"), bool(os.getenv("OPENAI_API_KEY")))
    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    if chunk_size < 100 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_SIZE must be at least 100 and CHUNK_OVERLAP must be smaller.")
    chroma_dir = Path(os.getenv("CHROMA_DIR", str(ROOT_DIR / "data" / "chroma")))
    if not chroma_dir.is_absolute():
        chroma_dir = ROOT_DIR / chroma_dir
    # Each provider produces vectors of a different size, so each always gets its own collection.
    base_name = os.getenv("CHROMA_COLLECTION") or "study_materials"
    for name in PROVIDERS:
        base_name = base_name.removesuffix(f"_{name}")
    collection_name = f"{base_name}_{provider}"
    prefix = provider.upper()
    retrieval_mode = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"RETRIEVAL_MODE must be one of {', '.join(RETRIEVAL_MODES)}.")
    return Settings(
        provider=provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        chat_model=os.getenv(f"{prefix}_CHAT_MODEL", DEFAULT_MODELS[provider]["chat"]),
        embedding_model=os.getenv(f"{prefix}_EMBEDDING_MODEL", DEFAULT_MODELS[provider]["embedding"]),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=max(1, min(10, int(os.getenv("RETRIEVAL_K", "5")))),
        max_file_mb=max(1, int(os.getenv("MAX_FILE_MB", "25"))),
        relevance_cutoff=float(os.getenv("RELEVANCE_CUTOFF") or DEFAULT_CUTOFF[provider]),
        retrieval_mode=retrieval_mode,
        repair_citations=os.getenv("REPAIR_CITATIONS", "true").strip().lower() not in ("0", "false", "no"),
    )
