"""Document parsing, chunking, and persistent Chroma indexing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

SUPPORTED_TYPES = {"pdf", "docx", "txt"}


class StudyEmbeddings:
    """Ollama embeddings with the task prefixes expected by Nomic models."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.client = OllamaEmbeddings(model=model, base_url=base_url)
        self.nomic = model.lower().startswith("nomic-embed-text")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.nomic:
            texts = [f"search_document: {text}" for text in texts]
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.nomic:
            text = f"search_query: {text}"
        return self.client.embed_query(text)


@dataclass(frozen=True)
class ParsedPage:
    text: str
    page: int


@dataclass(frozen=True)
class IngestResult:
    filename: str
    chunks_added: int
    pages_read: int


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_document(filename: str, raw: bytes) -> list[ParsedPage]:
    """Extract text while retaining page numbers when the format supports them."""
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type .{extension or '(none)'}; use PDF, DOCX, or TXT.")
    if not raw:
        raise ValueError("The uploaded file is empty.")
    if extension == "txt":
        text = _decode_text(raw).strip()
        pages = [ParsedPage(part.strip(), index + 1) for index, part in enumerate(text.split("\f"))]
    elif extension == "pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw), strict=False)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ValueError("This PDF is password-protected and could not be opened.") from exc
            pages = [ParsedPage(page.extract_text() or "", index + 1) for index, page in enumerate(reader.pages)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read this PDF: {exc}") from exc
    else:
        try:
            from docx import Document

            document = Document(io.BytesIO(raw))
            blocks = [p.text.strip() for p in document.paragraphs if p.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    line = " | ".join(cell for cell in cells if cell)
                    if line:
                        blocks.append(line)
            pages = [ParsedPage("\n".join(blocks), 1)]
        except Exception as exc:
            raise ValueError(f"Could not read this DOCX: {exc}") from exc
    pages = [
        ParsedPage(re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", page.text)).strip(), page.page)
        for page in pages
    ]
    pages = [page for page in pages if page.text]
    if not pages:
        raise ValueError("No readable text was found. Scanned PDFs need OCR before they can be searched.")
    return pages


def get_collection(settings: Settings):
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine", "description": "Study materials indexed for question answering"},
    )


def _batches(items: list, size: int = 64):
    for offset in range(0, len(items), size):
        yield offset, items[offset : offset + size]


def ingest_document(filename: str, raw: bytes, settings: Settings) -> IngestResult:
    pages = parse_document(filename, raw)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    documents = []
    for page in pages:
        if len(page.text) <= settings.chunk_size:
            documents.append((page.text, page.page, 0))
            continue
        for chunk in splitter.create_documents([page.text], metadatas=[{"page": page.page}]):
            content = chunk.page_content.strip()
            if content:
                documents.append((content, page.page, int(chunk.metadata.get("start_index", 0))))
    if not documents:
        raise ValueError("No usable text chunks were produced from this file.")
    collection = get_collection(settings)
    safe_name = Path(filename.replace("\\", "/")).name
    # Re-uploading the same filename replaces its old chunks and avoids stale duplicates.
    embedder = StudyEmbeddings(settings.embedding_model, settings.ollama_base_url)
    texts = [text for text, _, _ in documents]
    embeddings = []
    # Bound each local Ollama request for large documents, including 1K+ page files.
    for _, text_batch in _batches(texts):
        embeddings.extend(embedder.embed_documents(text_batch))
    # Preserve an existing indexed copy if parsing or the embedding request fails.
    existing = collection.get(where={"source": safe_name}, include=[])
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)
    ids, metadatas = [], []
    for index, (text, page, start) in enumerate(documents):
        digest = hashlib.sha256(f"{safe_name}\0{text}\0{page}\0{start}".encode()).hexdigest()
        ids.append(digest)
        metadatas.append({"source": safe_name, "page": int(page), "chunk": index + 1, "start_index": int(start)})
    for offset, _ in _batches(texts):
        collection.upsert(
            ids=ids[offset : offset + 64],
            documents=texts[offset : offset + 64],
            embeddings=embeddings[offset : offset + 64],
            metadatas=metadatas[offset : offset + 64],
        )
    return IngestResult(safe_name, len(texts), len(pages))
