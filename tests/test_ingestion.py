from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chromadb
from docx import Document
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.config import Settings
from app.ingestion import StudyEmbeddings, get_collection, ingest_document, parse_document


def make_text_pdf(pages: list[str]) -> bytes:
    writer = PdfWriter()
    for value in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        content = DecodedStreamObject()
        escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class ParseDocumentTests(unittest.TestCase):
    def test_text_file_preserves_form_feed_page_boundaries_and_paragraphs(self):
        pages = parse_document("notes.TXT", "First paragraph\n\nSecond line\fNext page".encode())
        self.assertEqual([p.page for p in pages], [1, 2])
        self.assertIn("First paragraph\n\nSecond line", pages[0].text)
        self.assertEqual(pages[1].text, "Next page")

    def test_text_file_decodes_windows_1252(self):
        pages = parse_document("notes.txt", b"caf\xe9")
        self.assertEqual(pages[0].text, "café")

    def test_pdf_extracts_text_and_page_numbers(self):
        pages = parse_document("lecture.pdf", make_text_pdf(["First page text", "Second page text"]))
        self.assertEqual([(p.page, p.text) for p in pages], [(1, "First page text"), (2, "Second page text")])

    def test_blank_or_scanned_pdf_is_rejected_as_not_readable(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(ValueError, "Scanned PDFs need OCR"):
            parse_document("scan.pdf", output.getvalue())

    def test_malformed_pdf_gets_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Could not read this PDF"):
            parse_document("broken.pdf", b"not a pdf")

    def test_docx_extracts_paragraphs_and_tables(self):
        document = Document()
        document.add_paragraph("Course concept")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Term"
        table.cell(0, 1).text = "Definition"
        output = BytesIO()
        document.save(output)
        page = parse_document("course.docx", output.getvalue())[0]
        self.assertIn("Course concept", page.text)
        self.assertIn("Term | Definition", page.text)
        self.assertEqual(page.page, 1)

    def test_empty_and_unsupported_uploads_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_document("empty.txt", b"")
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            parse_document("image.png", b"data")


class StudyEmbeddingTests(unittest.TestCase):
    def test_nomic_models_receive_document_and_query_task_prefixes(self):
        with patch("app.ingestion.OllamaEmbeddings") as ollama:
            embeddings = StudyEmbeddings("nomic-embed-text", "http://localhost:11434")
            embeddings.embed_documents(["lecture notes"])
            embeddings.embed_query("What are the notes about?")
        ollama.assert_called_once_with(model="nomic-embed-text", base_url="http://localhost:11434")
        client = ollama.return_value
        client.embed_documents.assert_called_once_with(["search_document: lecture notes"])
        client.embed_query.assert_called_once_with("search_query: What are the notes about?")

    def test_non_nomic_model_receives_unmodified_text(self):
        with patch("app.ingestion.OllamaEmbeddings") as ollama:
            embeddings = StudyEmbeddings("other-embedder", "http://localhost:11434")
            embeddings.embed_documents(["document"])
            embeddings.embed_query("question")
        ollama.return_value.embed_documents.assert_called_once_with(["document"])
        ollama.return_value.embed_query.assert_called_once_with("question")


    def test_openai_provider_uses_openai_embeddings_without_task_prefixes(self):
        with patch("langchain_openai.OpenAIEmbeddings") as openai_embeddings:
            embeddings = StudyEmbeddings("text-embedding-3-small", "http://unused", provider="openai")
            embeddings.embed_documents(["lecture notes"])
            embeddings.embed_query("question")
        openai_embeddings.assert_called_once_with(model="text-embedding-3-small")
        openai_embeddings.return_value.embed_documents.assert_called_once_with(["lecture notes"])
        openai_embeddings.return_value.embed_query.assert_called_once_with("question")

class PersistentIngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            chroma_dir=Path(self.temp_dir.name) / "chroma",
            collection_name="test_study_materials",
            chunk_size=64,
            chunk_overlap=12,
        )
        self.embedder_patch = patch("app.ingestion.StudyEmbeddings", FakeEmbeddings)
        self.embedder_patch.start()

    def tearDown(self):
        self.embedder_patch.stop()
        self.temp_dir.cleanup()

    def test_chunks_text_and_persists_source_page_metadata(self):
        text = ("Retrieval augmented generation combines search and generation. " * 14).encode()
        result = ingest_document("folder/lecture.txt", text, self.settings)
        self.assertGreater(result.chunks_added, 1)
        self.assertEqual(result.filename, "lecture.txt")
        collection = get_collection(self.settings)
        records = collection.get(include=["documents", "metadatas"])
        self.assertEqual(collection.count(), result.chunks_added)
        self.assertTrue(all(len(document) <= 64 for document in records["documents"]))
        self.assertTrue(all(item["source"] == "lecture.txt" and item["page"] == 1 for item in records["metadatas"]))
        # A newly opened client can read the same on-disk collection.
        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self.assertEqual(client.get_collection("test_study_materials").count(), result.chunks_added)

    def test_reupload_replaces_old_chunks_for_same_filename(self):
        ingest_document("notes.txt", b"old material " * 30, self.settings)
        ingest_document("notes.txt", b"replacement material only", self.settings)
        records = get_collection(self.settings).get(include=["documents", "metadatas"])
        self.assertEqual(len(records["documents"]), 1)
        self.assertEqual(records["documents"][0], "replacement material only")

    def test_failed_embedding_keeps_previously_indexed_document(self):
        ingest_document("notes.txt", b"original notes", self.settings)
        with patch("app.ingestion.StudyEmbeddings", side_effect=RuntimeError("Ollama unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Ollama unavailable"):
                ingest_document("notes.txt", b"new notes", self.settings)
        records = get_collection(self.settings).get(include=["documents"])
        self.assertEqual(records["documents"], ["original notes"])

    def test_pdf_page_numbers_survive_chunking(self):
        pdf = make_text_pdf(["biology cell membrane", "physics wave motion"])
        ingest_document("course.pdf", pdf, self.settings)
        records = get_collection(self.settings).get(include=["documents", "metadatas"])
        by_page = {meta["page"] for meta in records["metadatas"]}
        self.assertEqual(by_page, {1, 2})

    def test_more_than_one_thousand_pages_are_embedded_in_bounded_batches(self):
        batch_sizes = []

        class TrackingEmbeddings(FakeEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                batch_sizes.append(len(texts))
                return super().embed_documents(texts)

        page_count = 1024
        raw = "\f".join(f"Biology note page {page}: mitochondria produce ATP." for page in range(page_count)).encode()
        with patch("app.ingestion.StudyEmbeddings", TrackingEmbeddings):
            result = ingest_document("large-course.txt", raw, self.settings)
        records = get_collection(self.settings).get(include=["metadatas"])
        self.assertEqual(result.pages_read, page_count)
        self.assertEqual(result.chunks_added, page_count)
        self.assertEqual(len(records["metadatas"]), page_count)
        self.assertEqual(max(batch_sizes), 64)
        self.assertEqual(batch_sizes[-1], 64)
        self.assertEqual(max(item["page"] for item in records["metadatas"]), page_count)


if __name__ == "__main__":
    unittest.main()
