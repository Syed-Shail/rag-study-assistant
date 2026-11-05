from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.ingestion import ingest_document
from app.qa import answer_question


class KeywordEmbeddings:
    TERMS = ("mitochondria", "photosynthesis", "gravity")

    def __init__(self, *args, **kwargs):
        pass

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if term in lowered else 0.0 for term in self.TERMS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LocalRagFlowTests(unittest.TestCase):
    def test_ingest_retrieve_answer_and_trace_source_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                chroma_dir=Path(directory) / "chroma",
                collection_name="full_rag_flow",
            )
            chat_prompts = []

            def generate(prompt_value):
                messages = prompt_value.to_messages()
                chat_prompts.extend(messages)
                return AIMessage(content="Mitochondria produce ATP during cellular respiration [Source: biology.txt, p. 1].")

            with patch("app.ingestion.StudyEmbeddings", KeywordEmbeddings), patch(
                "app.qa.StudyEmbeddings", KeywordEmbeddings
            ), patch("app.qa.ChatOllama", return_value=RunnableLambda(generate)):
                indexed = ingest_document(
                    "biology.txt",
                    b"Mitochondria produce ATP during cellular respiration.\fPhotosynthesis converts light into chemical energy.",
                    settings,
                )
                answer = answer_question("What do mitochondria produce?", [], settings)

        self.assertEqual(indexed.pages_read, 2)
        self.assertEqual(indexed.chunks_added, 2)
        self.assertIn("[Source: biology.txt, p. 1]", answer.text)
        self.assertEqual([(source.filename, source.page) for source in answer.sources], [("biology.txt", 1)])
        retrieved_material = chat_prompts[0].content.split("Retrieved study material:\n", 1)[1]
        self.assertIn("Mitochondria produce ATP", retrieved_material)
        self.assertNotIn("Photosynthesis converts", retrieved_material)


if __name__ == "__main__":
    unittest.main()
