from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.ingestion import get_collection
from app.qa import answer_question


class FakeQueryEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, question: str) -> list[float]:
        return [1.0, 0.0]


class GroundedQaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings = Settings(
            chroma_dir=Path(self.temp_dir.name) / "chroma",
            collection_name="test_qa",
            retrieval_k=3,
        )
        self.collection = get_collection(self.settings)
        self.collection.add(
            ids=["supported-1", "unrelated-1"],
            documents=["Mitochondria produce most cellular ATP during aerobic respiration.", "Baking uses flour, yeast, and heat."],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[
                {"source": "biology.pdf", "page": 4, "chunk": 1},
                {"source": "cooking.txt", "page": 1, "chunk": 1},
            ],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_answers_from_retrieved_context_with_citations_and_history(self):
        seen = {}

        def local_model(prompt_value):
            seen["messages"] = prompt_value.to_messages()
            return AIMessage(content="Mitochondria produce most ATP during aerobic respiration [Source: biology.pdf, p. 4].")

        with patch("app.qa.StudyEmbeddings", FakeQueryEmbeddings), patch(
            "app.qa.ChatOllama", return_value=RunnableLambda(local_model)
        ) as chat:
            result = answer_question(
                "Where is most ATP produced?",
                [{"role": "user", "content": "Tell me about cells."}, {"role": "assistant", "content": "Cells have organelles."}],
                self.settings,
            )
        self.assertIn("[Source: biology.pdf, p. 4]", result.text)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].filename, "biology.pdf")
        system = seen["messages"][0].content
        human = seen["messages"][1].content
        self.assertIn("Few-shot examples", system)
        self.assertIn("Mitochondria produce most cellular ATP", system)
        self.assertNotIn("Baking uses flour", system)
        self.assertIn("[Source: biology.pdf, p. 4", system)
        self.assertIn("Cells have organelles", human)
        chat.assert_called_once_with(model="phi3:latest", base_url="http://localhost:11434", temperature=0, num_predict=180)

    def test_unrelated_retrieval_abstains_without_calling_chat_model(self):
        self.collection.update(ids=["supported-1", "unrelated-1"], embeddings=[[0.0, 1.0], [0.0, 1.0]])
        with patch("app.qa.StudyEmbeddings", FakeQueryEmbeddings), patch("app.qa.ChatOllama") as chat:
            result = answer_question("How do I bake sourdough?", [], self.settings)
        self.assertIn("couldn’t find relevant information", result.text)
        self.assertEqual(result.sources, [])
        chat.assert_not_called()

    def test_empty_library_fails_with_clear_instruction(self):
        self.collection.delete(ids=["supported-1", "unrelated-1"])
        with self.assertRaisesRegex(ValueError, "Upload at least one study document"):
            answer_question("question", [], self.settings)


if __name__ == "__main__":
    unittest.main()
