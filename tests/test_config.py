import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.config import get_settings


class ConfigurationTests(unittest.TestCase):
    def test_ollama_defaults_do_not_require_an_api_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "PYTHON_DOTENV_DISABLED": "1",
                "CHROMA_DIR": directory,
                "OLLAMA_BASE_URL": "http://localhost:11434/",
                "OLLAMA_CHAT_MODEL": "phi3:latest",
                "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
                "CHROMA_COLLECTION": "study_materials",
            },
            clear=True,
        ):
            settings = get_settings()
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434")
        self.assertEqual(settings.chat_model, "phi3:latest")
        self.assertEqual(settings.embedding_model, "nomic-embed-text")
        self.assertEqual(settings.chroma_dir, Path(directory))
        # Prevent reuse of the old collection that could contain OpenAI-sized vectors.
        self.assertEqual(settings.collection_name, "study_materials_ollama")

    def test_auto_provider_uses_openai_when_a_key_is_set(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"PYTHON_DOTENV_DISABLED": "1", "CHROMA_DIR": directory, "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            settings = get_settings()
        self.assertEqual(settings.provider, "openai")
        self.assertEqual(settings.chat_model, "gpt-4o-mini")
        self.assertEqual(settings.embedding_model, "text-embedding-3-small")
        # OpenAI and Ollama vectors have different sizes, so they never share a collection.
        self.assertEqual(settings.collection_name, "study_materials_openai")

    def test_provider_can_be_forced_to_local_ollama(self):
        with patch.dict(
            os.environ,
            {"PYTHON_DOTENV_DISABLED": "1", "OPENAI_API_KEY": "sk-test", "LLM_PROVIDER": "ollama"},
            clear=True,
        ):
            settings = get_settings()
        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.chat_model, "phi3:latest")

    def test_openai_provider_without_a_key_is_rejected(self):
        with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1", "LLM_PROVIDER": "openai"}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                get_settings()

    def test_unknown_provider_is_rejected(self):
        with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1", "LLM_PROVIDER": "bard"}, clear=True):
            with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
                get_settings()

    def test_invalid_chunk_settings_are_rejected(self):
        with patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1", "CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, clear=True):
            with self.assertRaisesRegex(ValueError, "CHUNK_OVERLAP"):
                get_settings()


if __name__ == "__main__":
    unittest.main()
