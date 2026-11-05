import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitSmokeTests(unittest.TestCase):
    def test_app_starts_and_shows_chat_and_upload_controls_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "PYTHON_DOTENV_DISABLED": "1",
                "CHROMA_DIR": directory,
                "CHROMA_COLLECTION": "streamlit_smoke_test",
            },
            clear=True,
        ):
            app = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0, [str(item.value) for item in app.exception])
        self.assertTrue(any("RAG Study Assistant" in title.value for title in app.title))
        self.assertTrue(any("Add study material" in uploader.label for uploader in app.get("file_uploader")))
        self.assertEqual(len(app.chat_input), 1)


if __name__ == "__main__":
    unittest.main()
