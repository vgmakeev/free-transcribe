import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from free_transcribe.gemini import _prompt, load_glossary, proofread_segments


class GeminiProofreadingTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            SimpleNamespace(start=0, end=2, speaker="Владимир", text="Открой джиру"),
            SimpleNamespace(start=2, end=4, speaker="Иван", text="Проверю свагер"),
        ]

    def test_prompt_contains_complete_dialogue_and_speaker_labels(self):
        prompt = _prompt(self.segments, {"terms": ["Jira", "Swagger"]})

        self.assertIn("Владимир", prompt)
        self.assertIn("Иван", prompt)
        self.assertIn("Открой джиру", prompt)
        self.assertIn("Проверю свагер", prompt)

    def test_applies_only_unique_exact_non_overlapping_edits(self):
        response = {
            "edits": [
                {"id": 0, "old": "джиру", "new": "Jira"},
                {"id": 1, "old": "свагер", "new": "Swagger"},
                {"id": 1, "old": "не существует", "new": "API"},
            ]
        }
        with patch(
            "free_transcribe.gemini._generate",
            return_value=(response, {"totalTokenCount": 42}),
        ):
            result = proofread_segments(self.segments, api_key="test-key")

        self.assertEqual(result.texts, ["Открой Jira", "Проверю Swagger"])
        self.assertEqual(result.accepted_edits, 2)
        self.assertEqual(result.rejected_edits, 1)

    def test_operator_glossary_replaces_bundled_glossary(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as glossary:
            json.dump({"terms": ["InternalName"]}, glossary)
            glossary.flush()
            self.assertEqual(load_glossary(glossary.name), {"terms": ["InternalName"]})


if __name__ == "__main__":
    unittest.main()
