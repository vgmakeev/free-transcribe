import contextlib
import io
import json
import unittest
from unittest.mock import patch

from free_transcribe.cli import _normalized_argv, _speaker_request, main


class CliContractTests(unittest.TestCase):
    def test_media_path_is_implicit_run(self):
        self.assertEqual(
            _normalized_argv(["meeting.mp4", "--speakers"]),
            ["run", "meeting.mp4", "--speakers"],
        )

    def test_named_stage_is_unchanged(self):
        self.assertEqual(
            _normalized_argv(["asr", "meeting.mp4"]),
            ["asr", "meeting.mp4"],
        )

    def test_speakers_accepts_auto_or_count(self):
        self.assertEqual(_speaker_request("auto"), (True, None))
        self.assertEqual(_speaker_request("3"), (True, 3))

    def test_doctor_is_machine_readable(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("free_transcribe.cli.importlib.util.find_spec", return_value=None),
        ):
            main(["doctor"])

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["schema"], "free-transcribe/doctor/v1")
        self.assertIn("capabilities", payload)


if __name__ == "__main__":
    unittest.main()
