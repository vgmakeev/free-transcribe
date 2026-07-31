import time
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    from free_transcribe.api import create_app
    from free_transcribe.core import TranscriptResult, TranscriptSegment
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "API dependencies are not installed")
class ApiTests(unittest.TestCase):
    def test_authenticated_background_transcription(self):
        result = TranscriptResult(
            text="Привет",
            segments=[TranscriptSegment(0.0, 1.0, "Привет")],
            language="ru",
            duration_min=1 / 60,
            device="test",
            model="test/model",
        )
        app = create_app(token="secret")
        headers = {"Authorization": "Bearer secret"}

        with (
            patch("free_transcribe.api.transcribe_file", return_value=result),
            TestClient(app) as client,
        ):
            self.assertEqual(client.get("/health").status_code, 200)
            unauthorized = client.post(
                "/v1/transcriptions", files={"file": ("sample.wav", b"audio")}
            )
            self.assertEqual(unauthorized.status_code, 401)

            submitted = client.post(
                "/v1/transcriptions",
                headers=headers,
                files={"file": ("sample.wav", b"audio")},
                data={"language": "ru", "speakers": "false"},
            )
            self.assertEqual(submitted.status_code, 202)
            job_id = submitted.json()["id"]

            status = {}
            for _ in range(100):
                status = client.get(
                    f"/v1/transcriptions/{job_id}", headers=headers
                ).json()
                if status["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.01)

            self.assertEqual(status["status"], "succeeded")
            response = client.get(
                f"/v1/transcriptions/{job_id}/result", headers=headers
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Привет", response.text)
            self.assertEqual(
                client.delete(
                    f"/v1/transcriptions/{job_id}", headers=headers
                ).status_code,
                204,
            )


if __name__ == "__main__":
    unittest.main()
