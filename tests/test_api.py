import threading
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
    def test_web_ui_is_served(self):
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Free Transcribe", response.text)
            self.assertIn("default-src 'self'", response.headers["content-security-policy"])
            self.assertEqual(client.get("/assets/app.js").status_code, 200)
            health = client.get("/health").json()
            self.assertEqual(set(health["ready"]["engines"]), {"qwen", "parakeet"})
            self.assertIsInstance(health["ready"]["speakers"], bool)

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
            events = client.get(
                f"/v1/transcriptions/{job_id}/events", headers=headers
            )
            self.assertEqual(events.status_code, 200)
            self.assertTrue(events.headers["content-type"].startswith("text/event-stream"))
            self.assertIn('"status": "succeeded"', events.text)
            self.assertEqual(
                client.delete(
                    f"/v1/transcriptions/{job_id}", headers=headers
                ).status_code,
                204,
            )

    def test_queue_position_and_capacity(self):
        result = TranscriptResult(
            text="Queued result",
            segments=[TranscriptSegment(0.0, 1.0, "Queued result")],
            language="en",
            duration_min=1 / 60,
            device="test",
            model="test/model",
        )
        started = threading.Event()
        release = threading.Event()

        def blocking_transcription(*_args, **_kwargs):
            started.set()
            release.wait(timeout=3)
            return result

        app = create_app(concurrency=1, max_queue=1)
        with (
            patch(
                "free_transcribe.api.transcribe_file",
                side_effect=blocking_transcription,
            ),
            TestClient(app) as client,
        ):
            first = client.post(
                "/v1/transcriptions", files={"file": ("first.wav", b"audio")}
            )
            self.assertEqual(first.status_code, 202)
            self.assertTrue(started.wait(timeout=1))

            second = client.post(
                "/v1/transcriptions", files={"file": ("second.wav", b"audio")}
            )
            self.assertEqual(second.status_code, 202)
            second_status = client.get(
                f"/v1/transcriptions/{second.json()['id']}"
            ).json()
            self.assertEqual(second_status["status"], "queued")
            self.assertEqual(second_status["queue_position"], 1)

            rejected = client.post(
                "/v1/transcriptions", files={"file": ("third.wav", b"audio")}
            )
            self.assertEqual(rejected.status_code, 429)
            self.assertEqual(rejected.headers["retry-after"], "30")
            release.set()


if __name__ == "__main__":
    unittest.main()
