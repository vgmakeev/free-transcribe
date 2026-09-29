import threading
import time
import unittest
from pathlib import Path
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
            self.assertEqual(set(health["ready"]["engines"]), {"parakeet"})
            self.assertIsInstance(health["ready"]["speakers"], bool)
            self.assertIsInstance(health["ready"]["gemini"], bool)

    def test_gemini_request_requires_server_key(self):
        app = create_app()
        with (
            patch("free_transcribe.api.gemini_available", return_value=False),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/transcriptions",
                files={"file": ("sample.wav", b"audio")},
                data={"gemini": "true"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

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


def _wait_finished(client, job_id, headers=None):
    status = {}
    for _ in range(200):
        status = client.get(f"/v1/transcriptions/{job_id}", headers=headers).json()
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.01)
    return status


@unittest.skipIf(TestClient is None, "API dependencies are not installed")
class RetentionTests(unittest.TestCase):
    result = None

    def setUp(self):
        self.result = TranscriptResult(
            text="Итог",
            segments=[TranscriptSegment(0.0, 1.0, "Итог")],
            language="ru",
            duration_min=1 / 60,
            device="test",
            model="test/model",
        )
        self.now = 1000.0
        self.sources = []

    def clock(self):
        return self.now

    def recording_transcription(self, source, **_kwargs):
        self.sources.append(Path(source))
        return self.result

    def test_media_is_deleted_and_result_expires(self):
        app = create_app(result_ttl_hours=24, clock=self.clock)
        with (
            patch(
                "free_transcribe.api.transcribe_file",
                side_effect=self.recording_transcription,
            ),
            TestClient(app) as client,
        ):
            job_id = client.post(
                "/v1/transcriptions", files={"file": ("meeting.ogg", b"audio")}
            ).json()["id"]
            status = _wait_finished(client, job_id)

            self.assertEqual(status["status"], "succeeded")
            self.assertIn("expires_at", status)
            self.assertFalse(self.sources[0].exists())
            work_dir = self.sources[0].parent
            self.assertTrue((work_dir / "transcript.md").exists())

            self.now += 23 * 3600
            result = client.get(f"/v1/transcriptions/{job_id}/result")
            self.assertEqual(result.status_code, 200)
            self.assertIn("Итог", result.text)

            self.now += 3600
            self.assertEqual(client.get(f"/v1/transcriptions/{job_id}").status_code, 404)
            self.assertFalse(work_dir.exists())

    def test_failed_job_deletes_media(self):
        def failing(source, **_kwargs):
            self.sources.append(Path(source))
            raise RuntimeError("decoder crashed")

        app = create_app(result_ttl_hours=24, clock=self.clock)
        with (
            patch("free_transcribe.api.transcribe_file", side_effect=failing),
            TestClient(app) as client,
        ):
            job_id = client.post(
                "/v1/transcriptions", files={"file": ("meeting.ogg", b"audio")}
            ).json()["id"]
            status = _wait_finished(client, job_id)

            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error"], "decoder crashed")
            self.assertFalse(self.sources[0].exists())

    def test_zero_ttl_keeps_results(self):
        app = create_app(result_ttl_hours=0, clock=self.clock)
        with (
            patch(
                "free_transcribe.api.transcribe_file",
                side_effect=self.recording_transcription,
            ),
            TestClient(app) as client,
        ):
            job_id = client.post(
                "/v1/transcriptions", files={"file": ("meeting.ogg", b"audio")}
            ).json()["id"]
            status = _wait_finished(client, job_id)
            self.assertNotIn("expires_at", status)

            self.now += 365 * 24 * 3600
            response = client.get(f"/v1/transcriptions/{job_id}/result")
            self.assertEqual(response.status_code, 200)

    def test_negative_ttl_is_rejected(self):
        with self.assertRaises(ValueError):
            create_app(result_ttl_hours=-1)


if __name__ == "__main__":
    unittest.main()
