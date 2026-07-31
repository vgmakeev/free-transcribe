import os
import tempfile
import unittest

from free_transcribe.artifacts import (
    ASR_SCHEMA,
    DIARIZATION_SCHEMA,
    SPEAKER_LABELS_SCHEMA,
    TRANSCRIPT_SCHEMA,
    apply_speaker_labels,
    asr_artifact,
    diarization_artifact,
    merge_artifacts,
    transcript_from_artifact,
)
from free_transcribe.core import (
    DiarizationResult,
    SpeakerTurn,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)


class AgentArtifactTests(unittest.TestCase):
    def setUp(self):
        with tempfile.NamedTemporaryFile(delete=False) as media:
            media.write(b"same media")
            self.media_path = media.name
        self.addCleanup(os.unlink, self.media_path)
        self.asr_result = TranscriptResult(
            text="Hello there",
            segments=[TranscriptSegment(0.0, 1.0, "Hello there")],
            words=[
                TranscriptWord(0.0, 0.4, "Hello"),
                TranscriptWord(0.6, 1.0, "there"),
            ],
            language="English",
            duration_min=1 / 60,
            device="mlx",
            model="test/asr",
            engine="qwen",
        )
        self.diarization_result = DiarizationResult(
            turns=[
                SpeakerTurn(0.0, 0.5, "A"),
                SpeakerTurn(0.5, 1.0, "B"),
            ],
            exclusive_turns=[
                SpeakerTurn(0.0, 0.5, "A"),
                SpeakerTurn(0.5, 1.0, "B"),
            ],
            model="test/diarization",
            device="mps",
        )

    def test_round_trip_artifacts(self):
        asr = asr_artifact(self.asr_result, self.media_path)
        diarization = diarization_artifact(
            self.diarization_result, self.media_path
        )
        merged = merge_artifacts(asr, diarization, ["Alice", "Bob"])
        result = transcript_from_artifact(merged)

        self.assertEqual(asr["schema"], ASR_SCHEMA)
        self.assertEqual(diarization["schema"], DIARIZATION_SCHEMA)
        self.assertEqual(merged["schema"], TRANSCRIPT_SCHEMA)
        self.assertTrue(asr["id"].startswith("sha256:"))
        self.assertEqual(asr["timestamps"], "word")
        self.assertEqual(asr["source"]["sha256"], diarization["source"]["sha256"])
        self.assertEqual(merged["parents"], [asr["id"], diarization["id"]])
        self.assertEqual(
            [segment["speaker"] for segment in merged["segments"]],
            ["Speaker 1", "Speaker 2"],
        )
        self.assertEqual(merged["speakers"][0]["identity"]["name"], "Alice")
        self.assertEqual(
            [segment.speaker for segment in result.segments], ["Alice", "Bob"]
        )
        self.assertEqual(result.diarization_model, "test/diarization")

    def test_rejects_different_sources(self):
        asr = asr_artifact(self.asr_result, self.media_path)
        with tempfile.NamedTemporaryFile() as other_media:
            other_media.write(b"different media")
            other_media.flush()
            diarization = diarization_artifact(
                self.diarization_result, other_media.name
            )

        with self.assertRaisesRegex(ValueError, "different media"):
            merge_artifacts(asr, diarization)

    def test_applies_auditable_speaker_labels(self):
        asr = asr_artifact(self.asr_result, self.media_path)
        diarization = diarization_artifact(
            self.diarization_result, self.media_path
        )
        merged = merge_artifacts(asr, diarization)
        labels = {
            "schema": SPEAKER_LABELS_SCHEMA,
            "transcript_id": merged["id"],
            "speakers": [
                {
                    "id": "Speaker 2",
                    "identity": {
                        "name": "Victor",
                        "source": "inferred",
                        "confidence": 0.93,
                        "evidence": ["00:42 direct address"],
                    },
                }
            ],
        }

        labelled = apply_speaker_labels(merged, labels)
        rendered = transcript_from_artifact(labelled)

        self.assertEqual(labelled["parents"], [merged["id"]])
        self.assertNotEqual(labelled["id"], merged["id"])
        self.assertEqual(rendered.segments[1].speaker, "Victor")

    def test_inferred_identity_requires_evidence(self):
        asr = asr_artifact(self.asr_result, self.media_path)
        diarization = diarization_artifact(
            self.diarization_result, self.media_path
        )
        merged = merge_artifacts(asr, diarization)
        labels = {
            "schema": SPEAKER_LABELS_SCHEMA,
            "transcript_id": merged["id"],
            "speakers": [
                {
                    "id": "Speaker 2",
                    "identity": {
                        "name": "Victor",
                        "source": "inferred",
                        "confidence": 0.5,
                        "evidence": [],
                    },
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "requires evidence"):
            apply_speaker_labels(merged, labels)


if __name__ == "__main__":
    unittest.main()
