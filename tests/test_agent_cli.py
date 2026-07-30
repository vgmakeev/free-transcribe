import unittest

from whisper_transcribe.agent_cli import (
    ASR_SCHEMA,
    DIARIZATION_SCHEMA,
    TRANSCRIPT_SCHEMA,
    asr_artifact,
    diarization_artifact,
    merge_artifacts,
    transcript_from_artifact,
)
from whisper_transcribe.core import (
    DiarizationResult,
    SpeakerTurn,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)


class AgentArtifactTests(unittest.TestCase):
    def setUp(self):
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
        asr = asr_artifact(self.asr_result, "meeting.wav")
        diarization = diarization_artifact(
            self.diarization_result, "meeting.wav"
        )
        merged = merge_artifacts(asr, diarization, ["Alice", "Bob"])
        result = transcript_from_artifact(merged)

        self.assertEqual(asr["schema"], ASR_SCHEMA)
        self.assertEqual(diarization["schema"], DIARIZATION_SCHEMA)
        self.assertEqual(merged["schema"], TRANSCRIPT_SCHEMA)
        self.assertEqual(
            [segment.speaker for segment in result.segments], ["Alice", "Bob"]
        )
        self.assertEqual(result.diarization_model, "test/diarization")

    def test_rejects_different_sources(self):
        asr = asr_artifact(self.asr_result, "one.wav")
        diarization = diarization_artifact(self.diarization_result, "two.wav")

        with self.assertRaisesRegex(ValueError, "different sources"):
            merge_artifacts(asr, diarization)


if __name__ == "__main__":
    unittest.main()
