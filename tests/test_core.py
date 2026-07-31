import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from free_transcribe.core import (
    DEFAULT_MODELS,
    SpeakerTurn,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    _EngineOutput,
    _normalized_audio_path,
    _pyannote_progress_hook,
    _transcribe_parakeet,
    _transcribe_parakeet_cuda,
    _transcribe_qwen,
    _transcribe_qwen_mlx,
    assign_speakers_to_words,
    format_timestamp,
    restore_segment_punctuation,
    result_to_markdown,
    transcribe_file,
)


class SpeakerAssignmentTests(unittest.TestCase):
    def test_splits_one_asr_segment_at_speaker_change(self):
        words = [
            TranscriptWord(0.0, 0.4, " Hello"),
            TranscriptWord(0.4, 0.9, " there."),
            TranscriptWord(1.0, 1.3, " Hi"),
            TranscriptWord(1.3, 1.8, " Alice."),
        ]
        turns = [
            SpeakerTurn(0.0, 0.95, "SPEAKER_00"),
            SpeakerTurn(0.95, 2.0, "SPEAKER_01"),
        ]

        segments = assign_speakers_to_words(words, turns)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "Speaker 1")
        self.assertEqual(segments[0].text, "Hello there.")
        self.assertEqual(segments[1].speaker, "Speaker 2")
        self.assertEqual(segments[1].text, "Hi Alice.")

    def test_uses_names_in_order_of_first_appearance(self):
        words = [
            TranscriptWord(0.0, 0.4, " First"),
            TranscriptWord(0.5, 0.9, " Second"),
            TranscriptWord(1.0, 1.4, " Again"),
        ]
        turns = [
            SpeakerTurn(0.0, 0.45, "B"),
            SpeakerTurn(0.45, 0.95, "A"),
            SpeakerTurn(0.95, 1.5, "B"),
        ]

        segments = assign_speakers_to_words(words, turns, ["Анна", "Борис"])

        self.assertEqual(
            [segment.speaker for segment in segments], ["Анна", "Борис", "Анна"]
        )

    def test_uses_nearest_turn_for_small_timestamp_gap(self):
        words = [TranscriptWord(1.01, 1.05, " word")]
        turns = [SpeakerTurn(0.0, 1.0, "A")]

        segments = assign_speakers_to_words(words, turns)

        self.assertEqual(segments[0].speaker, "Speaker 1")

    def test_does_not_assign_speaker_across_long_silence(self):
        words = [TranscriptWord(10.0, 10.2, " word")]
        turns = [SpeakerTurn(0.0, 1.0, "A")]

        segments = assign_speakers_to_words(words, turns)

        self.assertIsNone(segments[0].speaker)

    def test_bridges_short_vad_hole_between_speaker_turns(self):
        words = [TranscriptWord(5.0, 5.2, " word")]
        turns = [
            SpeakerTurn(0.0, 1.0, "A"),
            SpeakerTurn(9.0, 10.0, "B"),
        ]

        segments = assign_speakers_to_words(words, turns)

        self.assertEqual(segments[0].speaker, "Speaker 1")

    def test_fills_long_vad_hole_when_bounded_by_same_speaker(self):
        words = [
            TranscriptWord(0.5, 0.7, "before"),
            TranscriptWord(20.0, 20.2, "quiet"),
            TranscriptWord(40.1, 40.3, "after"),
        ]
        turns = [
            SpeakerTurn(0.0, 1.0, "A"),
            SpeakerTurn(40.0, 41.0, "A"),
        ]

        segments = assign_speakers_to_words(words, turns)

        self.assertTrue(all(segment.speaker == "Speaker 1" for segment in segments))


class MarkdownTests(unittest.TestCase):
    def test_diarized_metadata_and_labels(self):
        result = TranscriptResult(
            text="Привет",
            segments=[TranscriptSegment(3.2, 4.0, "Привет", "Анна")],
            language="ru",
            duration_min=1.0,
            device="mlx",
            model=DEFAULT_MODELS["qwen"],
            engine="qwen",
            speaker_count=2,
            diarization_model="pyannote/community",
        )

        markdown = result_to_markdown(result, "meeting.wav", date_str="2026-07-30")

        self.assertIn("speakers: 2", markdown)
        self.assertIn("diarization_model: pyannote/community", markdown)
        self.assertIn("**[00:03] Анна:** Привет", markdown)

    def test_long_timestamp_includes_hours(self):
        self.assertEqual(format_timestamp(3661), "01:01:01")

    def test_restores_punctuation_without_changing_speakers(self):
        segments = [
            TranscriptSegment(0.0, 1.0, "привет какихто", "Speaker 1"),
            TranscriptSegment(1.0, 2.0, "новостей нет", "Speaker 2"),
        ]

        restored = restore_segment_punctuation(
            segments, "Привет, каких-то новостей нет."
        )

        self.assertEqual(restored[0].text, "Привет, каких-то")
        self.assertEqual(restored[1].text, "новостей нет.")
        self.assertEqual(restored[1].speaker, "Speaker 2")


class TranscriptionPipelineTests(unittest.TestCase):
    def test_pyannote_reports_real_embedding_batch_progress(self):
        events = []
        hook = _pyannote_progress_hook(
            lambda stage, message: events.append((stage, message))
        )

        self.assertIsNotNone(hook)
        hook("embeddings", None, total=20, completed=7)

        self.assertEqual(events[0][0], "diarizing")
        self.assertIn("35% · 7/20 batches", events[0][1])

        hook("embeddings", None, total=20, completed=32)
        self.assertIn("100% · 20/20 batches", events[1][1])

    def test_video_audio_is_extracted_as_mono_16khz_flac(self):
        with (
            patch("free_transcribe.core.subprocess.run") as run,
            _normalized_audio_path("meeting.webm") as audio_path,
        ):
            self.assertEqual(Path(audio_path).suffix, ".flac")
            command = run.call_args.args[0]
            self.assertIn("-vn", command)
            self.assertEqual(command[command.index("-ac") + 1], "1")
            self.assertEqual(command[command.index("-ar") + 1], "16000")

        self.assertFalse(Path(audio_path).exists())

    def test_prepared_audio_is_shared_by_asr_and_diarization(self):
        engine_output = _EngineOutput(
            text="Hello",
            segments=[TranscriptSegment(0.0, 0.5, "Hello")],
            words=[TranscriptWord(0.0, 0.5, "Hello")],
            language="English",
            duration_min=0.5 / 60,
            device="mlx",
        )
        turns = [SpeakerTurn(0.0, 0.5, "A")]

        with (
            tempfile.NamedTemporaryFile(suffix=".webm") as media_file,
            patch(
                "free_transcribe.core._normalized_audio_path",
                return_value=nullcontext("prepared.flac"),
            ),
            patch(
                "free_transcribe.core._transcribe_parakeet",
                return_value=engine_output,
            ) as parakeet_transcribe,
            patch(
                "free_transcribe.core.diarize_file", return_value=turns
            ) as diarize,
        ):
            transcribe_file(media_file.name, engine="parakeet", diarize=True)

        self.assertEqual(parakeet_transcribe.call_args.args[0], "prepared.flac")
        self.assertEqual(diarize.call_args.args[0], "prepared.flac")

    def test_mlx_reports_real_processed_audio_progress(self):
        events = []

        def fake_transcribe(_path, **options):
            options["on_progress"](
                {
                    "event": "chunks_prepared",
                    "total_chunks": 4,
                    "audio_duration_sec": 120.0,
                    "progress": 0.0,
                }
            )
            options["on_progress"](
                {
                    "event": "chunk_completed",
                    "chunk_index": 1,
                    "total_chunks": 4,
                    "audio_duration_sec": 120.0,
                    "processed_audio_sec": 30.0,
                    "progress": 0.25,
                }
            )
            return SimpleNamespace(
                text="Hello",
                language="English",
                segments=[],
                chunks=[{"start": 0.0, "end": 30.0, "text": "Hello"}],
            )

        with (
            patch("free_transcribe.core._is_apple_silicon", return_value=True),
            patch.dict(
                sys.modules,
                {"mlx_qwen3_asr": SimpleNamespace(transcribe=fake_transcribe)},
            ),
        ):
            _transcribe_qwen_mlx(
                "meeting.wav",
                model_name="test/model",
                language=None,
                context=None,
                need_words=False,
                on_progress=lambda stage, message: events.append((stage, message)),
            )

        self.assertEqual(events[0][0], "transcribing")
        self.assertIn("0% · 00:00 / 02:00 · chunk 0/4", events[0][1])
        self.assertIn("25% · 00:30 / 02:00 · chunk 1/4", events[1][1])

    def test_qwen_dispatches_to_mlx_on_apple_silicon(self):
        expected = _EngineOutput("text", [], [], "ru", 0.0, "mlx")
        with (
            patch("free_transcribe.core._is_apple_silicon", return_value=True),
            patch(
                "free_transcribe.core._transcribe_qwen_mlx", return_value=expected
            ) as backend,
        ):
            result = _transcribe_qwen(
                "meeting.wav",
                model_name="test/model",
                language="ru",
                context=None,
                need_words=True,
            )

        self.assertIs(result, expected)
        self.assertTrue(backend.call_args.kwargs["need_words"])

    def test_qwen_dispatches_to_torch_off_apple_silicon(self):
        expected = _EngineOutput("text", [], [], "ru", 0.0, "cuda")
        with (
            patch("free_transcribe.core._is_apple_silicon", return_value=False),
            patch(
                "free_transcribe.core._transcribe_qwen_torch", return_value=expected
            ) as backend,
        ):
            result = _transcribe_qwen(
                "meeting.wav",
                model_name="test/model",
                language="ru",
                context=None,
                need_words=False,
            )

        self.assertIs(result, expected)
        backend.assert_called_once()

    def test_qwen_requests_alignment_and_assigns_speakers(self):
        engine_output = _EngineOutput(
            text="Hello Hi",
            segments=[TranscriptSegment(0.0, 1.0, "Hello Hi")],
            words=[
                TranscriptWord(0.0, 0.4, " Hello"),
                TranscriptWord(0.6, 1.0, " Hi"),
            ],
            language="English",
            duration_min=1 / 60,
            device="mlx",
        )
        turns = [
            SpeakerTurn(0.0, 0.5, "A"),
            SpeakerTurn(0.5, 1.0, "B"),
        ]

        with (
            tempfile.NamedTemporaryFile() as media_file,
            patch(
                "free_transcribe.core._transcribe_qwen",
                return_value=engine_output,
            ) as qwen_transcribe,
            patch("free_transcribe.core.diarize_file", return_value=turns),
        ):
            result = transcribe_file(
                media_file.name,
                engine="qwen",
                language="en",
                diarize=True,
                speaker_names=["Alice", "Bob"],
            )

        options = qwen_transcribe.call_args.kwargs
        self.assertTrue(options["need_words"])
        self.assertEqual(options["model_name"], DEFAULT_MODELS["qwen"])
        self.assertEqual(
            [segment.speaker for segment in result.segments], ["Alice", "Bob"]
        )
        self.assertEqual(result.speaker_count, 2)
        self.assertEqual(result.engine, "qwen")

    def test_parakeet_engine_uses_its_default_model(self):
        engine_output = _EngineOutput(
            text="Fast",
            segments=[TranscriptSegment(0.0, 0.5, "Fast")],
            words=[TranscriptWord(0.0, 0.5, "Fast")],
            language="ru",
            duration_min=0.5 / 60,
            device="mlx",
        )
        with (
            tempfile.NamedTemporaryFile() as media_file,
            patch(
                "free_transcribe.core._transcribe_parakeet",
                return_value=engine_output,
            ) as parakeet_transcribe,
        ):
            result = transcribe_file(media_file.name, engine="parakeet")

        self.assertEqual(
            parakeet_transcribe.call_args.kwargs["model_name"],
            DEFAULT_MODELS["parakeet"],
        )
        self.assertEqual(result.engine, "parakeet")

    def test_parakeet_dispatches_to_mlx_on_apple_silicon(self):
        expected = _EngineOutput("text", [], [], "ru", 0.0, "mlx")
        with (
            patch("free_transcribe.core._is_apple_silicon", return_value=True),
            patch(
                "free_transcribe.core._transcribe_parakeet_mlx",
                return_value=expected,
            ) as backend,
        ):
            result = _transcribe_parakeet(
                "meeting.wav",
                model_name="test/model",
                language="ru",
                context=None,
            )

        self.assertIs(result, expected)
        backend.assert_called_once()

    def test_parakeet_dispatches_to_cuda_off_apple_silicon(self):
        expected = _EngineOutput("text", [], [], "ru", 0.0, "cuda")
        with (
            patch("free_transcribe.core._is_apple_silicon", return_value=False),
            patch(
                "free_transcribe.core._transcribe_parakeet_cuda",
                return_value=expected,
            ) as backend,
        ):
            result = _transcribe_parakeet(
                "meeting.wav",
                model_name="test/model",
                language="ru",
                context=None,
            )

        self.assertIs(result, expected)
        backend.assert_called_once()

    def test_cuda_parakeet_normalizes_nemo_timestamps(self):
        class FakeModel:
            def cuda(self):
                return self

            def eval(self):
                return self

            def transcribe(self, _paths, timestamps=False):
                self.timestamps = timestamps
                return [
                    SimpleNamespace(
                        text="Hello world",
                        timestamp={
                            "word": [
                                {"start": 0.0, "end": 0.4, "word": "Hello"},
                                {"start": 0.5, "end": 0.9, "word": "world"},
                            ],
                            "segment": [
                                {
                                    "start": 0.0,
                                    "end": 0.9,
                                    "segment": "Hello world",
                                }
                            ],
                        },
                    )
                ]

        model = FakeModel()
        asr = ModuleType("nemo.collections.asr")
        asr.models = SimpleNamespace(
            ASRModel=SimpleNamespace(
                from_pretrained=lambda **_options: model,
            )
        )
        collections = ModuleType("nemo.collections")
        collections.__path__ = []
        collections.asr = asr
        nemo = ModuleType("nemo")
        nemo.__path__ = []
        nemo.collections = collections
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

        with (
            patch.dict(
                sys.modules,
                {
                    "nemo": nemo,
                    "nemo.collections": collections,
                    "nemo.collections.asr": asr,
                    "torch": fake_torch,
                },
            ),
            patch(
                "free_transcribe.core._chunked_audio_paths",
                return_value=nullcontext([("chunk.flac", 0.0)]),
            ),
            patch("free_transcribe.core._media_duration_seconds", return_value=1.0),
        ):
            result = _transcribe_parakeet_cuda(
                "meeting.flac",
                model_name="nvidia/test",
                language="en",
                context=None,
            )

        self.assertTrue(model.timestamps)
        self.assertEqual(result.device, "cuda")
        self.assertEqual(result.text, "Hello world")
        self.assertEqual([word.text for word in result.words], ["Hello", "world"])
        self.assertEqual(result.segments[0].text, "Hello world")

    def test_qwen_can_request_words_without_diarization(self):
        engine_output = _EngineOutput(
            text="Word",
            segments=[TranscriptSegment(0.0, 0.5, "Word")],
            words=[TranscriptWord(0.0, 0.5, "Word")],
            language="English",
            duration_min=0.5 / 60,
            device="mlx",
        )
        with (
            tempfile.NamedTemporaryFile() as media_file,
            patch(
                "free_transcribe.core._transcribe_qwen",
                return_value=engine_output,
            ) as qwen_transcribe,
        ):
            result = transcribe_file(
                media_file.name, engine="qwen", word_timestamps=True
            )

        self.assertTrue(qwen_transcribe.call_args.kwargs["need_words"])
        self.assertEqual(result.words, engine_output.words)

    def test_rejects_unknown_engine(self):
        with (
            tempfile.NamedTemporaryFile() as media_file,
            self.assertRaisesRegex(ValueError, "Invalid engine"),
        ):
            transcribe_file(media_file.name, engine="unknown")


if __name__ == "__main__":
    unittest.main()
