"""Free Transcribe - local multi-engine audio/video transcription."""

__version__ = "0.1.0"

from .core import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    DEFAULT_MODEL,
    DEFAULT_MODELS,
    SUPPORTED_FORMATS,
    DiarizationResult,
    SpeakerTurn,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    assign_speakers_to_words,
    diarize_file,
    diarize_media,
    restore_segment_punctuation,
    result_to_markdown,
    save_transcript,
    transcribe_file,
)

__all__ = [
    "AVAILABLE_ENGINES",
    "DEFAULT_ENGINE",
    "DEFAULT_MODEL",
    "DEFAULT_MODELS",
    "SUPPORTED_FORMATS",
    "DiarizationResult",
    "SpeakerTurn",
    "TranscriptResult",
    "TranscriptSegment",
    "TranscriptWord",
    "assign_speakers_to_words",
    "diarize_file",
    "diarize_media",
    "restore_segment_punctuation",
    "result_to_markdown",
    "save_transcript",
    "transcribe_file",
]
