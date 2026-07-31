"""ASR, alignment, diarization, and transcript rendering."""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SUPPORTED_FORMATS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".wma",
    ".aac",
    ".mp4",
    ".webm",
    ".mkv",
    ".avi",
    ".mov",
}
VIDEO_FORMATS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

AVAILABLE_ENGINES = ("qwen", "parakeet")
DEFAULT_ENGINE = (
    "parakeet"
    if platform.system() == "Darwin" and platform.machine() == "arm64"
    else "qwen"
)
DEFAULT_MODELS = {
    "qwen": "Qwen/Qwen3-ASR-1.7B",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",
}
DEFAULT_MODEL = DEFAULT_MODELS[DEFAULT_ENGINE]
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_FORCED_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "uk": "Ukrainian",
}

ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class SpeakerTurn:
    """A time interval assigned to one speaker by a diarization model."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptWord:
    """A word with timestamps produced by an ASR engine or aligner."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptSegment:
    """A single segment of transcription."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptResult:
    """Result of transcription."""

    text: str
    segments: list[TranscriptSegment]
    language: str
    duration_min: float
    device: str
    model: str
    engine: str = DEFAULT_ENGINE
    speaker_count: int = 0
    diarization_model: str | None = None
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass
class DiarizationResult:
    """Raw and exclusive pyannote turns from one diarization run."""

    turns: list[SpeakerTurn]
    exclusive_turns: list[SpeakerTurn]
    model: str
    device: str


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS for long recordings."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _overlap(start: float, end: float, turn: SpeakerTurn) -> float:
    return max(0.0, min(end, turn.end) - max(start, turn.start))


def _speaker_for_interval(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> str | None:
    """Choose the speaker with the greatest overlap with a timed interval."""
    if not turns:
        return None

    best_turn = max(turns, key=lambda turn: _overlap(start, end, turn))
    if _overlap(start, end, best_turn) > 0:
        return best_turn.speaker

    # Timestamp boundaries from the two models can differ slightly. In that
    # case, use the nearest diarization turn rather than dropping the speaker.
    midpoint = (start + end) / 2

    def distance(turn: SpeakerTurn) -> float:
        if midpoint < turn.start:
            return turn.start - midpoint
        if midpoint > turn.end:
            return midpoint - turn.end
        return 0.0

    nearest = min(turns, key=distance)
    if distance(nearest) <= 1.0:
        return nearest.speaker

    # VAD can occasionally leave a short hole even while quiet speech is
    # present. Bridge only bounded holes that have diarization turns on both
    # sides; genuine long silence (or text outside the diarized range) stays
    # unlabeled.
    previous = [turn for turn in turns if turn.end <= midpoint]
    following = [turn for turn in turns if turn.start >= midpoint]
    if previous and following:
        left = max(previous, key=lambda turn: turn.end)
        right = min(following, key=lambda turn: turn.start)
        if right.start - left.end <= 15.0:
            return left.speaker if distance(left) <= distance(right) else right.speaker
    return None


def _display_speaker_map(
    speakers: Iterable[str],
    speaker_names: list[str] | None,
) -> dict[str, str]:
    """Map model labels to stable labels in order of first appearance."""
    mapping: dict[str, str] = {}
    cleaned_names = [name.strip() for name in (speaker_names or []) if name.strip()]
    for speaker in speakers:
        if speaker in mapping:
            continue
        index = len(mapping)
        mapping[speaker] = (
            cleaned_names[index]
            if index < len(cleaned_names)
            else f"Speaker {index + 1}"
        )
    return mapping


def _join_word_texts(words: list[TranscriptWord]) -> str:
    """Preserve leading-space and punctuation conventions across engines."""
    pieces = [word.text for word in words]
    if any(piece[:1].isspace() for piece in pieces):
        return "".join(pieces).strip()
    return " ".join(piece.strip() for piece in pieces).strip()


def assign_speakers_to_words(
    words: list[TranscriptWord],
    turns: list[SpeakerTurn],
    speaker_names: list[str] | None = None,
    max_merge_gap: float = 1.5,
) -> list[TranscriptSegment]:
    """Assign words to speakers and merge adjacent words into speaker turns."""
    if not words:
        return []

    assigned = [
        (word, _speaker_for_interval(word.start, word.end, turns)) for word in words
    ]
    for index, (word, speaker) in enumerate(assigned):
        if speaker is not None:
            continue
        previous = next(
            (item[1] for item in reversed(assigned[:index]) if item[1] is not None),
            None,
        )
        following = next(
            (item[1] for item in assigned[index + 1 :] if item[1] is not None),
            None,
        )
        if previous is not None and previous == following:
            assigned[index] = (word, previous)
    mapping = _display_speaker_map(
        (speaker for _, speaker in assigned if speaker is not None),
        speaker_names,
    )

    grouped: list[tuple[list[TranscriptWord], str | None]] = []
    for word, raw_speaker in assigned:
        speaker = mapping.get(raw_speaker) if raw_speaker is not None else None
        if (
            grouped
            and grouped[-1][1] == speaker
            and word.start - grouped[-1][0][-1].end <= max_merge_gap
        ):
            grouped[-1][0].append(word)
        else:
            grouped.append(([word], speaker))

    return [
        TranscriptSegment(
            start=group_words[0].start,
            end=group_words[-1].end,
            text=_join_word_texts(group_words),
            speaker=speaker,
        )
        for group_words, speaker in grouped
    ]


def restore_segment_punctuation(
    segments: list[TranscriptSegment],
    punctuated_text: str,
) -> list[TranscriptSegment]:
    """Restore punctuation/casing from ASR text without changing timestamps."""
    source_tokens = punctuated_text.split()
    indexed_target_tokens = [
        (segment_index, token)
        for segment_index, segment in enumerate(segments)
        for token in segment.text.split()
    ]
    if not source_tokens or not indexed_target_tokens:
        return segments

    def normalized(token: str) -> str:
        return "".join(character for character in token.casefold() if character.isalnum())

    target_items = [
        (index, normalized(token))
        for index, (_, token) in enumerate(indexed_target_tokens)
        if normalized(token)
    ]
    source_items = [
        (index, normalized(token))
        for index, token in enumerate(source_tokens)
        if normalized(token)
    ]
    matcher = SequenceMatcher(
        None,
        [item[1] for item in target_items],
        [item[1] for item in source_items],
    )
    replacements: dict[int, str] = {}
    for target_start, source_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            target_index = target_items[target_start + offset][0]
            source_index = source_items[source_start + offset][0]
            replacements[target_index] = source_tokens[source_index]

    tokens_by_segment: list[list[str]] = [[] for _ in segments]
    for target_index, (segment_index, token) in enumerate(indexed_target_tokens):
        tokens_by_segment[segment_index].append(replacements.get(target_index, token))

    return [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=" ".join(tokens_by_segment[index]),
            speaker=segment.speaker,
        )
        for index, segment in enumerate(segments)
    ]


def _fallback_assign_segments(
    segments: list[TranscriptSegment],
    turns: list[SpeakerTurn],
    speaker_names: list[str] | None,
) -> list[TranscriptSegment]:
    raw_speakers = [
        _speaker_for_interval(segment.start, segment.end, turns) for segment in segments
    ]
    mapping = _display_speaker_map(
        (speaker for speaker in raw_speakers if speaker is not None),
        speaker_names,
    )
    return [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=mapping.get(speaker) if speaker is not None else None,
        )
        for segment, speaker in zip(segments, raw_speakers)
    ]


def _words_from_items(items: Iterable[Any]) -> list[TranscriptWord]:
    """Normalize dicts or aligned-token objects to timestamped words."""
    words: list[TranscriptWord] = []
    for item in items:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
            text = item.get("text", item.get("word", ""))
        else:
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            text = getattr(item, "text", "")
        if start is None or end is None:
            continue
        words.append(TranscriptWord(float(start), float(end), str(text)))
    return words


@dataclass
class _EngineOutput:
    text: str
    segments: list[TranscriptSegment]
    words: list[TranscriptWord]
    language: str
    duration_min: float
    device: str


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _require_apple_silicon(engine: str) -> None:
    if not _is_apple_silicon():
        raise RuntimeError(
            f"The {engine} MLX backend requires Apple Silicon. "
            "This engine does not have a backend for the current platform."
        )


def _qwen_language(language: str | None) -> str | None:
    if not language:
        return None
    return LANGUAGE_NAMES.get(language.casefold(), language)


def _transcribe_qwen_mlx(
    file_path: str,
    *,
    model_name: str,
    language: str | None,
    context: str | None,
    need_words: bool,
    on_progress: ProgressCallback | None = None,
) -> _EngineOutput:
    _require_apple_silicon("Qwen3-ASR")
    try:
        from mlx_qwen3_asr import transcribe as qwen_transcribe
    except ImportError as exc:
        raise RuntimeError(
            "Qwen support is not installed. Install the 'qwen' or 'quality' extra."
        ) from exc

    def mlx_progress(payload: dict[str, Any]) -> None:
        if on_progress is None or payload.get("event") not in {
            "chunks_prepared",
            "chunk_completed",
        }:
            return
        fraction = min(max(float(payload.get("progress", 0.0)), 0.0), 1.0)
        percent = int(fraction * 100)
        processed = float(payload.get("processed_audio_sec", 0.0))
        duration = float(payload.get("audio_duration_sec", 0.0))
        chunk = int(payload.get("chunk_index", 0))
        chunks = int(payload.get("total_chunks", 0))
        details = f"{format_timestamp(processed)} / {format_timestamp(duration)}"
        if chunks:
            details += f" · chunk {chunk}/{chunks}"
        on_progress("transcribing", f"Transcribing… {percent}% · {details}")

    result = qwen_transcribe(
        file_path,
        model=model_name,
        context=context or "",
        language=_qwen_language(language),
        return_timestamps=need_words,
        return_chunks=True,
        on_progress=mlx_progress,
    )
    raw_words = list(result.segments or [])
    words = _words_from_items(raw_words)
    chunks = list(result.chunks or [])
    segments = [
        TranscriptSegment(
            start=float(chunk.get("start", 0.0)),
            end=float(chunk.get("end", chunk.get("start", 0.0))),
            text=str(chunk.get("text", "")).strip(),
        )
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    ]
    if not segments and words:
        segments = [
            TranscriptSegment(word.start, word.end, word.text.strip()) for word in words
        ]
    duration = max(
        [segment.end for segment in segments] + [word.end for word in words] + [0.0]
    )
    return _EngineOutput(
        text=str(result.text).strip(),
        segments=segments,
        words=words,
        language=str(result.language or language or "unknown"),
        duration_min=duration / 60,
        device="mlx",
    )


def _media_duration_seconds(file_path: str) -> float:
    """Read duration without loading media into Python memory."""
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(0.0, float(process.stdout.strip()))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0.0


@contextmanager
def _normalized_audio_path(file_path: str):
    """Extract video audio once; leave native audio files unchanged."""
    if Path(file_path).suffix.casefold() not in VIDEO_FORMATS:
        yield file_path
        return

    with tempfile.TemporaryDirectory(prefix="free-transcribe-audio-") as directory:
        audio_path = str(Path(directory) / "audio.flac")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    file_path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "flac",
                    "-y",
                    audio_path,
                ],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to process video files") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Could not extract audio from media") from exc
        yield audio_path


def _transcribe_qwen_torch(
    file_path: str,
    *,
    model_name: str,
    language: str | None,
    context: str | None,
    need_words: bool,
    on_progress: ProgressCallback | None = None,
) -> _EngineOutput:
    """Official Qwen Transformers backend for NVIDIA CUDA (Windows/Linux)."""
    del on_progress
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen CUDA support is not installed. Install the 'qwen' or "
            "'quality' extra."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The Qwen backend on Windows/Linux currently requires an NVIDIA "
            "GPU with a working CUDA build of PyTorch."
        )

    device = "cuda:0"
    options: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": device,
        "max_inference_batch_size": 8,
        "max_new_tokens": 4096,
    }
    if need_words:
        options.update(
            {
                "forced_aligner": DEFAULT_FORCED_ALIGNER_MODEL,
                "forced_aligner_kwargs": {
                    "dtype": torch.bfloat16,
                    "device_map": device,
                },
            }
        )

    model = Qwen3ASRModel.from_pretrained(model_name, **options)
    # Contextual prompting is currently an MLX-only feature. Keeping this
    # argument in the adapter boundary makes it possible to add when the
    # official Transformers backend exposes it.
    del context
    results = model.transcribe(
        audio=file_path,
        language=_qwen_language(language),
        return_time_stamps=need_words,
    )
    if not results:
        raise RuntimeError("Qwen returned no transcription result")
    result = results[0]
    raw_timestamps = list(getattr(result, "time_stamps", None) or [])
    words = [
        TranscriptWord(
            float(item.start_time),
            float(item.end_time),
            str(item.text),
        )
        for item in raw_timestamps
        if getattr(item, "start_time", None) is not None
        and getattr(item, "end_time", None) is not None
    ]
    duration = max(
        [word.end for word in words] + [_media_duration_seconds(file_path), 0.0]
    )
    text = str(getattr(result, "text", "")).strip()
    segments = (
        [TranscriptSegment(word.start, word.end, word.text.strip()) for word in words]
        if words
        else [TranscriptSegment(0.0, duration, text)]
    )
    return _EngineOutput(
        text=text,
        segments=[segment for segment in segments if segment.text],
        words=words,
        language=str(getattr(result, "language", None) or language or "unknown"),
        duration_min=duration / 60,
        device="cuda",
    )


def _transcribe_qwen(
    file_path: str,
    *,
    model_name: str,
    language: str | None,
    context: str | None,
    need_words: bool,
    on_progress: ProgressCallback | None = None,
) -> _EngineOutput:
    if _is_apple_silicon():
        return _transcribe_qwen_mlx(
            file_path,
            model_name=model_name,
            language=language,
            context=context,
            need_words=need_words,
            on_progress=on_progress,
        )
    return _transcribe_qwen_torch(
        file_path,
        model_name=model_name,
        language=language,
        context=context,
        need_words=need_words,
        on_progress=on_progress,
    )


def _transcribe_parakeet(
    file_path: str,
    *,
    model_name: str,
    language: str | None,
    context: str | None,
    on_progress: ProgressCallback | None = None,
) -> _EngineOutput:
    _require_apple_silicon("Parakeet")
    try:
        from mlx_audio.stt.utils import load
    except ImportError as exc:
        raise RuntimeError(
            "Parakeet support is not installed. Install the 'parakeet' extra."
        ) from exc

    if context:
        # Parakeet does not currently expose contextual biasing in mlx-audio.
        context = None
    model = load(model_name)
    previous_chunk_end = 0

    def chunk_started(current_position: int, total_position: int) -> None:
        nonlocal previous_chunk_end
        if on_progress is not None and previous_chunk_end and total_position:
            fraction = min(previous_chunk_end / total_position, 1.0)
            on_progress(
                "transcribing",
                f"Transcribing… {int(fraction * 100)}% · "
                f"{format_timestamp(previous_chunk_end / 16000)} / "
                f"{format_timestamp(total_position / 16000)}",
            )
        previous_chunk_end = current_position

    result = model.generate(
        file_path,
        chunk_duration=300.0,
        overlap_duration=2.0,
        chunk_callback=chunk_started,
    )
    if on_progress is not None:
        duration = _media_duration_seconds(file_path)
        on_progress(
            "transcribing",
            f"Transcribing… 100% · {format_timestamp(duration)} / "
            f"{format_timestamp(duration)}",
        )
    sentences = list(getattr(result, "sentences", []) or [])
    segments = [
        TranscriptSegment(
            start=float(sentence.start),
            end=float(sentence.end),
            text=str(sentence.text).strip(),
        )
        for sentence in sentences
        if str(sentence.text).strip()
    ]
    words = _words_from_items(
        token for sentence in sentences for token in (sentence.tokens or [])
    )
    duration = max(
        [segment.end for segment in segments] + [word.end for word in words] + [0.0]
    )
    return _EngineOutput(
        text=str(getattr(result, "text", "")).strip(),
        segments=segments,
        words=words,
        language=language or str(getattr(result, "language", "auto")),
        duration_min=duration / 60,
        device="mlx",
    )


def _validate_speaker_counts(
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    for name, value in (
        ("num_speakers", num_speakers),
        ("min_speakers", min_speakers),
        ("max_speakers", max_speakers),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be at least 1")
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise ValueError("min_speakers cannot be greater than max_speakers")


def _pyannote_progress_hook(
    on_progress: ProgressCallback | None,
) -> Callable[..., None] | None:
    """Adapt pyannote's internal batch hook to our progress contract."""
    if on_progress is None:
        return None

    labels = {
        "segmentation": "Speech segmentation complete",
        "speaker_counting": "Speaker counting complete",
        "embeddings": "Speaker embeddings complete",
        "discrete_diarization": "Finalizing speaker turns",
    }

    def hook(
        step_name: str,
        _artifact: Any,
        *,
        file: Any = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        del file
        if total is not None and completed is not None and total > 0:
            bounded_completed = min(completed, total)
            percent = round(bounded_completed / total * 100)
            on_progress(
                "diarizing",
                f"Identifying speakers… embeddings {percent}% · "
                f"{bounded_completed}/{total} batches",
            )
            return
        on_progress(
            "diarizing",
            labels.get(step_name, step_name.replace("_", " ").capitalize()),
        )

    return hook


def diarize_media(
    file_path: str,
    *,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    token: str | None = None,
    device: str = "auto",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> DiarizationResult:
    """Run pyannote and retain regular plus exclusive speaker turns."""
    _validate_speaker_counts(num_speakers, min_speakers, max_speakers)
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("diarization device must be one of: auto, cpu, mps, cuda")

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Speaker diarization is not installed. Install the optional dependencies "
            "with: uv tool install 'free-transcribe[quality] @ "
            "git+https://github.com/vgmakeev/free-transcribe.git'"
        ) from exc

    if on_progress:
        on_progress("diarization_loading", f"Loading diarization model {model_name}...")

    access_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=access_token)
        if pipeline is None:
            raise RuntimeError("The model repository did not return a pipeline")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the pyannote diarization model. Accept its terms at "
            "https://huggingface.co/pyannote/speaker-diarization-community-1 and "
            "set HF_TOKEN, or run `hf auth login`."
        ) from exc

    if on_progress:
        on_progress("diarization_loading", "Diarization model ready · 100%")

    resolved_device = device
    if resolved_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    try:
        pipeline.to(torch.device(resolved_device))
    except RuntimeError:
        if device != "auto" or resolved_device == "cpu":
            raise
        resolved_device = "cpu"
        pipeline.to(torch.device(resolved_device))

    diarization_options = {
        key: value
        for key, value in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if value is not None
    }

    if on_progress:
        on_progress(
            "diarizing", f"Identifying speakers on {resolved_device.upper()}..."
        )

    progress_hook = _pyannote_progress_hook(on_progress)
    with _normalized_audio_path(file_path) as diarization_path:
        try:
            output = pipeline(
                diarization_path,
                hook=progress_hook,
                **diarization_options,
            )
        except RuntimeError:
            if device != "auto" or resolved_device == "cpu":
                raise
            if on_progress:
                on_progress(
                    "diarizing",
                    f"{resolved_device.upper()} diarization failed; retrying on CPU...",
                )
            pipeline.to(torch.device("cpu"))
            resolved_device = "cpu"
            output = pipeline(
                diarization_path,
                hook=progress_hook,
                **diarization_options,
            )

    annotation = getattr(output, "speaker_diarization", output)
    turns = [
        SpeakerTurn(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive_annotation is None:
        exclusive_turns = turns
    else:
        exclusive_turns = [
            SpeakerTurn(float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in exclusive_annotation.itertracks(yield_label=True)
        ]
    if on_progress:
        on_progress("diarization_complete", "Speaker diarization complete")
    return DiarizationResult(
        turns=turns,
        exclusive_turns=exclusive_turns,
        model=model_name,
        device=resolved_device,
    )


def diarize_file(
    file_path: str,
    *,
    model_name: str = DEFAULT_DIARIZATION_MODEL,
    token: str | None = None,
    device: str = "auto",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[SpeakerTurn]:
    """Run pyannote and return exclusive turns for ASR word assignment."""
    result = diarize_media(
        file_path,
        model_name=model_name,
        token=token,
        device=device,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        on_progress=on_progress,
    )
    return result.exclusive_turns


def _transcribe_prepared(
    file_path: str,
    *,
    resolved_model: str,
    language: str | None,
    prompt: str | None,
    on_progress: ProgressCallback | None,
    engine: str,
    diarize: bool,
    diarization_model: str,
    diarization_device: str,
    hf_token: str | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    speaker_names: list[str] | None,
    word_timestamps: bool,
) -> TranscriptResult:
    """Run model stages on already normalized audio."""
    if engine == "qwen":
        engine_output = _transcribe_qwen(
            file_path,
            model_name=resolved_model,
            language=language,
            context=prompt,
            need_words=diarize or word_timestamps,
            on_progress=on_progress,
        )
    else:
        engine_output = _transcribe_parakeet(
            file_path,
            model_name=resolved_model,
            language=language,
            context=prompt,
            on_progress=on_progress,
        )

    speaker_count = 0
    used_diarization_model: str | None = None
    segments = engine_output.segments
    if diarize:
        if engine == "parakeet":
            # The ASR model is no longer needed. Release Metal allocations
            # before loading pyannote so peak memory is bounded by one model.
            gc.collect()
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:
                pass
        turns = diarize_file(
            file_path,
            model_name=diarization_model,
            token=hf_token,
            device=diarization_device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            on_progress=on_progress,
        )
        if engine_output.words:
            segments = assign_speakers_to_words(
                engine_output.words, turns, speaker_names
            )
        else:
            segments = _fallback_assign_segments(segments, turns, speaker_names)
        segments = restore_segment_punctuation(segments, engine_output.text)
        speaker_count = len({turn.speaker for turn in turns})
        used_diarization_model = diarization_model

    if on_progress:
        on_progress("complete", "Transcription complete")

    return TranscriptResult(
        text=engine_output.text,
        segments=segments,
        language=engine_output.language,
        duration_min=engine_output.duration_min,
        device=engine_output.device,
        model=resolved_model,
        engine=engine,
        speaker_count=speaker_count,
        diarization_model=used_diarization_model,
        words=engine_output.words,
    )


def transcribe_file(
    file_path: str,
    model_name: str | None = None,
    language: str | None = None,
    prompt: str | None = None,
    on_progress: ProgressCallback | None = None,
    *,
    engine: str = DEFAULT_ENGINE,
    diarize: bool = False,
    diarization_model: str = DEFAULT_DIARIZATION_MODEL,
    diarization_device: str = "auto",
    hf_token: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    speaker_names: list[str] | None = None,
    word_timestamps: bool = False,
) -> TranscriptResult:
    """Transcribe media with the selected engine and optional pyannote."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    engine = engine.casefold()
    if engine not in AVAILABLE_ENGINES:
        raise ValueError(f"Invalid engine: {engine}. Available: {AVAILABLE_ENGINES}")
    resolved_model = model_name or DEFAULT_MODELS[engine]

    _validate_speaker_counts(num_speakers, min_speakers, max_speakers)
    if on_progress:
        backend = "Apple Silicon MLX" if _is_apple_silicon() else "NVIDIA CUDA"
        on_progress("device", f"Using the {backend} backend")
        if Path(file_path).suffix.casefold() in VIDEO_FORMATS:
            on_progress("preparing", "Extracting mono audio with ffmpeg...")

    with _normalized_audio_path(file_path) as prepared_path:
        if on_progress:
            on_progress("loading", f"Loading {engine} model {resolved_model}...")
            on_progress("transcribing", "Transcribing...")
        return _transcribe_prepared(
            prepared_path,
            resolved_model=resolved_model,
            language=language,
            prompt=prompt,
            on_progress=on_progress,
            engine=engine,
            diarize=diarize,
            diarization_model=diarization_model,
            diarization_device=diarization_device,
            hf_token=hf_token,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            speaker_names=speaker_names,
            word_timestamps=word_timestamps,
        )


def result_to_markdown(
    result: TranscriptResult,
    source_filename: str,
    date_str: str | None = None,
) -> str:
    """Convert TranscriptResult to Markdown string."""
    if date_str is None:
        date_str = datetime.now(UTC).astimezone().strftime("%Y-%m-%d")

    file_name = os.path.splitext(source_filename)[0]
    lines = [
        "---",
        "type: transcript",
        f"date: {date_str}",
        f'source: "[[{source_filename}]]"',
        f"duration: {result.duration_min:.1f} min",
        f"language: {result.language}",
        f"device: {result.device}",
        f"engine: {result.engine}",
        f"model: {result.model}",
    ]
    if result.diarization_model:
        lines.extend(
            [
                f"speakers: {result.speaker_count}",
                f"diarization_model: {result.diarization_model}",
            ]
        )
    lines.extend(["---", "", f"# Transcript: {file_name}", ""])

    for segment in result.segments:
        timestamp = format_timestamp(segment.start)
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        lines.append(f"**[{timestamp}]{speaker}** {segment.text}")
        lines.append("")
    return "\n".join(lines)


def save_transcript(
    result: TranscriptResult,
    source_path: str,
    output_path: str | None = None,
) -> str:
    """Save a transcript as Markdown and return the output path."""
    if output_path:
        output_file = output_path
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        file_dir = os.path.dirname(os.path.abspath(source_path))
        transcripts_dir = os.path.join(file_dir, "Transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)
        date_str = datetime.now(UTC).astimezone().strftime("%Y-%m-%d")
        file_name = os.path.splitext(os.path.basename(source_path))[0]
        output_file = os.path.join(
            transcripts_dir, f"{date_str} {file_name} Transcript.md"
        )

    markdown = result_to_markdown(result, os.path.basename(source_path))
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(os.path.abspath(output_file)),
            prefix=f".{os.path.basename(output_file)}.",
            delete=False,
        ) as file_handle:
            file_handle.write(markdown)
            temporary_name = file_handle.name
        os.replace(temporary_name, output_file)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return output_file
