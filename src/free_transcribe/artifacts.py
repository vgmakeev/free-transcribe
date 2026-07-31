"""Versioned, deterministic JSON artifacts for composable workflows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .core import (
    DEFAULT_ENGINE,
    DiarizationResult,
    SpeakerTurn,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    assign_speakers_to_words,
    restore_segment_punctuation,
)

ASR_SCHEMA = "free-transcribe/asr/v2"
DIARIZATION_SCHEMA = "free-transcribe/diarization/v2"
TRANSCRIPT_SCHEMA = "free-transcribe/transcript/v2"
SPEAKER_LABELS_SCHEMA = "free-transcribe/speaker-labels/v1"


def describe_source(source: str) -> dict[str, Any]:
    """Return a content-addressed media descriptor shared by all stages."""
    path = Path(source).expanduser().resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Media changed while fingerprinting: {path}")
    return {
        "path": str(path),
        "name": path.name,
        "size": after.st_size,
        "sha256": digest.hexdigest(),
    }


def _with_id(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    artifact_id = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return {
        "schema": payload["schema"],
        "id": artifact_id,
        **{key: value for key, value in payload.items() if key != "schema"},
    }


def _word_payload(word: TranscriptWord) -> dict[str, Any]:
    return {"start": word.start, "end": word.end, "text": word.text}


def _segment_payload(segment: TranscriptSegment) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }
    if segment.speaker is not None:
        payload["speaker"] = segment.speaker
    return payload


def _turn_payload(turn: SpeakerTurn) -> dict[str, Any]:
    return {"start": turn.start, "end": turn.end, "speaker": turn.speaker}


def _speaker_metadata(
    segments: list[TranscriptSegment], speaker_names: list[str] | None
) -> list[dict[str, Any]]:
    speaker_ids = list(
        dict.fromkeys(segment.speaker for segment in segments if segment.speaker)
    )
    names = speaker_names or []
    return [
        {
            "id": speaker_id,
            "identity": (
                {
                    "name": names[index],
                    "source": "provided",
                    "confidence": 1.0,
                    "evidence": [],
                }
                if index < len(names)
                else None
            ),
            "role": None,
            "voiceprint": None,
        }
        for index, speaker_id in enumerate(speaker_ids)
    ]


def asr_artifact(result: TranscriptResult, source: str) -> dict[str, Any]:
    """Serialize ASR output and its explicit timestamp precision."""
    return _with_id(
        {
            "schema": ASR_SCHEMA,
            "source": describe_source(source),
            "engine": result.engine,
            "model": result.model,
            "language": result.language,
            "device": result.device,
            "duration_seconds": result.duration_min * 60,
            "timestamps": "word" if result.words else "segment",
            "text": result.text,
            "segments": [_segment_payload(segment) for segment in result.segments],
            "words": [_word_payload(word) for word in result.words],
        }
    )


def diarization_artifact(result: DiarizationResult, source: str) -> dict[str, Any]:
    """Serialize overlap-aware and exclusive pyannote annotations."""
    labels = {turn.speaker for turn in result.exclusive_turns}
    return _with_id(
        {
            "schema": DIARIZATION_SCHEMA,
            "source": describe_source(source),
            "model": result.model,
            "device": result.device,
            "speaker_count": len(labels),
            "turns": [_turn_payload(turn) for turn in result.turns],
            "exclusive_turns": [
                _turn_payload(turn) for turn in result.exclusive_turns
            ],
        }
    )


def _require_schema(payload: dict[str, Any], expected: str, label: str) -> None:
    if payload.get("schema") != expected:
        raise ValueError(
            f"Unsupported {label} schema {payload.get('schema')!r}; expected {expected}"
        )


def merge_artifacts(
    asr: dict[str, Any],
    diarization: dict[str, Any],
    speaker_names: list[str] | None = None,
) -> dict[str, Any]:
    """Merge saved ASR words with saved exclusive speaker turns."""
    _require_schema(asr, ASR_SCHEMA, "ASR")
    _require_schema(diarization, DIARIZATION_SCHEMA, "diarization")
    asr_source = asr.get("source", {})
    diarization_source = diarization.get("source", {})
    if asr_source.get("sha256") != diarization_source.get("sha256"):
        raise ValueError("ASR and diarization artifacts refer to different media")

    words = [
        TranscriptWord(float(item["start"]), float(item["end"]), str(item["text"]))
        for item in asr.get("words", [])
    ]
    if not words:
        raise ValueError(
            "ASR artifact has no word timestamps; rerun `ft asr --timestamps word`"
        )
    turns = [
        SpeakerTurn(
            float(item["start"]), float(item["end"]), str(item["speaker"])
        )
        for item in diarization.get("exclusive_turns", [])
    ]
    segments = assign_speakers_to_words(words, turns)
    segments = restore_segment_punctuation(segments, str(asr.get("text", "")))

    return _with_id(
        {
            "schema": TRANSCRIPT_SCHEMA,
            "source": asr_source,
            "parents": [asr.get("id"), diarization.get("id")],
            "engine": asr["engine"],
            "model": asr["model"],
            "language": asr["language"],
            "device": asr["device"],
            "duration_seconds": asr["duration_seconds"],
            "text": asr["text"],
            "speaker_count": int(diarization.get("speaker_count", 0)),
            "speakers": _speaker_metadata(segments, speaker_names),
            "diarization_model": diarization["model"],
            "segments": [_segment_payload(segment) for segment in segments],
            "words": asr["words"],
            "diarization": {
                "device": diarization.get("device"),
                "turns": diarization.get("turns", []),
                "exclusive_turns": diarization.get("exclusive_turns", []),
            },
        }
    )


def transcript_from_artifact(payload: dict[str, Any]) -> TranscriptResult:
    """Deserialize a merged transcript artifact for rendering."""
    _require_schema(payload, TRANSCRIPT_SCHEMA, "transcript")
    display_names: dict[str, str] = {}
    for speaker in payload.get("speakers", []):
        identity = speaker.get("identity")
        if isinstance(identity, dict) and identity.get("name"):
            display_names[str(speaker["id"])] = str(identity["name"])
    return TranscriptResult(
        text=str(payload.get("text", "")),
        segments=[
            TranscriptSegment(
                float(item["start"]),
                float(item["end"]),
                str(item["text"]),
                (
                    display_names.get(str(item["speaker"]), str(item["speaker"]))
                    if item.get("speaker") is not None
                    else None
                ),
            )
            for item in payload.get("segments", [])
        ],
        language=str(payload.get("language", "unknown")),
        duration_min=float(payload.get("duration_seconds", 0)) / 60,
        device=str(payload.get("device", "unknown")),
        model=str(payload.get("model", "unknown")),
        engine=str(payload.get("engine", DEFAULT_ENGINE)),
        speaker_count=int(payload.get("speaker_count", 0)),
        diarization_model=payload.get("diarization_model"),
        words=[
            TranscriptWord(
                float(item["start"]), float(item["end"]), str(item["text"])
            )
            for item in payload.get("words", [])
        ],
    )


def apply_speaker_labels(
    transcript: dict[str, Any], labels: dict[str, Any]
) -> dict[str, Any]:
    """Apply auditable identity/role assertions to a transcript artifact."""
    _require_schema(transcript, TRANSCRIPT_SCHEMA, "transcript")
    _require_schema(labels, SPEAKER_LABELS_SCHEMA, "speaker labels")
    if labels.get("transcript_id") != transcript.get("id"):
        raise ValueError("Speaker labels refer to a different transcript artifact")

    existing = {
        str(speaker["id"]): copy.deepcopy(speaker)
        for speaker in transcript.get("speakers", [])
    }
    updates = labels.get("speakers", [])
    if not isinstance(updates, list):
        raise TypeError("speaker labels 'speakers' must be an array")

    def validate_assertion(value: Any, field: str) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise TypeError(f"speaker {field} must be an object or null")
        if not isinstance(value.get("name"), str) or not value["name"].strip():
            raise ValueError(f"speaker {field} requires a non-empty name")
        if value.get("source") not in {"provided", "inferred", "voiceprint"}:
            raise ValueError(
                f"speaker {field} source must be provided, inferred, or voiceprint"
            )
        confidence = value.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"speaker {field} confidence must be between 0 and 1")
        evidence = value.get("evidence")
        if not isinstance(evidence, list) or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            raise TypeError(f"speaker {field} evidence must be an array of strings")
        if value["source"] == "inferred" and not evidence:
            raise ValueError(f"inferred speaker {field} requires evidence")

    for update in updates:
        if not isinstance(update, dict):
            raise TypeError("each speaker label must be an object")
        speaker_id = str(update.get("id", ""))
        if speaker_id not in existing:
            raise ValueError(f"Unknown speaker ID: {speaker_id}")
        for field in ("identity", "role"):
            if field in update:
                validate_assertion(update[field], field)
        for field in ("identity", "role", "voiceprint"):
            if field in update:
                existing[speaker_id][field] = update[field]

    body = {
        key: value
        for key, value in transcript.items()
        if key not in {"id", "parents", "speakers"}
    }
    body["parents"] = [transcript["id"]]
    body["speakers"] = list(existing.values())
    return _with_id(body)
