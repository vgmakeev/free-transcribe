"""Composable JSON CLI for agents and scripted transcription workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .core import (
    AVAILABLE_ENGINES,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_ENGINE,
    DiarizationResult,
    SpeakerTurn,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    assign_speakers_to_words,
    diarize_media,
    restore_segment_punctuation,
    result_to_markdown,
    save_transcript,
    transcribe_file,
)

ASR_SCHEMA = "free-transcribe/asr/v1"
DIARIZATION_SCHEMA = "free-transcribe/diarization/v1"
TRANSCRIPT_SCHEMA = "free-transcribe/transcript/v1"


def _progress(stage: str, message: str) -> None:
    print(f"{stage}: {message}", file=sys.stderr, flush=True)


def _write_text(text: str, output: str | None) -> None:
    if not output or output == "-":
        print(text)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(str(path), file=sys.stderr)


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    _write_text(json.dumps(payload, ensure_ascii=False, indent=2), output)


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


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


def asr_artifact(result: TranscriptResult, source: str) -> dict[str, Any]:
    """Serialize ASR output, including aligned words when requested."""
    return {
        "schema": ASR_SCHEMA,
        "source": source,
        "engine": result.engine,
        "model": result.model,
        "language": result.language,
        "device": result.device,
        "duration_seconds": result.duration_min * 60,
        "text": result.text,
        "segments": [_segment_payload(segment) for segment in result.segments],
        "words": [_word_payload(word) for word in result.words],
    }


def diarization_artifact(result: DiarizationResult, source: str) -> dict[str, Any]:
    """Serialize both overlap-aware and exclusive pyannote annotations."""
    labels = {turn.speaker for turn in result.exclusive_turns}
    return {
        "schema": DIARIZATION_SCHEMA,
        "source": source,
        "model": result.model,
        "device": result.device,
        "speaker_count": len(labels),
        "turns": [_turn_payload(turn) for turn in result.turns],
        "exclusive_turns": [
            _turn_payload(turn) for turn in result.exclusive_turns
        ],
    }


def merge_artifacts(
    asr: dict[str, Any],
    diarization: dict[str, Any],
    speaker_names: list[str] | None = None,
) -> dict[str, Any]:
    """Merge previously saved word timestamps and pyannote turns."""
    if asr.get("schema") != ASR_SCHEMA:
        raise ValueError(f"Unsupported ASR schema: {asr.get('schema')!r}")
    if diarization.get("schema") != DIARIZATION_SCHEMA:
        raise ValueError(
            f"Unsupported diarization schema: {diarization.get('schema')!r}"
        )
    if asr.get("source") != diarization.get("source"):
        raise ValueError("ASR and diarization artifacts refer to different sources")

    words = [
        TranscriptWord(float(item["start"]), float(item["end"]), str(item["text"]))
        for item in asr.get("words", [])
    ]
    if not words:
        raise ValueError("ASR artifact has no words; rerun `free-transcribe asr`")
    turns = [
        SpeakerTurn(
            float(item["start"]), float(item["end"]), str(item["speaker"])
        )
        for item in diarization.get("exclusive_turns", [])
    ]
    segments = assign_speakers_to_words(words, turns, speaker_names)
    segments = restore_segment_punctuation(segments, str(asr.get("text", "")))

    return {
        "schema": TRANSCRIPT_SCHEMA,
        "source": asr["source"],
        "engine": asr["engine"],
        "model": asr["model"],
        "language": asr["language"],
        "device": asr["device"],
        "duration_seconds": asr["duration_seconds"],
        "text": asr["text"],
        "speaker_count": int(diarization.get("speaker_count", 0)),
        "diarization_model": diarization["model"],
        "segments": [_segment_payload(segment) for segment in segments],
        "words": asr["words"],
        "diarization": {
            "device": diarization.get("device"),
            "turns": diarization.get("turns", []),
            "exclusive_turns": diarization.get("exclusive_turns", []),
        },
    }


def transcript_from_artifact(payload: dict[str, Any]) -> TranscriptResult:
    """Deserialize a merged transcript artifact for rendering."""
    if payload.get("schema") != TRANSCRIPT_SCHEMA:
        raise ValueError(f"Unsupported transcript schema: {payload.get('schema')!r}")
    return TranscriptResult(
        text=str(payload.get("text", "")),
        segments=[
            TranscriptSegment(
                float(item["start"]),
                float(item["end"]),
                str(item["text"]),
                str(item["speaker"]) if item.get("speaker") is not None else None,
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


def _add_asr_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=AVAILABLE_ENGINES, default=DEFAULT_ENGINE)
    parser.add_argument("--model")
    parser.add_argument("--lang")
    parser.add_argument("--prompt")


def _add_diarization_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL)
    parser.add_argument(
        "--diarization-device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)


def _parse_names(value: str | None) -> list[str] | None:
    return [name.strip() for name in value.split(",")] if value else None


def _run_asr(args: argparse.Namespace) -> None:
    result = transcribe_file(
        args.file,
        engine=args.engine,
        model_name=args.model,
        language=args.lang,
        prompt=args.prompt,
        word_timestamps=not args.no_words,
        on_progress=_progress,
    )
    _write_json(asr_artifact(result, args.file), args.output)


def _run_diarize(args: argparse.Namespace) -> None:
    result = diarize_media(
        args.file,
        model_name=args.diarization_model,
        device=args.diarization_device,
        num_speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        on_progress=_progress,
    )
    _write_json(diarization_artifact(result, args.file), args.output)


def _run_merge(args: argparse.Namespace) -> None:
    payload = merge_artifacts(
        _read_json(args.asr),
        _read_json(args.diarization),
        _parse_names(args.speaker_names),
    )
    _write_json(payload, args.output)


def _run_render(args: argparse.Namespace) -> None:
    payload = _read_json(args.transcript)
    if args.format == "json":
        _write_json(payload, args.output)
        return
    result = transcript_from_artifact(payload)
    source = os.path.basename(str(payload.get("source", "transcript")))
    _write_text(result_to_markdown(result, source), args.output)


def _run_one_shot(args: argparse.Namespace) -> None:
    result = transcribe_file(
        args.file,
        engine=args.engine,
        model_name=args.model,
        language=args.lang,
        prompt=args.prompt,
        diarize=args.diarize,
        diarization_model=args.diarization_model,
        diarization_device=args.diarization_device,
        num_speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        speaker_names=_parse_names(args.speaker_names),
        on_progress=_progress,
    )
    if args.output == "-":
        _write_text(result_to_markdown(result, os.path.basename(args.file)), "-")
    else:
        path = save_transcript(result, args.file, args.output)
        print(path, file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="free-transcribe",
        description="Composable local ASR and pyannote CLI with versioned JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    asr = subparsers.add_parser("asr", help="Transcribe and save timestamped JSON")
    asr.add_argument("file")
    _add_asr_options(asr)
    asr.add_argument("--no-words", action="store_true")
    asr.add_argument("-o", "--output", default="-")
    asr.set_defaults(handler=_run_asr)

    diarize = subparsers.add_parser(
        "diarize", help="Run pyannote and save speaker turns as JSON"
    )
    diarize.add_argument("file")
    _add_diarization_options(diarize)
    diarize.add_argument("-o", "--output", default="-")
    diarize.set_defaults(handler=_run_diarize)

    merge = subparsers.add_parser(
        "merge", help="Merge ASR words with saved pyannote turns"
    )
    merge.add_argument("asr")
    merge.add_argument("diarization")
    merge.add_argument("--speaker-names")
    merge.add_argument("-o", "--output", default="-")
    merge.set_defaults(handler=_run_merge)

    render = subparsers.add_parser("render", help="Render a merged transcript")
    render.add_argument("transcript")
    render.add_argument("--format", choices=["markdown", "json"], default="markdown")
    render.add_argument("-o", "--output", default="-")
    render.set_defaults(handler=_run_render)

    run = subparsers.add_parser("run", help="Run the complete pipeline in one call")
    run.add_argument("file")
    _add_asr_options(run)
    _add_diarization_options(run)
    run.add_argument("--diarize", action="store_true")
    run.add_argument("--speaker-names")
    run.add_argument("-o", "--output")
    run.set_defaults(handler=_run_one_shot)
    return parser


def main() -> None:
    """Agent CLI entry point."""
    args = _build_parser().parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
