"""One human-friendly and agent-first command-line interface."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import (
    apply_speaker_labels,
    asr_artifact,
    diarization_artifact,
    merge_artifacts,
    transcript_from_artifact,
)
from .core import (
    AVAILABLE_ENGINES,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_ENGINE,
    diarize_media,
    result_to_markdown,
    save_transcript,
    transcribe_file,
)

COMMANDS = {
    "run",
    "asr",
    "diarize",
    "merge",
    "label",
    "render",
    "doctor",
    "serve",
}


def _progress(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", file=sys.stderr, flush=True)


def _write_text(text: str, output: str | None) -> None:
    if not output or output == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return

    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(path, file=sys.stderr)


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    _write_text(json.dumps(payload, ensure_ascii=False, indent=2), output)


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _parse_names(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [name.strip() for name in value.split(",") if name.strip()] or None


def _speaker_request(value: str | None) -> tuple[bool, int | None]:
    if value is None:
        return False, None
    if value == "auto":
        return True, None
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError("--speakers must be omitted, 'auto', or a positive integer") from exc
    if count < 1:
        raise ValueError("--speakers must be a positive integer")
    return True, count


def _run_asr(args: argparse.Namespace) -> None:
    result = transcribe_file(
        args.file,
        engine=args.engine,
        model_name=args.model,
        language=args.lang,
        prompt=args.prompt,
        word_timestamps=args.timestamps == "word",
        on_progress=_progress,
    )
    if args.timestamps != "word":
        result.words = []
    _write_json(asr_artifact(result, args.file), args.output)


def _run_diarize(args: argparse.Namespace) -> None:
    _, num_speakers = _speaker_request(args.speakers)
    result = diarize_media(
        args.file,
        model_name=args.diarization_model,
        device=args.diarization_device,
        num_speakers=num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        on_progress=_progress,
    )
    _write_json(diarization_artifact(result, args.file), args.output)


def _run_merge(args: argparse.Namespace) -> None:
    payload = merge_artifacts(
        _read_json(args.asr),
        _read_json(args.diarization),
        _parse_names(args.names),
    )
    _write_json(payload, args.output)


def _run_render(args: argparse.Namespace) -> None:
    payload = _read_json(args.transcript)
    if args.format == "json":
        _write_json(payload, args.output)
        return
    result = transcript_from_artifact(payload)
    source = payload.get("source", {})
    source_name = str(source.get("name", "transcript"))
    _write_text(result_to_markdown(result, source_name), args.output)


def _run_label(args: argparse.Namespace) -> None:
    payload = apply_speaker_labels(
        _read_json(args.transcript), _read_json(args.labels)
    )
    _write_json(payload, args.output)


def _run_one_shot(args: argparse.Namespace) -> None:
    speakers_requested, num_speakers = _speaker_request(args.speakers)
    diarize = bool(
        speakers_requested
        or args.min_speakers
        or args.max_speakers
        or args.names
    )
    result = transcribe_file(
        args.file,
        engine=args.engine,
        model_name=args.model,
        language=args.lang,
        prompt=args.prompt,
        diarize=diarize,
        diarization_model=args.diarization_model,
        diarization_device=args.diarization_device,
        num_speakers=num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        speaker_names=_parse_names(args.names),
        on_progress=_progress,
    )
    if args.output == "-":
        _write_text(result_to_markdown(result, os.path.basename(args.file)), "-")
        return
    output = save_transcript(result, args.file, args.output)
    print(output, file=sys.stderr)


def _run_doctor(args: argparse.Namespace) -> None:
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    mlx_qwen = importlib.util.find_spec("mlx_qwen3_asr") is not None
    qwen_torch = importlib.util.find_spec("qwen_asr") is not None
    parakeet_mlx = importlib.util.find_spec("mlx_audio") is not None
    parakeet_cuda = importlib.util.find_spec("nemo") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    cuda = False
    if torch_installed and not apple_silicon:
        try:
            import torch

            cuda = torch.cuda.is_available()
        except (ImportError, RuntimeError):
            cuda = False
    capabilities = {
        "qwen": mlx_qwen or qwen_torch,
        "qwen_mlx": mlx_qwen,
        "qwen_transformers": qwen_torch,
        "parakeet": parakeet_mlx or parakeet_cuda,
        "parakeet_mlx": parakeet_mlx,
        "parakeet_cuda": parakeet_cuda,
        "diarization": importlib.util.find_spec("pyannote.audio") is not None,
        "mcp": importlib.util.find_spec("mcp") is not None,
        "api": importlib.util.find_spec("fastapi") is not None
        and importlib.util.find_spec("uvicorn") is not None,
    }
    qwen_ready = (apple_silicon and mlx_qwen) or (cuda and qwen_torch)
    parakeet_ready = (apple_silicon and parakeet_mlx) or (cuda and parakeet_cuda)
    payload = {
        "schema": "free-transcribe/doctor/v1",
        "version": __version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "apple_silicon": apple_silicon,
            "cuda": cuda,
        },
        "commands": {"ffmpeg": shutil.which("ffmpeg")},
        "capabilities": capabilities,
        "hugging_face_token": bool(
            os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        ),
        "ready": {
            "qwen": qwen_ready,
            "parakeet": parakeet_ready,
            "diarization": capabilities["diarization"],
            "word_alignment": qwen_ready,
            "speaker_transcription": capabilities["diarization"]
            and (qwen_ready or parakeet_ready),
            "mcp": capabilities["mcp"],
            "api": capabilities["api"],
            "artifact_tools": True,
        },
    }
    _write_json(payload, args.output)


def _run_serve(args: argparse.Namespace) -> None:
    try:
        from .api import run
    except ImportError as exc:
        raise RuntimeError("Install the 'api' extra to run the HTTP server") from exc
    run(host=args.host, port=args.port)


def _add_asr_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e", "--engine", choices=AVAILABLE_ENGINES, default=DEFAULT_ENGINE
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model path or Hugging Face ID; selected engine default if omitted",
    )
    parser.add_argument("-l", "--lang", help="language code/name; auto if omitted")
    parser.add_argument("-p", "--prompt", help="known terms or names for Qwen")


def _add_speaker_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--speakers",
        nargs="?",
        const="auto",
        metavar="N",
        help="identify speakers automatically, or specify the exact count",
    )
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--names", help="comma-separated names by first appearance")
    parser.add_argument(
        "--diarization-model", default=DEFAULT_DIARIZATION_MODEL, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--diarization-device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help=argparse.SUPPRESS,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ft",
        description="Local transcription. Human-simple, agent-composable.",
        epilog="Shortcut: `ft MEDIA` is equivalent to `ft run MEDIA`.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="transcribe media to Markdown")
    run.add_argument("file")
    _add_asr_options(run)
    _add_speaker_options(run)
    run.add_argument("-o", "--output", help="Markdown path; '-' writes to stdout")
    run.set_defaults(handler=_run_one_shot)

    asr = subparsers.add_parser("asr", help="create a versioned ASR JSON artifact")
    asr.add_argument("file")
    _add_asr_options(asr)
    asr.add_argument(
        "--timestamps",
        choices=["segment", "word"],
        default="segment",
        help="word invokes the Qwen ForcedAligner and is required by merge",
    )
    asr.add_argument("-o", "--output", default="-")
    asr.set_defaults(handler=_run_asr)

    diarize = subparsers.add_parser(
        "diarize", help="create a versioned pyannote JSON artifact"
    )
    diarize.add_argument("file")
    _add_speaker_options(diarize)
    diarize.add_argument("-o", "--output", default="-")
    diarize.set_defaults(handler=_run_diarize)

    merge = subparsers.add_parser("merge", help="merge ASR and speaker artifacts")
    merge.add_argument("asr")
    merge.add_argument("diarization")
    merge.add_argument("--names", help="comma-separated names by first appearance")
    merge.add_argument("-o", "--output", default="-")
    merge.set_defaults(handler=_run_merge)

    label = subparsers.add_parser(
        "label", help="apply evidence-backed speaker identities and roles"
    )
    label.add_argument("transcript")
    label.add_argument("labels")
    label.add_argument("-o", "--output", default="-")
    label.set_defaults(handler=_run_label)

    render = subparsers.add_parser("render", help="render a transcript artifact")
    render.add_argument("transcript")
    render.add_argument("--format", choices=["markdown", "json"], default="markdown")
    render.add_argument("-o", "--output", default="-")
    render.set_defaults(handler=_run_render)

    doctor = subparsers.add_parser("doctor", help="report local capabilities as JSON")
    doctor.add_argument("-o", "--output", default="-")
    doctor.set_defaults(handler=_run_doctor)

    serve = subparsers.add_parser("serve", help="run the optional HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_run_serve)
    return parser


def _normalized_argv(argv: list[str]) -> list[str]:
    if not argv or argv[0] in {"-h", "--help", "--version"}:
        return argv
    return argv if argv[0] in COMMANDS else ["run", *argv]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(_normalized_argv(list(argv or sys.argv[1:])))
    try:
        args.handler(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
