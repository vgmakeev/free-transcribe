"""
CLI for local audio/video transcription.
"""

import argparse
import os
import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .core import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    DEFAULT_MODELS,
    SUPPORTED_FORMATS,
    save_transcript,
    transcribe_file,
)

console = Console()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio/video locally with Qwen or Parakeet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  transcribe meeting.webm
  transcribe podcast.mp3 --engine qwen --lang en
  transcribe meeting.mp4 --engine parakeet
  transcribe interview.m4a --diarize --speakers 2
  transcribe meeting.mp4 --diarize --speaker-names "Анна,Борис"
  transcribe interview.m4a --output ./result.md
  transcribe call.wav --prompt "Technical discussion about Python"
        """,
    )

    parser.add_argument(
        "file",
        help="Path to audio/video file (mp3, wav, m4a, mp4, webm, etc.)",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=AVAILABLE_ENGINES,
        default=DEFAULT_ENGINE,
        help=f"ASR engine (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "Custom local/Hugging Face model ID. Defaults: "
            + ", ".join(f"{key}={value}" for key, value in DEFAULT_MODELS.items())
        ),
    )
    parser.add_argument(
        "-l",
        "--lang",
        default=None,
        help="Language code (e.g., ru, en). Auto-detect if not specified.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. Default: ./Transcripts/<date> <name> Transcript.md",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Qwen context terms and proper names that may improve accuracy.",
    )
    parser.add_argument(
        "-d",
        "--diarize",
        action="store_true",
        help="Identify speaker turns with the optional local pyannote model.",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Exact number of speakers, when known (improves diarization accuracy).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum expected number of speakers.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum expected number of speakers.",
    )
    parser.add_argument(
        "--speaker-names",
        default=None,
        help='Comma-separated names in order of first appearance, e.g. "Анна,Борис".',
    )
    parser.add_argument(
        "--diarization-device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device for speaker diarization (default: auto; CUDA, MPS, then CPU).",
    )

    args = parser.parse_args()

    # Validate file exists
    if not os.path.exists(args.file):
        console.print(f"[red]❌ File not found: {args.file}[/red]")
        sys.exit(1)

    # Validate format
    ext = os.path.splitext(args.file)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        console.print(f"[yellow]⚠️ Unknown format {ext}, attempting anyway...[/yellow]")

    # Progress tracking
    progress_ctx = None
    task_id = None

    def on_progress(stage: str, message: str) -> None:
        nonlocal progress_ctx, task_id

        if stage == "device":
            console.print("[green]🚀 Using MLX (Apple Silicon GPU)[/green]")
        elif stage == "loading":
            progress_ctx = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            )
            progress_ctx.start()
            task_id = progress_ctx.add_task(f"[cyan]{message}", total=None)
        elif stage == "loaded":
            if progress_ctx and task_id is not None:
                progress_ctx.update(
                    task_id, completed=True, description=f"[green]✅ {message}"
                )
        elif stage == "transcribing":
            if progress_ctx and task_id is not None:
                progress_ctx.update(task_id, description=f"[cyan]{message}")
        elif stage in {"diarization_loading", "diarizing"}:
            if progress_ctx and task_id is not None:
                progress_ctx.update(task_id, completed=True)
                task_id = progress_ctx.add_task(f"[cyan]{message}", total=None)
        elif stage == "diarization_complete":
            if progress_ctx and task_id is not None:
                progress_ctx.update(
                    task_id, completed=True, description=f"[green]✅ {message}"
                )
        elif stage == "complete" and progress_ctx and task_id is not None:
            progress_ctx.update(
                task_id, completed=True, description=f"[green]✅ {message}"
            )
            progress_ctx.stop()

    try:
        diarize_requested = bool(
            args.diarize
            or args.speakers
            or args.min_speakers
            or args.max_speakers
            or args.speaker_names
        )
        result = transcribe_file(
            file_path=args.file,
            engine=args.engine,
            model_name=args.model,
            language=args.lang,
            prompt=args.prompt,
            on_progress=on_progress,
            diarize=diarize_requested,
            diarization_device=args.diarization_device,
            num_speakers=args.speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            speaker_names=(
                [name.strip() for name in args.speaker_names.split(",")]
                if args.speaker_names
                else None
            ),
        )

        output_file = save_transcript(
            result=result,
            source_path=args.file,
            output_path=args.output,
        )

        console.print(f"\n[green]✅ Saved:[/green] {output_file}")
        speaker_stats = (
            f" | Speakers: {result.speaker_count}" if result.speaker_count else ""
        )
        console.print(
            f"[dim]Duration: {result.duration_min:.1f} min | "
            f"Segments: {len(result.segments)} | Engine: {result.engine} | "
            f"Language: {result.language}{speaker_stats}[/dim]"
        )

    except FileNotFoundError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - CLI boundary reports model/runtime errors.
        if progress_ctx:
            progress_ctx.stop()
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
