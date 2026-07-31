"""Optional MCP adapter for Free Transcribe."""

import logging
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .core import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    DEFAULT_MODELS,
    SUPPORTED_FORMATS,
    result_to_markdown,
    save_transcript,
    transcribe_file,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("free-transcribe")

# Create MCP server
server = Server("free-transcribe")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="transcribe",
            description="""Transcribe audio/video locally with Qwen or Parakeet.

Supports formats: mp3, wav, m4a, flac, ogg, mp4, webm, mkv, avi, mov.
Apple Silicon uses MLX acceleration. Speaker diarization uses pyannote.

The transcript is saved as a Markdown file next to the source file (in ./Transcripts/ folder)
and the content is also returned directly.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Absolute path to audio/video file",
                    },
                    "engine": {
                        "type": "string",
                        "enum": list(AVAILABLE_ENGINES),
                        "default": DEFAULT_ENGINE,
                        "description": "ASR engine; qwen prioritizes quality",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional local/Hugging Face model override",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g., 'ru', 'en'). Auto-detect if not specified.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Context, terminology, and proper names that improve accuracy",
                    },
                    "diarize": {
                        "type": "boolean",
                        "default": False,
                        "description": "Identify and label speaker turns locally with pyannote",
                    },
                    "num_speakers": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Exact number of speakers, if known",
                    },
                    "min_speakers": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Minimum expected number of speakers",
                    },
                    "max_speakers": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum expected number of speakers",
                    },
                    "speaker_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Known names in order of first appearance",
                    },
                    "diarization_device": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                        "description": "Device used for speaker diarization",
                    },
                    "output": {
                        "type": "string",
                        "description": "Custom output file path. Default: ./Transcripts/<date> <name> Transcript.md",
                    },
                    "save_file": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to save transcript to file (default: true)",
                    },
                },
                "required": ["file"],
            },
        ),
        Tool(
            name="transcribe_info",
            description="Get information about available models and supported formats.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "transcribe_info":
        info = f"""# Free Transcribe

## Supported Formats
**Audio:** {", ".join(sorted(f for f in SUPPORTED_FORMATS if f in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac"}))}
**Video:** {", ".join(sorted(f for f in SUPPORTED_FORMATS if f in {".mp4", ".webm", ".mkv", ".avi", ".mov"}))}

## Engines
| Engine | Default model | Profile |
|--------|---------------|---------|
| **qwen** | {DEFAULT_MODELS["qwen"]} | Best quality (default) |
| **parakeet** | {DEFAULT_MODELS["parakeet"]} | Fast |

Models are downloaded automatically from Hugging Face on first use.
Apple Silicon is optimized with MLX. Other platform adapters are experimental.

Speaker diarization is available with `diarize: true` after installing the
`diarization` extra and authorizing the pyannote Community-1 model.
"""
        return [TextContent(type="text", text=info)]

    if name == "transcribe":
        file_path = arguments.get("file")
        engine = arguments.get("engine", DEFAULT_ENGINE)
        model_name = arguments.get("model")
        language = arguments.get("language")
        prompt = arguments.get("prompt")
        diarize = arguments.get("diarize", False)
        num_speakers = arguments.get("num_speakers")
        min_speakers = arguments.get("min_speakers")
        max_speakers = arguments.get("max_speakers")
        speaker_names = arguments.get("speaker_names")
        diarization_device = arguments.get("diarization_device", "auto")
        diarize = bool(
            diarize or num_speakers or min_speakers or max_speakers or speaker_names
        )
        output = arguments.get("output")
        save_file = arguments.get("save_file", True)

        if not file_path:
            return [
                TextContent(type="text", text="❌ Error: 'file' parameter is required")
            ]

        # Expand path
        file_path = os.path.expanduser(file_path)

        if not os.path.exists(file_path):
            return [
                TextContent(type="text", text=f"❌ Error: File not found: {file_path}")
            ]

        logger.info("Transcribing %s with engine=%s model=%s", file_path, engine, model_name)

        try:
            # Progress callback for logging
            def on_progress(stage: str, message: str) -> None:
                logger.info(f"[{stage}] {message}")

            result = transcribe_file(
                file_path=file_path,
                engine=engine,
                model_name=model_name,
                language=language,
                prompt=prompt,
                on_progress=on_progress,
                diarize=diarize,
                diarization_device=diarization_device,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                speaker_names=speaker_names,
            )

            # Generate markdown content
            source_filename = os.path.basename(file_path)
            markdown = result_to_markdown(result, source_filename)

            # Save file if requested
            output_info = ""
            if save_file:
                if output:
                    output = os.path.expanduser(output)
                output_file = save_transcript(
                    result=result,
                    source_path=file_path,
                    output_path=output,
                )
                output_info = f"\n\n---\n✅ Saved to: {output_file}"

            response = f"""{markdown}{output_info}

---
**Stats:** {result.duration_min:.1f} min | {len(result.segments)} segments | Engine: {result.engine} | Language: {result.language} | Speakers: {result.speaker_count or "not detected"} | Device: {result.device}"""

            return [TextContent(type="text", text=response)]

        except FileNotFoundError as e:
            return [TextContent(type="text", text=f"❌ Error: {e}")]
        except ValueError as e:
            return [TextContent(type="text", text=f"❌ Error: {e}")]
        except Exception as e:
            logger.exception("Transcription failed")
            return [TextContent(type="text", text=f"❌ Error: {e}")]

    return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Entry point for MCP server."""
    import asyncio

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
