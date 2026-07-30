# Free Transcribe

Local audio/video transcription with a quality-first Qwen backend, a fast
Parakeet backend, and optional pyannote speaker diarization.

The current production backend targets Apple Silicon through MLX. The public
engine interface is platform-independent, but CUDA/CPU adapters are still
experimental and must not be considered verified Windows/Linux support yet.

## What to use

| Profile | Pipeline | Use case |
|---|---|---|
| `qwen` (default) | Qwen3-ASR 1.7B | Best text quality |
| `qwen --diarize` | Qwen + ForcedAligner + pyannote | Best text with speakers |
| `parakeet` | Parakeet TDT 0.6B v3 | Very fast transcription |
| `parakeet --diarize` | Parakeet + pyannote | Fast text with speakers |

Models are downloaded lazily from Hugging Face and are not bundled in the
Python package or Git repository.

## Requirements

- Apple Silicon Mac
- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` for video and compressed audio

```bash
brew install ffmpeg uv
```

## Run with uvx

Quality-first transcription:

```bash
uvx --from git+https://github.com/vgmakeev/free-transcribe.git \
  transcribe meeting.webm --lang ru
```

Fast Parakeet transcription:

```bash
uvx --from "free-transcribe[parakeet] @ git+https://github.com/vgmakeev/free-transcribe.git" \
  transcribe meeting.webm --engine parakeet --lang ru
```

Qwen with speaker labels:

```bash
uvx --from "free-transcribe[diarization] @ git+https://github.com/vgmakeev/free-transcribe.git" \
  transcribe meeting.webm --diarize
```

For repeated use, install the tool once:

```bash
uv tool install "free-transcribe[all] @ git+https://github.com/vgmakeev/free-transcribe.git"
```

## Pyannote authorization

Before the first diarized transcription:

1. Accept the conditions for
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create a read token in [Hugging Face settings](https://huggingface.co/settings/tokens).
3. Set `HF_TOKEN` or run `hf auth login`.

```bash
export HF_TOKEN=hf_your_read_token
```

The token is used to download the gated model. It is not written to transcript
files. Cached models can subsequently run offline.

Pyannote clusters voices as `Speaker 1`, `Speaker 2`, and so on. It does not
infer real names, gender, or job roles. Known names can be supplied in order of
first appearance:

```bash
transcribe interview.m4a --diarize --speaker-names "Анна,Виктор,Борис"
```

## CLI

```bash
# Qwen3-ASR 1.7B is the default
transcribe meeting.mp4

# Automatic speaker count
transcribe meeting.mp4 --diarize

# A known count improves consistency
transcribe meeting.mp4 --diarize --speakers 3

# Optional terms known before transcription
transcribe meeting.mp4 --prompt "Stenova 1С PostgreSQL"

# Fast backend
transcribe meeting.mp4 --engine parakeet

# Custom compatible Hugging Face model
transcribe meeting.mp4 --engine qwen --model Qwen/Qwen3-ASR-1.7B
```

| Option | Description |
|---|---|
| `-e, --engine qwen\|parakeet` | ASR engine; Qwen is the default |
| `-m, --model` | Compatible local path or Hugging Face model ID |
| `-l, --lang` | Language name/code; auto-detected when omitted |
| `-p, --prompt` | Optional Qwen context terms known in advance |
| `-d, --diarize` | Run pyannote and label speaker turns |
| `--speakers N` | Exact speaker count, if known |
| `--min-speakers N`, `--max-speakers N` | Automatic-count bounds |
| `--speaker-names "A,B"` | Display names in first-appearance order |
| `--diarization-device auto\|mps\|cuda\|cpu` | Pyannote device |
| `-o, --output` | Markdown output path |

## Agent CLI and reusable artifacts

`free-transcribe` exposes the expensive stages separately. JSON is written to
stdout by default; use `--output` to persist it. Progress and errors go to
stderr, so an agent can safely pipe stdout.

```bash
# Qwen ASR plus word alignment (free-transcribe/asr/v1)
free-transcribe asr meeting.mp4 --lang ru --output asr.json

# Pyannote only (free-transcribe/diarization/v1)
free-transcribe diarize meeting.mp4 --output speakers.json

# Cheap, repeatable merge; models are not loaded again
free-transcribe merge asr.json speakers.json \
  --speaker-names "Анна,Виктор,Борис" --output transcript.json

# Human-readable result
free-transcribe render transcript.json --output transcript.md
```

The diarization artifact contains both `turns`, where overlapping speakers are
preserved, and `exclusive_turns`, which provide one speaker per interval for
unambiguous word assignment. This lets an agent inspect interruptions, change
speaker names, infer roles, or render another format without rerunning pyannote.

The complete pipeline is still available as one command:

```bash
free-transcribe run meeting.mp4 --diarize --output transcript.md
```

The JSON contracts are versioned as `free-transcribe/asr/v1`,
`free-transcribe/diarization/v1`, and `free-transcribe/transcript/v1`.

## Model storage

Typical cache sizes:

| Component | Approximate disk size |
|---|---:|
| Qwen3-ASR 1.7B | 4.4 GB |
| Qwen ForcedAligner 0.6B | 1.7 GB |
| Parakeet TDT 0.6B v3 | 2.3 GB |
| pyannote Community-1 | tens of MB |

Hugging Face uses `~/.cache/huggingface` by default. Move the cache to another
disk when needed:

```bash
export HF_HOME=/Volumes/Models/huggingface
```

Useful cache commands:

```bash
uvx --from huggingface_hub hf cache ls
uvx --from huggingface_hub hf cache rm model/OWNER/MODEL
uv cache prune
```

## Benchmark on an M4 Pro, 24 GB

The same ten-minute Russian meeting excerpt was used for every model. Quality
is a manual comparison of obvious wording and domain-term errors, not a claimed
WER because no human reference transcript was available.

| Model | Processing time | Peak memory | Result |
|---|---:|---:|---|
| Parakeet TDT 0.6B v3 | 5.7 s | 3.68 GB | Very strong and fastest |
| Whisper Medium | 21.7 s | 0.77 GB | Fast, weaker text |
| Whisper Large-v3 Turbo | 19.8 s | 2.80 GB | Weaker than Qwen/Parakeet |
| Whisper Large-v3 | 61.8 s | 2.31 GB | Better Whisper, still weaker |
| MOSS Transcribe Diarize | 60.1 s | 3.76 GB | Two speakers, noisy text |
| VibeVoice-ASR 4-bit | 138.0 s | 11.44 GB | Three speakers, uneven text |
| Qwen3-ASR 1.7B | 179.7 s | 5.03 GB | Best overall domain text |
| pyannote Community-1 | 25.6 s | 1.47 GB | Correctly found three speakers |

Qwen alignment adds substantial latency. Parakeet already emits word timestamps,
so `parakeet --diarize` is the recommended fast profile.

## MCP server

```json
{
  "mcpServers": {
    "free-transcribe": {
      "command": "uvx",
      "args": [
        "--from",
        "free-transcribe[all] @ git+https://github.com/vgmakeev/free-transcribe.git",
        "free-transcribe-mcp"
      ],
      "env": {
        "HF_TOKEN": "hf_your_read_token"
      }
    }
  }
}
```

The MCP tool exposes the same engine, model, language, context, diarization,
speaker-count, speaker-name, and output options as the CLI.

## Portability

The normalized backend interface is ready for platform-specific implementations:

- macOS Apple Silicon: MLX (verified)
- Linux/NVIDIA: PyTorch/CUDA and NeMo adapters (planned)
- Windows/NVIDIA: PyTorch or WSL2 adapters (planned)
- CPU and Linux/AMD ROCm: experimental future targets

The README will only mark a platform verified after real inference tests, not
merely import-only CI.

## License

MIT
