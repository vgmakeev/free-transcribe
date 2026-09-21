# Free Transcribe

Transcription for NVIDIA GPUs with three sequential stages:

1. NVIDIA Parakeet TDT 0.6B v3 recognizes speech and produces word timestamps.
2. pyannote Community-1 identifies speaker turns.
3. Optional Gemini 3.8 Flash corrects recognition errors using the complete dialogue and an IT glossary.

The local models and their CUDA execution stay independent. Parakeet is released before pyannote is loaded, which limits peak GPU memory. Gemini receives transcript text only; audio and video are never sent to it.

## CUDA setup

Requirements: Linux, NVIDIA driver with CUDA support, Python 3.14, `ffmpeg`, and a Hugging Face token authorized for [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

```bash
git clone https://github.com/vgmakeev/free-transcribe.git
cd free-transcribe
uv sync --extra cuda --extra api

export HF_TOKEN='...'
uv run ft doctor
```

The CUDA extra keeps the established versions of `nemo-toolkit[asr]`, `numba`, and `pyannote.audio`.

## CLI

```bash
# Parakeet only
uv run ft meeting.webm

# Parakeet + automatic pyannote diarization
uv run ft meeting.webm --speakers

# Exact speaker count and deterministic names by first appearance
uv run ft meeting.webm --speakers 2 --names 'Владимир,Иван'

# Add full-context Gemini correction
export GEMINI_API_KEY='...'
uv run ft meeting.webm --speakers --gemini

# Use another compact glossary
uv run ft meeting.webm --speakers --gemini --glossary-file team-glossary.json
```

The result is Markdown in `Transcripts/` next to the source file. `-o result.md` selects another path and `-o -` writes Markdown to stdout.

## Gemini correction

The bundled glossary is [`src/free_transcribe/glossary.json`](src/free_transcribe/glossary.json). Set `FT_GLOSSARY_FILE` to replace it for a running service. Set `FT_GEMINI_MODEL` to override the default `gemini-3.8-flash` model.

The correction request contains every timestamped, speaker-labelled segment in one prompt. Gemini returns only edits with this shape:

```json
{"edits":[{"id":12,"old":"джиру","new":"Jira"}]}
```

The client applies an edit only when `old` occurs exactly once in the referenced source segment and does not overlap another edit. Invalid or ambiguous edits are rejected. Speaker labels, timestamps, numbers, negations, and unchanged text are never regenerated. Applied and rejected edit counts are written to the Markdown front matter.

Gemini correction is opt in. Keep `GEMINI_API_KEY` in the server environment; do not put it in the browser, glossary, repository, or command history.

## Web interface

```bash
export HF_TOKEN='...'
export GEMINI_API_KEY='...'       # optional
export FT_API_TOKEN='...'         # required for a non-local bind
uv run ft serve --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

For video uploads, the page uses the browser's built-in Web Audio API to decode the selected file and create mono 16 kHz WAV locally. Only that audio file is uploaded. This avoids downloading the large ffmpeg.wasm runtime. Codec support follows the browser; if the browser cannot decode a container, convert it locally to WAV or FLAC. Audio uploads pass through unchanged.

The page exposes speaker diarization and Gemini correction only when `/health` reports that each backend is configured. Jobs run through a bounded queue; the default concurrency is one to avoid competing GPU model loads.

## Docker on NVIDIA

```bash
export FT_API_TOKEN='...'
export HF_TOKEN='...'
export GEMINI_API_KEY='...'       # optional
docker compose -f compose.cuda.yaml up -d --build
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `FT_API_CONCURRENCY` | `1` | simultaneous inference jobs |
| `FT_API_MAX_QUEUE` | `20` | waiting jobs |
| `FT_MAX_UPLOAD_MB` | `4096` | upload limit |
| `FT_REQUIRE_CUDA` | `1` in the image | fail startup without CUDA |
| `FT_GEMINI_MODEL` | `gemini-3.8-flash` | proofreading model |
| `FT_GLOSSARY_FILE` | bundled glossary | custom JSON glossary |

## HTTP API

`POST /v1/transcriptions` accepts multipart form data:

| Field | Type | Description |
|---|---|---|
| `file` | file | supported audio or video |
| `engine` | string | `parakeet` |
| `language` | string | optional language hint |
| `speakers` | boolean | run pyannote |
| `speaker_count` | integer | optional exact count |
| `speaker_names` | string | comma-separated names by first appearance |
| `gemini` | boolean | run text correction |

The response is a job resource. Read progress with `GET /v1/transcriptions/{id}/events`, then download Markdown from the returned `result_url`.

## Agent commands

The staged commands preserve auditable intermediate artifacts:

```bash
uv run ft asr meeting.wav --timestamps word -o asr.json
uv run ft diarize meeting.wav --speakers auto -o speakers.json
uv run ft merge asr.json speakers.json --names 'Владимир,Иван' -o transcript.json
uv run ft render transcript.json -o transcript.md
```

An MCP adapter is available with the `mcp` extra:

```bash
uv sync --extra cuda --extra mcp
uv run free-transcribe-mcp
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
cd apps/desktop && npm ci && npm run build
```

Unit tests mock model inference and Gemini HTTP calls. Run a real CUDA smoke test on the deployment host before release.
