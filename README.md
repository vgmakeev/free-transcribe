# Free Transcribe

Accurate local transcription with Qwen, optional pyannote speakers, and a fast
Parakeet alternative. Optimized for Apple Silicon and designed to compose well
from agents and shell scripts.

```text
free_transcribe · Python library/core
├── CLI adapter · ft
├── HTTP adapter · FastAPI
├── agent adapter · MCP + JSON artifacts
└── desktop adapter · tiny Tauri app
```

The interfaces share models, caches, transcript logic, and versioned artifacts;
none of them contains a second transcription implementation.

The web and desktop interfaces expose two profiles everywhere they are
supported: **Fast** uses Parakeet, while **Maximum quality** uses Qwen 1.7B.
Both can add pyannote speaker labels. On Linux the same choices use native
NVIDIA CUDA backends inside the container.

## Quick start

```bash
# Recommended on Apple Silicon: fast text plus automatic speaker labels
uvx --from "free-transcribe[apple] @ git+https://github.com/vgmakeev/free-transcribe.git" \
  ft meeting.mp4 --speakers

# Maximum terminology accuracy: Qwen3-ASR 1.7B plus speakers
uvx --from "free-transcribe[apple] @ git+https://github.com/vgmakeev/free-transcribe.git" \
  ft meeting.mp4 --engine qwen --speakers
```

Models are downloaded lazily from Hugging Face. They are not bundled with the
package or repository. For video input, `ffmpeg` first extracts one temporary
mono 16 kHz FLAC; ASR and pyannote share it, and it is deleted after the job.

Requirements: Python 3.14, `uv`, and `ffmpeg`. Apple Silicon is the verified
inference platform; the Windows/NVIDIA adapter is experimental.

```bash
brew install uv ffmpeg
```

For repeated use:

```bash
uv tool install "free-transcribe[apple] @ git+https://github.com/vgmakeev/free-transcribe.git"
ft doctor
```

## Python library

```python
from free_transcribe import save_transcript, transcribe_file

result = transcribe_file(
    "meeting.mp4",
    engine="qwen",
    language="ru",
    diarize=True,
    num_speakers=3,
)
path = save_transcript(result, "meeting.mp4")
```

The returned `TranscriptResult` contains plain dataclasses for text, segments,
word timestamps, speakers, language, model, and device. The CLI and both server
adapters call this same public function.

## Tiny desktop app

`apps/desktop` is a Tauri 2 shell with no React and no embedded Python or model
weights. The built macOS application is 11 MB. Drop an audio/video file, choose
Quality or Fast, optionally enable speakers, and open the generated Markdown.

The desktop app deliberately reuses the same `ft` backend and lazy model cache:

```bash
uv tool install "free-transcribe[apple] @ git+https://github.com/vgmakeev/free-transcribe.git"
cd apps/desktop
npm ci
npm run tauri dev
```

It detects the engines reported by `ft doctor`, disables unavailable choices,
streams progress, supports cancellation, and recognizes uv's default tool
directory even when launched from Finder. Dependencies and models are still
downloaded only when their installation profile or engine needs them.

## Python HTTP backend

The API is another optional adapter over the same Python library. Its built-in
web UI at `/` accepts drag-and-drop uploads, shows upload/model/ASR/diarization
progress streamed live over SSE, and can copy or download the final transcript. The same operations
remain available as JSON endpoints under `/v1` and OpenAPI at `/docs`.

One worker is the safe default for a single GPU. Additional requests wait in a
bounded in-process queue and receive a queue position; a full queue returns
HTTP `429` with `Retry-After` instead of overcommitting GPU memory.

```bash
uv tool install "free-transcribe[api,apple] @ git+https://github.com/vgmakeev/free-transcribe.git"

# Local-only development server; web UI at / and OpenAPI at /docs
ft serve

# Network deployment must be authenticated
export FT_API_TOKEN='replace-with-a-long-random-token'
ft serve --host 0.0.0.0 --port 8000
```

```bash
job=$(curl -s -H "Authorization: Bearer $FT_API_TOKEN" \
  -F file=@meeting.mp4 -F speakers=true \
  http://localhost:8000/v1/transcriptions)

# Poll the ID returned above, then download its result_url.
curl -H "Authorization: Bearer $FT_API_TOKEN" \
  http://localhost:8000/v1/transcriptions/JOB_ID
```

Environment controls: `FT_API_CONCURRENCY` (default `1`), `FT_API_MAX_QUEUE`
(default `20` waiting jobs), and `FT_MAX_UPLOAD_MB` (default `4096`). Jobs and
uploads are local and ephemeral; they are removed explicitly with
`DELETE /v1/transcriptions/{id}` or when the server stops. Use a reverse proxy
for TLS. Multiple API replicas need a shared queue such as Redis instead of the
built-in single-process queue.

### Ubuntu + NVIDIA CUDA container

The host needs a CUDA 13-capable NVIDIA driver, Docker, Docker Compose, and NVIDIA
Container Toolkit. CUDA user-space libraries and Python 3.14 are included in
the image; the model cache persists in a Docker volume.

```bash
export FT_API_TOKEN="$(openssl rand -hex 32)"
export HF_TOKEN='your-hugging-face-token'  # required for pyannote speakers
docker compose -f compose.cuda.yaml up -d --build

curl http://localhost:8000/health
docker compose -f compose.cuda.yaml logs -f api
```

Compose reserves one NVIDIA GPU, refuses startup when CUDA is unavailable, and
keeps `FT_API_CONCURRENCY=1` by default. Set `FT_API_MAX_QUEUE` to cap waiting
work. Put TLS/auth rate limiting in a reverse proxy before exposing it publicly.

## CLI

```bash
ft meeting.mp4                         # Parakeet on Apple Silicon; Qwen elsewhere
ft meeting.mp4 --speakers              # automatic speaker count
ft meeting.mp4 --speakers 3            # exact speaker count
ft meeting.mp4 --speakers --names "Anna,Victor,Igor"
ft meeting.mp4 --engine parakeet       # fast profile
ft meeting.mp4 --prompt "1C MES литография"
ft meeting.mp4 -o transcript.md
ft meeting.mp4 -o -                    # Markdown to stdout
```

`--speakers` runs pyannote after the selected ASR engine. Qwen additionally
loads its ForcedAligner; Parakeet already supplies word timestamps.

## Agent pipeline

Every expensive stage can be persisted and reused:

```bash
ft asr meeting.mp4 --timestamps word -o asr.json
ft diarize meeting.mp4 -o speakers.json
ft merge asr.json speakers.json -o transcript.json
ft label transcript.json labels.json -o labelled.json   # optional
ft render labelled.json -o transcript.md
```

This makes changing names or rendering cheap:

```bash
ft merge asr.json speakers.json --names "Anna,Victor,Igor" -o named.json
ft render named.json -o named.md
```

JSON goes to stdout by default; progress and errors go to stderr. File outputs
are replaced atomically. Artifacts are deterministic and content-addressed:

- `free-transcribe/asr/v2`
- `free-transcribe/diarization/v2`
- `free-transcribe/transcript/v2`

Each artifact has an ID and a SHA-256 media fingerprint. `merge` refuses inputs
from different media. Transcript artifacts retain parent IDs, aligned words,
regular pyannote turns with overlaps, and exclusive turns used for assignment.

`ft asr` uses cheap segment timestamps by default. Request the expensive Qwen
ForcedAligner explicitly with `--timestamps word`; word timestamps are required
by `ft merge`.

## Speakers

Pyannote clusters voices as `Speaker 1`, `Speaker 2`, and so on. It does not
claim real names, gender, or roles.

- Known names can be supplied with `--names`.
- Names can be inferred only when the recording or external metadata identifies
  participants.
- Roles can be inferred later by an agent from the transcript and stored as
  separate, confidence-labelled metadata.
- Perceived voice gender requires another classifier and is not an identity fact.

Transcript artifacts keep stable `Speaker N` IDs separate from inferred labels.
An agent writes a separate, auditable `speaker-labels/v1` assertion:

```json
{
  "schema": "free-transcribe/speaker-labels/v1",
  "transcript_id": "sha256:...",
  "speakers": [{
    "id": "Speaker 2",
    "identity": {
      "name": "Виктор",
      "source": "inferred",
      "confidence": 0.93,
      "evidence": ["42:57 — «все вопросы Виктор ответит»"]
    },
    "role": {
      "name": "координатор внедрения",
      "source": "inferred",
      "confidence": 0.78,
      "evidence": ["54:07 — обсуждает архитектуру внедрения"]
    }
  }]
}
```

`ft label transcript.json labels.json` validates IDs and evidence provenance,
then creates a new content-addressed transcript whose parent is the unlabelled
artifact. `ft render` uses the identity for display while retaining stable IDs
in JSON. A voiceprint can be attached only after a person supplies a verified
reference recording.

Before the first pyannote run, accept the conditions for
[Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
and authenticate:

```bash
export HF_TOKEN=hf_your_read_token
```

The token is used for gated model download and is never written to artifacts.
Cached models can run offline afterward.

## Installation profiles

| Install | Includes |
|---|---|
| `free-transcribe` | zero-ML artifact CLI: merge, label, render, doctor |
| `free-transcribe[apple]` | both MLX engines and pyannote on Apple Silicon |
| `free-transcribe[cuda]` | Qwen, NeMo Parakeet, and pyannote on Linux/CUDA |
| `free-transcribe[windows]` | Qwen and pyannote on Windows/CUDA |
| `free-transcribe[diarization]` | pyannote |
| `free-transcribe[mcp]` | MCP server |
| `free-transcribe[api]` | FastAPI/uvicorn HTTP server |

Run `ft doctor` for a machine-readable capability report. It checks the
platform, Python, `ffmpeg`, installed engines, pyannote, MCP, and whether a
Hugging Face token is available without exposing its value.

## MCP

MCP is deliberately not part of the base installation:

```json
{
  "mcpServers": {
    "free-transcribe": {
      "command": "uvx",
      "args": [
        "--from",
        "free-transcribe[mcp,apple] @ git+https://github.com/vgmakeev/free-transcribe.git",
        "free-transcribe-mcp"
      ],
      "env": {
        "HF_TOKEN": "hf_your_read_token"
      }
    }
  }
}
```

## Measured profiles

Same ten-minute Russian meeting excerpt, M4 Pro with 24 GB. Quality is a manual
comparison because no human reference transcript was available.

| Pipeline | Time | Peak memory | Outcome |
|---|---:|---:|---|
| Parakeet TDT 0.6B v3 | 5.7 s | 3.68 GB | Fastest, strong text |
| Parakeet + pyannote | 50.2 s | 3.34 GB | Fast speakers, correctly found 3 |
| Qwen3-ASR 1.7B | 179.7 s | 5.03 GB | Best domain text |
| pyannote Community-1 | 25.6 s | 1.47 GB | Correctly found 3 speakers |

On the complete 54:56 recording, chunked Parakeet plus pyannote finished in
3:22, used 3.45 GB peak RSS, and found three speakers automatically. Qwen plus
ForcedAligner and pyannote took 19:43 and about 5 GB RSS, but produced the best
domain terminology. The practical Apple Silicon default is Parakeet plus
pyannote; use Qwen 1.7B when accuracy matters enough to accept the longer run.

## Model storage

Typical Hugging Face cache sizes:

| Model | Approximate size |
|---|---:|
| Qwen3-ASR 1.7B | 4.7 GB |
| Qwen ForcedAligner 0.6B | 1.8 GB |
| Parakeet TDT 0.6B v3 | 2.3 GB |
| pyannote Community-1 | 33 MB |

```bash
export HF_HOME=/Volumes/Models/huggingface
uvx --from huggingface_hub hf cache ls
uv cache prune
```

## Platforms

- macOS Apple Silicon with MLX: verified
- Windows/NVIDIA: official Qwen Transformers/CUDA adapter implemented; dependency
  resolution and Tauri compilation are tested, real GPU inference is not yet verified
- Linux/NVIDIA: Qwen and NVIDIA NeMo Parakeet CUDA adapters are packaged; real
  GPU inference is not yet verified
- CPU, ROCm, and other platforms: not production-supported yet

The CLI and artifact contracts are platform-neutral. A platform is only marked
verified after real inference tests.

## License

MIT
