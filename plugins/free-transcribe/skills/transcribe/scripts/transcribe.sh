#!/usr/bin/env bash
# Transcribe a local recording on a remote free-transcribe server.
#
# Compresses the audio track locally (when ffmpeg is available), uploads it to
# the HTTP API, waits for the job and saves the Markdown transcript next to the
# recording. The job id is printed as soon as the upload is accepted, so an
# interrupted run can be picked up again with --resume.
#
# Environment:
#   TRANSCRIBE_URL    server base URL, e.g. https://transcribe.example.com
#   TRANSCRIBE_AUTH   "user:password" for HTTP Basic auth (reverse proxy), or
#   TRANSCRIBE_TOKEN  API bearer token (direct access to the service)
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  transcribe.sh <media-file> [options]
  transcribe.sh --resume <job-id> --out <file.md>
  transcribe.sh --check

Options:
  --names "A,B"       speaker names in order of first appearance (implies diarization)
  --speakers N        exact number of speakers
  --no-speakers       skip speaker diarization
  --no-gemini         skip Gemini transcript correction
  --language CODE     language hint, e.g. ru
  --out FILE          where to save the transcript (default: <recording>.transcript.md)
  --keep-job          keep the job on the server after download (default: delete it)
  --no-compress       upload the original file as is
EOF
}

die() { echo "transcribe: $*" >&2; exit 1; }
log() { echo "transcribe: $*" >&2; }

url="${TRANSCRIBE_URL:-}"
url="${url%/}"
auth=()
if [[ -n "${TRANSCRIBE_AUTH:-}" ]]; then
  auth=(-u "$TRANSCRIBE_AUTH")
elif [[ -n "${TRANSCRIBE_TOKEN:-}" ]]; then
  auth=(-H "Authorization: Bearer $TRANSCRIBE_TOKEN")
fi

require_config() {
  [[ -n "$url" ]] || die "TRANSCRIBE_URL is not set"
  [[ ${#auth[@]} -gt 0 ]] || die "set TRANSCRIBE_AUTH (user:password) or TRANSCRIBE_TOKEN"
}

# Tiny JSON field readers: the API returns flat, predictable objects.
json_str() { sed -n "s/.*\"$1\":\"\([^\"]*\)\".*/\1/p" <<<"$2" | head -n 1; }

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
body="" http_code=""

api() {
  # api <method> <path> [curl args...]: sets $body and $http_code.
  # Runs in the current shell (no subshell) so both values reach the caller.
  local method=$1 path=$2
  shift 2
  http_code=$(curl -sS -X "$method" "${auth[@]}" -o "$workdir/response" \
    -w '%{http_code}' "$@" "$url$path") || return 1
  body=$(cat "$workdir/response" 2>/dev/null || true)
}

wait_and_fetch() {
  local job=$1 out_file=$2 status message last="" failures=0
  while true; do
    if ! api GET "/v1/transcriptions/$job"; then
      failures=$((failures + 1))
      ((failures <= 30)) || die "server unreachable; resume with: --resume $job --out '$out_file'"
      log "network error, retrying ($failures)"
      sleep 10
      continue
    fi
    failures=0
    case "$http_code" in
      200) ;;
      404) die "job $job not found (expired or deleted)" ;;
      *) die "status request failed with HTTP $http_code: $body" ;;
    esac
    status=$(json_str status "$body")
    message=$(json_str message "$body")
    if [[ "$message" != "$last" ]]; then
      log "$status: $message"
      last=$message
    fi
    case "$status" in
      succeeded) break ;;
      failed) die "transcription failed: $(json_str error "$body")" ;;
    esac
    sleep 10
  done

  local tmp="$out_file.part"
  curl -sS -f "${auth[@]}" -o "$tmp" "$url/v1/transcriptions/$job/result" \
    || die "could not download the result; resume with: --resume $job --out '$out_file'"
  mv "$tmp" "$out_file"
  if [[ "$keep_job" != 1 ]]; then
    api DELETE "/v1/transcriptions/$job" || true
  fi
  log "saved $out_file"
  echo "$out_file"
}

names="" speakers="" diarize=1 gemini=1 language="" out="" keep_job=0 compress=1
resume="" check=0 media=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --names) names=$2; shift 2 ;;
    --speakers) speakers=$2; shift 2 ;;
    --no-speakers) diarize=0; shift ;;
    --no-gemini) gemini=0; shift ;;
    --language) language=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --keep-job) keep_job=1; shift ;;
    --no-compress) compress=0; shift ;;
    --resume) resume=$2; shift 2 ;;
    --check) check=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) [[ -z "$media" ]] || die "only one media file at a time"; media=$1; shift ;;
  esac
done

require_config

if [[ "$check" == 1 ]]; then
  api GET /health || die "cannot reach $url"
  case "$http_code" in
    200) ;;
    401) die "authentication rejected by $url: check TRANSCRIBE_AUTH / TRANSCRIBE_TOKEN" ;;
    *) die "health check failed with HTTP $http_code" ;;
  esac
  # /health is public on the API itself; a protected endpoint proves the token.
  api GET /v1/transcriptions/credentials-check || die "cannot reach $url"
  [[ "$http_code" == 404 ]] || die "API rejected the credentials (HTTP $http_code)"
  echo "ok: $url is reachable, credentials accepted"
  exit 0
fi

if [[ -n "$resume" ]]; then
  [[ -n "$out" ]] || die "--resume needs --out"
  wait_and_fetch "$resume" "$out"
  exit 0
fi

[[ -n "$media" ]] || { usage >&2; exit 1; }
[[ -f "$media" ]] || die "no such file: $media"
[[ -n "$out" ]] || out="${media%.*}.transcript.md"
if [[ -e "$out" ]]; then
  base="${out%.md}" n=2
  while [[ -e "$base.$n.md" ]]; do n=$((n + 1)); done
  out="$base.$n.md"
fi

upload="$media"
ext=$(tr '[:upper:]' '[:lower:]' <<<"${media##*.}")
if [[ "$compress" == 1 && "$ext" != ogg ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    upload="$workdir/audio.ogg"
    log "extracting and compressing audio (opus 48 kbit/s, mono, 16 kHz)"
    # The server resamples to 16 kHz mono anyway; 48 kbit/s opus keeps the
    # recognised text practically identical to lossless input at ~20 MB/hour.
    ffmpeg -nostdin -v error -y -i "$media" -map 0:a:0 -vn -ac 1 -ar 16000 \
      -c:a libopus -b:a 48k "$upload" \
      || die "ffmpeg could not extract audio from $media"
  else
    log "ffmpeg not found: uploading the original file (slower for video)"
  fi
fi

size_mb=$(LC_ALL=C awk -v bytes="$(wc -c <"$upload")" 'BEGIN { printf "%.1f", bytes / 1048576 }')
log "uploading ${size_mb} MB to $url"
form=(-F "file=@$upload" -F "engine=parakeet")
[[ "$diarize" == 1 ]] && form+=(-F "speakers=true")
[[ -n "$speakers" ]] && form+=(-F "speaker_count=$speakers")
[[ -n "$names" ]] && form+=(-F "speaker_names=$names")
[[ "$gemini" == 1 ]] && form+=(-F "gemini=true")
[[ -n "$language" ]] && form+=(-F "language=$language")

for attempt in 1 2 3 4 5; do
  api POST /v1/transcriptions "${form[@]}" || die "upload failed (network)"
  case "$http_code" in
    202) break ;;
    429) log "server queue is full, retrying in 30 s"; sleep 30 ;;
    401) die "authentication rejected: check TRANSCRIBE_AUTH / TRANSCRIBE_TOKEN" ;;
    413) die "file is too large for the server" ;;
    415) die "unsupported media format" ;;
    *) die "upload rejected with HTTP $http_code: $body" ;;
  esac
  [[ $attempt -lt 5 ]] || die "server queue stayed full"
done

job=$(json_str id "$body")
[[ -n "$job" ]] || die "unexpected server response: $body"
log "job $job accepted; if interrupted, resume with: --resume $job --out '$out'"
wait_and_fetch "$job" "$out"
