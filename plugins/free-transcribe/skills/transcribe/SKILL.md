---
name: transcribe
description: >-
  Transcribe a local audio or video recording (meeting, interview, call, lecture) into a Markdown
  transcript with speaker labels on a remote free-transcribe server. Use whenever the user asks to
  transcribe, decode or get the text of a recording ("транскрибируй", "расшифруй запись",
  "сделай стенограмму встречи", "что говорили на созвоне" with a media file), or points at an
  mp3/m4a/wav/ogg/flac/mp4/webm/mkv/mov file and wants its contents as text.
argument-hint: "[media-file] [speaker names]"
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/scripts/transcribe.sh *)
---

# Transcribe a recording

The bundled script does the whole job: it extracts and compresses the audio track locally
(ffmpeg, opus 48 kbit/s, ~20 MB per hour instead of gigabytes of video), uploads it, waits for the
server and saves the transcript next to the recording as `<name>.transcript.md`. Do not build
curl requests by hand and do not convert the file yourself: the script already does both.

## 1. Check the setup once

```bash
${CLAUDE_SKILL_DIR}/scripts/transcribe.sh --check
```

It reads `TRANSCRIBE_URL` and either `TRANSCRIBE_AUTH` (`user:password`, HTTP Basic auth in front of
the server) or `TRANSCRIBE_TOKEN` (API bearer token) from the environment. If it reports that a
variable is missing or the credentials are rejected, stop and ask the user to set them, for example
in `~/.claude/settings.json`:

```json
{ "env": { "TRANSCRIBE_URL": "https://transcribe.example.com", "TRANSCRIBE_AUTH": "user:password" } }
```

Never ask the user to paste the password into the chat and never print it.

## 2. Collect what the server needs

- **File**: an absolute path. If the user names a file vaguely, find it (Downloads, Desktop, the
  current project) and confirm which one when several match.
- **Speaker names** (optional but valuable): names in the order people first speak, e.g. for an
  interview the interviewer usually speaks first. Pass them with `--names "Артём,Кандидат"`; the
  count then follows from the list. If the user only knows how many people talk, use
  `--speakers N`. Without either, speakers are detected automatically.
- **Gemini correction** is on by default and fixes terms, names and punctuation using the whole
  dialogue; pass `--no-gemini` only if the user asks for raw ASR output.

## 3. Run it

An hour of audio takes several minutes (upload plus processing), longer than a normal command
timeout, so run it in the background and wait for it to finish:

```bash
${CLAUDE_SKILL_DIR}/scripts/transcribe.sh "/path/to/recording.mp4" --names "Артём,Кандидат"
```

The script prints progress to stderr and the path of the saved transcript as the last line of
stdout. Options: `--out FILE` to choose the destination, `--language ru` as a language hint,
`--keep-job` to leave the result on the server (it is deleted after download by default; the
server itself keeps results for at most 24 hours).

## 4. If something goes wrong

- The job id is printed right after the upload (`job <id> accepted`). If the run is interrupted
  (network drop, timeout, closed session), do **not** upload again: resume with
  `${CLAUDE_SKILL_DIR}/scripts/transcribe.sh --resume <id> --out "<same output path>"`.
- `404` on resume means the job expired or was already downloaded.
- `queue is full` is retried automatically; `too large` or `unsupported media format` means the
  file needs converting to audio first (install ffmpeg, or ask the user for an audio export).
- If the server rejects parameters after an update, read `$TRANSCRIBE_URL/openapi.json` (with the
  same credentials) to see the current API.

## 5. Report back

Tell the user where the transcript was saved and summarise its size (duration, number of
speakers from the front matter). Offer a summary or action items only if they want it: the
transcript itself is the deliverable.
