"""Restore ASR punctuation in an already diarized Markdown transcript."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from free_transcribe.core import TranscriptSegment, restore_segment_punctuation

LINE_PATTERN = re.compile(
    r"^\*\*\[([^]]+)\](?: ([^:*]+):)?\*\*\s+(.*)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    parser.add_argument("asr", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.transcript.read_text(encoding="utf-8").splitlines()
    entries: list[tuple[int, str, str | None, str]] = []
    for line_index, line in enumerate(lines):
        match = LINE_PATTERN.match(line)
        if match:
            timestamp, speaker, text = match.groups()
            entries.append((line_index, timestamp, speaker, text))

    # A missed VAD span bounded by the same speaker is unambiguous.
    mutable_entries = [list(entry) for entry in entries]
    for index, entry in enumerate(mutable_entries):
        if entry[2] is not None:
            continue
        previous = next(
            (item[2] for item in reversed(mutable_entries[:index]) if item[2]), None
        )
        following = next(
            (item[2] for item in mutable_entries[index + 1 :] if item[2]), None
        )
        if previous == following:
            entry[2] = previous

    segments = [
        TranscriptSegment(float(index), float(index + 1), str(entry[3]), entry[2])
        for index, entry in enumerate(mutable_entries)
    ]
    asr = json.loads(args.asr.read_text(encoding="utf-8"))
    restored = restore_segment_punctuation(segments, str(asr["text"]))

    for entry, segment in zip(mutable_entries, restored):
        line_index, timestamp, speaker, _ = entry
        label = f" {speaker}:" if speaker else ""
        lines[int(line_index)] = f"**[{timestamp}]{label}** {segment.text}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
