"""Run pyannote Community-1 diarization and save benchmark JSON."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speakers", type=int)
    parser.add_argument(
        "--model", default="pyannote/speaker-diarization-community-1"
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    if pipeline is None:
        raise RuntimeError("pyannote did not return a pipeline")
    loaded = time.perf_counter()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline.to(torch.device(device))
    try:
        options = {"num_speakers": args.speakers} if args.speakers else {}
        result = pipeline(str(args.audio), **options)
    except RuntimeError:
        if device != "mps":
            raise
        device = "cpu"
        pipeline.to(torch.device(device))
        result = pipeline(str(args.audio), **options)
    finished = time.perf_counter()

    annotation = getattr(result, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(result, "speaker_diarization", result)
    segments = [
        {"start": turn.start, "end": turn.end, "speaker": str(speaker)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    payload = {
        "model": args.model,
        "device": device,
        "requested_speakers": args.speakers,
        "speakers": sorted({item["speaker"] for item in segments}),
        "load_time_seconds": loaded - started,
        "diarization_time_seconds": finished - loaded,
        "total_time_seconds": finished - started,
        "segments": segments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
