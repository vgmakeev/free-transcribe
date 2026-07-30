"""Run the open MLX Sortformer diarizer and save compact benchmark JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_audio.vad import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--model",
        default="mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    model = load(args.model)
    loaded = time.perf_counter()
    result = model.generate(
        str(args.audio), threshold=args.threshold, verbose=True
    )
    finished = time.perf_counter()

    payload = {
        "model": args.model,
        "threshold": args.threshold,
        "num_speakers": result.num_speakers,
        "model_time_seconds": result.total_time,
        "load_time_seconds": loaded - started,
        "total_time_seconds": finished - started,
        "segments": [
            {"start": item.start, "end": item.end, "speaker": item.speaker}
            for item in result.segments
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
