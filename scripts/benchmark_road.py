"""Compare general-only and road intervals on the same video/device, sequentially."""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, RoadDamageConfig
from src.main import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("video1.mp4"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    directory = Path("outputs/validation/v2_benchmark")
    directory.mkdir(parents=True, exist_ok=True)
    comparison = {}
    baseline_tracks = None
    for interval in (0, 2, 3):
        name = "general" if interval == 0 else f"road_interval_{interval}"
        config = Config(args.input, directory / f"{name}.mp4", device=args.device,
                        road_damage=RoadDamageConfig(enabled=bool(interval), frame_interval=interval or 2))
        report = process_video(config, frame_limit=args.frames)
        if baseline_tracks is None:
            baseline_tracks = report["tracks"]
        comparison[name] = {"device": report["device"], "frames": report["frames_processed"],
                            "processing_fps": report["processing_fps"], "counts": report["counts"],
                            "general_tracks_unchanged": report["tracks"] == baseline_tracks,
                            "general_performance": report["general_performance"],
                            "road_damage": report.get("road_damage")}
    path = directory / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
