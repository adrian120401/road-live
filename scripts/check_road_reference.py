"""Positive pipeline smoke check on one labeled public image, not local-road accuracy."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import urllib.request

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RoadDamageConfig
from src.road_damage import RoadDamageDetector, overlap
from src.road_analytics import RoadAnalytics

DATASET = "Ryukijano/Pothole-detection-Yolov8"
IMAGE = "0004_jpg.rf.f92ab952cd8544f887caf35fcccbcd10.jpg"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Urban-Vision/2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    directory = Path("outputs/validation/road_reference")
    directory.mkdir(parents=True, exist_ok=True)
    provenance = directory / "source.json"
    if not provenance.is_file():
        revision = json.loads(fetch(f"https://huggingface.co/api/datasets/{DATASET}"))["sha"]
        base = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/test"
        image_url = f"{base}/images/{IMAGE}"
        label_url = f"{base}/labels/{Path(IMAGE).with_suffix('.txt').name}"
        (directory / "source.jpg").write_bytes(fetch(image_url))
        (directory / "labels.txt").write_bytes(fetch(label_url))
        provenance.write_text(json.dumps({"dataset": DATASET, "revision": revision,
                                          "image_url": image_url, "label_url": label_url}, indent=2), encoding="utf-8")
    frame = cv2.imread(str(directory / "source.jpg"))
    if frame is None:
        raise RuntimeError("Reference image cannot be decoded")
    height, width = frame.shape[:2]
    config = replace(RoadDamageConfig(), enabled=True, roi=(0, 0, 1, 1))
    detector = RoadDamageDetector(config, args.device, 30, width, height)
    detector.warmup(frame)
    analytics = RoadAnalytics(config, directory / "reference.mp4", 30)
    state = detector.update(frame, 1)
    analytics.update(state, frame, 1)
    # Repeated image is intentional: exercise confirmation/JSON/photo persistence only.
    state = detector.update(frame, 3)
    analytics.update(state, frame, 3)
    summary = analytics.finish(True, "reference_smoke_check")
    ground_truth = []
    for row in (directory / "labels.txt").read_text().splitlines():
        _, cx, cy, bw, bh = map(float, row.split())
        ground_truth.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                             (cx + bw / 2) * width, (cy + bh / 2) * height))
    matches = [max((overlap(d.box, box) for box in ground_truth), default=0) for d in state.detections]
    result = {**summary, "source": json.loads(provenance.read_text()),
              "confidence_threshold": config.confidence, "ground_truth_boxes": ground_truth,
              "predicted_boxes": [list(d.box) for d in state.detections],
              "confidence": [d.confidence for d in state.detections], "best_ground_truth_iou": matches,
              "positive_check_passed": any(iou >= 0.5 for iou in matches),
              "limitation": "Single public dataset image repeated to exercise events; not independent accuracy validation or local-video evidence."}
    (directory / "check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["positive_check_passed"]:
        raise RuntimeError("No reference detection overlaps a labeled pothole at IoU >= 0.5")


if __name__ == "__main__":
    main()
