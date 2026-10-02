"""Download one reviewed, revision-pinned checkpoint; runtime inference stays offline."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "101b80987b3fa86f9747d4170f936760ebad3a9f"
SHA256 = "af2ac6ce7bfec72e71643659ac946caf80ced84869e526a60135c457abfbb200"
URL = f"https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/{REVISION}/best.pt"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or digest(destination) != SHA256:
        temporary = destination.with_suffix(".download")
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "Urban-Vision/2"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if digest(temporary) != SHA256:
                raise RuntimeError("Road checkpoint checksum mismatch; refusing installation.")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    metadata = {
        "author": "PeterHdd / peterhdd", "architecture": "YOLOv8s", "classes": ["pothole"],
        "source": URL, "revision": REVISION, "sha256": SHA256,
        "model_card": "https://huggingface.co/peterhdd/pothole-detection-yolov8",
        "training_code": "https://github.com/PeterHdd/pothole-detection-yolo",
        "declared_dataset": "https://huggingface.co/datasets/Ryukijano/Pothole-detection-Yolov8",
        "dataset_original": "https://universe.roboflow.com/project-ssayl/potholes-detection-d4rma/dataset/1",
        "checkpoint_class_names": {"0": "0"}, "verified_class_alias": {"0": "pothole"},
        "limitations": "Published checkpoint, not independently validated on local roads. Dataset card lacks collection details.",
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Verified road model: {destination.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/pothole_yolov8s.pt"))
    download(parser.parse_args().output)
