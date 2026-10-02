"""Download the official revision-tagged depth checkpoint; inference stays local."""

import hashlib
import json
from pathlib import Path
import urllib.request

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-depth.pt"
SHA256 = "befed1b8561d8b2eaa66274b070cdfc9c44853bda6d6d62a04759f9383af74e9"


def download(path: Path = Path("models/yolo26n-depth.pt")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(".download")
        try:
            urllib.request.urlretrieve(URL, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
                raise ValueError("Official depth checkpoint checksum mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Depth checkpoint checksum mismatch")
    path.with_suffix(".json").write_text(json.dumps({"source": URL, "sha256": SHA256,
        "model": "YOLO26n-depth", "docs": "https://docs.ultralytics.com/tasks/depth",
        "units": "nominal meters; uncalibrated for local iPhone"}, indent=2), encoding="utf-8")
    print("Verified depth model:", path)


if __name__ == "__main__":
    download()
