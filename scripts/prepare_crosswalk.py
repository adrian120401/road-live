"""Prepare a reviewed legacy YOLOv5 checkpoint in an isolated export process.

Inference only needs the ONNX file, JSON manifest and existing OpenCV dependency.
Install export dependencies with --target tools/crosswalk-export-deps (README).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5ba1e01d946e1a5cc9dcf396b3b8fc62ae2666d5"
SOURCE = "https://github.com/kairess/crosswalk-traffic-light-detection-yolov5"
CHECKPOINT_SHA = "7eb9e22c31e6eef68dce2cfd656fa27614e21203371cdf9255d942c86e29f127"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare() -> None:
    archive = ROOT / "tools/crosswalk-source.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        url = f"https://codeload.github.com/kairess/crosswalk-traffic-light-detection-yolov5/zip/{REVISION}"
        urllib.request.urlretrieve(url, archive)
    weights = ROOT / "models/crosswalk_yolov5s6.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    vendor = ROOT / "tools/legacy_crosswalk"
    with zipfile.ZipFile(archive) as files:
        prefix = files.namelist()[0].split("/")[0]
        for name in files.namelist():
            relative = Path(name).relative_to(prefix)
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("Unsafe archive path")
            if name.endswith("runs/train/exp4/weights/best.pt"):
                weights.write_bytes(files.read(name))
            elif name.endswith(".py") and (relative.parts[0] in {"models", "utils"} or relative.name == "export.py") or relative.name == "LICENSE":
                path = vendor / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files.read(name))
    if digest(weights) != CHECKPOINT_SHA:
        raise ValueError("Reviewed crosswalk checkpoint checksum mismatch")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--export-child"], cwd=ROOT, check=True)


def export_child() -> None:
    sys.path[:0] = [str(ROOT / "tools/legacy_crosswalk"), str(ROOT / "tools/crosswalk-export-deps")]
    os.environ["YOLOV5_CONFIG_DIR"] = str(ROOT / "tools/legacy_crosswalk/config")
    os.environ["MPLCONFIGDIR"] = str(ROOT / "tools/legacy_crosswalk/config")
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    import cv2
    import numpy as np
    import onnx
    import torch
    from torch import nn

    torch.set_num_threads(2)
    weights = ROOT / "models/crosswalk_yolov5s6.pt"
    if digest(weights) != CHECKPOINT_SHA:
        raise ValueError("Unreviewed checkpoint")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = checkpoint["model"].float().eval()
    # Fixed 640px input: use constant resize scales. Legacy recomputation emits
    # shape/Floor nodes unsupported by OpenCV; parity below checks equivalence.
    for module in model.modules():
        if isinstance(module, nn.Upsample):
            module.recompute_scale_factor = False
    model.model[-1].inplace = False
    names = dict(enumerate(model.names)) if isinstance(model.names, list) else model.names
    print("Checkpoint classes:", names, flush=True)
    if "Zebra_Cross" not in names.values():
        raise ValueError("Checkpoint does not contain documented Zebra_Cross class")
    # Return decoded predictions only, without the legacy model's raw feature maps.
    class Decoded(nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, image):
            return self.detector(image)[0]

    wrapped = Decoded(model).eval()
    output = ROOT / "models/crosswalk_yolov5s6.onnx"
    sample = torch.zeros((1, 3, 640, 640))
    with torch.inference_mode():
        wrapped(sample)
        torch.onnx.export(wrapped, sample, str(output), opset_version=12,
                          input_names=["images"], output_names=["predictions"], dynamo=False)
    wrapped.eval()
    onnx.checker.check_model(onnx.load(output))
    network = cv2.dnn.readNetFromONNX(str(output))
    capture = cv2.VideoCapture(str(ROOT / "video2.MOV"))
    checks = []
    try:
        for frame in (1, 240, 270, 300, 1033, 1191):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ok, image = capture.read()
            if not ok:
                raise ValueError("Cannot read reference frame")
            h, w = image.shape[:2]
            image = image[round(h*.42):round(h*.74)]
            scale = min(640/image.shape[1], 640/image.shape[0])
            resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
            canvas = np.full((640, 640, 3), 114, np.uint8)
            left, top = (640-resized.shape[1])//2, (640-resized.shape[0])//2
            canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
            blob = cv2.dnn.blobFromImage(canvas, 1/255, swapRB=True)
            with torch.inference_mode():
                reference = wrapped(torch.from_numpy(blob)).numpy()
            network.setInput(blob)
            actual = network.forward()
            # Coordinate tolerances are in 640px model space; check scores independently.
            score_error = float(np.max(np.abs(reference[...,4:] - actual[...,4:])))
            coordinate_error = float(np.max(np.abs(reference[...,:4] - actual[...,:4])))
            if score_error > .01 or coordinate_error > 1:
                raise ValueError(f"ONNX parity failure at frame {frame}: {score_error}, {coordinate_error}")
            checks.append({"frame": frame, "max_score_error": score_error, "max_coordinate_error_px": coordinate_error})
    finally:
        capture.release()
    manifest = {"source": SOURCE, "revision": REVISION, "checkpoint_sha256": CHECKPOINT_SHA,
                "sha256": digest(output), "architecture": "YOLOv5s6", "input_size": 640,
                "names": names, "crosswalk_class": "Zebra_Cross", "license": "GPL-3.0 (code)",
                "declared_dataset": "SelectStar Intersection and currency information datasets",
                "dataset_source": "https://open.selectstar.ai/", "parity": checks,
                "limitations": "Author-reported training; pedestrian viewpoint. Local dashcam quality requires review."}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Exported and parity checked:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    export_child() if args.export_child else prepare()
