"""Ultralytics inference boundary; analytics never depends on its Results API."""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Keep Ultralytics settings in this project unless explicitly configured elsewhere.
config_directory = Path(os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics")))
# Ultralytics checks its parent directory before creating its own subdirectory.
config_directory.expanduser().mkdir(parents=True, exist_ok=True)
from ultralytics import YOLO

from .config import CLASSES, Config, Detection, is_presentable

LOG = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    calls: int = 0
    inference_ms: float = 0.0
    stage_seconds: float = 0.0

    def record(self, inference_ms: float, stage_seconds: float) -> None:
        self.calls += 1
        self.inference_ms += inference_ms
        self.stage_seconds += stage_seconds

    def report(self) -> dict:
        return {
            "inferences": self.calls,
            "mean_inference_ms": round(self.inference_ms / self.calls, 3) if self.calls else 0,
            "inference_fps": round(self.calls * 1000 / self.inference_ms, 3) if self.inference_ms else 0,
            "mean_stage_ms": round(self.stage_seconds * 1000 / self.calls, 3) if self.calls else 0,
            "stage_seconds": round(self.stage_seconds, 3),
        }


def select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA")
        # Availability alone doesn't prove this wheel supports the GPU architecture.
        sample = torch.ones((8, 8), device="cuda:0")
        _ = sample @ sample
        torch.cuda.synchronize()
        LOG.info("GPU: %s (%s)", torch.cuda.get_device_name(0), torch.__version__)
        return "cuda:0"
    except (RuntimeError, AssertionError) as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA requested but unusable: {exc}") from exc
        LOG.warning("CUDA unavailable; using CPU: %s", exc)
        return "cpu"


class ObjectTracker:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = select_device(config.device)
        self.stats = InferenceStats()
        self.model = YOLO(config.model)
        if self.model.task != "detect":
            raise ValueError("Urban Vision V1 requires a detection model.")
        names = self.model.names
        allowed = set(CLASSES)
        if not config.traffic.traffic_lights:
            allowed.discard("traffic light")
        self.class_ids = [i for i, name in names.items() if name in allowed]
        missing = set(CLASSES) - set(names.values())
        if missing:
            LOG.warning("Classes absent from this model: %s", ", ".join(sorted(missing)))
        if not self.class_ids:
            raise ValueError("The model has none of the requested urban classes.")

    def warmup(self, frame: np.ndarray) -> None:
        """Warm up inference without creating tracking IDs or counting objects."""
        self.model.predict(frame, device=self.device, imgsz=self.config.image_size,
                           classes=self.class_ids, verbose=False, save=False)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def update(self, frame: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        result = next(self.model.track(
            frame, persist=True, stream=True, tracker=self.config.tracker,
            device=self.device, imgsz=self.config.image_size,
            conf=self.config.conf, classes=self.class_ids,
            verbose=False, save=False, show=False,
        ))
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self.stats.record(result.speed.get("inference", 0), time.perf_counter() - started)
            return []
        # Transfer once; row format is normalized independently of Ultralytics.
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidence = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        detections = [Detection(
            track_id=int(ids[i]) if ids is not None else None,
            class_name=result.names[int(classes[i])],
            confidence=float(confidence[i]),
            box=tuple(float(v) for v in xyxy[i]),
        ) for i in range(len(xyxy))]
        self.stats.record(result.speed.get("inference", 0), time.perf_counter() - started)
        return [detection for detection in detections if is_presentable(detection)]
