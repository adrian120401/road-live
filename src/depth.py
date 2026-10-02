"""Small working-resolution depth maps and robust per-track observations."""

from dataclasses import dataclass
import hashlib
import json
import time

import cv2
import numpy as np

from .association import overlap
from .config import Detection, ProximityConfig
from .tracker import InferenceStats, YOLO

VEHICLES = frozenset({"car", "motorcycle", "bus", "truck"})


@dataclass(frozen=True)
class DepthObservation:
    track_id: int
    estimate: float
    box: tuple[float, float, float, float]
    timestamp: float
    valid_fraction: float


def sample_depth(depth: np.ndarray, detection: Detection, source_shape: tuple[int, int],
                 timestamp: float) -> DepthObservation | None:
    height, width = source_shape
    x1, y1, x2, y2 = detection.box
    # Stay inside the vehicle; reject empty/non-finite samples instead of inventing distance.
    ax, bx = x1+.25*(x2-x1), x1+.75*(x2-x1)
    ay, by = y1+.40*(y2-y1), y1+.85*(y2-y1)
    a, c = (int(np.clip(v/width*depth.shape[1], 0, depth.shape[1])) for v in (ax,bx))
    b, d = (int(np.clip(v/height*depth.shape[0], 0, depth.shape[0])) for v in (ay,by))
    region = depth[b:d, a:c]
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size < 16 or valid.size / max(1, region.size) < .75 or detection.track_id is None:
        return None
    low, high = np.percentile(valid, [10,90])
    trimmed = valid[(valid >= low) & (valid <= high)]
    return DepthObservation(detection.track_id, float(np.median(trimmed)), detection.box,
                            timestamp, valid.size/region.size)


class DepthEstimator:
    def __init__(self, config: ProximityConfig, device: str, fps: float) -> None:
        self.config, self.device, self.fps = config, device, fps
        manifest_path = config.depth_model.with_suffix(".json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with config.depth_model.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
                    raise ValueError("Depth model checksum mismatch")
        self.model = YOLO(str(config.depth_model), task="depth")
        if self.model.task != "depth":
            raise ValueError("Proximity requires a depth checkpoint")
        self.stats = InferenceStats()
        self.observations: dict[int, DepthObservation] = {}
        self.last_frame = -config.depth_interval
        self.map: np.ndarray | None = None

    def _predict(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = min(1, self.config.depth_image_size/max(h,w))
        working = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        return next(self.model.predict(working, imgsz=self.config.depth_image_size,
                    device=self.device, stream=True, verbose=False, save=False, show=False))

    def warmup(self, frame: np.ndarray) -> None:
        self._predict(frame)

    def update(self, frame: np.ndarray, detections: list[Detection], number: int, timestamp: float,
               *, has_candidate: bool) -> tuple[dict[int, DepthObservation], bool]:
        fresh = has_candidate and number-self.last_frame >= self.config.depth_interval
        if fresh:
            started = time.perf_counter()
            result = self._predict(frame)
            self.map = result.depth.data.detach().cpu().numpy()
            observations = [sample_depth(self.map,d,frame.shape[:2],timestamp) for d in detections
                            if d.track_id is not None and d.class_name in VEHICLES]
            self.observations = {o.track_id:o for o in observations if o is not None}
            self.last_frame = number
            self.stats.record(result.speed.get("inference",0), time.perf_counter()-started)
        max_age = min(.5, 1.5*self.config.depth_interval/self.fps)
        current = {d.track_id:d for d in detections}
        valid = {i:o for i,o in self.observations.items() if i in current
                 and timestamp-o.timestamp <= max_age and overlap(o.box,current[i].box) >= .5}
        return valid, fresh

    def report(self) -> dict:
        return {"model":str(self.config.depth_model), "frame_interval":self.config.depth_interval,
                "image_size":self.config.depth_image_size, "performance":self.stats.report(),
                "units":"nominal meters; not calibrated for this camera", "metric_calibrated":False}
