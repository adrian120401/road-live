"""Local road inference and isolated short-lived IDs; never resets general tracking."""

from dataclasses import dataclass, replace
import logging
import hashlib
import math
import time

import numpy as np

from .config import (Detection, RoadDamageConfig, RoadFrame, is_valid_road_detection,
                     MIN_POTHOLE_CONFIDENCE, ROAD_DIAGNOSTIC_CONFIDENCE)
from .tracker import InferenceStats, YOLO
from .association import geometry, overlap, match_boxes
from .road_region import RoadRegion

LOG = logging.getLogger(__name__)
REVIEWED_CHECKPOINT_SHA256 = "af2ac6ce7bfec72e71643659ac946caf80ced84869e526a60135c457abfbb200"


def road_class_ids(names: dict[int, str], model_path) -> list[int]:
    classes = [i for i, name in names.items() if name.strip().lower() in {"pothole", "potholes", "d40"}]
    if not classes and names == {0: "0"}:
        # The author's dataset YAML names its sole pothole class '0'. Only this
        # exact reviewed checkpoint may use that alias; never infer semantics by ID.
        with model_path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum == REVIEWED_CHECKPOINT_SHA256:
            classes = [0]
    return classes


def roi_pixels(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    bounds = (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))
    if bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
        raise ValueError("Road ROI is too small for this video.")
    return bounds


def restore_box(box, roi: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = roi
    return (float(np.clip(box[0] + x1, x1, x2)), float(np.clip(box[1] + y1, y1, y2)),
            float(np.clip(box[2] + x1, x1, x2)), float(np.clip(box[3] + y1, y1, y2)))


@dataclass
class RoadTrack:
    detection: Detection
    last_frame: int
    last_step: int
    consecutive_hits: int = 1
    confirmed: bool = False
    last_timestamp: float = 0.0


class RoadAssociator:
    def __init__(self, config: RoadDamageConfig, source_fps: float) -> None:
        self.config, self.source_fps = config, source_fps
        self.tracks: dict[int, RoadTrack] = {}
        self.next_id = 1
        self.step = 0

    def expire(self, frame: int, timestamp: float | None = None) -> tuple[int, ...]:
        now = timestamp if timestamp is not None else (frame - 1) / self.source_fps
        expired = tuple(track_id for track_id, track in self.tracks.items()
                        if now - track.last_timestamp > self.config.max_gap_seconds)
        for track_id in expired:
            del self.tracks[track_id]
        return expired

    def update(self, detections: list[Detection], frame: int, timestamp: float | None = None) -> RoadFrame:
        detections = [d for d in detections if is_valid_road_detection(d, self.config.confidence)]
        now = timestamp if timestamp is not None else (frame - 1) / self.source_fps
        expired = self.expire(frame, now)
        self.step += 1
        matched = match_boxes({i: t.detection.box for i, t in self.tracks.items()},
                              [d.box for d in detections])
        observed = []
        for index, detection in enumerate(detections):
            track_id = matched.get(index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                track = RoadTrack(replace(detection, track_id=track_id), frame, self.step)
                self.tracks[track_id] = track
            else:
                track = self.tracks[track_id]
                track.consecutive_hits = track.consecutive_hits + 1 if track.last_step == self.step - 1 else 1
                track.detection = replace(detection, track_id=track_id)
                track.last_frame, track.last_step = frame, self.step
            track.confirmed |= track.consecutive_hits >= self.config.confirmation_hits
            track.last_timestamp = now
            observed.append(track.detection)
        confirmed = frozenset(track_id for track_id, track in self.tracks.items() if track.confirmed)
        visible = tuple(d for d in observed if d.track_id in confirmed)
        return RoadFrame(visible, tuple(observed), confirmed, expired, True, tuple(detections))


class RoadDamageDetector:
    def __init__(self, config: RoadDamageConfig, device: str, source_fps: float,
                 width: int, height: int) -> None:
        self.config, self.device = config, device
        self.roi = roi_pixels(config.roi, width, height)
        self.region = RoadRegion(config, width, height)
        # Existing local files only: never pass a URL or allow a runtime download.
        if not config.model.is_file():
            raise ValueError(f"Road model not found: {config.model}")
        self.model = YOLO(str(config.model))
        if self.model.task != "detect":
            raise ValueError("Road model must be an object detection checkpoint.")
        self.class_ids = road_class_ids(self.model.names, config.model)
        if not self.class_ids:
            raise ValueError("Road model has no pothole / D40 class; COCO weights cannot detect road damage.")
        self.associator = RoadAssociator(config, source_fps)
        self.stats = InferenceStats()
        self.last = RoadFrame(roi=self.roi, road_area=self.region.points)
        self.discarded_below_50 = 0
        self.discarded_below_threshold = 0
        self.discarded_outside_road = 0

    def _predict(self, frame):
        x1, y1, x2, y2 = self.roi
        return next(self.model.predict(
            frame[y1:y2, x1:x2], device=self.device, imgsz=self.config.image_size,
            conf=ROAD_DIAGNOSTIC_CONFIDENCE, iou=self.config.iou_threshold, classes=self.class_ids, stream=True,
            verbose=False, save=False, show=False,
        ))

    def warmup(self, frame: np.ndarray) -> None:
        self._predict(frame)

    def update(self, frame: np.ndarray, frame_number: int, timestamp: float | None = None) -> RoadFrame:
        if (frame_number - 1) % self.config.frame_interval:
            expired = self.associator.expire(frame_number, timestamp)
            return replace(self.last, inferred=False, observed=(), raw=(), outside_road=(), expired_ids=expired,
                           detections=tuple(d for d in self.last.detections if d.track_id in self.associator.tracks))
        started = time.perf_counter()
        result = self._predict(frame)
        raw = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, confidence):
                restored = restore_box(box, self.roi)
                if geometry(restored)[2] > 0:
                    raw.append(Detection(None, "pothole", float(score), restored))
        elapsed = time.perf_counter() - started
        self.stats.record(result.speed.get("inference", 0), elapsed)
        self.discarded_below_50 += sum(ROAD_DIAGNOSTIC_CONFIDENCE <= d.confidence < MIN_POTHOLE_CONFIDENCE for d in raw)
        self.discarded_below_threshold += sum(not is_valid_road_detection(d, self.config.confidence) for d in raw)
        eligible = [d for d in raw if is_valid_road_detection(d, self.config.confidence)]
        accepted, rejected = [], []
        for detection in eligible:
            (accepted if self.region.accepts(detection) else rejected).append(detection)
        self.discarded_outside_road += len(rejected)
        self.last = replace(self.associator.update(accepted, frame_number, timestamp), roi=self.roi,
                            road_area=self.region.points,
                            outside_road=tuple(rejected) if self.config.debug else ())
        if self.config.debug:
            LOG.info("ROAD frame=%d ROI=%s raw=%d scores=%s calls=%d stage=%.1fms outside_road=%d",
                     frame_number, self.roi, len(raw), [round(d.confidence, 3) for d in raw],
                     self.stats.calls, elapsed * 1000, len(rejected))
        return self.last
