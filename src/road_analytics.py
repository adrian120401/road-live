"""Confirmed road events and one representative evidence image per event."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .config import Detection, RoadDamageConfig, RoadFrame, is_valid_road_detection
from .location import LocationProvider
from .renderer import render_road_evidence
from .road_region import RoadRegion


@dataclass
class Representative:
    detection: Detection
    frame: int
    timestamp: float
    image: np.ndarray | None


class RoadAnalytics:
    def __init__(self, config: RoadDamageConfig, output: Path, source_fps: float,
                 location: LocationProvider | None = None) -> None:
        self.config, self.source_fps = config, source_fps
        self.location = location
        self.events_path = output.with_name(output.stem + "_events.json")
        self.evidence_dir = output.with_name(output.stem + "_events")
        self.candidates_dir = config.candidates_dir / "potholes" / "unreviewed" / output.stem
        self.events: dict[int, dict] = {}
        self.best: dict[int, Representative] = {}
        self.finalized: set[int] = set()
        self.observations = 0
        self.track_stats: dict[int, tuple[int, int]] = {}

    @property
    def count(self) -> int:
        return len(self.events)

    @staticmethod
    def _record(track_id: int, best: Representative) -> dict:
        return {"type": "pothole", "track_id": track_id, "confidence": best.detection.confidence,
                "frame": best.frame, "timestamp": best.timestamp, "bbox": list(best.detection.box),
                "evidence_path": None}

    def update(self, state: RoadFrame, frame: np.ndarray, frame_number: int,
               timestamp: float | None = None) -> None:
        if state.inferred:
            region = RoadRegion(self.config, frame.shape[1], frame.shape[0])
            for detection in state.observed:
                track_id = detection.track_id
                if (track_id is None or not is_valid_road_detection(detection, self.config.confidence)
                        or not region.accepts(detection)):
                    continue
                self.observations += 1
                previous_count, first_frame = self.track_stats.get(track_id, (0, frame_number))
                self.track_stats[track_id] = (previous_count + 1, first_frame)
                previous = self.best.get(track_id)
                if previous is None or detection.confidence > previous.detection.confidence:
                    keep_image = self.config.save_evidence or self.config.collect_candidates
                    self.best[track_id] = Representative(detection, frame_number,
                                                        timestamp if timestamp is not None else (frame_number - 1) / self.source_fps,
                                                        frame.copy() if keep_image else None)
                if track_id in state.confirmed_ids:
                    previous_event = self.events.get(track_id, {})
                    confirmed_frame = previous_event.get("confirmed_frame", frame_number)
                    self.events[track_id] = {**self._record(track_id, self.best[track_id]),
                                             "event_id": previous_event.get("event_id", len(self.events) + 1),
                                             "confirmed_frame": confirmed_frame,
                                             "observations": previous_count + 1,
                                             "first_frame": first_frame, "last_frame": frame_number}
        for track_id in state.expired_ids:
            self._finalize(track_id)

    def _finalize(self, track_id: int) -> None:
        best = self.best.get(track_id)
        self.track_stats.pop(track_id, None)
        event = self.events.get(track_id)
        if best is None or event is None or track_id in self.finalized:
            self.best.pop(track_id, None)
            return
        name = f"pothole_{track_id:03d}.jpg"
        if self.location is not None:
            point = self.location.point_at(best.timestamp)
            event.update(latitude=point.latitude, longitude=point.longitude,
                         location_source=self.location.source)
            if hasattr(self.location, "accuracy_at"):
                event["location_accuracy_m"] = self.location.accuracy_at(best.timestamp)
        if self.config.save_evidence and best.image is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            path = self.evidence_dir / name
            image = render_road_evidence(best.image, best.detection, best.frame, best.timestamp)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Cannot save road evidence: {path}")
            event["evidence_path"] = str(path.relative_to(self.events_path.parent).as_posix())
        event["evidence_image"] = event["evidence_path"]
        if self.config.collect_candidates and best.image is not None:
            self.candidates_dir.mkdir(parents=True, exist_ok=True)
            path = self.candidates_dir / name
            if not cv2.imwrite(str(path), best.image):
                raise OSError(f"Cannot save road candidate: {path}")
            metadata = {**event, "review_status": "unreviewed", "image_path": name}
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            event["candidate_path"] = str(path.resolve())
        self.finalized.add(track_id)
        self.best.pop(track_id, None)

    def finish(self, complete: bool, stop_reason: str) -> dict:
        for track_id in list(self.best):
            self._finalize(track_id)
        if self.location is not None:
            for event in self.events.values():
                point = self.location.point_at(event["timestamp"])
                event.update(latitude=point.latitude, longitude=point.longitude,
                             location_source=self.location.source)
                if hasattr(self.location, "accuracy_at"):
                    event["location_accuracy_m"] = self.location.accuracy_at(event["timestamp"])
        payload = {"complete": complete, "stop_reason": stop_reason,
                   "unique_potholes": self.count, "count_is_estimate": True,
                   "confidence_threshold": self.config.confidence,
                   "road_area": self.config.road_area,
                   "road_area_min_overlap": self.config.road_area_min_overlap,
                   "events": list(self.events.values())}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"potholes": self.count, "observations": self.observations,
                "events_path": str(self.events_path.resolve()),
                "saved_evidence": sum(event["evidence_path"] is not None for event in self.events.values()),
                "saved_candidates": sum("candidate_path" in event for event in self.events.values())}
