"""Experimental relative proximity, independent from inference and general tracking."""

from collections import deque
from dataclasses import dataclass
import json
import math

import cv2
import numpy as np

from .config import Detection, ProximityConfig
from .depth import DepthObservation, VEHICLES

LEVELS = ("SAFE", "CAUTION", "NEAR")


@dataclass(frozen=True)
class ProximityFrame:
    track_id: int | None = None
    state: str = "UNKNOWN"
    estimate: float | None = None
    persistent_frames: int = 0
    fresh_samples: int = 0
    box: tuple[float,float,float,float] | None = None
    reason: str = "no_candidate"


def in_corridor(detection: Detection, config: ProximityConfig, shape: tuple[int,int]) -> bool:
    if (detection.track_id is None or detection.class_name not in VEHICLES
            or detection.confidence < config.candidate_confidence):
        return False
    h, w = shape
    polygon = np.asarray(config.corridor, np.float32)
    x1,y1,x2,y2 = detection.box
    # Both anchor and center must agree: a large lateral box is not a forward vehicle.
    for point in (((x1+x2)/2/w,y2/h), ((x1+x2)/2/w,(y1+y2)/2/h)):
        if cv2.pointPolygonTest(polygon,point,False) < 0:
            return False
    return True


class ProximityAnalyzer:
    def __init__(self, config: ProximityConfig) -> None:
        self.config = config
        self.profile = json.loads(config.profile.read_text(encoding="utf-8")) if config.profile else None
        if self.profile:
            values = [self.profile[k] for k in ("near_threshold","caution_threshold","hysteresis")]
            if not all(math.isfinite(v) for v in values) or not 0 < values[0] < values[1] or values[2] < 0:
                raise ValueError("Invalid measured proximity thresholds")
            if self.profile.get("depth_image_size",config.depth_image_size) != config.depth_image_size:
                raise ValueError("Depth size differs from measured profile; regenerate the profile.")
            corridor = self.profile.get("corridor",config.corridor)
            if tuple(map(tuple,corridor)) != config.corridor:
                raise ValueError("Road corridor differs from measured profile; regenerate the profile.")
            if self.profile.get("candidate_confidence",config.candidate_confidence) != config.candidate_confidence:
                raise ValueError("Candidate confidence differs from measured profile; regenerate the profile.")
        self.current_id: int | None = None
        self.frames = self.samples = self.level = self.pending = self.pending_hits = 0
        self.depths: deque = deque(maxlen=5)
        self.transitions: list[dict] = []

    def reset(self) -> None:
        self.current_id = None
        self.frames = self.samples = self.level = self.pending = self.pending_hits = 0
        self.depths.clear()

    def update(self, detections: list[Detection], observations: dict[int,DepthObservation],
               fresh: bool, shape: tuple[int,int], number: int, timestamp: float) -> ProximityFrame:
        eligible = {d.track_id:d for d in detections if in_corridor(d,self.config,shape)
                    and d.track_id in observations}
        if not eligible:
            self.reset()
            return ProximityFrame()
        selected = min(eligible, key=lambda i: observations[i].estimate)
        # Preserve the current candidate within the observed depth-noise band.
        margin = self.profile["hysteresis"] if self.profile else 0
        if self.current_id in eligible and observations[self.current_id].estimate <= observations[selected].estimate+margin:
            selected = self.current_id
        if selected != self.current_id:
            self.reset()
            self.current_id = selected
        self.frames += 1
        if fresh:
            self.samples += 1
            self.depths.append(observations[selected].estimate)
        estimate = float(np.median(self.depths)) if self.depths else observations[selected].estimate
        detection = eligible[selected]
        if not self.profile:
            return ProximityFrame(selected,"UNKNOWN",estimate,self.frames,self.samples,detection.box,"profile_missing")
        near, caution = self.profile["near_threshold"], self.profile["caution_threshold"]
        desired = 2 if estimate <= near else 1 if estimate <= caution else 0
        if desired < self.level:
            threshold = near if self.level == 2 else caution
            if estimate <= threshold+self.profile["hysteresis"]:
                desired = self.level
        if desired == self.level:
            self.pending_hits = 0
        else:
            if self.pending != desired:
                self.pending_hits = 0
                self.pending = desired
            if fresh:
                self.pending_hits += 1
        required = self.config.confirmation_samples if desired > self.level else self.config.release_samples
        if (desired != self.level and self.pending_hits >= required
                and self.frames >= self.config.warning_frames):
            self.level = desired
            self.pending_hits = 0
            self.transitions.append({"frame":number,"timestamp":timestamp,"track_id":selected,
                                     "state":LEVELS[self.level],"depth_estimate":estimate,
                                     "persistent_frames":self.frames})
        return ProximityFrame(selected,LEVELS[self.level],estimate,self.frames,self.samples,detection.box,"estimated")

    def report(self) -> dict:
        return {"profile":str(self.config.profile) if self.config.profile else None,
                "thresholds":self.profile,"corridor":self.config.corridor,
                "warning_frames":self.config.warning_frames,"transitions":self.transitions,
                "experimental":True,"limitations":"Fixed corridor does not identify lanes, orientation or parked status; camera/profile specific."}
