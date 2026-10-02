"""Optional perception layers; bounded state and independent failure handling."""

from dataclasses import asdict, dataclass, field
import json
import logging

import cv2
import numpy as np

from .config import Config, Detection
from .crosswalk import CrosswalkDetector, CrossingFrame
from .depth import DepthEstimator
from .proximity import ProximityAnalyzer, ProximityFrame, in_corridor

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneFrame:
    crossings: CrossingFrame = field(default_factory=CrossingFrame)
    proximity: ProximityFrame = field(default_factory=ProximityFrame)
    depth_map: np.ndarray | None = None
    debug: dict = field(default_factory=dict)
    crosswalks_seen: int = 0


class SceneAnalyzer:
    def __init__(self, config: Config, device: str, fps: float, width: int, height: int) -> None:
        self.config, self.fps = config, fps
        self.crosswalk = CrosswalkDetector(config.traffic.crosswalks,fps,width,height,device) if config.traffic.crosswalks.enabled else None
        self.depth = DepthEstimator(config.proximity,device,fps) if config.proximity.enabled and config.proximity.depth_enabled else None
        self.proximity = ProximityAnalyzer(config.proximity)
        self.errors: dict[str,str] = {}
        self.trace: list[dict] = []
        if self.crosswalk:
            LOG.info("Cebras: %s | Confidence %.2f | Interval %d | Backend %s",
                     config.traffic.crosswalks.model,config.traffic.crosswalks.confidence,
                     config.traffic.crosswalks.frame_interval,self.crosswalk.backend)
        if self.depth:
            LOG.info("Depth: %s | Size %d | Interval %d | Candidate confidence %.2f | Profile %s",
                     config.proximity.depth_model,config.proximity.depth_image_size,
                     config.proximity.depth_interval,config.proximity.candidate_confidence,config.proximity.profile)

    def warmup(self, frame: np.ndarray) -> None:
        if self.crosswalk:
            self.crosswalk.warmup(frame)
        if self.depth:
            self.depth.warmup(frame)

    def update(self, frame: np.ndarray, detections: list[Detection], number: int,
               timestamp: float) -> SceneFrame:
        crossings, proximity = CrossingFrame(), ProximityFrame()
        observed, fresh = {}, False
        if self.crosswalk and "crosswalk" not in self.errors:
            try:
                crossings = self.crosswalk.update(frame,number)
            except (RuntimeError,ValueError,cv2.error) as exc:
                self._disable("crosswalk",exc)
        if self.depth and "depth" not in self.errors:
            try:
                candidates = any(in_corridor(d,self.config.proximity,frame.shape[:2]) for d in detections)
                observed,fresh = self.depth.update(frame,detections,number,timestamp,has_candidate=candidates)
                proximity = self.proximity.update(detections,observed,fresh,frame.shape[:2],number,timestamp)
            except (RuntimeError,ValueError,cv2.error) as exc:
                self._disable("depth",exc)
                self.proximity.reset()
        debug = {"fresh_depth":fresh,"observations":{i:o.estimate for i,o in observed.items()},
                 "depth_boxes":{i:o.box for i,o in observed.items()},
                 "thresholds":self.proximity.profile,"errors":self.errors.copy(),
                 "crosswalk_ms":self.crosswalk.stats.report()["mean_inference_ms"] if self.crosswalk else 0,
                 "depth_ms":self.depth.stats.report()["mean_inference_ms"] if self.depth else 0}
        if self.config.proximity.enabled:
            self.trace.append({"frame":number,"timestamp":timestamp,**asdict(proximity),
                               "fresh_depth":fresh,"observations":debug["observations"]})
        return SceneFrame(crossings,proximity,self.depth.map if self.depth else None,debug,
                          len(self.crosswalk.seen) if self.crosswalk else 0)

    def _disable(self, name: str, error: Exception) -> None:
        self.errors[name] = str(error)
        LOG.warning("Capa %s desactivada después de un error; continúa video general/pozos: %s",name,error)

    def finish(self) -> dict:
        report = {"errors":self.errors,"traffic_lights_enabled":self.config.traffic.traffic_lights}
        if self.crosswalk:
            report["crosswalk"] = self.crosswalk.report()
        if self.depth:
            path = self.config.output_path.with_name(self.config.output_path.stem+"_proximity.jsonl")
            with path.open("w",encoding="utf-8") as stream:
                for row in self.trace:
                    stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            report["depth"] = self.depth.report()
            report["proximity"] = {**self.proximity.report(),"trace_path":str(path.resolve())}
        return report
