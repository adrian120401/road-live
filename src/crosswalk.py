"""Local specialized ONNX road-marking inference, short tracks and unique counts."""

from dataclasses import dataclass, replace
import hashlib
import json
import time

import cv2
import numpy as np

from .association import match_boxes
from .config import CrosswalkConfig, Detection
from .road_damage import roi_pixels, restore_box
from .tracker import InferenceStats


@dataclass(frozen=True)
class CrossingFrame:
    detections: tuple[Detection, ...] = ()
    raw: tuple[Detection, ...] = ()
    inferred: bool = False
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class CrossingTrack:
    detection: Detection
    last_frame: int
    last_step: int
    hits: int = 1
    confirmed: bool = False
    first_frame: int = 0


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    scale = min(size / image.shape[1], size / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    left, top = (size-resized.shape[1])//2, (size-resized.shape[0])//2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
    return canvas, scale, left, top


def decode_predictions(rows: np.ndarray, class_id: int, confidence: float, scale: float,
                       left: int, top: int, roi: tuple[int, int, int, int]) -> tuple[Detection, ...]:
    rows = rows.reshape(-1, rows.shape[-1])
    scores = rows[:, 4] * rows[:, 5+class_id]
    selected = rows[scores >= confidence]
    scores = scores[scores >= confidence]
    if not len(selected):
        return ()
    boxes = np.column_stack((selected[:,0]-selected[:,2]/2, selected[:,1]-selected[:,3]/2,
                             selected[:,2], selected[:,3]))
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence, .45)
    observations = []
    for i in np.asarray(keep).reshape(-1):
        x, y, w, h = boxes[i]
        box = restore_box(((x-left)/scale, (y-top)/scale,
                           (x+w-left)/scale, (y+h-top)/scale), roi)
        if box[2] > box[0] and box[3] > box[1]:
            observations.append(Detection(None, "crosswalk", float(scores[i]), box))
    return tuple(observations)


class CrosswalkDetector:
    def __init__(self, config: CrosswalkConfig, fps: float, width: int, height: int,
                 device: str = "cpu") -> None:
        self.config, self.fps = config, fps
        self.device = device
        manifest = json.loads(config.model.with_suffix(".json").read_text(encoding="utf-8"))
        self.provenance = manifest
        with config.model.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
                raise ValueError("Crosswalk ONNX checksum does not match reviewed manifest")
        names = {int(k): v for k, v in manifest["names"].items()}
        self.class_id = next(i for i, name in names.items() if name == manifest["crosswalk_class"])
        self.class_ids = list(names)
        self.size = int(manifest["input_size"])
        self.backend = "YOLOE / " + device if config.model.suffix.lower() == ".pt" else "OpenCV DNN / CPU"
        if config.model.suffix.lower() == ".pt":
            from ultralytics import YOLOE
            prompt_path = config.model.parent / manifest["prompt_embeddings"]
            with prompt_path.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest() != manifest["prompt_sha256"]:
                    raise ValueError("Crosswalk prompt checksum mismatch")
            self.model = YOLOE(str(config.model))
            self.model.load_prompt_embeddings(prompt_path)
        else:
            self.net = cv2.dnn.readNetFromONNX(str(config.model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.roi = roi_pixels(config.roi, width, height)
        self.stats = InferenceStats()
        self.tracks: dict[int, CrossingTrack] = {}
        self.seen: set[int] = set()
        self.observed_tracks: dict[int, dict] = {}
        self.next_id, self.step = 1, 0
        self.last = CrossingFrame(roi=self.roi)

    def _predict(self, frame: np.ndarray, *, record: bool) -> tuple[Detection, ...]:
        started = time.perf_counter()
        x1, y1, x2, y2 = self.roi
        if self.config.model.suffix.lower() == ".pt":
            # YOLOE needs scene context: cropped pavement failed in the reviewed
            # crossing. Keep context at inference, gate resulting boxes to the ROI.
            result = next(self.model.predict(frame,imgsz=self.size,device=self.device,
                                            conf=self.config.confidence,classes=self.class_ids,
                                            stream=True,verbose=False,save=False,show=False))
            detections = []
            for box,score in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                a,b,c,d = map(float,box)
                clipped = max(x1,a),max(y1,b),min(x2,c),min(y2,d)
                area = max(0,clipped[2]-clipped[0])*max(0,clipped[3]-clipped[1])
                if area/max(1,(c-a)*(d-b)) >= .8:
                    detections.append(Detection(None,"crosswalk",float(score),clipped))
            # Synonymous prompts describe one class; suppress overlapping proposals.
            boxes = [[d.box[0],d.box[1],d.box[2]-d.box[0],d.box[3]-d.box[1]] for d in detections]
            keep = cv2.dnn.NMSBoxes(boxes,[d.confidence for d in detections],self.config.confidence,self.config.nms_iou) if boxes else []
            detections = tuple(detections[i] for i in np.asarray(keep).reshape(-1))
            if record:
                self.stats.record(result.speed.get("inference",0),time.perf_counter()-started)
            return detections
        image, scale, left, top = letterbox(frame[y1:y2, x1:x2], self.size)
        self.net.setInput(cv2.dnn.blobFromImage(image, 1/255, swapRB=True))
        inferred = time.perf_counter()
        rows = self.net.forward()
        inference_ms = (time.perf_counter() - inferred) * 1000
        detections = decode_predictions(rows, self.class_id, self.config.confidence, scale, left, top, self.roi)
        if record:
            self.stats.record(inference_ms, time.perf_counter() - started)
        return detections

    def warmup(self, frame: np.ndarray) -> None:
        self._predict(frame, record=False)

    def update(self, frame: np.ndarray, number: int) -> CrossingFrame:
        self.tracks = {i:t for i,t in self.tracks.items() if (number-t.last_frame)/self.fps <= self.config.max_gap_seconds}
        if (number-1) % self.config.frame_interval:
            fresh = tuple(d for d in self.last.detections if d.track_id in self.tracks and
                          (number-self.tracks[d.track_id].last_frame)/self.fps <= self.config.frame_interval/self.fps)
            return replace(self.last, detections=fresh, raw=(), inferred=False)
        raw = self._predict(frame, record=True)
        self.step += 1
        matches = match_boxes({i:t.detection.box for i,t in self.tracks.items()}, [d.box for d in raw])
        visible = []
        for index, detection in enumerate(raw):
            track_id = matches.get(index)
            if track_id is None:
                track_id, self.next_id = self.next_id, self.next_id+1
                self.tracks[track_id] = CrossingTrack(replace(detection, track_id=track_id), number, self.step,
                                                     first_frame=number)
            else:
                track = self.tracks[track_id]
                track.hits = track.hits+1 if track.last_step == self.step-1 else 1
                track.detection = replace(detection, track_id=track_id)
                track.last_frame, track.last_step = number, self.step
            track = self.tracks[track_id]
            track.confirmed |= track.hits >= self.config.confirmation_hits
            if track.confirmed:
                self.seen.add(track_id)
                summary = self.observed_tracks.setdefault(track_id, {"first_frame":track.first_frame,
                    "confirmed_frame":number,"observations":0,"max_confidence":0})
                summary["last_frame"],summary["bbox"] = number,detection.box
                summary["observations"] += 1
                summary["max_confidence"] = max(summary["max_confidence"],detection.confidence)
                visible.append(track.detection)
        self.last = CrossingFrame(tuple(visible), raw, True, self.roi)
        return self.last

    def report(self) -> dict:
        return {"model": str(self.config.model), "backend": self.backend, "crosswalks_seen": len(self.seen),
                "confidence": self.config.confidence, "roi": self.config.roi,
                "nms_iou":self.config.nms_iou,"provenance":self.provenance,
                "tracks":self.observed_tracks,
                "frame_interval": self.config.frame_interval, "performance": self.stats.report(),
                "counting_limitations": "Short-lived image-space tracks can fragment or merge physical crossings."}
