"""Bounded comparison of road profiles on identical, timestamped source segments."""

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import time

import cv2

from .config import RoadDamageConfig
from .offline_media import atomic_json
from .renderer import render_road_evidence
from .road_damage import RoadDamageDetector, RoadAssociator
from .tracker import select_device
from .video_reader import working_size

LOG = logging.getLogger(__name__)
ROAD_ROI = (.03, .48, .97, .80)
ROAD_AREA = ((.36, .52), (.74, .52), (1., .80), (0., .80))
PROFILES = (("1080_640_i2", 1920, 640, 2), ("4k_640_i2", 0, 640, 2),
            ("4k_960_i1", 0, 960, 1), ("4k_1280_i1", 0, 1280, 1))


def calibrate(source, root, device="auto", seconds=1., starts=(22, 39, 53, 72, 92)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el video para comparar perfiles.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    device = select_device(device)
    comparison = {}
    try:
        for name, max_side, size, interval in PROFILES:
            output = root / name
            output.mkdir(exist_ok=True)
            config = RoadDamageConfig(enabled=True, roi=ROAD_ROI, road_area=ROAD_AREA,
                                      image_size=size, frame_interval=interval)
            detector = None
            events, processed, elapsed = [], 0, 0.
            for segment, start in enumerate(starts):
                capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                best = {}
                for number in range(1, round(seconds * fps) + 1):
                    ok, original = capture.read()
                    if not ok:
                        break
                    w, h = original.shape[1], original.shape[0]
                    size_wh = working_size(w, h, max_side)
                    frame = cv2.resize(original, size_wh, interpolation=cv2.INTER_AREA) if size_wh != (w, h) else original
                    if detector is None:
                        detector = RoadDamageDetector(config, device, fps, *size_wh)
                        detector.warmup(frame)
                    if number == 1:
                        detector.associator = RoadAssociator(config, fps)
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    tick = time.perf_counter()
                    state = detector.update(frame, number, timestamp)
                    elapsed += time.perf_counter() - tick
                    processed += 1
                    for detection in state.observed:
                        previous = best.get(detection.track_id)
                        if previous is None or previous[0].confidence < detection.confidence:
                            best[detection.track_id] = (detection, frame.copy(), timestamp, number)
                    for track_id in state.confirmed_ids:
                        if track_id in best:
                            d, image, t, n = best[track_id]
                            item = {"segment": segment, "track_id": track_id, "timestamp": t,
                                    "confidence": d.confidence, "box": list(d.box),
                                    "image": f"{name}/segment_{segment}_pozo_{track_id}.jpg"}
                            existing = next((i for i, e in enumerate(events) if e["segment"] == segment and e["track_id"] == track_id), None)
                            if existing is None:
                                events.append(item)
                            else:
                                events[existing] = item
                for item in (e for e in events if e["segment"] == segment):
                    d, image, t, n = best[item["track_id"]]
                    evidence = render_road_evidence(image, d, n, t)
                    cv2.imwrite(str(root / item["image"]), evidence)
            comparison[name] = {"processing_max_side": max_side, "image_size": size,
                                "frame_interval": interval, "roi": list(ROAD_ROI),
                                "road_area": [list(p) for p in ROAD_AREA], "frames": processed,
                                "detector_seconds": elapsed, "events": events,
                                "performance": detector.stats.report() if detector else {},
                                "device": device, "starts": list(starts), "segment_seconds": seconds}
            atomic_json(root / "comparison.json", comparison)
            LOG.info("Perfil %s: %d eventos, %.1f ms por frame de detector", name, len(events), elapsed / max(processed, 1) * 1000)
            del detector
        return comparison
    finally:
        capture.release()
