"""Custom ADAS-inspired overlays; no Ultralytics plotting."""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analytics import Analytics
from .config import DISPLAY_NAMES, HUD_COUNTS, MOVING_CLASSES, Detection, is_valid_road_detection

if TYPE_CHECKING:
    from .config import RoadFrame
    from .scene import SceneFrame

WHITE = (238, 244, 244)
MUTED = (173, 190, 194)
ACCENT = (204, 230, 106)  # soft cyan in BGR
DARK = (25, 24, 19)
ROAD_ACCENT = (117, 194, 238)  # restrained amber, BGR
NEAR_ACCENT = (69, 137, 245)  # orange, BGR
AA = cv2.LINE_AA


@lru_cache(maxsize=32)
def _font(pixel_size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, pixel_size)
        except OSError:
            continue
    return ImageFont.load_default(size=pixel_size)


@lru_cache(maxsize=512)
def _text_mask(value: str, pixel_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    font = _font(pixel_size)
    left, top, right, bottom = font.getbbox(value, anchor="ls")
    image = Image.new("L", (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(image).text((-left, -top), value, font=font, fill=255, anchor="ls")
    return np.asarray(image), (left, top, right, bottom)


@dataclass(frozen=True)
class FrameMetrics:
    frame: int
    total_frames: int
    source_fps: float
    processing_fps: float
    elapsed_seconds: float | None = None


class Renderer:
    def __init__(self, width: int, height: int, display_confidence: float = 0.30) -> None:
        self.width, self.height = width, height
        self.display_confidence = display_confidence
        self.scale = min(width / 464, height / 832)
        self.margin = self.px(14)
        self.header_bottom = self.margin + self.px(55)
        self.footer_top = height - self.margin - self.px(119)

    def px(self, value: float) -> int:
        return max(1, round(value * self.scale))

    def text(self, frame: np.ndarray, value: str, position: tuple[int, int],
             size: float = 0.40, color: tuple[int, int, int] = WHITE) -> None:
        # Rasterize only a small cached text tile, never convert the whole 4K frame.
        mask, (left, top, _, _) = _text_mask(value, self.px(30 * size))
        x, y = position[0] + left, position[1] + top
        h, w = mask.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return
        alpha = mask[y1-y:y2-y, x1-x:x2-x, None].astype(np.float32) / 255
        region = frame[y1:y2, x1:x2]
        region[:] = np.clip(region * (1 - alpha) + np.asarray(color) * alpha, 0, 255).astype(np.uint8)

    def text_size(self, value: str, size: float) -> tuple[tuple[int, int], int]:
        _, (left, top, right, bottom) = _text_mask(value, self.px(30 * size))
        return (right - left, max(1, -top)), max(0, bottom)

    def panel(self, frame: np.ndarray, bounds: tuple[int, int, int, int],
              opacity: float = 0.77) -> None:
        x1, y1, x2, y2 = bounds
        roi = frame[y1:y2, x1:x2]
        tint = np.full_like(roi, DARK)
        cv2.addWeighted(tint, opacity, roi, 1 - opacity, 0, dst=roi)

    def render(self, frame: np.ndarray, detections: list[Detection],
               analytics: Analytics, metrics: FrameMetrics, *, road: "RoadFrame | None" = None,
               potholes: int = 0, debug_road: bool = False, scene: "SceneFrame | None" = None,
               debug_scene: bool = False, corridor: tuple = (), infrastructure: bool = False) -> np.ndarray:
        self.footer_top = self.height - self.margin - self.px(
            119 + (30 if road is not None else 0) + (23 if infrastructure else 0))
        annotated = frame.copy()
        active = [d for d in detections if d.track_id is not None and
                  (d.confidence >= self.display_confidence or d.class_name in {"traffic light","stop sign"})]
        self._trails(annotated, active, analytics)
        label_regions: list[tuple[int, int, int, int]] = []
        # Put labels for the largest, most legible objects first.
        for detection in sorted(active, key=lambda d: self._area(d), reverse=True):
            self._box(annotated, detection, analytics, label_regions)
        if road is not None:
            for detection in road.detections:
                if is_valid_road_detection(detection):
                    self._box(annotated, detection, None, label_regions, road=True)
            if debug_road:
                x1, y1, x2, y2 = road.roi
                cv2.rectangle(annotated, (x1, y1), (x2, y2), MUTED, self.px(1), AA)
                self.text(annotated, "REGIÓN DE CALLE / CANDIDATOS", (x1 + self.px(4), y1 + self.px(14)), 0.29)
                if road.road_area:
                    polygon = np.round(road.road_area).astype(np.int32)
                    cv2.polylines(annotated, [polygon], True, ROAD_ACCENT, self.px(1), AA)
                    self.text(annotated, "CALZADA / FILTRO DE VEREDAS", tuple(polygon[0]), 0.29, ROAD_ACCENT)
                for rejected in road.outside_road:
                    a, b, c, d = map(round, rejected.box)
                    cv2.rectangle(annotated, (a, b), (c, d), MUTED, self.px(1), AA)
                    self.text(annotated, "FUERA DE CALZADA", (a, max(12, b - 3)), 0.29, MUTED)
                for detection in road.raw:
                    if not is_valid_road_detection(detection):
                        continue
                    a, b, c, d = (round(v) for v in detection.box)
                    cv2.rectangle(annotated, (a, b), (c, d), MUTED, self.px(1), AA)
                    self.text(annotated, f"DETECCIÓN {detection.confidence:.0%}", (a, max(12, b - 3)), 0.29, MUTED)
        if scene is not None:
            for detection in scene.crossings.detections:
                self._box(annotated, detection, None, label_regions)
            self._scene_overlay(annotated, scene, debug_scene, corridor, metrics)
        active_tracks = len(active) + (sum(is_valid_road_detection(d) for d in road.detections) if road is not None else 0)
        self._hud(annotated, analytics, metrics, active_tracks, potholes if road is not None else None,
                  scene.crosswalks_seen if infrastructure and scene is not None else None)
        return annotated

    @staticmethod
    def _area(detection: Detection) -> float:
        x1, y1, x2, y2 = detection.box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _trails(self, frame: np.ndarray, active: list[Detection], analytics: Analytics) -> None:
        histories = [list(analytics.histories.get(d.track_id, ())) for d in active
                     if analytics.class_for(d) in MOVING_CLASSES]
        points = [p for history in histories for _, p in history]
        if not points:
            return
        pad = self.px(5)
        xs, ys = zip(*points)
        x0, y0 = max(0, int(min(xs))-pad), max(0, int(min(ys))-pad)
        x3, y3 = min(self.width, int(max(xs))+pad+1), min(self.height, int(max(ys))+pad+1)
        if x3 <= x0 or y3 <= y0:
            return
        region = frame[y0:y3, x0:x3]
        trail_layer = region.copy()
        def local(point):
            return round(point[0])-x0, round(point[1])-y0
        for history in histories:
            for (f1, p1), (f2, p2) in zip(history, history[1:]):
                if f2 != f1 + 1:
                    continue  # don't bridge occlusions
                if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) > self.width * 0.12:
                    continue  # don't draw screen-wide jumps after an ID switch
                age = analytics.current_frame - f2
                strength = 0.20 + 0.80 * (1 - age / analytics.trail_frames)
                color = tuple(round(c * strength) for c in ACCENT)
                cv2.line(trail_layer, local(p1), local(p2), color, self.px(2), AA)
            if history:
                point = local(history[-1][1])
                cv2.circle(trail_layer, point, self.px(3), ACCENT, -1, AA)
        cv2.addWeighted(trail_layer, 0.72, region, 0.28, 0, dst=region)

    def _proximity_banner(self, frame: np.ndarray, proximity) -> None:
        accent = ROAD_ACCENT if proximity.state == "CAUTION" else NEAR_ACCENT
        width = round(self.width * .80)
        left = (self.width - width) // 2
        top = self.header_bottom + self.px(10)
        bottom = top + self.px(65)
        self.panel(frame, (left, top, left + width, bottom), .86)
        cv2.rectangle(frame, (left, top), (left + self.px(3), bottom), accent, -1)
        def centered(value, baseline, size, color):
            text_width = self.text_size(value, size)[0][0]
            self.text(frame, value, ((self.width - text_width)//2, baseline), size, color)
        centered("VEHÍCULO DELANTE", top + self.px(24), .63, WHITE)
        state = "PRECAUCIÓN" if proximity.state == "CAUTION" else "MUY CERCA"
        centered(state, top + self.px(51), .78, accent)
        track = f"#{proximity.track_id}"
        track_width = self.text_size(track, .32)[0][0]
        self.text(frame, track, (left + width - track_width - self.px(8), bottom - self.px(6)), .32, MUTED)
        if proximity.box is not None:
            x1, y1, x2, y2 = map(round, proximity.box)
            x1, x2 = max(0,x1), min(self.width-1,x2)
            y1, y2 = max(0,y1), min(self.height-1,y2)
            if x2 > x1 and y2 > y1:
                length = min(self.px(16), (x2-x1)//3, (y2-y1)//3)
                for x, y, dx, dy in ((x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)):
                    cv2.line(frame,(x,y),(x+dx*length,y),accent,self.px(2),AA)
                    cv2.line(frame,(x,y),(x,y+dy*length),accent,self.px(2),AA)

    def _scene_overlay(self, frame: np.ndarray, scene: "SceneFrame", debug: bool, corridor: tuple,
                       metrics: FrameMetrics) -> None:
        proximity = scene.proximity
        if proximity.state in {"CAUTION", "NEAR"}:
            self._proximity_banner(frame, proximity)
        if not debug:
            return
        if corridor:
            polygon = np.asarray([(round(x*self.width), round(y*self.height)) for x,y in corridor],np.int32)
            cv2.polylines(frame,[polygon],True,MUTED,self.px(1),AA)
        for detection in scene.crossings.raw:
            x1,y1,x2,y2 = map(round,detection.box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),MUTED,self.px(1),AA)
            self.text(frame,f"CEBRA RAW {detection.confidence:.0%}",(x1,y1-self.px(3)),.29,MUTED)
        for track,value in scene.debug.get("observations",{}).items():
            box = scene.debug.get("depth_boxes",{}).get(track)
            if box:
                self.text(frame,f"DEPTH #{track}: {value:.2f}",(round(box[0]),round(box[3])),.28,MUTED)
        top = self.header_bottom+self.px(85)
        self.panel(frame,(self.margin,top,self.width-self.margin,top+self.px(140)),0.75)
        estimate = f"{proximity.estimate:.2f}" if proximity.estimate is not None else "—"
        lines = [f"CANDIDATO #{proximity.track_id} / DEPTH {estimate} / {proximity.state}",
                 f"PERSISTENCIA {proximity.persistent_frames} / MUESTRAS {proximity.fresh_samples}",
                 f"CEBRAS RAW {len(scene.crossings.raw)} / DEPTH NUEVO {scene.debug.get('fresh_depth',False)}"]
        thresholds = scene.debug.get("thresholds") or {}
        if thresholds:
            lines.append(f"UMBRALES {thresholds['near_threshold']:.2f} / {thresholds['caution_threshold']:.2f} / H {thresholds['hysteresis']:.2f}")
        lines.append(f"CEBRAS {scene.debug.get('crosswalk_ms',0):.1f}ms / DEPTH {scene.debug.get('depth_ms',0):.1f}ms")
        def rate(name):
            milliseconds = scene.debug.get(name,0)
            return f"{1000/milliseconds:.1f}" if milliseconds else "—"
        lines.append(f"FPS INF. GENERAL {rate('general_ms')} / POZOS {rate('road_ms')}")
        lines.append(f"VIDEO {metrics.source_fps:.1f} FPS / PROCESAMIENTO {metrics.processing_fps:.1f} FPS")
        for index,line in enumerate(lines):
            self.text(frame,line,(self.margin+self.px(9),top+self.px(17+18*index)),.30,MUTED)
        if scene.depth_map is not None:
            depth = scene.depth_map
            valid = depth[np.isfinite(depth) & (depth > 0)]
            if valid.size:
                low, high = np.percentile(valid,[5,95])
                normalized = np.nan_to_num(np.clip((depth-low)/max(.001,high-low),0,1))
                inset = cv2.applyColorMap((normalized*255).astype(np.uint8),cv2.COLORMAP_MAGMA)
                width = self.px(100)
                inset = cv2.resize(inset,(width,max(1,round(width*depth.shape[0]/depth.shape[1]))))
                x,y = self.width-self.margin-inset.shape[1],top+self.px(146)
                if y+inset.shape[0] < self.footer_top:
                    frame[y:y+inset.shape[0],x:x+inset.shape[1]] = inset

    def _box(self, frame: np.ndarray, detection: Detection, analytics: Analytics | None,
             occupied: list[tuple[int, int, int, int]], *, road: bool = False) -> None:
        accent = ROAD_ACCENT if road else ACCENT
        x1, y1, x2, y2 = (round(v) for v in detection.box)
        x1, x2 = sorted((max(0, min(self.width - 1, x1)), max(0, min(self.width - 1, x2))))
        y1, y2 = sorted((max(0, min(self.height - 1, y1)), max(0, min(self.height - 1, y2))))
        if x2 <= x1 or y2 <= y1:
            return
        # Quiet full rectangle, with brighter short corner marks.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (91, 135, 160) if road else (113, 150, 120), self.px(1), AA)
        corner = max(2, min(self.px(10), (x2 - x1) // 4, (y2 - y1) // 4))
        for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                              (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (x, y), (x + dx * corner, y), accent, self.px(1), AA)
            cv2.line(frame, (x, y), (x, y + dy * corner), accent, self.px(1), AA)
        category = analytics.class_for(detection) if analytics is not None else detection.class_name
        label = f"{DISPLAY_NAMES.get(category, category.upper())} #{detection.track_id}"
        if road or x2 - x1 >= self.px(105):
            label += f"  {detection.confidence:.0%}"
        (text_w, text_h), baseline = self.text_size(label, 0.36)
        pad = self.px(4)
        label_w, label_h = text_w + 2 * pad, text_h + baseline + 2 * pad
        label_x = max(self.margin, min(x1, self.width - self.margin - label_w))
        candidates = (y1 - label_h - self.px(3), y1 + self.px(3), y2 + self.px(3))
        for label_y in candidates:
            bounds = (label_x, label_y, label_x + label_w, label_y + label_h)
            if label_y <= self.header_bottom or bounds[3] >= self.footer_top:
                continue
            if any(self._overlap(bounds, other) for other in occupied):
                continue
            self.panel(frame, bounds, 0.80)
            self.text(frame, label, (label_x + pad, label_y + pad + text_h), 0.36)
            occupied.append(bounds)
            break

    @staticmethod
    def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def _hud(self, frame: np.ndarray, analytics: Analytics, metrics: FrameMetrics,
             active_count: int, potholes: int | None = None, crossings: int | None = None) -> None:
        left, right = self.margin, self.width - self.margin
        inner = left + self.px(12)
        self.panel(frame, (left, self.margin, right, self.header_bottom), 0.70)
        cv2.line(frame, (left, self.margin), (left + self.px(42), self.margin), ACCENT, self.px(2), AA)
        self.text(frame, "URBAN VISION", (inner, self.margin + self.px(24)), 0.62)
        self.text(frame, "ANÁLISIS EN VIVO", (inner, self.margin + self.px(43)), 0.32, MUTED)
        status = f"{active_count:02d} ACTIVOS"
        status_w = self.text_size(status, 0.32)[0][0]
        self.text(frame, status, (right - self.px(12) - status_w, self.margin + self.px(43)), 0.32, ACCENT)

        top = self.footer_top
        self.panel(frame, (left, top, right, self.height - self.margin), 0.82)
        self.text(frame, "OBJETOS ÚNICOS / EST.", (inner, top + self.px(17)), 0.29, MUTED)
        column_width = (right - left - self.px(24)) // 2
        for i, (label, category) in enumerate(HUD_COUNTS):
            x = inner + (i % 2) * column_width
            y = top + self.px(43 + (i // 2) * 29)
            self.text(frame, f"{analytics.counts[category]:02d}", (x, y), 0.68, ACCENT)
            self.text(frame, label, (x + self.px(44), y - self.px(2)), 0.32, WHITE)

        extra = 30 if potholes is not None else 0
        if potholes is not None:
            self.text(frame, "ESTADO DE LA CALLE", (inner, top + self.px(98)), 0.30, MUTED)
            self.text(frame, f"POZOS  {potholes:02d}",
                      (inner + column_width, top + self.px(98)), 0.36, ROAD_ACCENT)
        if crossings is not None:
            self.text(frame, f"SEMÁFOROS {analytics.counts['traffic light']:02d}   /   CEBRAS {crossings:02d}",
                      (inner,top+self.px(100+extra)),.32,MUTED)
            extra += 23
        divider_y = top + self.px(84 + extra)
        cv2.line(frame, (inner, divider_y), (right - self.px(12), divider_y), (72, 77, 65), self.px(1), AA)
        elapsed = metrics.elapsed_seconds if metrics.elapsed_seconds is not None else metrics.frame / metrics.source_fps
        minutes, seconds = divmod(int(elapsed), 60)
        frame_total = str(metrics.total_frames) if metrics.total_frames else "?"
        self.text(frame, f"F {metrics.frame:04d}/{frame_total}",
                  (inner, top + self.px(103 + extra)), 0.33, MUTED)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        timestamp_w = self.text_size(timestamp, 0.38)[0][0]
        self.text(frame, timestamp, (right - self.px(12) - timestamp_w, top + self.px(103 + extra)), 0.38)


def render_road_evidence(frame: np.ndarray, detection: Detection, frame_number: int,
                         timestamp: float) -> np.ndarray:
    if not is_valid_road_detection(detection):
        raise ValueError("Road evidence requires pothole confidence >= 0.50.")
    annotated = frame.copy()
    height, width = frame.shape[:2]
    renderer = Renderer(width, height)
    renderer.footer_top = height - renderer.margin
    renderer._box(annotated, detection, None, [], road=True)
    renderer.panel(annotated, (renderer.margin, renderer.margin,
                              width - renderer.margin, renderer.header_bottom), 0.75)
    renderer.text(annotated, "URBAN VISION / EVIDENCIA VIAL",
                  (renderer.margin + renderer.px(10), renderer.margin + renderer.px(23)), 0.39)
    renderer.text(annotated, f"POZO #{detection.track_id}  {detection.confidence:.1%} / F {frame_number} / {timestamp:.2f}s",
                  (renderer.margin + renderer.px(10), renderer.margin + renderer.px(43)), 0.34, ROAD_ACCENT)
    return annotated
