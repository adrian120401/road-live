"""Small, explicit defaults for inference and presentation."""

from dataclasses import dataclass, field
import math
from pathlib import Path


CLASSES = (
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "traffic light", "stop sign",
)
MOVING_CLASSES = frozenset(CLASSES[:6])
# In the source video, yellow speed-bump signs were predicted as STOP at <= 0.61.
# This gate is applied AFTER tracking, preserving low-score association candidates.
CLASS_MIN_CONFIDENCE = {"stop sign": 0.65}
MIN_POTHOLE_CONFIDENCE = 0.50
ROAD_DIAGNOSTIC_CONFIDENCE = 0.35
# Full carriageway approximation reviewed on the current windshield recording.
# Separate from the narrower driving corridor used by the proximity analyzer.
ROAD_AREA = ((0.48, 0.48), (0.72, 0.48), (1.0, 0.70), (0.0, 0.70))
DISPLAY_NAMES = {
    "person": "PERSONA", "car": "AUTO", "motorcycle": "MOTO", "bicycle": "BICICLETA",
    "bus": "ÓMNIBUS", "truck": "CAMIÓN", "traffic light": "SEMÁFORO",
    "stop sign": "PARE", "pothole": "POZO", "crosswalk": "PASO PEATONAL",
}
HUD_COUNTS = (("AUTOS", "car"), ("PERSONAS", "person"),
              ("MOTOS", "motorcycle"), ("BICICLETAS", "bicycle"))


@dataclass(frozen=True)
class Detection:
    track_id: int | None
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def ground_point(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


def is_presentable(detection: Detection) -> bool:
    if detection.class_name == "pothole":
        return is_valid_road_detection(detection)
    return detection.confidence >= CLASS_MIN_CONFIDENCE.get(detection.class_name, 0.0)


def is_valid_road_detection(detection: Detection, confidence: float = MIN_POTHOLE_CONFIDENCE) -> bool:
    return (detection.class_name == "pothole" and math.isfinite(detection.confidence)
            and detection.confidence >= max(MIN_POTHOLE_CONFIDENCE, confidence))


@dataclass(frozen=True)
class RoadFrame:
    """Normalized road observations; analytics has no dependency on a model API."""
    detections: tuple[Detection, ...] = ()
    observed: tuple[Detection, ...] = ()
    confirmed_ids: frozenset[int] = frozenset()
    expired_ids: tuple[int, ...] = ()
    inferred: bool = False
    raw: tuple[Detection, ...] = ()
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    road_area: tuple[tuple[float, float], ...] = ()
    outside_road: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class RoadDamageConfig:
    enabled: bool = False
    model: Path = Path("models/pothole_yolov8s.pt")
    confidence: float = MIN_POTHOLE_CONFIDENCE
    iou_threshold: float = 0.45
    frame_interval: int = 2
    roi: tuple[float, float, float, float] = (0.05, 0.45, 0.95, 0.65)
    image_size: int = 640
    debug: bool = False
    save_evidence: bool = True
    collect_candidates: bool = False
    candidates_dir: Path = Path("dataset_candidates")
    max_gap_seconds: float = 0.5
    confirmation_hits: int = 2
    road_area: tuple[tuple[float, float], ...] | None = ROAD_AREA
    road_area_min_overlap: float = 0.60

    def validate(self) -> None:
        if not MIN_POTHOLE_CONFIDENCE <= self.confidence <= 1:
            raise ValueError("--road-conf must be at least 0.50 and at most 1.")
        if self.frame_interval < 1:
            raise ValueError("--road-interval must be at least 1.")
        if not 0 < self.iou_threshold <= 1:
            raise ValueError("Road NMS IoU must be greater than 0 and at most 1.")
        x1, y1, x2, y2 = self.roi
        if not all(math.isfinite(v) for v in self.roi) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("--road-roi requires normalized LEFT TOP RIGHT BOTTOM within [0, 1].")
        if self.confirmation_hits < 2 or self.max_gap_seconds <= 0:
            raise ValueError("Road confirmation needs at least two observations and a positive gap.")
        if not math.isfinite(self.road_area_min_overlap) or not 0 < self.road_area_min_overlap <= 1:
            raise ValueError("--road-area-overlap must be greater than 0 and at most 1.")
        if self.road_area is not None:
            points = self.road_area
            if (len(points) != 4 or any(len(p) != 2 for p in points)
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p)):
                raise ValueError("--road-area requires four normalized X Y points.")
            turns = []
            for i in range(4):
                a, b, c = (points[(i + j) % 4] for j in range(3))
                turns.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
            if not (all(v > 0 for v in turns) or all(v < 0 for v in turns)):
                raise ValueError("--road-area must be a convex non-degenerate polygon.")
        if not self.model.is_file():
            raise ValueError(f"Road model not found: {self.model}. Run scripts/download_road_model.py first.")


@dataclass(frozen=True)
class LocationConfig:
    use_video_gps: bool = True
    force_mock_route: bool = False
    map_output: Path | None = None
    exiftool: Path | None = None
    max_gap_seconds: float = 10.0
    endpoint_tolerance_seconds: float = 2.0
    inspection_timeout_seconds: float = 60.0

    def validate(self) -> None:
        if self.map_output is not None and self.map_output.suffix.lower() != ".html":
            raise ValueError("--map-output must be an .html file.")
        if (not math.isfinite(self.max_gap_seconds) or self.max_gap_seconds <= 0
                or not math.isfinite(self.endpoint_tolerance_seconds) or self.endpoint_tolerance_seconds < 0
                or not math.isfinite(self.inspection_timeout_seconds) or self.inspection_timeout_seconds <= 0):
            raise ValueError("Location timing limits must be finite and valid.")


@dataclass(frozen=True)
class CrosswalkConfig:
    enabled: bool = False
    model: Path = Path("models/yoloe-26s-seg.pt")
    confidence: float = 0.35
    frame_interval: int = 3
    roi: tuple[float, float, float, float] = (0.0, 0.42, 1.0, 0.74)
    confirmation_hits: int = 3
    max_gap_seconds: float = 0.5
    nms_iou: float = 0.40

    def validate(self) -> None:
        if not 0 < self.confidence <= 1 or self.frame_interval < 1:
            raise ValueError("Crosswalk confidence/interval invalid.")
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Crosswalk ROI must be normalized within [0, 1].")
        if self.confirmation_hits < 2 or self.max_gap_seconds <= 0:
            raise ValueError("Crosswalk confirmation/gap invalid.")
        if not 0 < self.nms_iou <= 1:
            raise ValueError("Crosswalk NMS IoU invalid.")
        if not self.model.is_file():
            raise ValueError(f"Crosswalk model missing: {self.model}; run scripts/prepare_yoloe_crosswalk.py.")


@dataclass(frozen=True)
class ProximityConfig:
    enabled: bool = False
    depth_enabled: bool = True
    depth_model: Path = Path("models/yolo26n-depth.pt")
    depth_interval: int = 3
    depth_image_size: int = 768
    profile: Path | None = None
    corridor: tuple[tuple[float, float], ...] = ((0.45, 0.48), (0.55, 0.48), (0.70, 0.74), (0.30, 0.74))
    warning_frames: int = 5
    confirmation_samples: int = 2
    release_samples: int = 3
    candidate_confidence: float = 0.30

    def validate(self) -> None:
        if self.depth_interval < 1 or self.depth_image_size < 128 or self.warning_frames < 2:
            raise ValueError("Depth interval/size and proximity persistence invalid.")
        if self.confirmation_samples < 2 or self.release_samples < 2:
            raise ValueError("Proximity requires repeated fresh depth observations.")
        if not 0 < self.candidate_confidence <= 1:
            raise ValueError("Proximity candidate confidence invalid.")
        if (len(self.corridor) != 4 or any(not math.isfinite(v) or not 0 <= v <= 1
                                        for p in self.corridor for v in p)):
            raise ValueError("Road corridor requires four normalized points.")
        # Reject degenerate/self-intersecting polygons without depending on OpenCV.
        cross = []
        for i in range(4):
            a, b, c = (self.corridor[(i + j) % 4] for j in range(3))
            cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
        if not (all(v > 0 for v in cross) or all(v < 0 for v in cross)):
            raise ValueError("Road corridor must be a convex non-degenerate polygon.")
        if self.depth_enabled and not self.depth_model.is_file():
            raise ValueError(f"Depth model missing: {self.depth_model}; run scripts/download_depth_model.py.")
        if self.profile is not None and not self.profile.is_file():
            raise ValueError(f"Proximity profile missing: {self.profile}.")


@dataclass(frozen=True)
class TrafficConfig:
    traffic_lights: bool = True
    crosswalks: CrosswalkConfig = field(default_factory=CrosswalkConfig)


@dataclass(frozen=True)
class MapConfig:
    enhanced_popups: bool = True


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path = Path("outputs/urban_vision_output.mp4")
    model: str = "yolo26n.pt"
    tracker: str = "botsort.yaml"
    conf: float = 0.10
    device: str = "auto"
    show: bool = False
    image_size: int = 640
    trail_frames: int = 30
    road_damage: RoadDamageConfig = field(default_factory=RoadDamageConfig)
    location: LocationConfig = field(default_factory=LocationConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    map: MapConfig = field(default_factory=MapConfig)
    debug_scene: bool = False
    display_confidence: float = 0.30
    processing_max_side: int = 1920
    prefetch_frames: int = 2

    def validate(self) -> None:
        if not self.input_path.is_file():
            raise ValueError(f"Input video does not exist: {self.input_path}")
        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError("Input and output paths must be different.")
        if self.output_path.suffix.lower() != ".mp4":
            raise ValueError("Output must be an .mp4 file.")
        if not 0 < self.conf <= 1:
            raise ValueError("--conf must be greater than 0 and at most 1.")
        if not 0 < self.display_confidence <= 1:
            raise ValueError("--display-conf must be greater than 0 and at most 1.")
        if self.processing_max_side != 0 and self.processing_max_side < 128:
            raise ValueError("--processing-max-side must be 0 or at least 128.")
        if not 0 <= self.prefetch_frames <= 8:
            raise ValueError("--prefetch-frames must be between 0 and 8.")
        if self.road_damage.enabled:
            self.road_damage.validate()
        self.location.validate()
        if self.traffic.crosswalks.enabled:
            self.traffic.crosswalks.validate()
        if self.proximity.enabled:
            self.proximity.validate()
