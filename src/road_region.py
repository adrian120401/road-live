"""Configurable carriageway gate; this geometry does not classify road surfaces."""

import math

import cv2
import numpy as np

from .config import Detection, RoadDamageConfig


class RoadRegion:
    def __init__(self, config: RoadDamageConfig, width: int, height: int) -> None:
        self.points = tuple((x * width, y * height) for x, y in config.road_area) if config.road_area else ()
        self.contour = np.asarray(self.points, dtype=np.float32)
        self.min_overlap = config.road_area_min_overlap

    def accepts(self, detection: Detection) -> bool:
        x1, y1, x2, y2 = detection.box
        if not all(math.isfinite(v) for v in detection.box) or x2 <= x1 or y2 <= y1:
            return False
        if not self.points:
            return True
        # Potholes lie on a surface: use the box center rather than its bottom edge.
        # A curb-spanning box must also have most of its area inside the carriageway.
        if cv2.pointPolygonTest(self.contour, ((x1 + x2) / 2, (y1 + y2) / 2), False) < 0:
            return False
        box = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
        area, _ = cv2.intersectConvexConvex(self.contour, box)
        return area / ((x2 - x1) * (y2 - y1)) + 1e-6 >= self.min_overlap
