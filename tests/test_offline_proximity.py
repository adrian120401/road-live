"""Restoring a scene layer must retain cached detections and the manual map."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.analysis_cache import AnalysisCache, read_cache
from src.config import Detection, RoadFrame
from src.offline_media import atomic_json
from src.offline_proximity import augment_proximity, augmented_cache, file_digest
from src.proximity import ProximityFrame
from src.review_project import ProjectStore
from src.scene import SceneFrame


class FakeCapture:
    def read(self):
        return True, np.zeros((200, 100, 3), np.uint8)

    def release(self):
        pass


class FakeDepth:
    def __init__(self, *args):
        pass

    def update(self, *args, **kwargs):
        return {}, True

    def report(self):
        return {}


class FakeProximity:
    def __init__(self, *args):
        pass

    def update(self, *args):
        return ProximityFrame(42, "CAUTION", 10, 10, 3, (40, 100, 60, 140))

    def report(self):
        return {}


class OfflineProximityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.project_path = self.root / "project.json"
        self.project = {"version": 1, "revision": 33, "name": "Recorrido", "source": {"sha256": "original"},
            "analysis": {"complete": True, "frames": 3, "cache": "observations.jsonl", "source_path": "source.mp4"},
            "center": [-33.5, -56.9], "route": [[-33.5, -56.9], [-33.501, -56.9]],
            "events": [{"event_id": 7, "status": "accepted", "position": [-33.5005, -56.9],
                        "observation": {"event_id": 7, "track_id": 10, "confidence": .9}}]}
        atomic_json(self.project_path, self.project)
        cache = AnalysisCache(self.root / "observations.jsonl", {"resolution": [100, 200], "source_resolution": [100, 200],
                             "source_fps": 60, "source_total_frames": 3})
        for number in range(1, 4):
            cache.append(number, (number - 1) / 60, [Detection(42, "car", .9, (40, 100, 60, 140))],
                         RoadFrame(detections=(Detection(10, "pothole", .9, (10, 150, 30, 180)),)), SceneFrame())
        cache.finish(True)
        self.profile_path = self.root / "profile.json"
        atomic_json(self.profile_path, {"source_sha256": "original", "depth_image_size": 768,
                                      "corridor": [[.45, .48], [.55, .48], [.78, .82], [.22, .82]]})

    def augment(self):
        with patch("src.offline_proximity.DepthEstimator", FakeDepth), \
             patch("src.offline_proximity.ProximityAnalyzer", FakeProximity), \
             patch("src.offline_proximity.select_device", return_value="cpu"), \
             patch("src.offline_proximity.cv2.VideoCapture", return_value=FakeCapture()):
            return augment_proximity(self.project_path, self.profile_path)

    def test_add_layer_preserves_map_review_original_cache_ids_and_timestamps(self):
        original_project = self.project_path.read_bytes()
        original_cache = (self.root / "observations.jsonl").read_bytes()
        result = self.augment()
        self.assertEqual(self.project_path.read_bytes(), original_project)
        self.assertEqual((self.root / "observations.jsonl").read_bytes(), original_cache)
        old = list(read_cache(self.root / "observations.jsonl"))
        new = list(read_cache(self.root / result["cache"]))
        for before, after in zip(old[1:], new[1:]):
            for key in ("frame", "timestamp", "detections", "road"):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after["scene"]["proximity"]["state"], "CAUTION")
            self.assertEqual(before["scene"]["crossings"], after["scene"]["crossings"])
        self.assertEqual(augmented_cache(self.root, self.project), self.root / "observations_proximity.jsonl")

    def test_layer_from_another_video_cannot_be_used(self):
        self.augment()
        other = {**self.project, "source": {"sha256": "another"}}
        with self.assertRaises(ValueError):
            augmented_cache(self.root, other)

    def test_replacing_base_cache_invalidates_added_layer(self):
        self.augment()
        with (self.root / "observations.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            augmented_cache(self.root, self.project)

    def test_unaugmented_project_keeps_original_cache(self):
        self.assertEqual(augmented_cache(self.root, self.project), self.root / "observations.jsonl")
