"""CFR timing, review filtering and cache validation without downloading models."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.analysis_cache import AnalysisCache, read_cache
from src.analytics import Analytics
from src.config import Detection, RoadFrame
from src.offline_export import ManualRoute, repetitions, replay_render, write_reviewed_map
from src.renderer import Renderer
from src.scene import SceneFrame


class OfflineExportTests(unittest.TestCase):
    def test_variable_source_intervals_preserve_duration_at_60_fps(self):
        times = [0, .016, .034, .050, .120, .137]
        count = sum(repetitions(a, b) for a, b in zip(times, times[1:]))
        self.assertEqual(count, 9)
        self.assertLess(abs(count / 60 - times[-1]), 1 / 60)
        self.assertEqual(repetitions(0, 8), 480)

    def test_cache_completion_order_and_monotonic_times(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.jsonl"
            cache = AnalysisCache(path, {"resolution": [100, 200]})
            cache.append(1, 0., [], RoadFrame(), SceneFrame())
            cache.append(2, .02, [], RoadFrame(), SceneFrame())
            cache.finish(True)
            self.assertEqual(len(list(read_cache(path))), 3)
            content = path.read_text().replace('"frame":2', '"frame":3')
            path.write_text(content)
            with self.assertRaises(ValueError):
                list(read_cache(path))

    def test_partial_cache_is_never_consumed_as_complete_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.jsonl"
            cache = AnalysisCache(path, {})
            cache.append(1, 0., [], RoadFrame(), SceneFrame())
            cache.finish(False)
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                list(read_cache(path.with_suffix(".jsonl.partial")))

    def test_rejected_potholes_are_hidden_and_accepted_count_waits_for_confirmation(self):
        image = np.zeros((400, 200, 3), np.uint8)
        renderer = Renderer(200, 400)
        metadata = {"resolution": [100, 200], "source_total_frames": 10, "source_fps": 60,
                    "infrastructure": False}
        box = {"track_id": 1, "class_name": "pothole", "confidence": .9, "box": [10, 100, 30, 120]}
        scene = {"crossings": [], "proximity": {}, "crosswalks_seen": 0}
        row = {"frame": 1, "timestamp": 0, "detections": [], "road": [box, {**box, "track_id": 2}], "scene": scene}
        accepted = {1: {"confirmed_frame": 2, "event_id": 7}}
        with patch.object(renderer, "render", return_value=image) as render:
            replay_render(image, row, metadata, Analytics(), renderer, accepted)
        kwargs = render.call_args.kwargs
        self.assertEqual(kwargs["potholes"], 0)
        self.assertEqual([d.track_id for d in kwargs["road"].detections], [7])
        self.assertEqual(kwargs["road"].detections[0].box, (20, 200, 60, 240))
        row["frame"] = 2
        with patch.object(renderer, "render", return_value=image) as render:
            replay_render(image, row, metadata, Analytics(), renderer, accepted)
        self.assertEqual(render.call_args.kwargs["potholes"], 1)

    def test_manual_map_preserves_route_and_labels_manual_positions(self):
        project = {"revision": 3, "route": [[-33.5, -56.9], [-33.501, -56.9], [-33.501, -56.901]],
                   "analysis": {"events": "original.json"}, "events": []}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = write_reviewed_map(project, root, 20)
            text = path.read_text(encoding="utf-8")
            self.assertIn("MARCADA MANUALMENTE", text)
            self.assertNotIn("RECORRIDO CON GPS REAL", text)
            self.assertIn("Sin pozos confirmados", text)
            feature = json.loads((root / "reviewed_r3.geojson").read_text())["features"][0]
            self.assertEqual(feature["geometry"]["coordinates"], [[-56.9, -33.5], [-56.9, -33.501], [-56.901, -33.501]])
            self.assertFalse(feature["properties"]["temporal"])


if __name__ == "__main__":
    unittest.main()
