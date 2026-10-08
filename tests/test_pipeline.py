"""Exercise real MP4 I/O with deterministic tracks, including error finalization."""

import json
import tempfile
from dataclasses import replace
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.config import Config, Detection, RoadDamageConfig, LocationConfig
from src.main import process_video
from src.road_damage import RoadAssociator, roi_pixels
from src.tracker import InferenceStats
from src.analysis_cache import read_cache


class FixedTracker:
    device = "cpu"

    def __init__(self, config):
        self.frames = 0

    def warmup(self, frame):
        pass

    def update(self, frame):
        self.frames += 1
        return [Detection(12, "car", 0.94, (200, 350, 250, 420))]


class FailingTracker(FixedTracker):
    def update(self, frame):
        if self.frames == 1:
            raise RuntimeError("Injected inference failure")
        return super().update(frame)


class FixedRoadDetector:
    def __init__(self, config, device, fps, width, height):
        self.config = config
        self.associator = RoadAssociator(config, fps)
        self.stats = InferenceStats()
        self.roi = roi_pixels(config.roi, width, height)
        self.last = None

    def warmup(self, frame):
        pass

    def update(self, frame, frame_number, timestamp=None):
        if (frame_number - 1) % self.config.frame_interval:
            return replace(self.last, inferred=False, observed=())
        self.stats.record(1, 0.001)
        self.last = self.associator.update([Detection(None, "pothole", 0.9, (230, 450, 270, 480))], frame_number, timestamp)
        return replace(self.last, roi=self.roi)


class FailingRoadDetector(FixedRoadDetector):
    def update(self, frame, frame_number, timestamp=None):
        if frame_number == 3:
            raise RuntimeError("Injected road inference failure")
        return super().update(frame, frame_number, timestamp)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.mp4"
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (464, 832))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(3):
                writer.write(np.full((832, 464, 3), 100, dtype=np.uint8))
        finally:
            writer.release()

    def decoded_frames(self, video):
        capture = cv2.VideoCapture(str(video))
        self.assertTrue(capture.isOpened())
        frames = 0
        try:
            while capture.read()[0]:
                frames += 1
        finally:
            capture.release()
        return frames

    def test_full_pipeline_counts_one_id_across_three_frames(self):
        output = self.root / "nested" / "output.mp4"
        with patch("src.main.ObjectTracker", FixedTracker):
            report = process_video(Config(self.source, output))
        self.assertTrue(report["complete"])
        self.assertEqual(report["frames_processed"], 3)
        self.assertEqual(report["source_resolution"],[464,832])
        self.assertEqual(report["resolution"],[464,832])
        self.assertEqual(report["prefetch_frames"],2)
        self.assertEqual(report["counts"]["car"], 1)
        self.assertEqual(report["tracks"]["12"]["longest_consecutive_run"], 3)
        self.assertEqual(self.decoded_frames(output), 3)
        self.assertTrue(output.with_suffix(".json").is_file())

    def test_pipeline_analysis_and_export_use_one_scaled_resolution(self):
        output = self.root / "scaled.mp4"
        config = Config(self.source,output,processing_max_side=416)
        sizes = []
        class SizedTracker(FixedTracker):
            def update(self,frame):
                sizes.append(frame.shape)
                return []
        with patch("src.main.ObjectTracker",SizedTracker):
            report = process_video(config)
        self.assertEqual(report["source_resolution"],[464,832])
        self.assertEqual(report["resolution"],[232,416])
        self.assertEqual(sizes,[(416,232,3)]*3)
        self.assertEqual(self.decoded_frames(output),3)

    def test_review_analysis_caches_all_frames_without_assigning_fake_locations(self):
        output = self.root / "analysis.mp4"
        cache = self.root / "observations.jsonl"
        with patch("src.main.ObjectTracker", FixedTracker), patch("src.main.RoadDamageDetector", FixedRoadDetector):
            report = process_video(self.road_config(output), trace_path=cache,
                                   write_video=False, manual_location=True)
        self.assertTrue(report["complete"])
        self.assertFalse(output.exists())
        self.assertFalse(report["video_written"])
        self.assertEqual(report["location"]["source"], "manual_pending")
        self.assertEqual(report["location"]["trajectory"], [])
        rows = list(read_cache(cache))
        self.assertEqual([row["frame"] for row in rows[1:]], [1, 2, 3])
        self.assertEqual(rows[-1]["road"][0]["track_id"], 1)
        event = json.loads(self.root.joinpath("analysis_events.json").read_text())["events"][0]
        self.assertNotIn("latitude", event)
        self.assertNotIn("longitude", event)
        self.assertFalse(self.root.joinpath("analysis_map.html").exists())

    def road_config(self, output, **kwargs):
        weights = self.root / "model.pt"
        weights.touch()
        return Config(self.source, output, road_damage=RoadDamageConfig(enabled=True, model=weights, **kwargs))

    def test_v2_exports_one_event_one_photo_and_keeps_general_ids(self):
        output = self.root / "v2.mp4"
        with patch("src.main.ObjectTracker", FixedTracker), patch("src.main.RoadDamageDetector", FixedRoadDetector):
            report = process_video(self.road_config(output))
        self.assertEqual(report["counts"]["car"], 1)
        self.assertEqual(report["tracks"]["12"]["observations"], 3)
        self.assertEqual(report["road_damage"]["potholes"], 1)
        self.assertEqual(report["road_damage"]["saved_evidence"], 1)
        self.assertEqual(report["road_damage"]["performance"]["inferences"], 2)
        self.assertEqual(self.decoded_frames(output), 3)
        payload = json.loads(Path(report["road_damage"]["events_path"]).read_text())
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["events"][0]["observations"], 2)
        event = payload["events"][0]
        self.assertEqual(event["location_source"], "mock")
        self.assertEqual(event["event_id"], 1)
        self.assertEqual(event["evidence_image"], event["evidence_path"])
        self.assertEqual(report["location"]["geolocated_potholes"], 1)
        self.assertTrue(Path(report["location"]["map_path"]).is_file())
        self.assertEqual(len(report["location"]["trajectory"]),report["location"]["route_points"])
        self.assertFalse(output.with_name(output.stem+"_route.geojson").exists())

    def test_road_failure_preserves_partial_output_and_empty_events(self):
        output = self.root / "road_partial.mp4"
        with patch("src.main.ObjectTracker", FixedTracker), patch("src.main.RoadDamageDetector", FailingRoadDetector):
            with self.assertRaisesRegex(RuntimeError, "Injected road inference failure"):
                process_video(self.road_config(output))
        report = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(report["frames_processed"], 2)
        self.assertEqual(report["tracks"]["12"]["last_frame"], 2)
        self.assertEqual(self.decoded_frames(output), 2)
        payload = json.loads(Path(report["road_damage"]["events_path"]).read_text())
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["events"], [])

    def test_map_export_failure_reports_error_without_invalidating_finished_video(self):
        output = self.root / "map_failure.mp4"
        bad_map = self.root / "map.html"
        bad_map.mkdir()
        config = replace(self.road_config(output), location=LocationConfig(map_output=bad_map))
        with patch("src.main.ObjectTracker", FixedTracker), patch("src.main.RoadDamageDetector", FixedRoadDetector):
            with self.assertRaises(OSError):
                process_video(config)
        report = json.loads(output.with_suffix(".json").read_text())
        payload = json.loads(Path(report["road_damage"]["events_path"]).read_text())
        self.assertTrue(report["complete"])
        self.assertTrue(payload["complete"])
        self.assertIn("export_error", report["location"])
        self.assertEqual(self.decoded_frames(output), 3)

    def test_disabling_road_never_initializes_second_model_or_writes_events(self):
        output = self.root / "general_only.mp4"
        with patch("src.main.ObjectTracker", FixedTracker), patch("src.main.RoadDamageDetector") as road:
            report = process_video(Config(self.source, output))
        road.assert_not_called()
        self.assertNotIn("road_damage", report)
        self.assertFalse(output.with_name(output.stem + "_events.json").exists())

    def test_inference_error_finalizes_decodable_partial_video(self):
        output = self.root / "partial.mp4"
        with patch("src.main.ObjectTracker", FailingTracker):
            with self.assertRaisesRegex(RuntimeError, "Injected inference failure"):
                process_video(Config(self.source, output))
        report = json.loads(output.with_suffix(".json").read_text())
        self.assertFalse(report["complete"])
        self.assertEqual(report["stop_reason"], "error")
        self.assertEqual(report["frames_processed"], 1)
        self.assertEqual(self.decoded_frames(output), 1)
