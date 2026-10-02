"""Road IDs, actual observation counting, coordinate restoration and evidence lifecycle."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from src.config import Detection, RoadDamageConfig, RoadFrame
from src.road_damage import RoadAssociator, RoadDamageDetector, roi_pixels, restore_box, road_class_ids, REVIEWED_CHECKPOINT_SHA256
from src.road_analytics import RoadAnalytics
from src.location import MockRouteLocationProvider
from src.renderer import Renderer, FrameMetrics, render_road_evidence


def pothole(box=(100, 400, 140, 420), confidence=0.8):
    return Detection(None, "pothole", confidence, box)


class RoadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model_path = self.root / "model.pt"
        self.model_path.touch()
        # Legacy tracking tests isolate association/evidence from spatial calibration.
        self.config = RoadDamageConfig(enabled=True, model=self.model_path, road_area=None)
        self.associator = RoadAssociator(self.config, 30)
        self.frame = np.full((832, 464, 3), 100, np.uint8)

    def test_roi_and_crop_coordinates_are_restored_and_clamped(self):
        roi = roi_pixels((0.05, 0.45, 0.95, 0.65), 464, 832)
        self.assertEqual(roi, (23, 374, 441, 541))
        self.assertEqual(restore_box((-20, 10, 500, 190), roi), (23, 384, 441, 541))
        with self.assertRaises(ValueError):
            roi_pixels((0, 0, 0.001, 0.001), 464, 832)

    def test_invalid_config_and_missing_weights_fail(self):
        for config in (replace(self.config, confidence=0), replace(self.config, frame_interval=0),
                       replace(self.config, confidence=0.49),
                       replace(self.config, roi=(0, 0.7, 1, 0.5)),
                       replace(self.config, model=self.root / "missing.pt")):
            with self.assertRaises(ValueError):
                config.validate()

    def test_confidence_floor_filters_before_confirmation_and_evidence(self):
        for confidence in (0.43, 0.49, 0.50, 0.51):
            with self.subTest(confidence=confidence):
                associator = RoadAssociator(self.config, 30)
                output = self.root / f"threshold_{confidence}.mp4"
                analytics = RoadAnalytics(self.config, output, 30, MockRouteLocationProvider(10))
                for frame in (1, 3):
                    state = associator.update([pothole(confidence=confidence)], frame)
                    analytics.update(state, self.frame, frame)
                summary = analytics.finish(True, "end_of_video")
                self.assertEqual(summary["potholes"], int(confidence >= 0.5))
                self.assertEqual(summary["saved_evidence"], int(confidence >= 0.5))
                self.assertEqual(len(state.detections), int(confidence >= 0.5))

    def test_renderer_and_analytics_reject_low_confidence_bypassing_detector(self):
        detection = Detection(1, "pothole", 0.43, (100, 400, 140, 420))
        state = RoadFrame((detection,), (detection,), frozenset({1}), inferred=True, raw=(detection,))
        analytics = RoadAnalytics(self.config, self.root / "invalid.mp4", 30)
        analytics.update(state, self.frame, 1)
        self.assertEqual(analytics.finish(True, "end_of_video")["potholes"], 0)
        from src.analytics import Analytics
        renderer = Renderer(464, 832)
        with patch.object(renderer, "_box") as box:
            renderer.render(self.frame, [], Analytics(30), FrameMetrics(1, 3, 30, 10), road=state, debug_road=True)
        box.assert_not_called()
        with self.assertRaises(ValueError):
            render_road_evidence(self.frame, detection, 1, 0)

    def test_higher_threshold_and_geolocation_follow_representative_frame(self):
        config = replace(self.config, confidence=0.6)
        associator = RoadAssociator(config, 30)
        provider = MockRouteLocationProvider(10)
        analytics = RoadAnalytics(config, self.root / "geo.mp4", 30, provider)
        for frame, confidence, timestamp in ((1, 0.55, 0), (3, 0.61, 1), (5, 0.9, 2)):
            state = associator.update([pothole(confidence=confidence)], frame)
            analytics.update(state, self.frame, frame, timestamp)
        analytics.finish(False, "frame_limit")
        event = list(analytics.events.values())[0]
        self.assertEqual(event["event_id"], 1)
        self.assertEqual(event["frame"], 5)
        self.assertEqual(event["timestamp"], 2)
        self.assertEqual(event["latitude"], provider.point_at(2).latitude)
        self.assertEqual(event["location_source"], "mock")
        self.assertEqual(event["evidence_path"], event["evidence_image"])

    def test_diagnostic_proposals_are_counted_but_do_not_get_track_ids(self):
        detector = self.detector()
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[80.,30.,120.,50.]]),
                                                       conf=torch.tensor([0.43])), speed={"inference":3.0})
        detector.model.predict.side_effect = lambda *args, **kwargs: iter([result])
        for frame in (1, 3):
            state = detector.update(self.frame, frame)
        self.assertEqual(detector.discarded_below_50, 2)
        self.assertFalse(state.observed)
        self.assertFalse(state.raw)
        self.assertEqual(detector.associator.next_id, 1)

    def test_one_id_is_confirmed_only_after_actual_second_observation(self):
        first = self.associator.update([pothole()], 1)
        second = self.associator.update([pothole((102, 402, 142, 422))], 3)
        self.assertFalse(first.detections)
        self.assertEqual(second.detections[0].track_id, 1)
        self.assertEqual(second.confirmed_ids, {1})

    def test_two_close_potholes_match_one_to_one_even_if_order_reverses(self):
        left, right = pothole(), pothole((150, 400, 190, 420))
        self.associator.update([left, right], 1)
        state = self.associator.update([right, left], 3)
        self.assertEqual([d.track_id for d in state.detections], [2, 1])

    def test_gap_breaks_confirmation_but_confirmed_id_survives_short_loss(self):
        self.associator.update([pothole()], 1)
        self.associator.update([], 3)
        self.assertFalse(self.associator.update([pothole()], 5).detections)
        self.assertEqual(self.associator.update([pothole()], 7).detections[0].track_id, 1)
        self.associator.update([], 9)
        self.assertEqual(self.associator.update([pothole()], 11).detections[0].track_id, 1)

    def test_expired_track_gets_new_id_and_requires_confirmation(self):
        self.associator.update([pothole()], 1)
        self.associator.update([pothole()], 3)
        state = self.associator.update([pothole()], 20)
        self.assertEqual(state.expired_ids, (1,))
        self.assertEqual(state.observed[0].track_id, 2)
        self.assertFalse(state.detections)

    def test_road_association_does_not_reset_ultralytics_global_ids(self):
        from ultralytics.trackers.basetrack import BaseTrack
        before = BaseTrack._count
        self.associator.update([pothole()], 1)
        self.associator.update([pothole()], 3)
        self.assertEqual(BaseTrack._count, before)

    def detector(self, names=None):
        fake = MagicMock()
        fake.task = "detect"
        fake.names = names or {0: "pothole"}
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[80., 30., 120., 50.]]),
                                                       conf=torch.tensor([0.8])), speed={"inference": 3.0})
        fake.predict.side_effect = lambda *args, **kwargs: iter([result])
        with patch("src.road_damage.YOLO", return_value=fake):
            return RoadDamageDetector(self.config, "cpu", 30, 464, 832)

    def test_second_model_filters_classes_and_rejects_coco(self):
        with self.assertRaisesRegex(ValueError, "no pothole"):
            self.detector({0: "person", 1: "car"})
        self.assertEqual(self.detector({0: "D00", 1: "D40"}).class_ids, [1])

    def test_numeric_class_alias_requires_exact_reviewed_checkpoint(self):
        self.assertEqual(road_class_ids({0: "0"}, self.model_path), [])
        with patch("src.road_damage.hashlib.file_digest") as digest:
            digest.return_value.hexdigest.return_value = REVIEWED_CHECKPOINT_SHA256
            self.assertEqual(road_class_ids({0: "0"}, self.model_path), [0])
            self.assertEqual(road_class_ids({0: "person"}, self.model_path), [])

    def test_skipped_frames_do_not_count_as_inference_or_observation(self):
        detector = self.detector()
        analytics = RoadAnalytics(self.config, self.root / "video.mp4", 30)
        for frame_number in range(1, 7):
            state = detector.update(self.frame, frame_number)
            analytics.update(state, self.frame, frame_number)
        self.assertEqual(detector.stats.calls, 3)
        self.assertEqual(analytics.observations, 3)
        self.assertEqual(analytics.count, 1)
        self.assertEqual(detector.model.predict.call_count, 3)

    def test_road_nms_suppresses_overlapping_proposals_before_association(self):
        detector = self.detector()
        detector.update(self.frame, 1)
        self.assertEqual(detector.model.predict.call_args.kwargs["iou"], 0.45)

    def test_best_evidence_is_saved_once_and_matches_event_on_partial_finish(self):
        config = replace(self.config, collect_candidates=True, candidates_dir=self.root / "candidates")
        analytics = RoadAnalytics(config, self.root / "video.mp4", 30)
        for frame_number, confidence in ((1, 0.9), (3, 0.7), (5, 0.8)):
            state = self.associator.update([pothole(confidence=confidence)], frame_number)
            analytics.update(state, self.frame, frame_number)
        summary = analytics.finish(False, "interrupted")
        event = json.loads(analytics.events_path.read_text())["events"][0]
        self.assertEqual(summary["saved_evidence"], 1)
        self.assertEqual(event["frame"], 1)
        self.assertEqual(event["confirmed_frame"], 3)
        self.assertEqual(event["timestamp"], 0)
        self.assertEqual(event["confidence"], 0.9)
        path = analytics.events_path.parent / event["evidence_path"]
        self.assertEqual(cv2.imread(str(path)).shape, self.frame.shape)
        self.assertEqual(len(list(analytics.evidence_dir.glob("*.jpg"))), 1)
        self.assertEqual(len(list(analytics.candidates_dir.glob("*.jpg"))), 1)
        self.assertFalse(analytics.best)
        analytics.finish(False, "interrupted")
        self.assertEqual(len(list(analytics.evidence_dir.glob("*.jpg"))), 1)

    def test_unconfirmed_candidates_and_disabled_evidence_do_not_create_photos(self):
        config = replace(self.config, save_evidence=False)
        analytics = RoadAnalytics(config, self.root / "video.mp4", 30)
        state = self.associator.update([pothole()], 1)
        analytics.update(state, self.frame, 1)
        state = self.associator.update([pothole()], 3)
        analytics.update(state, self.frame, 3)
        self.assertEqual(analytics.finish(True, "end_of_video")["saved_evidence"], 0)
        self.assertFalse(analytics.evidence_dir.exists())
        empty = RoadAnalytics(self.config, self.root / "empty.mp4", 30)
        state = RoadAssociator(self.config, 30).update([pothole()], 1)
        empty.update(state, self.frame, 1)
        self.assertEqual(empty.finish(True, "end_of_video")["potholes"], 0)
        self.assertFalse(empty.evidence_dir.exists())


if __name__ == "__main__":
    unittest.main()
