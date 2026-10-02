"""Temporal perception checks independent of downloaded models and GPU."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.config import Config, CrosswalkConfig, Detection, ProximityConfig, TrafficConfig
from src.crosswalk import CrosswalkDetector, decode_predictions
from src.depth import DepthObservation, sample_depth
from src.proximity import ProximityAnalyzer, in_corridor
from src.scene import SceneAnalyzer


def vehicle(track=12, box=(42,50,58,68)):
    return Detection(track,"car",.9,box)


class DepthTests(unittest.TestCase):
    def test_sampling_ignores_background_extremes_and_invalid_pixels(self):
        depth = np.full((100,100),40.0)
        depth[57:65,46:54] = 7
        depth[59,49] = 9999
        depth[60,49] = float("nan")
        result = sample_depth(depth,vehicle(),(100,100),1)
        self.assertAlmostEqual(result.estimate,7)

    def test_insufficient_samples_do_not_invent_distance(self):
        for depth in (np.zeros((100,100)),np.full((100,100),np.nan)):
            self.assertIsNone(sample_depth(depth,vehicle(),(100,100),0))

    def test_lateral_vehicles_and_people_are_excluded(self):
        config = ProximityConfig()
        self.assertTrue(in_corridor(vehicle(),config,(100,100)))
        self.assertFalse(in_corridor(vehicle(box=(75,50,95,70)),config,(100,100)))
        self.assertFalse(in_corridor(replace(vehicle(),class_name="person"),config,(100,100)))
        self.assertFalse(in_corridor(replace(vehicle(),confidence=.22),config,(100,100)))


class ProximityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name)/"profile.json"
        path.write_text(json.dumps({"near_threshold":5,"caution_threshold":10,"hysteresis":1}))
        self.analyzer = ProximityAnalyzer(ProximityConfig(profile=path))

    def update(self, number, value=4, fresh=True, track=12, detections=None):
        d = vehicle(track)
        o = DepthObservation(track,value,d.box,number/30,1)
        return self.analyzer.update([d] if detections is None else detections,{track:o},fresh,
                                    (100,100),number,number/30)

    def test_cached_depth_does_not_confirm_an_alert(self):
        for frame in range(1,10):
            state = self.update(frame,fresh=frame==1)
        self.assertEqual(state.state,"SAFE")
        state = self.update(10)
        self.assertEqual(state.state,"NEAR")

    def test_hysteresis_and_release_need_fresh_evidence(self):
        for frame in range(1,6):
            state = self.update(frame)
        self.assertEqual(state.state,"NEAR")
        for frame in range(6,16):
            state = self.update(frame,value=5.7)
        self.assertEqual(state.state,"NEAR")
        for frame in range(16,26):
            state = self.update(frame,value=8)
        self.assertEqual(state.state,"CAUTION")

    def test_candidate_change_and_loss_reset_persistence(self):
        for frame in range(1,6):
            self.update(frame)
        self.assertEqual(self.update(6,track=20).state,"SAFE")
        self.assertEqual(self.update(7,detections=[]).state,"UNKNOWN")
        self.assertEqual(self.update(8).persistent_frames,1)

    def test_no_profile_records_depth_without_warnings(self):
        self.analyzer = ProximityAnalyzer(ProximityConfig())
        for frame in range(1,12):
            state = self.update(frame)
        self.assertEqual(state.state,"UNKNOWN")
        self.assertEqual(state.reason,"profile_missing")

    def test_profile_cannot_silently_use_different_depth_resolution(self):
        path = Path(self.directory.name)/"size_profile.json"
        path.write_text(json.dumps({"near_threshold":5,"caution_threshold":10,"hysteresis":1,
                                    "depth_image_size":768}))
        with self.assertRaisesRegex(ValueError,"size differs"):
            ProximityAnalyzer(ProximityConfig(profile=path,depth_image_size=640))

    def test_self_intersecting_corridor_is_rejected(self):
        config = ProximityConfig(corridor=((.2,.4),(.8,.7),(.8,.4),(.2,.7)))
        with self.assertRaisesRegex(ValueError,"convex"):
            config.validate()


class CrosswalkTests(unittest.TestCase):
    def test_scores_nms_and_crop_coordinates(self):
        # Confidence is objectness * class score, not either score alone.
        rows = np.array([[[50,50,40,20,.8,.8,.1,.1],
                          [51,50,40,20,.7,.8,.1,.1],
                          [80,80,20,10,.5,.4,.1,.1]]],np.float32)
        detections = decode_predictions(rows,0,.35,.5,10,20,(100,200,300,400))
        self.assertEqual(len(detections),1)
        self.assertAlmostEqual(detections[0].confidence,.64,places=5)
        self.assertEqual(detections[0].box,(140,240,220,280))

    def test_confirmed_crosswalk_is_counted_once_across_reused_frames(self):
        detector = CrosswalkDetector.__new__(CrosswalkDetector)
        detector.config = CrosswalkConfig(frame_interval=3)
        detector.fps,detector.roi = 30,(0,0,100,100)
        detector.tracks,detector.seen = {},set()
        detector.observed_tracks = {}
        detector.next_id,detector.step = 1,0
        from src.crosswalk import CrossingFrame
        detector.last = CrossingFrame()
        with patch.object(detector,"_predict",return_value=(Detection(None,"crosswalk",.8,(10,40,90,70)),)):
            for frame in range(1,13):
                output = detector.update(np.zeros((100,100,3),np.uint8),frame)
                if frame < 7:
                    self.assertFalse(output.detections)
            self.assertEqual(detector.seen,{1})
            self.assertEqual(output.detections[0].track_id,1)

    def test_optional_failure_does_not_break_general_pipeline(self):
        config = Config(Path(__file__),traffic=TrafficConfig(crosswalks=CrosswalkConfig(enabled=True)))
        with patch("src.scene.CrosswalkDetector") as detector:
            detector.return_value.update.side_effect = RuntimeError("inference failed")
            analyzer = SceneAnalyzer(config,"cpu",30,100,100)
            output = analyzer.update(np.zeros((100,100,3),np.uint8),[],1,0)
            self.assertFalse(output.crossings.detections)
            self.assertIn("crosswalk",analyzer.errors)
            analyzer.update(np.zeros((100,100,3),np.uint8),[],2,1/30)
            self.assertEqual(detector.return_value.update.call_count,1)
