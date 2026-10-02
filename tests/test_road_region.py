"""Spatial gating before tracking and evidence, including user-reviewed sidewalk cases."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.cli import parse_args
from src.config import Detection, RoadDamageConfig, RoadFrame
from src.road_analytics import RoadAnalytics
from src.road_damage import RoadDamageDetector
from src.road_region import RoadRegion


class RoadRegionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        model = self.root / "model.pt"
        model.touch()
        self.config = RoadDamageConfig(enabled=True, model=model, roi=(0, 0, 1, 1))

    def test_reviewed_sidewalk_boxes_excluded_and_road_potholes_preserved(self):
        region = RoadRegion(self.config, 2160, 3840)
        for box in ((108,2194,407,2320), (153,2155,356,2271),
                    (224,2081,475,2158), (567,2081,713,2122)):
            self.assertFalse(region.accepts(Detection(None, "pothole", .8, box)), box)
        for box in ((1466,2311,1818,2406), (1622,2482,2049,2668), (543,2446,708,2635)):
            self.assertTrue(region.accepts(Detection(None, "pothole", .8, box)), box)

    def test_center_and_overlap_both_required_and_resolution_independent(self):
        config = replace(self.config, road_area=((.2,.2),(.8,.2),(.8,.8),(.2,.8)))
        for size in (100, 1000):
            region = RoadRegion(config, size, size)
            for box, expected in (((.3,.3,.7,.7), True), ((.05,.3,.25,.5), False),
                                  ((0,0,1,1), False), ((.1,.3,.5,.6), True)):
                self.assertEqual(region.accepts(Detection(None, "pothole", .8,
                                   tuple(v*size for v in box))), expected)

    def test_invalid_polygon_overlap_and_geometry_rejected(self):
        for config in (replace(self.config, road_area=((0,0),(1,1),(1,0),(0,1))),
                       replace(self.config, road_area=((0,0),(1,0),(1,0),(0,1))),
                       replace(self.config, road_area=((0,0),(1,0),(1,1),(float("nan"),1))),
                       replace(self.config, road_area_min_overlap=0)):
            with self.assertRaises(ValueError):
                config.validate()
        region = RoadRegion(replace(self.config, road_area=None), 100, 100)
        self.assertTrue(region.accepts(Detection(None,"pothole",.8,(0,0,10,10))))
        for box in ((0,0,0,10), (0,0,10,float("nan"))):
            self.assertFalse(region.accepts(Detection(None,"pothole",.8,box)))

    def test_sidewalk_does_not_allocate_ids_count_or_save_evidence(self):
        model = MagicMock(task="detect", names={0:"pothole"})
        result = SimpleNamespace(boxes=SimpleNamespace(
            xyxy=torch.tensor([[5.,57.,15.,60.], [50.,57.,60.,60.]]),
            conf=torch.tensor([.9,.8])), speed={"inference":1})
        model.predict.side_effect = lambda *a, **k: iter([result])
        with patch("src.road_damage.YOLO", return_value=model):
            detector = RoadDamageDetector(self.config, "cpu", 30, 100, 100)
        analytics = RoadAnalytics(self.config, self.root / "output.mp4", 30)
        frame = np.zeros((100,100,3),np.uint8)
        for number in (1,2,3,4):
            state = detector.update(frame,number)
            analytics.update(state,frame,number)
        summary = analytics.finish(True,"end_of_video")
        self.assertEqual(detector.discarded_outside_road, 2)
        self.assertEqual(detector.associator.next_id, 2)
        self.assertEqual([d.track_id for d in state.detections], [1])
        self.assertEqual(summary["potholes"],1)
        self.assertEqual(summary["saved_evidence"],1)

    def test_analytics_defends_against_outside_adapter_observations(self):
        frame = np.zeros((100,100,3),np.uint8)
        detection = Detection(1,"pothole",.99,(5,57,15,60))
        state = RoadFrame((detection,),(detection,),frozenset({1}),inferred=True)
        analytics = RoadAnalytics(self.config,self.root / "outside.mp4",30)
        analytics.update(state,frame,1)
        summary = analytics.finish(True,"end_of_video")
        self.assertEqual(summary["potholes"],0)
        self.assertFalse(analytics.evidence_dir.exists())

    def test_cli_default_custom_polygon_and_opt_out(self):
        self.assertEqual(parse_args(["--input","video.mov"]).road_damage.road_area,self.config.road_area)
        self.assertIsNone(parse_args(["--input","video.mov","--no-road-area"]).road_damage.road_area)
        custom = parse_args(["--input","video.mov","--road-area","0","0","1","0","1","1","0","1"])
        self.assertEqual(custom.road_damage.road_area,((0,0),(1,0),(1,1),(0,1)))


if __name__ == "__main__":
    unittest.main()
