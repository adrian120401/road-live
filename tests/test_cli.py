"""The public CLI must map flags to the pipeline configuration."""

import unittest
from pathlib import Path
from unittest.mock import patch

from src.main import parse_args


class CliTests(unittest.TestCase):
    def test_analysis_resolution_and_prefetch_flags(self):
        config = parse_args(["--input","video2.MOV"])
        self.assertEqual(config.processing_max_side,1920)
        self.assertEqual(config.prefetch_frames,2)
        native = parse_args(["--input","video2.MOV","--processing-max-side","0","--prefetch-frames","0"])
        self.assertEqual(native.processing_max_side,0)
        self.assertEqual(native.prefetch_frames,0)

    def test_road_flags_and_automatic_evidence(self):
        with patch("sys.argv", ["uv", "--input", "video1.mp4", "--road-damage"]):
            config = parse_args()
        self.assertTrue(config.road_damage.enabled)
        self.assertTrue(config.road_damage.save_evidence)
        self.assertFalse(config.road_damage.collect_candidates)
        self.assertEqual(config.output_path, Path("outputs/urban_vision_v2_output.mp4"))
        with patch("sys.argv", ["uv", "--input", "video1.mp4", "--road-damage", "--no-road-evidence",
                                "--road-conf", "0.6", "--road-interval", "3",
                                "--road-roi", "0", "0.4", "1", "0.7", "--collect-road-candidates"]):
            config = parse_args()
        self.assertFalse(config.road_damage.save_evidence)
        self.assertTrue(config.road_damage.collect_candidates)
        self.assertEqual(config.road_damage.roi, (0, 0.4, 1, 0.7))
        self.assertEqual(config.road_damage.frame_interval, 3)

    def test_v3_location_flags_and_pothole_floor(self):
        with patch("sys.argv", ["uv", "--input", "video2.MOV", "--road-damage"]):
            config = parse_args()
        self.assertEqual(config.road_damage.confidence, 0.50)
        self.assertTrue(config.location.use_video_gps)
        with patch("sys.argv", ["uv", "--input", "video2.MOV", "--no-use-video-gps",
                                "--force-mock-route", "--map-output", "custom/map.html", "--exiftool", "tool.exe"]):
            config = parse_args()
        self.assertFalse(config.location.use_video_gps)
        self.assertTrue(config.location.force_mock_route)
        self.assertEqual(config.location.map_output, Path("custom/map.html"))
        self.assertEqual(config.location.exiftool, Path("tool.exe"))

    def test_default_cli_maps_input_and_output_paths(self):
        with patch("sys.argv", ["urban-vision", "--input", "video1.mp4"]):
            config = parse_args()
        self.assertEqual(config.input_path, Path("video1.mp4"))
        self.assertEqual(config.output_path, Path("outputs/urban_vision_output.mp4"))
        self.assertEqual(config.tracker, "botsort.yaml")

    def test_cli_overrides_are_preserved(self):
        argv = ["urban-vision", "--input", "video1.mp4", "--output", "outputs/test.mp4",
                "--model", "custom.pt", "--tracker", "bytetrack.yaml", "--conf", "0.25",
                "--device", "cpu", "--show"]
        with patch("sys.argv", argv):
            config = parse_args()
        self.assertEqual(config.output_path, Path("outputs/test.mp4"))
        self.assertEqual(config.model, "custom.pt")
        self.assertEqual(config.tracker, "bytetrack.yaml")
        self.assertEqual(config.conf, 0.25)
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.show)

    def test_scene_flags_can_be_enabled_and_configured_independently(self):
        config = parse_args(["--input","video2.MOV","--crosswalks","--proximity",
                             "--depth-interval","4","--crosswalk-interval","6",
                             "--proximity-profile","profile.json","--debug-scene",
                             "--no-traffic-lights","--no-enhanced-popups"])
        self.assertTrue(config.traffic.crosswalks.enabled)
        self.assertEqual(config.traffic.crosswalks.frame_interval,6)
        self.assertFalse(config.traffic.traffic_lights)
        self.assertTrue(config.proximity.enabled)
        self.assertEqual(config.proximity.depth_interval,4)
        self.assertEqual(config.proximity.profile,Path("profile.json"))
        self.assertTrue(config.debug_scene)
        self.assertFalse(config.map.enhanced_popups)
        self.assertEqual(config.road_damage.confidence,.50)
        self.assertEqual(config.output_path,Path("outputs/urban_vision_v3_1_output.mp4"))
