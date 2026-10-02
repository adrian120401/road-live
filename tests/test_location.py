"""GPS sample grouping, real/mock selection, interpolation and video clock behavior."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import LocationConfig
from src.location import (RoutePoint, VideoLocationMetadata, VideoMetadataLocationProvider,
                          MockRouteLocationProvider, parse_metadata, select_provider,
                          inspect_video_metadata, VideoClock, export_route)


class LocationTests(unittest.TestCase):
    def test_iphone_static_location_is_not_paired_with_unrelated_sample_times(self):
        metadata = parse_metadata([{"Keys:GPSCoordinates": "-33.5147 -56.8963 139.496",
                                    "Keys:CreationDate": "2026:10:01 17:05:51-03:00",
                                    "Doc1:Track3:SampleTime": 0, "Doc2:Track4:SampleTime": 10}])
        self.assertEqual(metadata.status, "single")
        self.assertEqual(metadata.points, [])
        self.assertEqual(metadata.single_location["latitude"], -33.5147)
        self.assertEqual(select_provider(metadata, 64, LocationConfig()).source, "mock")

    def test_embedded_samples_preserve_pairing_across_tag_families_and_copies(self):
        payload = {"Doc1:QuickTime:GPSLatitude": -33.5, "Doc1:Composite:Copy1:GPSLongitude": -56.9,
                   "Doc1:Track2:SampleTime": 0, "Doc2:QuickTime:GPSLatitude": -33.51,
                   "Doc2:QuickTime:GPSLongitude": -56.91, "Doc2:Track2:Copy2:SampleTime": 10}
        metadata = parse_metadata([payload])
        provider = select_provider(metadata, 10, LocationConfig())
        self.assertEqual(provider.source, "real")
        midpoint = provider.point_at(5)
        self.assertAlmostEqual(midpoint.latitude, -33.505)
        self.assertAlmostEqual(midpoint.longitude, -56.905)

    def test_absolute_gps_time_uses_creation_timezone(self):
        metadata = parse_metadata([{"Keys:CreationDate": "2026:10:01 17:00:00-03:00",
            "Doc1:GPS:GPSLatitude": -33.5, "Doc1:GPS:GPSLongitude": -56.9,
            "Doc1:GPS:GPSDateTime": "2026:10:01 20:00:05"}])
        self.assertEqual(metadata.points[0].timestamp, 5)

    def test_invalid_coordinates_and_timestamps_do_not_create_real_route(self):
        metadata = parse_metadata([{"Doc1:GPSLatitude": 99, "Doc1:GPSLongitude": -56, "Doc1:SampleTime": 0,
                                    "Doc2:GPSLatitude": float("nan"), "Doc2:GPSLongitude": -56, "Doc2:SampleTime": 10}])
        self.assertEqual(metadata.status, "absent")
        self.assertEqual(select_provider(metadata, 10, LocationConfig()).source, "mock")

    def test_conflicting_fixes_at_same_timestamp_reject_track(self):
        metadata = parse_metadata([{"Doc1:GPSLatitude": -33, "Doc1:GPSLongitude": -56, "Doc1:SampleTime": 0,
                                    "Doc2:GPSLatitude": -34, "Doc2:GPSLongitude": -56, "Doc2:SampleTime": 0}])
        self.assertEqual(metadata.status, "incomplete")

    def test_missing_endpoints_and_large_gaps_fall_back(self):
        for times in ((4, 10), (0, 6), (0, 20)):
            duration = 20 if times[-1] == 20 else 10
            metadata = VideoLocationMetadata(status="track", points=[RoutePoint(t, -33.5, -56.9) for t in times])
            self.assertEqual(select_provider(metadata, duration, LocationConfig()).source, "mock")
            self.assertEqual(metadata.status, "incomplete")

    def test_forced_mock_overrides_usable_real_route(self):
        metadata = VideoLocationMetadata(status="track", points=[RoutePoint(0, -33, -56), RoutePoint(10, -33.1, -56)])
        self.assertEqual(select_provider(metadata, 10, LocationConfig(force_mock_route=True)).source, "mock")
        self.assertEqual(select_provider(metadata, 10, LocationConfig(use_video_gps=False)).source, "mock")

    def test_mock_covers_fractional_duration_and_interpolates_deterministically(self):
        provider = MockRouteLocationProvider(12.3)
        self.assertEqual(provider.points[0].timestamp, 0)
        self.assertEqual(provider.points[-1].timestamp, 12.3)
        self.assertAlmostEqual(provider.point_at(6.15).latitude, -33.517)
        self.assertEqual(provider.point_at(6.15), MockRouteLocationProvider(12.3).point_at(6.15))
        self.assertAlmostEqual(provider.point_at(100).latitude, provider.points[-1].latitude)
        self.assertAlmostEqual(provider.point_at(-1).latitude, provider.points[0].latitude)

    def test_unavailable_and_failed_inspection_have_distinct_status_from_absent_gps(self):
        with patch("src.location.resolve_exiftool", return_value=None):
            self.assertEqual(inspect_video_metadata(Path("video.mov"), LocationConfig()).status, "unavailable")
        with patch("src.location.resolve_exiftool", return_value="tool"), patch("src.location.subprocess.run", side_effect=OSError("failed")):
            metadata = inspect_video_metadata(Path("video.mov"), LocationConfig())
        self.assertEqual(metadata.status, "unavailable")
        self.assertEqual(select_provider(metadata, 10, LocationConfig()).source, "mock")

    def test_clock_prefers_pts_and_never_moves_backwards_on_fallback(self):
        clock = VideoClock(30)
        values = [clock.timestamp(value, frame) for frame, value in enumerate((100, 134, 134, float("nan"), 250), 1)]
        self.assertEqual(values[0], 0)
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))
        self.assertEqual(clock.pts_frames, 3)
        self.assertEqual(clock.fallback_frames, 2)

    def test_geojson_uses_lon_lat_and_retains_timestamps_and_partial_status(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.geojson"
            provider = MockRouteLocationProvider(10)
            export_route(provider, path, complete=False, processed_seconds=3)
            data = json.loads(path.read_text())
        self.assertEqual(data["geometry"]["coordinates"][0], [provider.points[0].longitude, provider.points[0].latitude])
        self.assertFalse(data["properties"]["complete"])
        self.assertTrue(data["properties"]["simulated"])
