"""Behavior tests for counting and time-bounded trajectories; no model downloads."""

import unittest

from src.analytics import Analytics
from src.config import Config, Detection, is_presentable
from pathlib import Path


def observation(track_id: int | None, category: str = "car", x: float = 20) -> Detection:
    return Detection(track_id, category, 0.9, (x, 10, x + 20, 50))


class AnalyticsTests(unittest.TestCase):
    def test_repeated_id_is_counted_once(self):
        analytics = Analytics()
        for frame in range(1, 101):
            analytics.update([observation(12)], frame)
        self.assertEqual(analytics.counts["car"], 1)
        self.assertEqual(analytics.tracks[12].observations, 100)
        self.assertEqual(analytics.tracks[12].longest_run, 100)

    def test_categories_and_new_ids_are_independent(self):
        analytics = Analytics()
        analytics.update([observation(12), observation(13), observation(4, "person"),
                          observation(7, "motorcycle")], 1)
        analytics.update([observation(12), observation(4, "person")], 2)
        self.assertEqual(analytics.counts["car"], 2)
        self.assertEqual(analytics.counts["person"], 1)
        self.assertEqual(analytics.counts["motorcycle"], 1)
        self.assertEqual(analytics.report()["unique_objects"], 4)

    def test_missing_id_does_not_count_or_build_trails(self):
        analytics = Analytics()
        analytics.update([observation(None)], 1)
        self.assertEqual(len(analytics.tracks), 0)
        self.assertFalse(analytics.histories)
        self.assertEqual(analytics.untracked_observations, 1)

    def test_duplicate_rows_within_frame_are_deduplicated(self):
        analytics = Analytics()
        analytics.update([observation(1), observation(1)], 1)
        self.assertEqual(analytics.tracks[1].observations, 1)
        self.assertEqual(analytics.tracked_observations, 1)

    def test_class_change_does_not_create_second_count(self):
        analytics = Analytics()
        analytics.update([observation(1)], 1)
        analytics.update([observation(1, "truck")], 2)
        self.assertEqual(analytics.counts["car"], 1)
        self.assertEqual(analytics.counts["truck"], 0)
        self.assertEqual(analytics.class_for(observation(1, "truck")), "car")
        self.assertEqual(analytics.tracks[1].class_changes, 1)

    def test_history_is_bounded_by_frames_not_just_samples(self):
        analytics = Analytics(trail_frames=3)
        for frame in range(1, 7):
            analytics.update([observation(1, x=frame)], frame)
        self.assertEqual([f for f, _ in analytics.histories[1]], [4, 5, 6])
        analytics.update([], 7)
        self.assertEqual([f for f, _ in analytics.histories[1]], [5, 6])
        analytics.update([], 9)
        self.assertNotIn(1, analytics.histories)
        self.assertIn(1, analytics.tracks)

    def test_sustained_class_correction_transfers_count_without_recount(self):
        analytics = Analytics()
        analytics.update([observation(1,"motorcycle")],1)
        for frame in range(2,15):
            analytics.update([observation(1,"person")],frame)
        self.assertEqual(analytics.class_for(observation(1)),"person")
        self.assertEqual(analytics.counts["person"],1)
        self.assertEqual(analytics.counts["motorcycle"],0)
        self.assertEqual(analytics.report()["unique_objects"],1)

    def test_alternating_class_noise_does_not_flicker_label(self):
        analytics = Analytics()
        for frame in range(1,30):
            analytics.update([observation(1,"car" if frame%2 else "truck")],frame)
            self.assertEqual(analytics.class_for(observation(1)),"car")

    def test_reappearing_id_does_not_recount_after_history_expires(self):
        analytics = Analytics(trail_frames=3)
        analytics.update([observation(1)], 1)
        analytics.update([], 5)
        analytics.update([observation(1)], 6)
        self.assertEqual(analytics.counts["car"], 1)
        self.assertEqual(analytics.tracks[1].gaps, 1)
        self.assertEqual(analytics.tracks[1].longest_run, 1)
        self.assertEqual(list(analytics.histories[1])[0][0], 6)

    def test_reassigned_id_is_explicitly_approximate(self):
        analytics = Analytics()
        analytics.update([observation(1)], 1)
        analytics.update([observation(2)], 2)
        self.assertEqual(analytics.counts["car"], 2)
        self.assertIn("overcount", analytics.report()["counting_limitations"])

    def test_nonurban_classes_are_excluded(self):
        analytics = Analytics()
        analytics.update([observation(1, "dog")], 1)
        self.assertFalse(analytics.tracks)

    def test_frame_indices_must_increase(self):
        analytics = Analytics()
        analytics.update([], 1)
        with self.assertRaises(ValueError):
            analytics.update([observation(1)], 1)

    def test_ground_point_uses_box_bottom_center(self):
        self.assertEqual(observation(1).ground_point, (30, 50))


class ConfigTests(unittest.TestCase):
    def test_weak_stop_sign_predictions_are_rejected(self):
        self.assertFalse(is_presentable(Detection(1, "stop sign", 0.61, (0, 0, 20, 20))))
        self.assertTrue(is_presentable(Detection(1, "stop sign", 0.65, (0, 0, 20, 20))))
        self.assertTrue(is_presentable(Detection(1, "car", 0.15, (0, 0, 20, 20))))

    def test_input_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "different"):
            Config(Path(__file__), Path(__file__)).validate()

    def test_confidence_is_validated(self):
        for confidence in (0, -0.1, 1.1, float("nan")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                Config(Path(__file__), conf=confidence).validate()


if __name__ == "__main__":
    unittest.main()
