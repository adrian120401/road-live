"""Unicode masks and clipped compositing must preserve native BGR frames."""

import unittest
from unittest.mock import patch

import numpy as np

from src.renderer import Renderer, FrameMetrics
from src.analytics import Analytics
from src.config import Detection
from src.proximity import ProximityFrame
from src.scene import SceneFrame


class UnicodeTextTests(unittest.TestCase):
    def test_banner_only_for_confirmed_proximity_and_processing_fps_debug_only(self):
        frame = np.zeros((1920,1080,3),np.uint8)
        renderer = Renderer(1080,1920)
        metrics = FrameMetrics(100,1915,30,4.84)
        for state in ("UNKNOWN","SAFE","CAUTION","NEAR"):
            scene = SceneFrame(proximity=ProximityFrame(12,state,box=(450,1000,650,1300)))
            with patch.object(renderer,"text",wraps=renderer.text) as text:
                renderer.render(frame,[],Analytics(),metrics,scene=scene)
            strings = [call.args[1] for call in text.call_args_list]
            self.assertEqual("VEHÍCULO DELANTE" in strings,state in {"CAUTION","NEAR"})
            self.assertFalse(any("FPS" in value for value in strings))
            if state == "CAUTION":
                self.assertIn("PRECAUCIÓN",strings)
            if state == "NEAR":
                self.assertIn("MUY CERCA",strings)
        with patch.object(renderer,"text",wraps=renderer.text) as text:
            renderer.render(frame,[],Analytics(),metrics,scene=SceneFrame(),debug_scene=True)
        self.assertTrue(any("VIDEO 30.0 FPS / PROCESAMIENTO 4.8 FPS" == c.args[1]
                            for c in text.call_args_list))

    def test_accents_are_rendered_as_different_glyphs(self):
        renderer = Renderer(464, 832)
        accented = np.zeros((832, 464, 3), np.uint8)
        plain = accented.copy()
        renderer.text(accented, "CAMIÓN / SEMÁFORO", (20, 50), 0.6)
        renderer.text(plain, "CAMION / SEMAFORO", (20, 50), 0.6)
        self.assertGreater(np.count_nonzero(accented), 0)
        self.assertFalse(np.array_equal(accented, plain))

    def test_partially_clipped_text_and_native_resolution_are_preserved(self):
        renderer = Renderer(2160, 3840)
        frame = np.zeros((3840, 2160, 3), np.uint8)
        renderer.text(frame, "ANÁLISIS", (-10, 20), 0.6, (0, 0, 255))
        self.assertEqual(frame.shape, (3840, 2160, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(frame[:, :, 2]), 0)
        self.assertEqual(np.count_nonzero(frame[:, :, :2]), 0)
        before = frame.copy()
        renderer.text(frame, "POZO", (3000, 5000))
        self.assertTrue(np.array_equal(before, frame))

    def test_visual_gate_preserves_tracking_counts_and_small_traffic_lights(self):
        renderer = Renderer(464,832,display_confidence=.30)
        detections = [Detection(1,"car",.22,(100,400,200,500)),
                      Detection(2,"car",.90,(210,400,300,500)),
                      Detection(3,"traffic light",.20,(200,300,220,330))]
        analytics = Analytics()
        analytics.update(detections,1)
        with patch.object(renderer,"_box") as boxes:
            output = renderer.render(np.zeros((832,464,3),np.uint8),detections,analytics,FrameMetrics(1,10,30,5))
        self.assertEqual({call.args[1].track_id for call in boxes.call_args_list},{2,3})
        self.assertEqual(analytics.counts["car"],2)
        self.assertEqual(output.shape,(832,464,3))


if __name__ == "__main__":
    unittest.main()
