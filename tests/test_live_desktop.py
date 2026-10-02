"""Real codec/timing checks and native frame conversion without camera hardware."""

import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.live_video import LiveVideo, portrait_frame
from src.live_session import LiveSession
from test_live import FixedTracker, FixedRoad, update_location
from src.live_location import WindowsLocation
from src.config import RoadDamageConfig


class PortraitTests(unittest.TestCase):
    def test_landscape_center_crop_preserves_scale(self):
        image = np.zeros((160, 320, 3), np.uint8)
        image[:, :, 0] = np.arange(320, dtype=np.uint16) % 256
        result = portrait_frame(image, (90, 160))
        self.assertEqual(result.shape, (160, 90, 3))
        np.testing.assert_array_equal(result[:, :, 0], image[:, 115:205, 0])

    def test_portrait_center_crop_and_native_portrait(self):
        image = np.arange(320, dtype=np.uint16)[:, None] * np.ones((1, 90), np.uint16)
        image = np.repeat((image % 256).astype(np.uint8)[:, :, None], 3, axis=2)
        np.testing.assert_array_equal(portrait_frame(image, (90, 160)), image[80:240])
        np.testing.assert_array_equal(portrait_frame(image[80:240], (90, 160)), image[80:240])


class RecordingTests(unittest.TestCase):
    def test_cfr_duration_repeats_slow_frames_excludes_pause_and_decodes(self):
        with tempfile.TemporaryDirectory() as directory:
            video = LiveVideo(Path(directory) / 'clip.mp4', size=(90, 160))
            red = np.full((160, 90, 3), (0, 0, 240), np.uint8)
            blue = np.full((160, 90, 3), (240, 0, 0), np.uint8)
            first = time.time()
            video.update(red, first)
            time.sleep(.18)
            paused = time.time()
            video.pause(paused)
            time.sleep(.20)
            second = time.time()
            video.update(blue, second)
            time.sleep(.18)
            end = time.time()
            video.finish(end)
            video.finish(end)  # Idempotent close.
            cap = cv2.VideoCapture(str(video.path))
            self.addCleanup(cap.release)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 30)
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 90)
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 160)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 30,
                                   paused - first + end - second, delta=2 / 30)
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            self.assertEqual(len(frames), video.frames)
            self.assertGreater(len(frames), 2)
            self.assertGreater(frames[0][:, :, 2].mean(), 220)
            self.assertGreater(frames[-1][:, :, 0].mean(), 220)
            index = round(video.timestamp(second) * 30)
            self.assertGreater(frames[index][:, :, 0].mean(), 220)
            self.assertIsNone(video.timestamp(second - .1))
            cap.release()

    def test_encoder_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('src.live_video.cv2.VideoWriter') as encoder:
                encoder.return_value.isOpened.return_value = False
                video = LiveVideo(Path(directory) / 'clip.mp4', size=(90, 160))
                video.update(np.zeros((160, 90, 3), np.uint8), time.time())
                with self.assertRaisesRegex(RuntimeError, 'No se pudo abrir'):
                    video.finish()

    def test_finish_keypress_freezes_duration_during_slow_final_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            video = LiveVideo(Path(directory) / 'clip.mp4', size=(90, 160))
            image = np.zeros((160, 90, 3), np.uint8)
            first = time.time()
            video.update(image, first)
            time.sleep(.12)
            finish = time.time()
            video.end_at(finish)
            time.sleep(.18)  # A model may still be finishing this old capture.
            video.update(image, first + .05)
            video.finish(finish)
            self.assertAlmostEqual(video.frames / 30, finish - first, delta=2 / 30)

    def test_desktop_session_exports_video_and_evidence_playback_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'weights.pt'
            weights.touch()
            location = WindowsLocation()
            update_location(location)
            with patch('src.live_session.ObjectTracker', FixedTracker), patch('src.live_session.RoadDamageDetector', FixedRoad):
                session = LiveSession(location, root, road_config=RoadDamageConfig(enabled=True, model=weights, frame_interval=1,
                                                                                 roi=(0, 0, 1, 1), road_area=None), desktop=True)
                session.start()
                try:
                    image = np.full((832, 464, 3), 100, np.uint8)
                    for _ in range(4):
                        result = session.submit(image, time.time()).result(10)
                        time.sleep(.08)
                    self.assertEqual(result.shape, (1280, 720, 3))
                    session.finish()
                    self.assertTrue(session.done.wait(10))
                    report = json.loads((session.directory / 'recorrido.json').read_text())
                    self.assertTrue(report['complete'], report['export_errors'])
                    self.assertEqual(report['video']['width'], 1080)
                    self.assertEqual(report['video']['height'], 1920)
                    events = json.loads((session.directory / 'recorrido_events.json').read_text())['events']
                    self.assertIsNotNone(events[0]['video_timestamp'])
                    cap = cv2.VideoCapture(str(session.config.output_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, events[0]['video_frame'])
                    ok, frame = cap.read()
                    cap.release()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape, (1920, 1080, 3))
                finally:
                    session.close()


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Install requirements-desktop.txt for native UI checks')
class NativeFrameTests(unittest.TestCase):
    def test_rotation_mirroring_and_padded_rows(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage
        from PySide6.QtMultimedia import QVideoFrame, QtVideo
        from src.live_desktop import camera_image
        image = QImage(5, 3, QImage.Format.Format_BGR888)
        image.fill(Qt.GlobalColor.red)
        image.setPixelColor(0, 0, Qt.GlobalColor.blue)
        frame = QVideoFrame(image)
        frame.setRotation(QtVideo.Rotation.Clockwise90)
        frame.setMirrored(True)
        result = camera_image(frame)
        self.assertEqual(result.shape, (5, 3, 3))
        np.testing.assert_array_equal(result[0, 0], (255, 0, 0))


if __name__ == '__main__':
    unittest.main()
