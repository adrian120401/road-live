"""Ordered PTS packets, bounded buffers, scaling, and decoder ownership."""

from dataclasses import replace
from pathlib import Path
import tempfile
from threading import Event
import unittest

import numpy as np

from src.config import Config
from src.video_reader import VideoReader, working_size


class Capture:
    def __init__(self, count=8, fail_at=None):
        self.count, self.fail_at = count, fail_at
        self.index = self.releases = 0
        self.fourth_read = Event()

    def read(self):
        if self.fail_at == self.index + 1:
            raise RuntimeError("decode failed")
        if self.index == self.count:
            return False, None
        self.index += 1
        if self.index >= 4:
            self.fourth_read.set()
        return True, np.full((64,128,3),self.index,np.uint8)

    def get(self, prop):
        return (self.index - 1) * 33.7

    def release(self):
        self.releases += 1


class ReaderTests(unittest.TestCase):
    def test_size_preserves_aspect_no_upscale_and_native_opt_out(self):
        self.assertEqual(working_size(2160,3840,1920),(1080,1920))
        self.assertEqual(working_size(3840,2160,1920),(1920,1080))
        self.assertEqual(working_size(464,832,1920),(464,832))
        self.assertEqual(working_size(2160,3840,0),(2160,3840))
        self.assertTrue(all(v%2 == 0 for v in working_size(2157,3837,1920)))

    def test_sync_and_prefetch_keep_all_frames_pixels_and_pts_in_order(self):
        for prefetch in (0,2):
            with self.subTest(prefetch=prefetch):
                capture = Capture()
                reader = VideoReader(capture,max_side=64,prefetch_frames=prefetch)
                try:
                    first = reader.read()
                    reader.start()
                    packets = [first]
                    while (packet := reader.read()) is not None:
                        packets.append(packet)
                    self.assertIsNone(reader.read())
                finally:
                    reader.close()
                self.assertEqual([p.number for p in packets],list(range(1,9)))
                self.assertEqual([p.source_pts_ms for p in packets],[(i-1)*33.7 for i in range(1,9)])
                self.assertEqual([int(p.image[0,0,0]) for p in packets],list(range(1,9)))
                self.assertTrue(all(p.image.shape == (32,64,3) for p in packets))
                self.assertEqual(reader.source_size,(128,64))
                self.assertEqual(capture.releases,1)

    def test_close_full_queue_does_not_deadlock_and_releases_arrays(self):
        capture = Capture(count=20)
        reader = VideoReader(capture,prefetch_frames=2)
        reader.read()
        reader.start()
        self.assertTrue(capture.fourth_read.wait(2))
        self.assertEqual(reader.queue.qsize(),2)
        reader.close()
        self.assertFalse(reader.thread.is_alive())
        self.assertTrue(reader.queue.empty())
        self.assertEqual(capture.releases,1)
        self.assertIsNone(reader.read())

    def test_decoder_error_reaches_consumer_after_earlier_valid_frames(self):
        for prefetch in (0,2):
            capture = Capture(fail_at=3)
            reader = VideoReader(capture,prefetch_frames=prefetch)
            try:
                self.assertEqual(reader.read().number,1)
                reader.start()
                self.assertEqual(reader.read().number,2)
                with self.assertRaisesRegex(RuntimeError,"decode failed"):
                    reader.read()
            finally:
                reader.close()
            self.assertEqual(capture.releases,1)

    def test_processing_parameters_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.mp4"
            path.touch()
            config = Config(path,Path(directory)/"output.mp4")
            config.validate()
            for invalid in (replace(config,processing_max_side=-1),replace(config,processing_max_side=127),
                            replace(config,prefetch_frames=-1),replace(config,prefetch_frames=9)):
                with self.assertRaises(ValueError):
                    invalid.validate()


if __name__ == "__main__":
    unittest.main()
