"""One ordered decoder, bounded prefetch, and one shared analysis resolution."""

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import cv2
import numpy as np


def working_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side == 0 or max(width, height) <= max_side:
        return width, height
    ratio = (max_side - max_side % 2) / max(width, height)
    # VideoWriter needs even dimensions; rounding costs at most one pixel per edge.
    return max(2, round(width * ratio / 2) * 2), max(2, round(height * ratio / 2) * 2)


@dataclass(frozen=True)
class VideoFrame:
    number: int
    image: np.ndarray
    source_pts_ms: float
    decode_seconds: float
    resize_seconds: float


@dataclass(frozen=True)
class ReadFailure:
    error: Exception


class VideoReader:
    """Owns capture. Models and profiling remain exclusively on the consumer thread."""

    def __init__(self, capture, max_side: int = 1920, prefetch_frames: int = 2) -> None:
        self.capture = capture
        self.max_side = max_side
        self.prefetch_frames = prefetch_frames
        self.queue: Queue = Queue(maxsize=max(1, prefetch_frames))
        self.stopped = Event()
        self.thread: Thread | None = None
        self.number = 0
        self.finished = False
        self.source_size: tuple[int, int] | None = None

    def _decode(self) -> VideoFrame | None:
        started = time.perf_counter()
        ok, image = self.capture.read()
        pts = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        decoded = time.perf_counter() - started
        if not ok:
            return None
        height, width = image.shape[:2]
        if self.source_size is None:
            self.source_size = (width, height)
        elif self.source_size != (width, height):
            raise ValueError("Video resolution changed during decoding.")
        started = time.perf_counter()
        size = working_size(width, height, self.max_side)
        if size != (width, height):
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        resized = time.perf_counter() - started
        self.number += 1
        return VideoFrame(self.number, image, pts, decoded, resized)

    def start(self) -> None:
        # The first frame is read synchronously for metadata and model warmup.
        if self.prefetch_frames and self.thread is None and not self.stopped.is_set() and not self.finished:
            self.thread = Thread(target=self._produce, name="urban-vision-decoder", daemon=True)
            self.thread.start()

    def _put(self, item) -> bool:
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=.05)
                return True
            except Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            while not self.stopped.is_set():
                packet = self._decode()
                if not self._put(packet) or packet is None:
                    break
        except Exception as error:
            self._put(ReadFailure(error))
        finally:
            self.capture.release()

    def read(self) -> VideoFrame | None:
        if self.stopped.is_set() or self.finished:
            return None
        if self.thread is None:
            packet = self._decode()
            self.finished = packet is None
            return packet
        while not self.stopped.is_set():
            try:
                item = self.queue.get(timeout=.05)
            except Empty:
                continue
            if isinstance(item, ReadFailure):
                self.finished = True
                raise item.error
            self.finished = item is None
            return item
        return None

    def close(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
        else:
            self.capture.release()
        # Release queued arrays promptly, including on interruption or model failure.
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
