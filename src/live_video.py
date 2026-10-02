"""Portrait framing and a constant-rate recorder independent of inference speed."""

import math
from threading import Condition, Thread
import time

import cv2


def portrait_frame(frame, size=(720, 1280)):
    """Center crop to 9:16 before resizing, never stretch the camera image."""
    height, width = frame.shape[:2]
    target_width, target_height = size
    if width * target_height > height * target_width:
        cropped_width = max(1, round(height * target_width / target_height))
        left = (width - cropped_width) // 2
        frame = frame[:, left:left + cropped_width]
    else:
        cropped_height = max(1, round(width * target_height / target_width))
        top = (height - cropped_height) // 2
        frame = frame[top:top + cropped_height]
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


class LiveVideo:
    """Keep only the latest annotated image; duplicate it to preserve real time.

    Segments use capture epochs, so exported evidence can locate its frame in the
    clip even after pauses. Only this worker owns/releases the OpenCV encoder.
    """

    def __init__(self, path, fps=30, size=(1080, 1920)):
        self.path, self.fps, self.size = path, fps, size
        self.condition = Condition()
        self.latest = None
        self.latest_epoch = None
        self.positions = {}
        self.segments = []
        self.active = False
        self.stopped = False
        self.ending_epoch = None
        self.frames = 0
        self.error = None
        self.thread = Thread(target=self._run, name='live-mp4', daemon=True)
        self.thread.start()

    def update(self, image, epoch):
        with self.condition:
            if self.stopped or self.error:
                raise RuntimeError(self.error or 'La grabación ya terminó.')
            self.latest = image.copy()
            self.latest_epoch = epoch
            if not self.active and self.ending_epoch is None:
                self.segments.append([epoch, None])
                self.active = True
            elif not self.segments and self.ending_epoch is not None:
                self.segments.append([epoch, max(epoch, self.ending_epoch)])
            self.condition.notify_all()

    def pause(self, epoch=None):
        with self.condition:
            if self.active:
                self.segments[-1][1] = max(self.segments[-1][0], epoch or time.time())
                self.active = False
                self.condition.notify_all()

    def timestamp(self, epoch):
        """Actual first encoded frame of this observation, after inference latency."""
        with self.condition:
            return self.positions.get(epoch)

    def _duration(self):
        return sum(max(0, (end if end is not None else time.time()) - start)
                   for start, end in self.segments)

    def finish(self, epoch=None):
        self.end_at(epoch or time.time())
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()
        if self.error:
            raise RuntimeError(self.error)

    def end_at(self, epoch):
        """Freeze duration immediately on the keypress, before inference/export ends."""
        with self.condition:
            if self.ending_epoch is None:
                self.ending_epoch = epoch
                self.pause(epoch)

    def report(self):
        return {'path': self.path.name if self.frames else None, 'fps': self.fps,
                'width': self.size[0], 'height': self.size[1], 'frames': self.frames,
                'duration_seconds': round(self.frames / self.fps, 3),
                'codec': 'mp4v', 'audio': False, 'paused_intervals_excluded': True}

    def _run(self):
        writer = None
        encoded_source = encoded = None
        try:
            while True:
                with self.condition:
                    target = max(1, math.ceil(self._duration() * self.fps)) if self.latest is not None else 0
                    if self.frames >= target:
                        if self.stopped:
                            break
                        self.condition.wait(1 / self.fps)
                        continue
                    latest = self.latest
                    latest_epoch = self.latest_epoch
                if writer is None:
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
                    if not writer.isOpened():
                        raise RuntimeError('No se pudo abrir recorrido.mp4 para grabar.')
                if latest is not encoded_source:
                    encoded = portrait_frame(latest, self.size)
                    encoded_source = latest
                with self.condition:
                    self.positions.setdefault(latest_epoch, self.frames / self.fps)
                writer.write(encoded)
                self.frames += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer is not None:
                writer.release()
