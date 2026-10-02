"""Bounded sequential inference and trip exports for live camera capture."""

from concurrent.futures import Future
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, RLock, Thread
import time
from uuid import uuid4

import cv2
import numpy as np

from .analytics import Analytics
from .config import Config, RoadDamageConfig
from .live_location import LiveRoute
from .live_video import LiveVideo, portrait_frame
from .map_renderer import render_map
from .renderer import FrameMetrics, Renderer
from .road_analytics import RoadAnalytics
from .road_damage import RoadDamageDetector
from .tracker import ObjectTracker
from .video_reader import working_size


class FrameError(ValueError):
    """Malformed camera uploads are recoverable; pipeline failures are not."""


class LiveSession:
    TERMINAL = {'finished', 'error'}

    def __init__(self, location, output_root: Path, device='auto', road_config=None, *, desktop=False):
        self.location = location
        self.desktop = desktop
        self.video = None
        self.finish_epoch = None
        self.id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + uuid4().hex[:8]
        self.directory = output_root / self.id
        self.config = Config(Path('live-camera'), self.directory / 'recorrido.mp4',
                             model=str(Path(__file__).resolve().parents[1] / 'yolo26n.pt'),
                             device=device, processing_max_side=1280,
                             road_damage=road_config or RoadDamageConfig(enabled=True, frame_interval=1))
        self.queue = Queue(maxsize=1)
        self.lock = RLock()
        self.stop = Event()
        self.done = Event()
        self.state = 'preparing'
        self.reason = 'Preparando los modelos con la imagen de la cámara…'
        self.stop_reason = 'user_finished'
        self.error = None
        self.export_errors = []
        self.route = LiveRoute()
        self.analytics = Analytics()
        self.tracker = self.road = self.road_analytics = self.renderer = None
        self.frames = 0
        self.started_epoch = time.time()
        self.started_monotonic = self.last_contact = time.monotonic()
        self.elapsed = 0.0
        self.last_timestamp = -1.0
        self.stage_seconds = 0.0
        self.id_offset = self.max_general_id = 0
        self.camera_name = ''
        self.map_url = None
        self.thread = Thread(target=self._run, name='live-inference', daemon=True)

    def start(self, camera_name=''):
        gps, _ = self.location.snapshot()
        if not gps['valid']:
            raise ValueError(gps['reason'])
        self.config.road_damage.validate()
        self.camera_name = str(camera_name)[:200]
        self.directory.mkdir(parents=True, exist_ok=False)
        self.thread.start()

    def snapshot(self, heartbeat=False):
        gps, _ = self.location.snapshot()
        with self.lock:
            if heartbeat and self.state not in self.TERMINAL:
                self.last_contact = time.monotonic()
            elapsed = self.elapsed if self.done.is_set() else time.monotonic() - self.started_monotonic
            return {'id': self.id, 'state': self.state, 'reason': self.reason,
                    'location': gps, 'elapsed_seconds': round(elapsed, 1),
                    'frames_processed': self.frames,
                    'potholes': self.road_analytics.count if self.road_analytics else 0,
                    'processing_fps': round(self.frames / self.stage_seconds, 1) if self.stage_seconds else 0,
                    'map_url': self.map_url, 'error': self.error,
                    'export_errors': list(self.export_errors),
                    'summary_url': f'/outputs/{self.id}/recorrido.json'
                    if self.done.is_set() and (self.directory / 'recorrido.json').is_file() else None}

    def submit(self, data: bytes | np.ndarray, epoch: float):
        future = Future()
        with self.lock:
            if self.stop.is_set() or self.state in self.TERMINAL:
                raise ValueError('El recorrido ya está finalizando.')
            self.last_contact = time.monotonic()
            try:
                self.queue.put_nowait((data, epoch, future))
            except Full:
                raise ValueError('Ya hay una imagen pendiente. Esperá su respuesta.')
        return future

    def finish(self, reason='user_finished'):
        with self.lock:
            if self.stop.is_set() or self.state in self.TERMINAL:
                return
            self.stop_reason = reason
            self.finish_epoch = time.time()
            if self.video:
                self.video.end_at(self.finish_epoch)
            self.state = 'finishing'
            self.reason = 'Guardando recorrido y evidencias…'
            self.stop.set()

    def _pause(self, reason):
        if self.state != 'paused':
            if self.video:
                self.video.pause()
            if self.road:
                for track_id in list(self.road.associator.tracks):
                    self.road_analytics._finalize(track_id)
                self.road.associator.tracks.clear()
                self.road.last = replace(self.road.last, detections=(), observed=(), confirmed_ids=frozenset())
            if self.tracker:
                predictor = getattr(self.tracker.model, 'predictor', None)
                for tracker in getattr(predictor, 'trackers', ()):
                    tracker.reset()
                self.id_offset = self.max_general_id
                self.analytics.histories.clear()
            self.route.pause()
        with self.lock:
            if not self.stop.is_set():
                self.state, self.reason = 'paused', reason

    def _check_location(self):
        gps, fix = self.location.snapshot()
        if not gps['valid']:
            self._pause(gps['reason'])
            return None
        with self.lock:
            if self.tracker and not self.stop.is_set():
                self.state, self.reason = 'running', 'Recorrido en curso'
        return fix

    def _process(self, data, epoch):
        now = time.time()
        if not np.isfinite(epoch) or not -1 <= now - epoch <= 2:
            return None  # Never process a queued old camera image.
        frame = data if isinstance(data, np.ndarray) else cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or max(frame.shape[:2]) > 4096:
            raise FrameError('Imagen de cámara inválida o demasiado grande.')
        fix = self._check_location()
        if fix is None:
            return None
        if self.desktop:
            frame = portrait_frame(frame)
        height, width = frame.shape[:2]
        size = working_size(width, height, self.config.processing_max_side)
        if size != (width, height):
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        width, height = size
        if self.tracker is None:
            self.tracker = ObjectTracker(self.config)
            self.tracker.warmup(frame)
            self.road = RoadDamageDetector(self.config.road_damage, self.tracker.device, 10, width, height)
            self.road.warmup(frame)
            self.road_analytics = RoadAnalytics(self.config.road_damage, self.config.output_path, 10, self.route)
            self.renderer = Renderer(width, height, self.config.display_confidence)
            self.started_epoch = time.time()
            self.started_monotonic = time.monotonic()
            self._check_location()
            return None  # Warmup is not an observation and doesn't register a route point.
        if (width, height) != (self.renderer.width, self.renderer.height):
            raise ValueError('La cámara cambió de resolución. Finalizá y comenzá otro recorrido.')
        timestamp = epoch - self.started_epoch
        if timestamp < 0 or timestamp <= self.last_timestamp:
            return None
        # Require a fix valid at the image's capture time as well as now.
        if not -1 <= epoch - fix.epoch <= self.location.max_age:
            return None
        started = time.perf_counter()
        detections = self.tracker.update(frame)
        detections = [replace(d, track_id=d.track_id + self.id_offset) if d.track_id is not None else d
                      for d in detections]
        self.max_general_id = max([self.max_general_id] + [d.track_id for d in detections if d.track_id is not None])
        number = self.frames + 1
        road_state = self.road.update(frame, number, timestamp=timestamp)
        # A slow model must not register an observation after GPS was lost.
        if self._check_location() is None:
            return None
        self.route.append(timestamp, fix)
        self.road_analytics.update(road_state, frame, number, timestamp)
        self.analytics.update(detections, number)
        elapsed = time.perf_counter() - started
        image = self.renderer.render(frame, detections, self.analytics,
                                     FrameMetrics(number, 0, 30 if self.desktop else 10,
                                                  1 / max(elapsed, .000001), timestamp),
                                     road=road_state, potholes=self.road_analytics.count)
        if self.desktop:
            if self.video is None:
                self.video = LiveVideo(self.config.output_path)
                if self.finish_epoch is not None:
                    self.video.end_at(self.finish_epoch)
            self.video.update(image, epoch)
            ok, encoded = True, image
        else:
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError('No se pudo generar la vista de detecciones.')
        with self.lock:
            self.frames = number
            self.last_timestamp = timestamp
            self.stage_seconds += time.perf_counter() - started
        return encoded if self.desktop else encoded.tobytes()

    def _run(self):
        try:
            while not self.stop.is_set():
                if not self.desktop and time.monotonic() - self.last_contact > 10:
                    self.finish('client_disconnected')
                    break
                self._check_location()
                try:
                    data, epoch, future = self.queue.get(timeout=.2)
                except Empty:
                    continue
                if self.stop.is_set():
                    future.set_result(None)
                    break
                try:
                    future.set_result(self._process(data, epoch))
                except FrameError as exc:
                    # Bad uploads don't terminate a healthy trip.
                    future.set_exception(exc)
                except Exception as exc:
                    future.set_exception(exc)
                    raise
        except Exception as exc:
            self.error = str(exc)
            self.finish('error')
        finally:
            with self.lock:
                self.state = 'finishing'
                self.elapsed = time.monotonic() - self.started_monotonic
            while True:
                try:
                    _, _, future = self.queue.get_nowait()
                    future.set_result(None)
                except Empty:
                    break
            try:
                self._export()
            except Exception as exc:
                self.export_errors.append(str(exc))
            with self.lock:
                self.state = 'error' if self.error or self.export_errors else 'finished'
                self.reason = self.error or ('No se pudieron guardar todas las salidas.' if self.export_errors
                                             else 'Recorrido finalizado' if self.stop_reason == 'user_finished'
                                             else 'Recorrido parcial guardado')
            self.done.set()

    def _export(self):
        if self.video:
            try:
                self.video.finish(self.finish_epoch)
            except Exception as exc:
                self.export_errors.append(f'Video: {exc}')
        complete = self.stop_reason == 'user_finished' and not self.error
        events_path = self.directory / 'recorrido_events.json'
        road_summary = {'potholes': 0, 'saved_evidence': 0}
        if self.road_analytics:
            try:
                road_summary = self.road_analytics.finish(complete, self.stop_reason)
            except Exception as exc:
                self.export_errors.append(f'Evidencias: {exc}')
        events = list(self.road_analytics.events.values()) if self.road_analytics else []
        for event in events:
            if self.video:
                epoch = self.started_epoch + event['timestamp']
                position = self.video.timestamp(epoch)
                event['video_timestamp'] = round(position, 3) if position is not None else None
                event['video_frame'] = int(position * self.video.fps) if position is not None else None
            point = self.route.point_at(event['timestamp'])
            event.update(latitude=point.latitude, longitude=point.longitude, location_source='windows',
                         location_accuracy_m=self.route.accuracy_at(event['timestamp']))
        road_summary.update(potholes=len(events), saved_evidence=sum(bool(e.get('evidence_path')) for e in events))
        # Preserve event metadata even when one evidence file cannot be saved.
        events_path.write_text(json.dumps({'complete': complete and not self.export_errors, 'stop_reason': self.stop_reason,
                                          'unique_potholes': len(events), 'count_is_estimate': True,
                                          'confidence_threshold': self.config.road_damage.confidence,
                                          'events': events}, indent=2, ensure_ascii=False), encoding='utf-8')
        location_report = {'source': 'windows', 'max_accuracy_m': self.location.max_accuracy,
                           'max_age_seconds': self.location.max_age,
                           'trajectory': [asdict(p) for p in self.route.points],
                           'segments': [[asdict(p) for p in segment] for segment in self.route.segments]}
        try:
            location_report.update(render_map(self.route, events, events_path, self.directory / 'map.html',
                                             complete=complete and not self.export_errors,
                                             processed_seconds=self.elapsed))
            self.map_url = f'/outputs/{self.id}/map.html'
        except Exception as exc:
            self.export_errors.append(f'Mapa: {exc}')
        report = {'source': 'live_camera', 'camera': self.camera_name,
                  'started_at': datetime.fromtimestamp(self.started_epoch, timezone.utc).isoformat(),
                  'complete': complete and not self.export_errors, 'stop_reason': self.stop_reason,
                  'frames_processed': self.frames, 'duration_seconds': round(self.elapsed, 3),
                  'processing_fps': round(self.frames / self.stage_seconds, 3) if self.stage_seconds else 0,
                  'error': self.error, 'export_errors': self.export_errors,
                  'video': self.video.report() if self.video else None,
                  'location': location_report, 'road_damage': road_summary, **self.analytics.report()}
        (self.directory / 'recorrido.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    def close(self):
        self.finish('server_stopped')
        self.thread.join()
