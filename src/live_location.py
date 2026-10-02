"""Native Windows location and a segmented route with capture-aligned positions."""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread
import time

from .location import RoutePoint

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LocationFix:
    latitude: float
    longitude: float
    accuracy_m: float
    epoch: float


class WindowsLocation:
    def __init__(self, max_accuracy: float = 25, max_age: float = 5):
        self.max_accuracy, self.max_age = max_accuracy, max_age
        self.lock = Lock()
        self.fix: LocationFix | None = None
        self.reason = 'Esperando ubicación de Windows…'
        self.permission = 'Unknown'
        self.status = 'Initializing'
        self.process = None
        self.thread = None

    def start(self):
        if sys.platform != 'win32':
            self.reason = 'La ubicación nativa requiere ejecutar Python en Windows, fuera de WSL.'
            return
        try:
            self.process = subprocess.Popen(
                ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                 str(ROOT / 'scripts/windows_location.ps1')],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
            self.thread = Thread(target=self._read, name='windows-location', daemon=True)
            self.thread.start()
        except OSError as exc:
            self.reason = f'No se pudo consultar Windows: {exc}'

    def ingest(self, row):
        with self.lock:
            self.permission = row.get('permission', 'Unknown')
            self.status = row.get('status', 'Unknown')
            self.fix = None
            if row.get('permission') == 'Denied':
                self.reason = 'Windows denegó la ubicación. Revisá Configuración > Privacidad y seguridad > Ubicación.'
                return
            if row.get('status') != 'Ready':
                reasons = {
                    'Initializing': 'Windows está buscando una posición…',
                    'NoData': 'Windows no recibió una posición. Revisá el receptor o la señal de ubicación.',
                    'Disabled': 'El servicio de ubicación de Windows está desactivado.',
                }
                self.reason = row.get('error') or reasons.get(row.get('status'), 'Ubicación de Windows no disponible.')
                return
            try:
                lat, lon, accuracy = (float(row[k]) for k in ('latitude', 'longitude', 'accuracy_m'))
                epoch = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')).timestamp()
                if not all(math.isfinite(v) for v in (lat, lon, accuracy, epoch)):
                    raise ValueError('Valores no finitos')
                if not (-90 <= lat <= 90 and -180 <= lon <= 180 and accuracy > 0):
                    raise ValueError('Coordenadas inválidas')
                self.fix = LocationFix(lat, lon, accuracy, epoch)
                self.reason = ''
            except (KeyError, TypeError, ValueError, OverflowError):
                self.reason = 'Windows no entregó una posición con precisión válida.'

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self.ingest(json.loads(line))
                except (ValueError, TypeError):
                    self.ingest({'status': 'Error', 'error': 'Respuesta de ubicación inválida.'})
        finally:
            self.ingest({'status': 'Error', 'error': 'Se detuvo el servicio de ubicación.'})

    def snapshot(self, now: float | None = None):
        now = time.time() if now is None else now
        with self.lock:
            fix, reason = self.fix, self.reason
            permission, status = self.permission, self.status
        age = now - fix.epoch if fix else None
        valid = bool(fix and -1 <= age <= self.max_age and fix.accuracy_m <= self.max_accuracy)
        if fix and not valid:
            reason = ('Ubicación antigua; esperando una nueva posición.' if age > self.max_age or age < -1
                      else f'Precisión insuficiente: ±{fix.accuracy_m:.0f} m (máximo {self.max_accuracy:g} m).')
        return {'valid': valid, 'source': 'windows', 'reason': reason,
                'api': 'System.Device.Location.GeoCoordinateWatcher',
                'permission': permission, 'status': status,
                'position_source': 'Unknown',
                'accuracy_m': fix.accuracy_m if fix else None,
                'age_seconds': round(max(0, age), 1) if age is not None else None,
                'max_accuracy_m': self.max_accuracy, 'max_age_seconds': self.max_age}, fix

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=5)
        if self.thread:
            self.thread.join(timeout=5)
        if self.process and self.process.stdout:
            self.process.stdout.close()


class LiveRoute:
    source = 'windows'

    def __init__(self):
        self.samples: list[RoutePoint] = []
        self.times: list[float] = []
        self.accuracies: list[float] = []
        self.segment_starts: list[int] = []
        self.new_segment = True

    def append(self, timestamp: float, fix: LocationFix):
        if self.times and timestamp <= self.times[-1]:
            raise ValueError('Los tiempos de captura deben ser crecientes.')
        if self.new_segment:
            self.segment_starts.append(len(self.samples))
            self.new_segment = False
        self.samples.append(RoutePoint(timestamp, fix.latitude, fix.longitude))
        self.times.append(timestamp)
        self.accuracies.append(fix.accuracy_m)

    def pause(self):
        self.new_segment = True

    def point_at(self, timestamp: float):
        index = bisect_right(self.times, timestamp) - 1
        if index < 0 or abs(self.times[index] - timestamp) > .001:
            raise ValueError('El evento no tiene una posición registrada al capturar la imagen.')
        return self.samples[index]

    def accuracy_at(self, timestamp: float):
        self.point_at(timestamp)
        return self.accuracies[bisect_right(self.times, timestamp) - 1]

    @property
    def segments(self):
        ends = self.segment_starts[1:] + [len(self.samples)]
        result = []
        for start, end in zip(self.segment_starts, ends):
            samples = self.samples[start:end]
            if not samples:
                continue
            # Keep one point per second plus both endpoints; events retain exact positions.
            points = [samples[0]]
            for point in samples[1:-1]:
                if point.timestamp - points[-1].timestamp >= 1:
                    points.append(point)
            if samples[-1] != points[-1]:
                points.append(samples[-1])
            result.append(tuple(points))
        return tuple(result)

    @property
    def points(self):
        return tuple(point for segment in self.segments for point in segment)
