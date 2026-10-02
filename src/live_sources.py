"""Windows -> paired iPhone -> explicit simulation, selected before each trip."""

from dataclasses import replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import socket
from threading import RLock, Thread
import time
from urllib.parse import urlsplit

from .live_location import LocationFix, WindowsLocation
from .location import MockRouteLocationProvider


class SimulatedLocation:
    max_accuracy, max_age = 25, 5

    def __init__(self, duration=120):
        self.route = MockRouteLocationProvider(duration)
        self.origin = time.time()

    def start(self):
        pass

    def begin_trip(self):
        self.origin = time.time()

    def end_trip(self):
        pass

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        point = self.route.point_at(max(0, now - self.origin))
        return {'valid': True, 'source': 'mock', 'simulated': True,
                'reason': 'Modo sin GPS · ruta simulada de Trinidad',
                'accuracy_m': None, 'age_seconds': 0,
                'max_accuracy_m': self.max_accuracy, 'max_age_seconds': self.max_age}, LocationFix(
                    point.latitude, point.longitude, None, now, 'mock')

    def close(self):
        pass


class PhoneLocation(WindowsLocation):
    """Accept measured accuracy and original fix time from OwnTracks HTTP."""

    def __init__(self, port=8766, host='0.0.0.0'):
        super().__init__()
        self.lock = RLock()
        self.port, self.host = port, host
        if not 0 <= port <= 65535:
            raise ValueError('El puerto del iPhone debe estar entre 0 y 65535.')
        self.token = secrets.token_urlsafe(24)
        self.server = self.server_thread = None
        self.receiver_error = None
        self.reason = 'Esperando ubicación del iPhone · P para configurar'

    @property
    def endpoint_path(self):
        return '/phone/' + self.token

    def urls(self):
        addresses = set()
        try:
            for row in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = row[4][0]
                if not ip.startswith('127.'):
                    addresses.add(ip)
        except OSError:
            pass
        return [f'http://{ip}:{self.port}{self.endpoint_path}' for ip in sorted(addresses)]

    def ingest_phone(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Se requiere un objeto OwnTracks.')
        if payload.get('_type') != 'location':
            return False  # OwnTracks also sends transitions/cards; acknowledge those.
        try:
            latitude, longitude, accuracy, epoch = (float(payload[k]) for k in ('lat', 'lon', 'acc', 'tst'))
            if not all(math.isfinite(v) for v in (latitude, longitude, accuracy, epoch)):
                raise ValueError()
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180 and accuracy > 0):
                raise ValueError()
            if epoch > time.time() + 1:
                raise ValueError()
            timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            raise ValueError('Ubicación inválida: faltan coordenadas, precisión real o tiempo original.')
        with self.lock:
            if self.fix and epoch <= self.fix.epoch:
                return False  # Delayed retries must not replace a newer position.
            self.ingest({'status': 'Ready', 'permission': 'Granted', 'latitude': latitude,
                         'longitude': longitude, 'accuracy_m': accuracy, 'timestamp': timestamp})
            self.fix = replace(self.fix, source='phone')
        return True

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Do not log the pairing token or coordinates.

            def do_POST(self):
                self.connection.settimeout(5)
                status, response = 200, b'[]'
                if urlsplit(self.path).path != owner.endpoint_path:
                    status, response = 403, b'[]'
                else:
                    try:
                        length = int(self.headers.get('Content-Length', '0'))
                        if not 0 <= length <= 16384:
                            raise ValueError('Tamaño inválido')
                        if length:
                            owner.ingest_phone(json.loads(self.rfile.read(length)))
                    except (ValueError, TypeError, OSError):
                        status, response = 400, b'[]'
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        try:
            self.server = ThreadingHTTPServer((self.host, self.port), Handler)
            self.server.daemon_threads = True
            self.port = self.server.server_port
            self.server_thread = Thread(target=self.server.serve_forever, name='iphone-location', daemon=True)
            self.server_thread.start()
        except OSError as exc:
            self.receiver_error = f'No se pudo abrir el receptor del iPhone: {exc}'

    def snapshot(self, now=None):
        state, fix = super().snapshot(now)
        state.update(source='phone', api='OwnTracks HTTP / iPhone', position_source='iPhone Location Services')
        if self.receiver_error:
            state.update(valid=False, reason=self.receiver_error)
        elif fix is None:
            state['reason'] = 'Esperando ubicación del iPhone · P para configurar'
        return state, fix

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join()
            self.server = None


class AutomaticLocation:
    def __init__(self, windows=None, phone=None, *, mode='auto', windows_wait=8, phone_wait=10):
        self.windows = windows or WindowsLocation()
        self.phone = phone or PhoneLocation()
        self.mock = SimulatedLocation()
        self.mode, self.windows_wait, self.phone_wait = mode, windows_wait, phone_wait
        self.max_age, self.max_accuracy = 5, 25
        self.bad_since = None
        self.active_provider = None
        self.lock = RLock()

    def start(self):
        if self.mode in {'auto', 'windows'}:
            self.windows.start()
        if self.mode in {'auto', 'phone'}:
            self.phone.start()

    def set_mode(self, mode):
        if self.active_provider is not None:
            raise ValueError('Finalizá el recorrido antes de cambiar de ubicación.')
        if mode not in {'auto', 'windows', 'phone', 'mock'}:
            raise ValueError('Modo de ubicación inválido.')
        self.mode = mode
        self.bad_since = None
        if mode in {'auto', 'windows'} and self.windows.process is None:
            self.windows.start()
        if mode in {'auto', 'phone'} and self.phone.server is None:
            self.phone.start()

    def _select(self):
        if self.active_provider:
            return self.active_provider, None
        if self.mode != 'auto':
            return {'windows': self.windows, 'phone': self.phone, 'mock': self.mock}[self.mode], None
        state, _ = self.windows.snapshot()
        if state['valid']:
            self.bad_since = None
            return self.windows, None
        if self.bad_since is None:
            self.bad_since = time.monotonic()
        elapsed = time.monotonic() - self.bad_since
        if elapsed < self.windows_wait:
            return self.windows, f'Probando Windows ({max(0, self.windows_wait - elapsed):.0f} s)\n{state["reason"]}'
        phone_state, _ = self.phone.snapshot()
        if phone_state['valid']:
            return self.phone, None
        if elapsed < self.windows_wait + self.phone_wait:
            return self.phone, f'Probando iPhone ({max(0, self.windows_wait + self.phone_wait - elapsed):.0f} s)\n{phone_state["reason"]}'
        return self.mock, None

    def snapshot(self, now=None):
        with self.lock:
            provider, reason = self._select()
            state, fix = provider.snapshot(now)
            if reason:
                state['reason'] = reason
            return state, fix

    def begin_trip(self):
        with self.lock:
            provider, _ = self._select()
            state, _ = provider.snapshot()
            if not state['valid']:
                raise ValueError(state['reason'])
            self.active_provider = provider
            if provider is self.mock:
                self.mock.begin_trip()

    def end_trip(self):
        with self.lock:
            self.active_provider = None

    def close(self):
        self.windows.close()
        self.phone.close()
