"""Native desktop entrypoint; legacy headless HTTP adapter for integration tests."""

import argparse
from concurrent.futures import TimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from threading import Lock

from .config import RoadDamageConfig
from .live_location import WindowsLocation
from .live_session import LiveSession
from .live_sources import AutomaticLocation, PhoneLocation

ROOT = Path(__file__).resolve().parents[1]


class LiveApplication:
    def __init__(self, location, output_root, device='auto', road_config=None):
        self.location, self.output_root, self.device = location, Path(output_root).resolve(), device
        self.road_config = road_config
        self.lock = Lock()
        self.session = None

    def start(self, camera_name):
        with self.lock:
            if self.session and not self.session.done.is_set():
                raise ValueError('Ya hay un recorrido activo.')
            session = LiveSession(self.location, self.output_root, self.device, self.road_config)
            session.start(camera_name)
            self.session = session
            return session.snapshot()

    def get_session(self, session_id):
        if not self.session or self.session.id != session_id:
            raise ValueError('El recorrido no existe o ya se inició otro.')
        return self.session

    def close(self):
        if self.session and self.session.thread.is_alive():
            self.session.close()
        self.location.close()


def create_server(app, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if self.path != '/api/state' and '/frames' not in self.path:
                logging.getLogger('urban_vision.live').debug(format, *args)

        def respond(self, status, body=b'', content_type='application/json; charset=utf-8'):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer-when-downgrade')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/favicon.ico':
                self.respond(204)
                return
            if url.path == '/api/state':
                session = app.session
                query_id = parse_qs(url.query).get('session', [''])[0]
                self.respond(200, {'location': app.location.snapshot()[0],
                                   'session': session.snapshot(heartbeat=query_id == session.id) if session else None})
                return
            if url.path.startswith('/outputs/'):
                path = (app.output_root / unquote(url.path[len('/outputs/'):])).resolve()
                if not path.is_relative_to(app.output_root) or path.suffix.lower() not in {'.html', '.json', '.jpg'}:
                    self.respond(404, {'error': 'Archivo no disponible.'})
                    return
            elif url.path in {'/', '/live.js', '/live.css'}:
                name = 'index.html' if url.path == '/' else url.path[1:]
                path = ROOT / 'assets/live' / name
            else:
                self.respond(404, {'error': 'Página no disponible.'})
                return
            types = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
                     '.css': 'text/css; charset=utf-8', '.jpg': 'image/jpeg', '.json': 'application/json; charset=utf-8'}
            try:
                self.respond(200, path.read_bytes(), types[path.suffix.lower()])
            except OSError:
                self.respond(404, {'error': 'Archivo no disponible.'})

        def do_POST(self):
            # Restrict mutations to this local origin, including uploads and finish beacons.
            origin = self.headers.get('Origin')
            expected = f'http://127.0.0.1:{self.server.server_port}'
            if origin and origin != expected:
                self.respond(403, {'error': 'Origen no autorizado.'})
                return
            self.connection.settimeout(15)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 4 * 1024 * 1024:
                    self.respond(413, {'error': 'Imagen demasiado grande.'})
                    return
                data = self.rfile.read(length)
                path = urlsplit(self.path).path
                if path == '/api/start':
                    body = json.loads(data or b'{}')
                    self.respond(201, app.start(body.get('camera_name', 'Cámara')))
                    return
                parts = path.strip('/').split('/')
                if len(parts) != 3 or parts[0] != 'api' or parts[2] not in {'frames', 'finish', 'disconnect'}:
                    self.respond(404, {'error': 'Operación desconocida.'})
                    return
                session = app.get_session(parts[1])
                if parts[2] in {'finish', 'disconnect'}:
                    session.finish('user_finished' if parts[2] == 'finish' else 'camera_disconnected')
                    self.respond(202, session.snapshot())
                else:
                    if self.headers.get('Content-Type') != 'image/jpeg':
                        raise ValueError('Se requiere una imagen JPEG.')
                    epoch = float(self.headers.get('X-Capture-Time', 'nan')) / 1000
                    if not math.isfinite(epoch):
                        raise ValueError('Tiempo de captura inválido.')
                    image = session.submit(data, epoch).result(timeout=120)
                    self.respond(200 if image else 204, image or b'', 'image/jpeg')
            except TimeoutError:
                self.respond(504, {'error': 'El procesamiento está tardando demasiado.'})
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception as exc:
                logging.exception('Error de sesión en vivo')
                self.respond(500, {'error': str(exc)})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server, f'http://127.0.0.1:{server.server_port}/'


def main():
    parser = argparse.ArgumentParser(description='Urban Vision · recorrido en vivo en Windows')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/live')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--location', choices=('auto', 'windows', 'phone', 'mock'), default='auto')
    modes.add_argument('--no-gps', action='store_true', help='Iniciar directamente con ruta simulada')
    parser.add_argument('--phone-port', type=int, default=8766)
    parser.add_argument('--road-roi', nargs=4, type=float, default=RoadDamageConfig.roi)
    parser.add_argument('--no-road-area', action='store_true', help='Desactivar el filtro de veredas para otra cámara')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    location = AutomaticLocation(phone=PhoneLocation(port=args.phone_port),
                                 mode='mock' if args.no_gps else args.location)
    road = RoadDamageConfig(enabled=True, frame_interval=1, model=ROOT / 'models/pothole_yolov8s.pt',
                            roi=tuple(args.road_roi), road_area=None if args.no_road_area else RoadDamageConfig.road_area)
    app = LiveApplication(location, args.output_root, args.device, road)
    try:
        road.validate()
        # Explicit local model avoids an implicit download during a demonstration.
        if not (ROOT / 'yolo26n.pt').is_file():
            raise ValueError('Falta yolo26n.pt. Descargalo siguiendo la guía de Windows del README.')
        location.start()
        try:
            from .live_desktop import run_desktop
        except ImportError as exc:
            raise ValueError('Instalá la interfaz con pip install -r requirements-desktop.txt') from exc
        print('Urban Vision | C: cámara · Esc/F: finalizar y ver mapa', flush=True)
        return run_desktop(location, args.output_root.resolve(), args.device, road)
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', flush=True)
        return 1
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(main())
