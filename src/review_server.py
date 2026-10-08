"""Loopback-only route editor, atomic review API and background export status."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from threading import RLock, Thread
from urllib.parse import unquote, urlsplit
import webbrowser

from .review_project import ProjectStore, review_progress, browser_project

ASSETS = Path(__file__).resolve().parents[1] / "assets"


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, project_path, port=8765):
        self.store = ProjectStore(project_path)
        self.root = self.store.path.parent
        self.export_lock = RLock()
        self.export_status = {"state": "idle", "message": "", "progress": 0}
        super().__init__(("127.0.0.1", port), ReviewHandler)

    def status(self):
        with self.export_lock:
            return dict(self.export_status)

    def start_export(self):
        with self.export_lock:
            if self.export_status["state"] == "running":
                raise RuntimeError("La exportación ya está en curso.")
            project = self.store.export_snapshot()
            self.export_status = {"state": "running", "message": "Preparando el mapa…", "progress": 0,
                                  "revision": project["revision"]}

        def worker():
            try:
                from .offline_export import export_review

                def update(progress, message):
                    with self.export_lock:
                        self.export_status.update(progress=progress, message=message)

                result = export_review(self.store.path, project=project, progress=update)
                with self.export_lock:
                    self.export_status.update(state="done", progress=1, message="Video y mapa listos.", **result)
            except Exception as exc:
                with self.export_lock:
                    self.export_status.update(state="error", message=str(exc))

        Thread(target=worker, name="review-export", daemon=False).start()
        return self.status()


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Leave exports/progress readable; avoid one access log line per poll.
        pass

    def trusted(self):
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin not in {"http://" + h for h in allowed}):
            self.send_json(403, {"error": "La revisión solo admite solicitudes locales del editor."})
            return False
        return True

    def send_json(self, status, data):
        content = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def json_body(self):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise ValueError("Se requiere Content-Type application/json.")
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= 2_000_000:
            raise ValueError("Tamaño del proyecto inválido.")
        return json.loads(self.rfile.read(size))

    def do_GET(self):
        if not self.trusted():
            return
        path = unquote(urlsplit(self.path).path)
        if path == "/api/project":
            project = self.server.store.snapshot()
            return self.send_json(200, {"project": browser_project(project), "progress": review_progress(project)})
        if path == "/api/export":
            return self.send_json(200, self.server.status())
        if path in {"/", "/index.html"}:
            return self.send_file(ASSETS / "review/index.html")
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        for prefix, root in (("/assets/review/", ASSETS / "review"),
                             ("/assets/leaflet/", ASSETS / "leaflet"), ("/files/", self.server.root)):
            if path.startswith(prefix):
                file = (root / path[len(prefix):]).resolve()
                if not file.is_relative_to(root.resolve()) or file.suffix.lower() not in {".html", ".css", ".js", ".jpg", ".png", ".mp4", ".json", ".geojson"}:
                    return self.send_json(404, {"error": "Archivo no disponible."})
                return self.send_file(file)
        self.send_json(404, {"error": "Recurso no disponible."})

    def do_PUT(self):
        if not self.trusted():
            return
        if urlsplit(self.path).path != "/api/project":
            return self.send_json(404, {"error": "Recurso no disponible."})
        try:
            project = self.server.store.save(self.json_body())
            self.send_json(200, {"project": browser_project(project), "progress": review_progress(project)})
        except RuntimeError as exc:
            self.send_json(409, {"error": str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            self.send_json(400, {"error": str(exc)})
        except OSError:
            self.send_json(500, {"error": "No se pudo guardar. Revisá el espacio y los permisos de la carpeta."})

    def do_POST(self):
        if not self.trusted():
            return
        if urlsplit(self.path).path != "/api/export":
            return self.send_json(404, {"error": "Recurso no disponible."})
        try:
            self.json_body()
            self.send_json(202, self.server.start_export())
        except RuntimeError as exc:
            self.send_json(409, {"error": str(exc)})
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})

    def send_file(self, path):
        if not path.is_file():
            return self.send_json(404, {"error": "Archivo no disponible."})
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if not match or int(match[1]) >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start = int(match[1])
            end = min(int(match[2]) if match[2] else size - 1, size - 1)
            if end < start:
                return self.send_json(416, {"error": "Rango inválido."})
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Referrer-Policy", "no-referrer-when-downgrade")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(remaining, 256 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def serve(project_path, port=8765, open_browser=True):
    with ReviewServer(project_path, port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Editor de recorrido: {url}", flush=True)
        print("Dejá esta terminal abierta. Ctrl+C cierra el editor.", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
