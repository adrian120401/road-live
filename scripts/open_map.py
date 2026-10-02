"""Open an exported map through a genuine loopback HTTP origin, as OSM requires."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


class MapRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # Send the actual local page URL on HTTP->HTTPS requests, including OSM tiles.
        self.send_header("Referrer-Policy", "no-referrer-when-downgrade")
        super().end_headers()


def create_server(map_path: Path, root: Path, port: int) -> tuple[ThreadingHTTPServer, str]:
    map_path, root = map_path.resolve(), root.resolve()
    if not map_path.is_file() or map_path.suffix.lower() != ".html":
        raise ValueError(f"Map HTML does not exist: {map_path}")
    try:
        relative = map_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Map is outside the viewer root; choose its parent directory with --root.") from exc
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(MapRequestHandler, directory=str(root)))
    url = f"http://127.0.0.1:{server.server_port}/" + quote(relative.as_posix(), safe="/")
    return server, url


def main() -> int:
    parser = argparse.ArgumentParser(description="Visor local de mapas de Urban Vision")
    parser.add_argument("--map", required=True, type=Path, help="Mapa HTML exportado")
    parser.add_argument("--root", type=Path, default=ROOT, help="Raíz de archivos estáticos; default: proyecto")
    parser.add_argument("--port", type=int, default=8765, help="Puerto local (0 elige uno libre)")
    parser.add_argument("--no-open", action="store_true", help="Mostrar URL sin abrir el navegador")
    args = parser.parse_args()
    try:
        server, url = create_server(args.map, args.root, args.port)
        with server:
            print(f"Urban Vision | Mapa local: {url}", flush=True)
            print("Dejá esta terminal abierta. Ctrl+C cierra el visor.", flush=True)
            if not args.no_open:
                if not webbrowser.open(url):
                    print("Abrí la URL anterior en tu navegador.", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print(f"Error del visor: {exc}. Si el puerto está ocupado, usá --port 0.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
