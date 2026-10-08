"""Replay cached observations onto source frames and append a reviewed route map."""

import json
import logging
import math
import os
from pathlib import Path
import subprocess
from threading import Thread

import cv2
from dataclasses import replace

from .analysis_cache import read_cache
from .analytics import Analytics
from .config import CLASSES, Detection, RoadFrame
from .crosswalk import CrossingFrame
from .location import RoutePoint
from .map_renderer import render_map
from .offline_media import atomic_json, ffmpeg_executable, verify_source
from .proximity import ProximityFrame
from .renderer import FrameMetrics, Renderer
from .review_project import ProjectStore
from .scene import SceneFrame

LOG = logging.getLogger(__name__)


class ManualRoute:
    source = "manual"

    def __init__(self, coordinates):
        # Vertex indexes are ordering only; no timeline or assumed speed is inferred.
        self.points = tuple(RoutePoint(i, p[0], p[1]) for i, p in enumerate(coordinates))

    def point_at(self, timestamp):
        raise ValueError("La ruta manual no tiene tiempos GPS ni interpolación temporal.")


def reviewed_events(project):
    result = []
    for event in project["events"]:
        if event["status"] == "accepted":
            result.append({**event["observation"], "latitude": event["position"][0],
                           "longitude": event["position"][1], "location_source": "manual"})
    return result


def write_reviewed_map(project, root, duration):
    events = reviewed_events(project)
    path = root / f"reviewed_r{project['revision']}_map.html"
    render_map(ManualRoute(project["route"]), events, root / project["analysis"]["events"], path,
               complete=True, processed_seconds=duration)
    # Keep route + markers clear of the summary in the portrait closing shot.
    addition = """<style>
    body.capture .summary{top:32px;left:32px;width:calc(100% - 64px);padding:28px}
    body.capture h1{font-size:32px}body.capture .description,body.capture .status{font-size:17px}
    body.capture .badge{font-size:17px}body.capture .count{font-size:44px}
    body.capture .count span,body.capture .metrics,body.capture .legend{font-size:17px}
    body.capture #events,body.capture .scan,body.capture .leaflet-control-zoom{display:none}
    body.capture .leaflet-control-attribution{font-size:14px}
    body.capture .route-end{font-size:15px;padding:5px 9px}
    </style><script>
    window.urbanVisionMap=map;
    if(new URLSearchParams(location.search).has('capture')){
      document.body.classList.add('capture');
      const panel=document.querySelector('.summary').getBoundingClientRect();
      const bounds=L.latLngBounds(data.route.concat(data.events.map(e=>[e.latitude,e.longitude])));
      map.fitBounds(bounds,{animate:false,paddingTopLeft:[65,panel.bottom+70],paddingBottomRight:[65,85],maxZoom:19});
      map.eachLayer(layer=>{if(layer instanceof L.CircleMarker && layer.options.radius===7){
        layer.setRadius(10);layer.bindTooltip(String(layer.getPopup().getContent().match(/EVENTO #(\\d+)/)?.[1]||''),{permanent:true,direction:'right'});
      }});
    }
    </script>"""
    document = path.read_text(encoding="utf-8").replace("</body>", addition + "</body>")
    path.write_text(document, encoding="utf-8")
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"location_source": "manual", "temporal": False},
         "geometry": {"type": "LineString", "coordinates": [[p[1], p[0]] for p in project["route"]]}},
        *[{"type": "Feature", "properties": {"event_id": e["event_id"], "location_source": "manual",
                                                 "timestamp": e["timestamp"]},
           "geometry": {"type": "Point", "coordinates": [e["longitude"], e["latitude"]]}} for e in events]]}
    atomic_json(root / f"reviewed_r{project['revision']}.geojson", geojson)
    atomic_json(root / f"reviewed_r{project['revision']}_events.json", {"events": events, "revision": project["revision"]})
    return path


def chrome_executable():
    configured = os.environ.get("URBAN_VISION_CHROME")
    if configured:
        return configured
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            path = Path(base) / "Google/Chrome/Application/chrome.exe"
            if path.is_file():
                return str(path)
    return None


def capture_map(path, root, target):
    from playwright.sync_api import sync_playwright
    from scripts.open_map import create_server
    server, url = create_server(path, root, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            executable = chrome_executable()
            options = {"headless": True}
            if executable:
                options["executable_path"] = executable
            browser = playwright.chromium.launch(**options)
            try:
                page = browser.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=2)
                page.goto(url + "?capture=1", wait_until="load", timeout=30000)
                try:
                    page.wait_for_function("""() => {
                      const images=[...document.querySelectorAll('.leaflet-tile')];
                      return !window.urbanVisionMap._animatingZoom && images.length>0
                        && images.every(i=>i.complete && i.naturalWidth>0 && Number(getComputedStyle(i).opacity)>.99);
                    }""", timeout=25000)
                except Exception as exc:
                    raise RuntimeError("No se pudo cargar el fondo de calles del mapa. Comprobá internet y reintentá la exportación.") from exc
                page.screenshot(path=str(target), animations="disabled")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def decode_detection(row, scale):
    return Detection(row["track_id"], row["class_name"], row["confidence"],
                     tuple(value * scale[i % 2] for i, value in enumerate(row["box"])))


def replay_render(image, row, metadata, analytics, renderer, accepted):
    h, w = image.shape[:2]
    scale = (w / metadata["resolution"][0], h / metadata["resolution"][1])
    detections = [decode_detection(d, scale) for d in row["detections"] if d["class_name"] in CLASSES]
    analytics.update(detections, row["frame"])
    road = RoadFrame(detections=tuple(
        replace(decode_detection(d, scale), track_id=accepted[d["track_id"]].get("event_id", d["track_id"]))
        for d in row["road"] if d["track_id"] in accepted))
    count = sum(row["frame"] >= event["confirmed_frame"] for event in accepted.values())
    scene_data = row["scene"]
    proximity = scene_data["proximity"].copy()
    if proximity.get("box"):
        proximity["box"] = tuple(v * scale[i % 2] for i, v in enumerate(proximity["box"]))
    scene = SceneFrame(crossings=CrossingFrame(tuple(decode_detection(d, scale) for d in scene_data["crossings"])),
                       proximity=ProximityFrame(**proximity), crosswalks_seen=scene_data["crosswalks_seen"])
    metrics = FrameMetrics(row["frame"], metadata["source_total_frames"], metadata["source_fps"], 0,
                           elapsed_seconds=row["timestamp"])
    return renderer.render(image, detections, analytics, metrics, road=road, potholes=count,
                           scene=scene, infrastructure=metadata["infrastructure"])


def repetitions(start, end, fps=60):
    """Sample source frame intervals on a CFR timeline without speeding the clip up."""
    return max(0, math.ceil(end * fps - 1e-7) - math.ceil(start * fps - 1e-7))


def export_review(project_path, *, project=None, progress=None, output_path=None, reuse_map=False, encoder="libx264"):
    store = ProjectStore(project_path)
    snapshot = store.export_snapshot() if project is None else project
    # Validate a supplied immutable snapshot just as strictly as a direct CLI export.
    store._validate(snapshot)
    from .review_project import review_progress
    if not review_progress(snapshot)["ready"]:
        raise ValueError("La revisión no está completa.")
    root = Path(project_path).resolve().parent
    progress = progress or (lambda value, message: LOG.info("%s", message))
    verify_source(snapshot["source"])
    from .offline_proximity import augmented_cache
    cache_path = augmented_cache(root, snapshot)
    has_added_proximity = cache_path != root / snapshot["analysis"]["cache"]
    default_name = f"recorrido_final_r{snapshot['revision']}{'_precaucion' if has_added_proximity else ''}.mp4"
    output = Path(output_path).resolve() if output_path else root / default_name
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("Elegí un archivo MP4 nuevo; no se reemplazan exportaciones existentes.")
    if output.resolve() in {Path(snapshot["source"]["path"]).resolve(), Path(snapshot["analysis"]["source_path"]).resolve()}:
        raise ValueError("La salida no puede reemplazar el video original.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preflight the complete cache before publishing any output.
    duration, count, last_delta = 0., 0, 1 / snapshot["analysis"]["fps"]
    for item in read_cache(cache_path):
        if item["type"] == "frame":
            if count:
                last_delta = item["timestamp"] - duration
            duration = item["timestamp"]
            count += 1
    duration += last_delta
    if count == 0:
        raise ValueError("El análisis no contiene fotogramas.")
    if count != snapshot["analysis"]["frames"]:
        raise ValueError("El número de fotogramas no coincide con el proyecto.")
    progress(.01, "Preparando el mapa final…")
    map_path = root / f"reviewed_r{snapshot['revision']}_map.html"
    closing = root / f"reviewed_r{snapshot['revision']}_map.png"
    if reuse_map:
        if not map_path.is_file() or not closing.is_file():
            raise ValueError("No se encontró el mapa exportado de esta revisión para reutilizarlo.")
    else:
        map_path = write_reviewed_map(snapshot, root, duration)
        capture_map(map_path, root, closing)
    source = Path(snapshot["analysis"]["source_path"])
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el video preparado para exportar.")
    aw, ah = snapshot["analysis"]["source_resolution"]
    # The reviewed output is always a portrait 4K canvas. Landscape inputs are rejected.
    if (aw, ah) != (2160, 3840):
        capture.release()
        raise ValueError("La exportación 4K de esta herramienta requiere un original vertical de 2160×3840.")
    filename = output.name
    temporary = output.with_name(output.stem + ".partial.mp4")
    if encoder not in {"libx264", "h264_nvenc"}:
        capture.release()
        raise ValueError("Encoder H.264 no admitido.")
    encoding = (["-c:v", "libx264", "-threads", "8", "-preset", "slow", "-crf", "16"]
                if encoder == "libx264" else
                ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", "14", "-b:v", "0", "-profile:v", "high"])
    command = [ffmpeg_executable(), "-hide_banner", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "2160x3840", "-r", "60", "-i", "pipe:0", "-an",
        *encoding, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(temporary)]
    accepted = {e["observation"]["track_id"]: e["observation"] for e in snapshot["events"] if e["status"] == "accepted"}
    rows = read_cache(cache_path)
    metadata = next(rows)
    renderer = Renderer(2160, 3840, metadata["display_confidence"], "ANÁLISIS DEL RECORRIDO")
    analytics = Analytics()
    frames_written = 0
    log_stem = f"export_r{snapshot['revision']}{'_precaucion' if has_added_proximity else ''}"
    log_path = root / (log_stem + ".log")
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            current = next(rows, None)
            while current is not None:
                following = next(rows, None)
                ok, image = capture.read()
                if not ok or image.shape[:2] != (3840, 2160):
                    raise ValueError("El video preparado no coincide con los fotogramas del análisis.")
                rendered = replay_render(image, current, metadata, analytics, renderer, accepted)
                end = following["timestamp"] if following else duration
                for _ in range(repetitions(current["timestamp"], end)):
                    process.stdin.write(rendered.tobytes())
                    frames_written += 1
                if current["frame"] % 150 == 0:
                    progress(.05 + .87 * current["frame"] / count, f"Exportando recorrido: {current['frame']} / {count} fotogramas…")
                current = following
            closing_image = cv2.imread(str(closing))
            if closing_image is None or closing_image.shape[:2] != (3840, 2160):
                raise ValueError("La imagen del mapa final no tiene resolución 4K vertical.")
            progress(.94, "Agregando los ocho segundos de mapa final…")
            for _ in range(480):
                process.stdin.write(closing_image.tobytes())
                frames_written += 1
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg no pudo finalizar el video. Revisá " + str(log_path))
        check = cv2.VideoCapture(str(temporary))
        try:
            if (not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != frames_written
                    or abs(check.get(cv2.CAP_PROP_FPS) - 60) > .01
                    or (int(check.get(cv2.CAP_PROP_FRAME_WIDTH)), int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (2160, 3840)):
                raise RuntimeError("El MP4 no pasó la verificación de resolución, FPS y duración.")
        finally:
            check.release()
        temporary.replace(output)
        result = {"video": filename, "map": map_path.name, "frames": frames_written,
                  "fps": 60, "duration_seconds": frames_written / 60, "source_duration_seconds": duration,
                  "closing_seconds": 8, "revision": snapshot["revision"], "accepted_potholes": len(accepted),
                  "audio": False, "resolution": [2160, 3840],
                  "proximity_enabled": has_added_proximity or metadata.get("scene_layers", {}).get("proximity", False),
                  "cache": cache_path.name, "reused_map": reuse_map, "output_path": str(output), "encoder": encoder}
        atomic_json(root / (log_stem + ".json"), result)
        progress(1, "Video y mapa listos.")
        return result
    except BrokenPipeError as exc:
        raise RuntimeError("FFmpeg interrumpió la exportación. Revisá " + str(log_path)) from exc
    finally:
        capture.release()
        if process:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
