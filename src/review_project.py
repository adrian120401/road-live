"""Versioned manual route/review, independent from immutable model observations."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from threading import RLock

from .offline_media import atomic_json

JS_MAX_SAFE_INTEGER = 2**53 - 1


def browser_project(project):
    """Keep nanosecond timestamps exact across JSON.parse/JSON.stringify."""
    result = deepcopy(project)
    source = result.get("source", {})
    if "mtime_ns" in source:
        source["mtime_ns"] = str(source["mtime_ns"])
    return result


def same_source(incoming, original):
    if not isinstance(incoming, dict):
        return False
    normalized = deepcopy(incoming)
    expected = original.get("mtime_ns")
    received = incoming.get("mtime_ns")
    if expected is not None:
        if received == str(expected):
            normalized["mtime_ns"] = expected
        elif (type(expected) is int and abs(expected) > JS_MAX_SAFE_INTEGER
              and type(received) in (int, float) and math.isfinite(received)
              and float(received) == float(expected)):
            # Existing tabs/drafts already rounded this field. Accept only that
            # exact IEEE-754 representation; preserve the server's original value.
            normalized["mtime_ns"] = expected
    return normalized == original


def valid_point(point):
    return (isinstance(point, list) and len(point) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in point)
            and -90 <= point[0] <= 90 and -180 <= point[1] <= 180)


def review_progress(project):
    accepted = sum(e["status"] == "accepted" and valid_point(e.get("position")) for e in project["events"])
    rejected = sum(e["status"] == "rejected" for e in project["events"])
    pending = len(project["events"]) - accepted - rejected
    route = project["route"]
    route_ready = len(route) >= 2 and len({tuple(p) for p in route}) >= 2
    ready = bool(project["analysis"]["complete"] and route_ready and not pending)
    return {"accepted": accepted, "rejected": rejected, "pending": pending,
            "total": len(project["events"]), "route_ready": route_ready, "ready": ready}


def create_project(root, source, analysis_source, report, media):
    if (root / "project.json").exists():
        raise ValueError("Ya existe una revisión en esta carpeta. Elegí otra carpeta para un nuevo análisis.")
    events_path = Path(report["road_damage"]["events_path"])
    events = json.loads(events_path.read_text(encoding="utf-8"))["events"]
    metadata = report["location"]
    center = metadata.get("single_location") or {"latitude": -33.5147, "longitude": -56.8984}
    # Location metadata report uses a separate field name in earlier versions.
    if not isinstance(center, dict):
        center = {"latitude": -33.5147, "longitude": -56.8984}
    project = {"version": 1, "revision": 0, "name": "Recorrido · Trinidad",
               "updated_at": datetime.now(timezone.utc).isoformat(), "source": source,
               "analysis": {"complete": report["complete"], "source_path": str(analysis_source.resolve()),
                            "cache": "observations.jsonl", "report": "analysis.json",
                            "events": "analysis_events.json", "preview": "preview.mp4",
                            "resolution": report["resolution"], "source_resolution": report["source_resolution"],
                            "fps": report["source_fps"], "frames": report["frames_processed"],
                            "hdr_converted": media["hdr"], "display_confidence": report["display_confidence_threshold"],
                            "infrastructure": bool(report.get("scene", {}).get("crosswalk"))},
               "center": [center.get("latitude", -33.5147), center.get("longitude", -56.8984)],
               "route": [], "events": [{"event_id": e["event_id"], "observation": e,
                                         "status": "pending", "position": None} for e in events]}
    atomic_json(root / "project.json", project)
    return project


class ProjectStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock = RLock()
        self.project = json.loads(self.path.read_text(encoding="utf-8"))
        if self.project.get("version") != 1:
            raise ValueError("Versión de proyecto incompatible.")
        self._validate(self.project)

    def snapshot(self):
        with self.lock:
            return deepcopy(self.project)

    def _validate(self, incoming):
        if not isinstance(incoming, dict):
            raise ValueError("Se requiere un proyecto JSON.")
        fixed = ("version", "name", "analysis", "center")
        if (any(incoming.get(key) != self.project.get(key) for key in fixed)
                or not same_source(incoming.get("source"), self.project["source"])):
            raise ValueError("Los datos originales del análisis no se pueden modificar.")
        route = incoming.get("route")
        if not isinstance(route, list) or len(route) > 10000 or not all(valid_point(p) for p in route):
            raise ValueError("El recorrido debe contener coordenadas válidas en orden.")
        events = incoming.get("events")
        originals = {e["event_id"]: e["observation"] for e in self.project["events"]}
        if not isinstance(events, list) or len(events) != len(originals):
            raise ValueError("La revisión debe conservar todas las detecciones originales.")
        seen = set()
        for event in events:
            if not isinstance(event, dict) or type(event.get("event_id")) is not int:
                raise ValueError("ID de detección inválido.")
            identifier = event["event_id"]
            if identifier in seen or identifier not in originals or event.get("observation") != originals[identifier]:
                raise ValueError("Las detecciones originales no se pueden alterar ni duplicar.")
            seen.add(identifier)
            if event.get("status") not in {"pending", "accepted", "rejected"}:
                raise ValueError("Estado de revisión inválido.")
            position = event.get("position")
            if position is not None and not valid_point(position):
                raise ValueError("Posición del pozo inválida.")
            if event["status"] == "accepted" and position is None:
                raise ValueError("Ubicá el pozo en el mapa antes de confirmarlo.")
            if event["status"] != "accepted" and position is not None:
                raise ValueError("Solo los pozos confirmados pueden tener una posición.")

    def save(self, incoming):
        with self.lock:
            self._validate(incoming)
            if type(incoming.get("revision")) is not int or incoming["revision"] != self.project["revision"]:
                raise RuntimeError("La revisión cambió en otra pestaña. Recargá antes de continuar.")
            clean = deepcopy(self.project)
            clean.update(route=incoming["route"], events=incoming["events"],
                         revision=clean["revision"] + 1, updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(self.path, clean)
            self.project = clean
            return self.snapshot()

    def export_snapshot(self):
        project = self.snapshot()
        progress = review_progress(project)
        if not progress["ready"]:
            raise ValueError("Dibujá el recorrido y ubicá o descartá todas las detecciones antes de exportar.")
        return project
