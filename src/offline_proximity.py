"""Add the existing proximity layer using cached tracks, retaining manual review."""

from dataclasses import asdict
import hashlib
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .analysis_cache import read_cache
from .config import ProximityConfig
from .depth import DepthEstimator
from .offline_export import decode_detection
from .offline_media import atomic_json
from .proximity import ProximityAnalyzer, in_corridor
from .review_project import ProjectStore
from .tracker import select_device

LOG = logging.getLogger(__name__)
RECORRIDO_CORRIDOR = ((.45, .48), (.55, .48), (.78, .82), (.22, .82))


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def profile_config(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProximityConfig(enabled=True, profile=Path(path),
        depth_image_size=profile["depth_image_size"],
        corridor=tuple(tuple(p) for p in profile["corridor"]),
        candidate_confidence=profile.get("candidate_confidence", .3))


def calibrate_cached(project_path, output, *, groups, corridor=RECORRIDO_CORRIDOR, device="auto"):
    project = ProjectStore(project_path).snapshot()
    root = Path(project_path).resolve().parent
    rows = read_cache(root / project["analysis"]["cache"])
    metadata = next(rows)
    config = ProximityConfig(enabled=True, corridor=corridor)
    config.validate()
    depth = DepthEstimator(config, select_device(device), metadata["source_fps"])
    capture = cv2.VideoCapture(project["analysis"]["source_path"])
    measured = {name: {"seconds": list(bounds), "observations": []} for name, bounds in groups.items()}
    scale = tuple(s / a for s, a in zip(metadata["source_resolution"], metadata["resolution"]))
    try:
        for row in rows:
            group = next((name for name, (a, b) in groups.items() if a <= row["timestamp"] <= b), None)
            if group is None or row["frame"] % 5:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"] - 1)
            ok, image = capture.read()
            if not ok:
                raise ValueError("No se pudo leer un fotograma de calibración.")
            detections = [decode_detection(d, scale) for d in row["detections"]]
            observations, _ = depth.update(image, detections, row["frame"], row["timestamp"], has_candidate=True)
            candidates = [observations[d.track_id] for d in detections if d.track_id in observations
                          and in_corridor(d, config, image.shape[:2])]
            if candidates:
                chosen = min(candidates, key=lambda o: o.estimate)
                measured[group]["observations"].append({"frame": row["frame"], **asdict(chosen)})
    finally:
        capture.release()
    for name, group in measured.items():
        if len(group["observations"]) < 5:
            raise ValueError(f"Se necesitan más muestras válidas del tramo {name}.")
        group["median"] = float(np.median([o["estimate"] for o in group["observations"]]))
    near, caution, safe = (measured[k]["median"] for k in ("NEAR", "CAUTION", "SAFE"))
    if not 0 < near < caution < safe:
        raise ValueError("Los tramos revisados no distinguen cercanía creciente. Revisá la calibración.")
    near_threshold, caution_threshold = math.sqrt(near * caution), math.sqrt(caution * safe)
    deltas = [abs(a["estimate"] - b["estimate"]) for group in measured.values()
              for a, b in zip(group["observations"], group["observations"][1:])]
    profile = {"near_threshold": near_threshold, "caution_threshold": caution_threshold,
               "hysteresis": min(float(np.percentile(deltas, 90)), (caution_threshold - near_threshold) / 4),
               "depth_image_size": config.depth_image_size, "corridor": config.corridor,
               "candidate_confidence": config.candidate_confidence, "groups": measured,
               "source_sha256": project["source"]["sha256"], "metric_calibrated": False,
               "review": "Chevrolet delantero: distante, aproximación y cercano; revisión visual del recorrido.",
               "method": "Mismo cálculo de medias geométricas e histéresis del perfil de proximidad existente."}
    atomic_json(output, profile)
    LOG.info("Calibración: SAFE %.2f / CAUTION %.2f / NEAR %.2f", safe, caution, near)
    return profile


def augment_proximity(project_path, profile_path, device="auto"):
    project_path = Path(project_path).resolve()
    project_bytes = project_path.read_bytes()
    project = ProjectStore(project_path).snapshot()
    root = project_path.parent
    base = root / project["analysis"]["cache"]
    base_hash = file_digest(base)
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if profile.get("source_sha256") != project["source"]["sha256"]:
        raise ValueError("El perfil debe estar calibrado para el video de este proyecto.")
    config = profile_config(profile_path)
    config.validate()
    analyzer = ProximityAnalyzer(config)
    rows = read_cache(base)
    metadata = next(rows)
    depth = DepthEstimator(config, select_device(device), metadata["source_fps"])
    capture = cv2.VideoCapture(project["analysis"]["source_path"])
    destination = root / "observations_proximity.jsonl"
    temporary = destination.with_suffix(".jsonl.partial")
    counts = {}
    total = 0
    scale = tuple(s / a for s, a in zip(metadata["source_resolution"], metadata["resolution"]))
    provenance = {"base_cache_sha256": base_hash, "source_sha256": project["source"]["sha256"],
                  "profile_sha256": file_digest(profile_path), "profile": str(Path(profile_path).resolve())}
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({**metadata, "proximity_layer": provenance}) + "\n")
            for row in rows:
                ok, image = capture.read()
                if not ok:
                    raise ValueError("El video preparado terminó antes que las observaciones guardadas.")
                detections = [decode_detection(d, scale) for d in row["detections"]]
                has_candidate = any(in_corridor(d, config, image.shape[:2]) for d in detections)
                observations, fresh = depth.update(image, detections, row["frame"], row["timestamp"], has_candidate=has_candidate)
                proximity = analyzer.update(detections, observations, fresh, image.shape[:2], row["frame"], row["timestamp"])
                state = asdict(proximity)
                if state["box"]:
                    state["box"] = [v / scale[i % 2] for i, v in enumerate(state["box"])]
                row["scene"]["proximity"] = state
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                counts[proximity.state] = counts.get(proximity.state, 0) + 1
                total += 1
                if total % 300 == 0:
                    LOG.info("Proximidad: %d / %d fotogramas", total, metadata["source_total_frames"])
            stream.write(json.dumps({"type": "end", "complete": True, "frames": total}) + "\n")
        if total != project["analysis"]["frames"]:
            raise ValueError("La capa nueva no cubre todos los fotogramas del proyecto.")
        if project_path.read_bytes() != project_bytes:
            # A concurrent manual edit is valid; this pass must never overwrite it.
            LOG.info("La revisión manual cambió durante el análisis; se conservaron esos cambios.")
        temporary.replace(destination)
        result = {"cache": destination.name, **provenance, "states": counts, "frames": total,
                  "depth": depth.report(), "proximity": analyzer.report()}
        atomic_json(root / "scene_layers.json", result)
        LOG.info("Proximidad restaurada: %s", counts)
        return result
    finally:
        capture.release()


def augmented_cache(root, project):
    manifest = root / "scene_layers.json"
    base = root / project["analysis"]["cache"]
    if not manifest.exists():
        return base
    layers = json.loads(manifest.read_text(encoding="utf-8"))
    cache = (root / layers["cache"]).resolve()
    if not cache.is_relative_to(root.resolve()):
        raise ValueError("Ruta de observaciones aumentadas inválida.")
    if (layers["source_sha256"] != project["source"]["sha256"]
            or layers["base_cache_sha256"] != file_digest(base)
            or layers["frames"] != project["analysis"]["frames"]):
        raise ValueError("La capa de proximidad no corresponde al análisis original.")
    header = next(read_cache(cache))
    if header.get("proximity_layer", {}).get("base_cache_sha256") != layers["base_cache_sha256"]:
        raise ValueError("Las observaciones aumentadas no coinciden con su registro.")
    return cache
