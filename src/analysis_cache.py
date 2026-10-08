"""Streaming observations for review and later rendering, without rerunning models."""

from dataclasses import asdict
import json
from pathlib import Path


class AnalysisCache:
    def __init__(self, path: Path, metadata: dict):
        self.path = path
        self.partial = path.with_suffix(path.suffix + ".partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.partial.open("w", encoding="utf-8")
        self.stream.write(json.dumps({"type": "metadata", "version": 1, **metadata}) + "\n")
        self.frames = 0

    def append(self, number, timestamp, detections, road, scene):
        payload = {"type": "frame", "frame": number, "timestamp": timestamp,
                   "detections": [asdict(d) for d in detections],
                   "road": [asdict(d) for d in road.detections] if road else [],
                   "scene": {"crossings": [asdict(d) for d in scene.crossings.detections],
                             "proximity": asdict(scene.proximity),
                             "crosswalks_seen": scene.crosswalks_seen}}
        self.stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        self.frames += 1
        if self.frames % 150 == 0:
            self.stream.flush()

    def finish(self, complete):
        self.stream.write(json.dumps({"type": "end", "complete": complete,
                                      "frames": self.frames}) + "\n")
        self.stream.close()
        if complete:
            self.partial.replace(self.path)


def read_cache(path: Path):
    """Validate ordering/completion as the caller consumes bounded frame records."""
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream, "{}"))
        if header.get("type") != "metadata" or header.get("version") != 1:
            raise ValueError("Caché de análisis incompatible.")
        yield header
        count, previous = 0, -1.0
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "end":
                if not row.get("complete") or row.get("frames") != count:
                    raise ValueError("El análisis está incompleto.")
                if stream.read().strip():
                    raise ValueError("Hay datos después del cierre del análisis.")
                return
            if (row.get("type") != "frame" or row.get("frame") != count + 1
                    or not isinstance(row.get("timestamp"), (int, float))
                    or not previous < row["timestamp"] < float("inf")):
                raise ValueError("Fotogramas o tiempos inválidos en el análisis.")
            count += 1
            previous = row["timestamp"]
            yield row
        raise ValueError("El análisis no tiene cierre completo.")
