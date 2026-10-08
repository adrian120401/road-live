"""Local video metadata inspection and interchangeable temporal location providers."""

from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Protocol

from .config import LocationConfig

LOG = logging.getLogger("urban_vision")
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RoutePoint:
    timestamp: float
    latitude: float
    longitude: float


class LocationProvider(Protocol):
    source: str
    points: tuple[RoutePoint, ...]

    def point_at(self, timestamp: float) -> RoutePoint: ...


class PendingManualRoute:
    """Offline review must never assign the mock corridor to real evidence."""
    source = "manual_pending"
    points = ()

    def point_at(self, timestamp: float) -> RoutePoint:
        raise ValueError("La ubicación debe marcarse manualmente en la revisión.")


class VideoMetadataLocationProvider:
    source = "real"

    def __init__(self, points: list[RoutePoint]) -> None:
        if len(points) < 2 or any(b.timestamp <= a.timestamp for a, b in zip(points, points[1:])):
            raise ValueError("A temporal route requires at least two increasing timestamps.")
        self.points = tuple(points)
        self.times = tuple(p.timestamp for p in points)

    def point_at(self, timestamp: float) -> RoutePoint:
        t = min(max(timestamp, self.times[0]), self.times[-1])
        index = min(max(bisect_right(self.times, t) - 1, 0), len(self.points) - 2)
        a, b = self.points[index:index + 2]
        ratio = (t - a.timestamp) / (b.timestamp - a.timestamp)
        return RoutePoint(timestamp, a.latitude + ratio * (b.latitude - a.latitude),
                          a.longitude + ratio * (b.longitude - a.longitude))


class MockRouteLocationProvider(VideoMetadataLocationProvider):
    source = "mock"

    def __init__(self, duration: float) -> None:
        data = json.loads((ROOT / "assets/trinidad_route.json").read_text(encoding="utf-8"))
        coordinates = data["coordinates"]
        lengths = [0.0]
        for (lat_a, lon_a), (lat_b, lon_b) in zip(coordinates, coordinates[1:]):
            dy = math.radians(lat_b - lat_a)
            dx = math.radians(lon_b - lon_a) * math.cos(math.radians((lat_a + lat_b) / 2))
            lengths.append(lengths[-1] + math.hypot(dx, dy) * 6371000)
        duration = max(duration, 0.001)
        vertices = [RoutePoint(length / lengths[-1] * duration, lat, lon)
                    for length, (lat, lon) in zip(lengths, coordinates)]
        super().__init__(vertices)
        samples = [self.point_at(float(t)) for t in range(math.ceil(duration))]
        samples.append(self.point_at(duration))
        super().__init__(samples)


@dataclass
class VideoLocationMetadata:
    status: str = "absent"
    creation_date: str | None = None
    single_location: dict | None = None
    points: list[RoutePoint] = field(default_factory=list)
    reason: str = "No usable GPS metadata found"
    warning: str | None = None

    def report(self) -> dict:
        return {"status": self.status, "metadata_found": self.single_location is not None or bool(self.points),
                "creation_date": self.creation_date, "single_location": self.single_location,
                "timed_points_found": len(self.points), "reason": self.reason, "warning": self.warning}


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def parse_datetime(value, *, gps_utc: bool = False) -> datetime | None:
    if not isinstance(value, str):
        return None
    # ExifTool dates use YYYY:MM:DD; naive dates cannot establish GPS synchronization.
    value = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", value)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None and gps_utc:
            result = result.replace(tzinfo=timezone.utc)
        return result if result.tzinfo is not None else None
    except ValueError:
        return None


def coordinates(tags: dict) -> tuple[float, float] | None:
    lat, lon = number(tags.get("GPSLatitude")), number(tags.get("GPSLongitude"))
    if lat is None or lon is None:
        values = re.findall(r"[+-]?\d+(?:\.\d+)?", str(tags.get("GPSCoordinates", "")))
        if len(values) >= 2:
            lat, lon = number(values[0]), number(values[1])
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def parse_metadata(payload: list[dict]) -> VideoLocationMetadata:
    metadata = VideoLocationMetadata()
    contexts: dict[str, dict] = {}
    for record in payload:
        for key, value in record.items():
            parts = key.split(":")
            tag = parts[-1]
            documents = [p for p in parts[:-1] if re.fullmatch(r"Doc\d+", p)]
            # Family names (QuickTime/Composite) may vary within one embedded sample.
            prefix = ":".join(documents or [p for p in parts[:-1] if not p.startswith("Copy")])
            contexts.setdefault(prefix, {})[tag] = value
            if tag in ("CreationDate", "CreateDate") and parse_datetime(value) is not None:
                if metadata.creation_date is None or tag == "CreationDate":
                    metadata.creation_date = value
    start = parse_datetime(metadata.creation_date)
    for tags in contexts.values():
        position = coordinates(tags)
        if position is None:
            continue
        lat, lon = position
        timestamp = number(tags.get("SampleTime"))
        if timestamp is None and start is not None:
            gps_date = parse_datetime(tags.get("GPSDateTime"), gps_utc=True)
            if gps_date is not None:
                timestamp = (gps_date - start).total_seconds()
        if timestamp is not None and timestamp >= 0:
            metadata.points.append(RoutePoint(timestamp, lat, lon))
        elif metadata.single_location is None:
            metadata.single_location = {"latitude": lat, "longitude": lon}
    # Conflicting samples at the same time cannot define a reliable single route.
    unique: dict[float, RoutePoint] = {}
    for point in metadata.points:
        previous = unique.get(point.timestamp)
        if previous and (previous.latitude, previous.longitude) != (point.latitude, point.longitude):
            metadata.status, metadata.reason = "incomplete", "Conflicting GPS samples at the same timestamp"
            return metadata
        unique[point.timestamp] = point
    metadata.points = sorted(unique.values(), key=lambda p: p.timestamp)
    if metadata.points:
        metadata.status, metadata.reason = "track", "GPS metadata found"
    elif metadata.single_location:
        metadata.status, metadata.reason = "single", "Single location found but no track available"
    return metadata


def resolve_exiftool(config: LocationConfig) -> str | None:
    if config.exiftool is not None:
        return str(config.exiftool.resolve())
    installed = shutil.which("exiftool")
    if installed:
        return installed
    bundled = sorted((ROOT / "tools/exiftool").glob("*/exiftool.exe"))
    return str(bundled[-1]) if bundled else None


def inspect_video_metadata(video: Path, config: LocationConfig) -> VideoLocationMetadata:
    executable = resolve_exiftool(config)
    if executable is None:
        return VideoLocationMetadata(status="unavailable", reason="ExifTool unavailable; GPS inspection not performed")
    command = [executable, "-j", "-n", "-ee3", "-G3:1:4", "-api", "QuickTimeUTC=1",
               "-api", "LargeFileSupport=1", "-GPS*", "-SampleTime", "-CreationDate", "-CreateDate",
               str(video.resolve())]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=config.inspection_timeout_seconds, check=True)
        metadata = parse_metadata(json.loads(result.stdout))
        metadata.warning = result.stderr.strip() or None
        return metadata
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError) as exc:
        return VideoLocationMetadata(status="unavailable", reason="GPS metadata inspection failed", warning=str(exc))


def select_provider(metadata: VideoLocationMetadata, duration: float,
                    config: LocationConfig) -> LocationProvider:
    points = metadata.points
    usable = (metadata.status == "track" and len(points) >= 2 and duration > 0
              and points[0].timestamp <= config.endpoint_tolerance_seconds
              and abs(points[-1].timestamp - duration) <= config.endpoint_tolerance_seconds
              and all(b.timestamp - a.timestamp <= config.max_gap_seconds for a, b in zip(points, points[1:])))
    if usable and not config.force_mock_route and config.use_video_gps:
        return VideoMetadataLocationProvider(points)
    if metadata.status == "track" and not usable:
        metadata.status, metadata.reason = "incomplete", "GPS track has insufficient temporal coverage"
    return MockRouteLocationProvider(duration)


def prepare_location(video: Path, duration: float, config: LocationConfig) -> tuple[LocationProvider, VideoLocationMetadata]:
    metadata = (inspect_video_metadata(video, config) if config.use_video_gps
                else VideoLocationMetadata(status="disabled", reason="Video GPS inspection disabled"))
    provider = select_provider(metadata, duration, config)
    LOG.info("GPS: %s", metadata.reason)
    if metadata.warning:
        LOG.warning("Metadata: %s", metadata.warning)
    if provider.source == "mock":
        LOG.info("Falling back to simulated Trinidad route%s", " (forced)" if config.force_mock_route else "")
    return provider, metadata


def export_route(provider: LocationProvider, path: Path, *, complete: bool, processed_seconds: float) -> None:
    payload = {"type": "Feature", "properties": {"location_source": provider.source,
               "simulated": provider.source == "mock", "complete": complete,
               "processed_seconds": processed_seconds, "points": [asdict(p) for p in provider.points]},
               "geometry": {"type": "LineString", "coordinates": [[p.longitude, p.latitude] for p in provider.points]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class VideoClock:
    """Use source PTS when monotonic, falling back without moving time backwards."""
    def __init__(self, fps: float) -> None:
        self.fps, self.origin, self.previous = fps, None, -1.0
        self.pts_frames = self.fallback_frames = 0

    def timestamp(self, milliseconds: float, frame: int) -> float:
        pts = number(milliseconds)
        if self.origin is None and pts is not None and pts >= 0:
            self.origin = pts / 1000
        t = pts / 1000 - self.origin if pts is not None and self.origin is not None else -1
        if t >= 0 and t > self.previous:
            self.pts_frames += 1
        else:
            t = max((frame - 1) / self.fps, self.previous + 1 / self.fps)
            self.fallback_frames += 1
        self.previous = t
        return t
