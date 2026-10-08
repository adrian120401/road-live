"""Local FFmpeg discovery, source inspection, browser previews and SDR preparation."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def ffmpeg_executable():
    configured = os.environ.get("URBAN_VISION_FFMPEG") or shutil.which("ffmpeg")
    if configured:
        return configured
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Instalá las dependencias: python -m pip install -r requirements-offline.txt") from exc


def run_ffmpeg(arguments, *, log_path=None):
    command = [ffmpeg_executable(), "-hide_banner", "-nostdin", "-y", *map(str, arguments)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    detail = result.stderr.decode("utf-8", errors="replace")
    if log_path:
        Path(log_path).write_text(detail, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("FFmpeg no pudo completar la operación: " + detail[-2000:])
    return detail


def inspect_source(path):
    result = subprocess.run([ffmpeg_executable(), "-hide_banner", "-i", str(path)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    description = result.stderr
    if "Video:" not in description:
        raise ValueError("FFmpeg no pudo leer la pista de video: " + description[-1000:])
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", description)
    seconds = (int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])) if duration else None
    return {"ffmpeg_description": description, "container_duration_seconds": seconds,
            "hdr": "smpte2084" in description.lower() or "arib-std-b67" in description.lower(),
            "has_audio": "Audio:" in description}


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "name": path.name, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def verify_source(identity):
    path = Path(identity["path"])
    if not path.is_file():
        raise ValueError("No se encontró el video original del proyecto.")
    if source_identity(path)["sha256"] != identity["sha256"]:
        raise ValueError("El video original cambió. Volvé a analizarlo antes de exportar.")
    return path


def prepare_sdr(source, root, info):
    """Tone-map HDR before OpenCV; never silently decode PQ/HLG as ordinary SDR."""
    if not info["hdr"]:
        return source
    target = root / "source_sdr.mp4"
    stamp = {"source": str(Path(source).resolve()), "size": Path(source).stat().st_size,
             "mtime_ns": Path(source).stat().st_mtime_ns, "conversion_version": 1}
    manifest = root / "source_sdr.json"
    if target.is_file() and manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) == stamp:
        return target
    temporary = root / "source_sdr.partial.mp4"
    run_ffmpeg(["-i", source, "-map", "0:v:0", "-an", "-vf",
        "zscale=t=linear:npl=100,format=gbrpf32le,tonemap=tonemap=hable:desat=0,"
        "zscale=p=bt709:t=bt709:m=bt709:r=limited,format=yuv420p",
        "-c:v", "libx264", "-threads", "8", "-preset", "fast", "-crf", "14", "-fps_mode", "passthrough",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", temporary],
        log_path=root / "sdr_conversion.log")
    temporary.replace(target)
    atomic_json(manifest, stamp)
    return target


def make_preview(source, root):
    target = root / "preview.mp4"
    run_ffmpeg(["-i", source, "-map", "0:v:0", "-an", "-vf", "scale=-2:960,fps=30",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", target], log_path=root / "preview.log")
    return target


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
