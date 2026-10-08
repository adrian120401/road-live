"""Offline route analysis, local review editor and reviewed 4K export."""

import argparse
import json
import logging
from pathlib import Path

from .offline_media import atomic_json, inspect_source, make_preview, prepare_sdr, source_identity
from .review_project import create_project

LOG = logging.getLogger(__name__)


def analyze(source, root, profile, device="auto", proximity_profile=None):
    from .config import Config, RoadDamageConfig, ProximityConfig
    from .main import process_video
    root = Path(root).resolve()
    if (root / "project.json").exists() or (root / "observations.jsonl").exists():
        raise ValueError("La carpeta ya contiene un análisis. Elegí otra carpeta para conservarlo.")
    root.mkdir(parents=True, exist_ok=True)
    identity = source_identity(source)
    media = inspect_source(source)
    atomic_json(root / "source_metadata.json", media)
    analysis_source = prepare_sdr(source.resolve(), root, media)
    road = RoadDamageConfig(enabled=True, roi=tuple(profile["roi"]),
        road_area=tuple(tuple(p) for p in profile["road_area"]), image_size=profile["image_size"],
        frame_interval=profile["frame_interval"])
    proximity = ProximityConfig()
    if proximity_profile:
        from .offline_proximity import profile_config
        proximity = profile_config(proximity_profile)
    config = Config(analysis_source, root / "analysis.mp4", device=device,
                    processing_max_side=profile["processing_max_side"], road_damage=road, proximity=proximity)
    report = process_video(config, trace_path=root / "observations.jsonl", write_video=False, manual_location=True)
    if not report["complete"]:
        raise RuntimeError("El análisis quedó parcial; se conservaron las observaciones y evidencias.")
    # Native evidence uses the same public event ID as the editor, map and final boxes.
    refresh_source_evidence(analysis_source, root, report)
    make_evidence_details(root)
    LOG.info("Creando video liviano para revisar las evidencias…")
    make_preview(analysis_source, root)
    atomic_json(root / "profile.json", profile)
    create_project(root, identity, analysis_source, report, media)
    LOG.info("Revisión preparada: python -m src.offline serve --project %s", root / "project.json")
    return report


def make_evidence_details(root):
    """Keep the whole photo and a readable contextual crop for each detection."""
    import cv2
    events_path = root / "analysis_events.json"
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    for event in payload["events"]:
        if not event.get("evidence_path"):
            continue
        path = root / event["evidence_path"]
        image = cv2.imread(str(path))
        if image is None:
            raise OSError("No se pudo leer una foto de evidencia: " + str(path))
        height, width = image.shape[:2]
        x1, y1, x2, y2 = event["bbox"]
        padx, pady = max(100, (x2 - x1) * .7), max(100, (y2 - y1) * .7)
        crop = image[max(0, int(y1 - pady)):min(height, int(y2 + pady)),
                     max(0, int(x1 - padx)):min(width, int(x2 + padx))]
        detail = path.with_name(path.stem + "_detail.jpg")
        if not cv2.imwrite(str(detail), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("No se pudo guardar el detalle de la evidencia.")
        event["detail_path"] = detail.relative_to(root).as_posix()
        event["evidence_resolution"] = [width, height]
    atomic_json(events_path, payload)


def refresh_source_evidence(source, root, report):
    import cv2
    from .config import Detection
    from .renderer import render_road_evidence
    events_path = root / "analysis_events.json"
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(source))
    sw, sh = report["source_resolution"]
    aw, ah = report["resolution"]
    try:
        for event in payload["events"]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, event["frame"] - 1)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError("No se pudo recuperar el fotograma original de una evidencia.")
            box = [v * (sw / aw if i % 2 == 0 else sh / ah) for i, v in enumerate(event["bbox"])]
            detection = Detection(event["event_id"], "pothole", event["confidence"], tuple(box))
            evidence = render_road_evidence(image, detection, event["frame"], event["timestamp"])
            if not cv2.imwrite(str(root / event["evidence_path"]), evidence):
                raise OSError("No se pudo guardar la evidencia original.")
            event["bbox"] = box
            event["evidence_resolution"] = [sw, sh]
        atomic_json(events_path, payload)
    finally:
        capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analizar y revisar recorridos de Urban Vision")
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="Comparar cuatro perfiles en los mismos fragmentos")
    calibration.add_argument("--input", type=Path, required=True)
    calibration.add_argument("--output", type=Path, default=Path("outputs/recorrido/calibration"))
    calibration.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    analysis = commands.add_parser("analyze", help="Guardar detecciones, evidencias y proyecto editable")
    analysis.add_argument("--input", type=Path, required=True)
    analysis.add_argument("--output", type=Path, default=Path("outputs/recorrido"))
    analysis.add_argument("--profile", type=Path, required=True)
    analysis.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    analysis.add_argument("--proximity-profile", type=Path, help="Activar los avisos de proximidad con un perfil de esta cámara")
    additions = commands.add_parser("add-proximity", help="Restaurar proximidad usando detecciones y mapa guardados")
    additions.add_argument("--project", type=Path, required=True)
    additions.add_argument("--profile", type=Path, required=True)
    additions.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    server = commands.add_parser("serve", help="Abrir el editor de recorrido y pozos")
    server.add_argument("--project", type=Path, required=True)
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--no-open", action="store_true")
    export = commands.add_parser("export", help="Exportar video 4K/60 y mapa de una revisión completa")
    export.add_argument("--project", type=Path, required=True)
    export.add_argument("--output", type=Path, help="Guardar una nueva versión sin reemplazar la anterior")
    export.add_argument("--reuse-map", action="store_true", help="Reutilizar la imagen exacta del mapa ya exportado")
    export.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="libx264",
                        help="H.264 CPU slow/CRF16 o NVIDIA p7/CQ14; ambos conservan 4K/60")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "calibrate":
            from .offline_calibration import calibrate
            args.output.parent.mkdir(parents=True, exist_ok=True)
            info = inspect_source(args.input)
            source = prepare_sdr(args.input, args.output.parent, info)
            calibrate(source, args.output, args.device)
        elif args.command == "analyze":
            analyze(args.input, args.output, json.loads(args.profile.read_text(encoding="utf-8")), args.device, args.proximity_profile)
        elif args.command == "add-proximity":
            from .offline_proximity import augment_proximity
            augment_proximity(args.project, args.profile, args.device)
        elif args.command == "serve":
            from .review_server import serve
            serve(args.project, args.port, not args.no_open)
        else:
            from .offline_export import export_review
            export_review(args.project, output_path=args.output, reuse_map=args.reuse_map, encoder=args.encoder)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        LOG.error("Error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
