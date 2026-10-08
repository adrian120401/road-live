"""CLI and streaming video pipeline for Urban Vision."""

import json
import logging
import math
import shlex
import sys
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

# Support both `python src/main.py` and `python -m src.main`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.analytics import Analytics
from src.config import CLASS_MIN_CONFIDENCE, Config
from src.cli import parse_args
from src.scene import SceneAnalyzer
from src.performance import StageTimings
from src.renderer import FrameMetrics, Renderer
from src.tracker import ObjectTracker
from src.road_damage import RoadDamageDetector
from src.road_analytics import RoadAnalytics
from src.location import prepare_location, select_provider, VideoClock, PendingManualRoute, inspect_video_metadata
from src.map_renderer import render_map
from src.video_reader import VideoReader
from src.analysis_cache import AnalysisCache

LOG = logging.getLogger("urban_vision")


def show_frame(frame, enabled: bool) -> tuple[bool, bool]:
    if not enabled:
        return False, False
    try:
        cv2.imshow("Urban Vision", frame)
        stop = cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27)
        if cv2.getWindowProperty("Urban Vision", cv2.WND_PROP_VISIBLE) < 1:
            stop = True
        return True, stop
    except cv2.error as exc:
        LOG.warning("Preview unavailable; continuing export without --show: %s", exc)
        return False, False


def process_video(config: Config, *, frame_limit: int | None = None,
                  trace_path: Path | None = None, write_video: bool = True,
                  manual_location: bool = False) -> dict:
    """Process all frames; frame_limit is only used by integration smoke checks."""
    config.validate()
    capture = cv2.VideoCapture(str(config.input_path))
    reader = VideoReader(capture, config.processing_max_side, config.prefetch_frames)
    writer = None
    cache = None
    analytics = Analytics(config.trail_frames)
    frames = 0
    duration = 0.0
    completed = False
    stop_reason = "end_of_video"
    failure: BaseException | None = None
    road_detector = None
    road_analytics = None
    road_summary = None
    general_stage_seconds = 0.0
    timings = StageTimings()
    scene_summary = None
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open video: {config.input_path}")
        packet = reader.read()
        if packet is None:
            raise ValueError("Video contains no decodable frames.")
        frame = packet.image
        height, width = frame.shape[:2]
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Video FPS is invalid; cannot preserve playback timing.")
        raw_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        source_rotation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
        total = int(raw_total) if math.isfinite(raw_total) and raw_total > 0 else 0
        if manual_location:
            location_provider = PendingManualRoute()
            location_metadata = inspect_video_metadata(config.input_path, config.location)
            LOG.info("Ubicación pendiente de revisión manual; no se asignará la ruta simulada.")
        else:
            location_provider, location_metadata = prepare_location(config.input_path, total / source_fps, config.location)
        clock = VideoClock(source_fps)
        tracker = ObjectTracker(config)
        LOG.info("Model: %s", config.model)
        LOG.info("Device: %s", tracker.device)
        LOG.info("Source resolution: %s | Analysis/output resolution: %s x %s | Prefetch: %d",
                 reader.source_size, width, height, config.prefetch_frames)
        LOG.info("FPS source: %.5f | Frames: %s", source_fps, total or "unknown")
        LOG.info("Tracker: %s | Confidence: %.2f", config.tracker, config.conf)
        tracker.warmup(frame)
        scene_analyzer = SceneAnalyzer(config,tracker.device,source_fps,width,height)
        scene_analyzer.warmup(frame)
        if config.proximity.enabled and config.proximity.profile is None:
            LOG.warning("Proximidad sin perfil medido: se registrará profundidad; avisos desactivados.")
        if config.road_damage.enabled:
            road_detector = RoadDamageDetector(config.road_damage, tracker.device, source_fps, width, height)
            road_analytics = RoadAnalytics(config.road_damage, config.output_path, source_fps,
                                           None if manual_location else location_provider)
            road_detector.warmup(frame)
            LOG.info("Urban Vision V3.1 | Road model: %s", config.road_damage.model)
            LOG.info("Road confidence: %.2f | Interval: %d | ROI: %s | Evidence: %s",
                     config.road_damage.confidence, config.road_damage.frame_interval,
                     config.road_damage.roi, config.road_damage.save_evidence)
            LOG.info("Road area: %s | Minimum box overlap: %.0f%%",
                     config.road_damage.road_area, config.road_damage.road_area_min_overlap * 100)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        if write_video:
            writer = cv2.VideoWriter(str(config.output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     source_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create MP4 writer: {config.output_path}")
        if trace_path:
            cache = AnalysisCache(trace_path, {"source": str(config.input_path.resolve()),
                "resolution": [width, height], "source_resolution": list(reader.source_size),
                "source_fps": source_fps, "source_total_frames": total,
                "display_confidence": config.display_confidence,
                "infrastructure": config.traffic.crosswalks.enabled,
                "scene_layers": {"proximity": config.proximity.enabled,
                                 "proximity_profile": str(config.proximity.profile) if config.proximity.profile else None}})
        renderer = Renderer(width, height,config.display_confidence,
                            "ANÁLISIS DEL RECORRIDO" if manual_location else "ANÁLISIS EN VIVO")
        rolling_times: deque[float] = deque(maxlen=30)
        show = config.show
        if show:
            try:
                cv2.namedWindow("Urban Vision", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Urban Vision", min(width, 720), min(height, 1000))
            except cv2.error as exc:
                LOG.warning("Preview unavailable; continuing export: %s", exc)
                show = False
        started = time.perf_counter()
        try:
            reader.start()
            while packet is not None:
                frame_started = time.perf_counter()
                if packet.number != frames + 1:
                    raise RuntimeError("Decoder returned unordered frames.")
                frame = packet.image
                timings.record("decode",packet.decode_seconds)
                timings.record("resize",packet.resize_seconds)
                timestamp = clock.timestamp(packet.source_pts_ms, packet.number)
                general_started = time.perf_counter()
                detections = tracker.update(frame)
                general_elapsed = time.perf_counter() - general_started
                general_stage_seconds += general_elapsed
                timings.record("general_tracking",general_elapsed)
                road_state = None
                if road_detector is not None and road_analytics is not None:
                    with timings.measure("road_damage"):
                        road_state = road_detector.update(frame, frames + 1, timestamp)
                    with timings.measure("road_analytics_evidence"):
                        road_analytics.update(road_state, frame, frames + 1, timestamp)
                with timings.measure("analytics"):
                    analytics.update(detections, frames + 1)
                with timings.measure("scene"):
                    scene = scene_analyzer.update(frame,detections,frames+1,timestamp)
                if config.debug_scene:
                    scene = replace(scene,debug={**scene.debug,
                        "general_ms":tracker.stats.report()["mean_inference_ms"],
                        "road_ms":road_detector.stats.report()["mean_inference_ms"] if road_detector else 0})
                processing_fps = (len(rolling_times) / sum(rolling_times)
                                  if rolling_times else 1 / max(time.perf_counter() - frame_started, 1e-6))
                metrics = FrameMetrics(frames + 1, total, source_fps, processing_fps)
                if cache:
                    cache.append(frames + 1, timestamp, detections, road_state, scene)
                rendered = frame
                if writer is not None or show:
                    with timings.measure("renderer"):
                        rendered = renderer.render(frame, detections, analytics, metrics, road=road_state,
                                                   potholes=road_analytics.count if road_analytics else 0,
                                                   debug_road=config.road_damage.debug,scene=scene,
                                                   debug_scene=config.debug_scene,
                                                   corridor=config.proximity.corridor,
                                                   infrastructure=config.traffic.crosswalks.enabled)
                if writer is not None:
                    with timings.measure("writer"):
                        writer.write(rendered)
                frames += 1
                show, stop = show_frame(rendered, show)
                if frames % 150 == 0:
                    elapsed = time.perf_counter() - started
                    LOG.info("Frame %d/%s | %.1f FPS | %d unique tracking IDs",
                             frames, total or "?", frames / elapsed, len(analytics.tracks))
                if stop or (frame_limit is not None and frames >= frame_limit):
                    timings.record("total_frame",time.perf_counter()-frame_started)
                    stop_reason = "preview_stopped" if stop else "frame_limit"
                    break
                with timings.measure("reader_wait"):
                    packet = reader.read()
                elapsed_frame = time.perf_counter()-frame_started
                rolling_times.append(elapsed_frame)
                timings.record("total_frame",elapsed_frame)
            else:
                completed = not total or frames >= total
                if not completed:
                    raise RuntimeError(f"Video ended early: decoded {frames} of {total} frames.")
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            LOG.warning("Interrupted; finalizing partial video (%d frames).", frames)
        except BaseException as exc:
            failure = exc
            stop_reason = "error"
        finally:
            if road_analytics is not None:
                try:
                    road_summary = road_analytics.finish(completed, stop_reason)
                except (OSError, cv2.error) as exc:
                    failure = failure or exc
                    completed, stop_reason = False, "error"
            duration = time.perf_counter() - started
            try:
                scene_summary = scene_analyzer.finish()
            except OSError as exc:
                failure = failure or exc
                completed, stop_reason = False, "error"
    finally:
        reader.close()
        if writer is not None:
            writer.release()
        if cache is not None:
            cache.finish(completed and failure is None)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    report = {
        "model": config.model, "tracker": config.tracker, "device": tracker.device,
        "input_path": str(config.input_path.resolve()),
        "output_path": str(config.output_path.resolve()),
        "resolution": [width, height], "source_fps": source_fps,
        "source_resolution": list(reader.source_size),
        "processing_max_side": config.processing_max_side, "prefetch_frames": config.prefetch_frames,
        "playback_fps": source_fps,
        "profiling_note": "Decode/resize run concurrently when prefetch is enabled; stage totals are not additive.",
        "source_rotation_degrees": source_rotation, "display_language": "es",
        "source_total_frames": total, "frames_processed": frames,
        "processing_seconds": round(duration, 3),
        "processing_fps": round(frames / duration, 3) if duration else 0,
        "complete": completed, "stop_reason": stop_reason,
        "confidence_threshold": config.conf, "inference_image_size": config.image_size,
        "display_confidence_threshold":config.display_confidence,
        "class_confidence_gates": CLASS_MIN_CONFIDENCE,
        "audio": False, **analytics.report(),
        "video_written": write_video,
        "cache_path": str(trace_path.resolve()) if trace_path else None,
        "general_performance": tracker.stats.report() if hasattr(tracker, "stats") else {},
        "general_tracking_stage_seconds": round(general_stage_seconds, 3),
        "stage_performance": timings.report(),
        "scene": scene_summary,
        "event_timing": {"source_pts_frames": clock.pts_frames, "fps_fallback_frames": clock.fallback_frames},
    }
    if road_detector is not None:
        report["road_damage"] = {
            "enabled": True, "model": str(config.road_damage.model), "classes": ["pothole"],
            "confidence": config.road_damage.confidence, "roi": config.road_damage.roi,
            "roi_pixels": road_detector.roi, "frame_interval": config.road_damage.frame_interval,
            "road_area": config.road_damage.road_area,
            "road_area_min_overlap": config.road_damage.road_area_min_overlap,
            "confirmation_hits": config.road_damage.confirmation_hits,
            "nms_iou": config.road_damage.iou_threshold,
            "save_evidence": config.road_damage.save_evidence,
            "max_gap_seconds": config.road_damage.max_gap_seconds,
            "performance": road_detector.stats.report(),
            "discarded_below_50_observations": getattr(road_detector, "discarded_below_50", 0),
            "discarded_below_threshold_observations": getattr(road_detector, "discarded_below_threshold", 0),
            "discarded_outside_road_observations": getattr(road_detector, "discarded_outside_road", 0),
            "discard_measurement_range": [0.35, 0.50],
            **(road_summary if road_summary is not None else {"evidence_error": str(failure)}),
        }
    processed_seconds = max(0, clock.previous + 1 / source_fps) if frames else 0
    if not total and not manual_location:
        location_provider = select_provider(location_metadata, max(processed_seconds, 1 / source_fps), config.location)
        if road_analytics is not None:
            road_analytics.location = location_provider
            road_summary = road_analytics.finish(completed, stop_reason)
            report["road_damage"].update(road_summary)
    map_path = config.location.map_output or config.output_path.with_name(config.output_path.stem + "_map.html")
    report["location"] = {**location_metadata.report(), "source": location_provider.source,
                          "forced_mock": config.location.force_mock_route,
                          "route_points": len(location_provider.points),
                          "trajectory": [asdict(p) for p in location_provider.points]}
    try:
        events = list(road_analytics.events.values()) if road_analytics is not None else []
        events_path = road_analytics.events_path if road_analytics is not None else config.output_path.with_suffix(".json")
        if manual_location:
            report["location"].update(source="manual_pending", trajectory=[], route_points=0,
                                      geolocated_potholes=0)
        else:
            report["location"].update(render_map(location_provider, events, events_path, map_path,
                                          complete=completed, processed_seconds=processed_seconds,
                                          confidence=config.road_damage.confidence,
                                          enhanced=config.map.enhanced_popups))
    except (OSError, ValueError) as exc:
        failure = failure or exc
        report["location"]["export_error"] = str(exc)
    summary_path = config.output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Frames processed: %d%s", frames, "" if completed else " (PARTIAL)")
    LOG.info("Processing FPS: %.2f", report["processing_fps"])
    LOG.info("Unique objects detected: %d (tracking IDs; approximate)", report["unique_objects"])
    LOG.info("Counts: %s", json.dumps(report["counts"]))
    if write_video:
        LOG.info("Output path: %s", config.output_path.resolve())
    else:
        LOG.info("Observaciones para exportar después de revisar: %s", trace_path)
    LOG.info("Analytics: %s", summary_path.resolve())
    LOG.info("General model inference: %.2f ms", report["general_performance"].get("mean_inference_ms", 0))
    if road_detector is not None:
        road = report["road_damage"]
        LOG.info("Road damage | Potholes detected: %d (estimated events)", road.get("potholes", 0))
        LOG.info("Road inference: %.2f ms | %.2f inference FPS | %d calls",
                 road["performance"]["mean_inference_ms"], road["performance"]["inference_fps"],
                 road["performance"]["inferences"])
        LOG.info("Saved events: %s | Photos: %d", road.get("events_path"), road.get("saved_evidence", 0))
        LOG.info("Discarded observations (35%% <= confidence < 50%%): %d", road["discarded_below_50_observations"])
        LOG.info("Discarded observations outside road area (confidence >= threshold): %d",
                 road["discarded_outside_road_observations"])
    LOG.info("Location source: %s | Metadata found: %s", report["location"]["source"], report["location"]["metadata_found"])
    LOG.info("Route points generated: %d | Geolocated potholes: %d", report["location"]["route_points"],
             report["location"].get("geolocated_potholes", 0))
    if manual_location:
        LOG.info("Mapa pendiente: dibujar recorrido y ubicar detecciones en el editor.")
    else:
        LOG.info("Interactive map: %s", report["location"].get("map_path", "ERROR"))
        LOG.info("Abrir mapa: python scripts/open_map.py --map %s", shlex.quote(str(map_path)))
    LOG.info("Semáforos vistos: %d | Pasos peatonales: %d",analytics.counts['traffic light'],
             (scene_summary or {}).get('crosswalk',{}).get('crosswalks_seen',0))
    for name in ('crosswalk','depth'):
        stats = (scene_summary or {}).get(name,{}).get('performance',{})
        if stats:
            LOG.info("%s inference: %.2f ms | Stage: %.2f ms | Calls: %d",name,
                     stats['mean_inference_ms'],stats['mean_stage_ms'],stats['inferences'])
    LOG.info("Stage profile: %s",json.dumps(report['stage_performance']))
    if failure is not None:
        raise failure
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = process_video(parse_args())
        return 0 if report["complete"] else 130
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        LOG.error("Error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
