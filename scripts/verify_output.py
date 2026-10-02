"""Decode the exported video and audit its compact analytics report."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.location import VideoClock, RoutePoint, VideoMetadataLocationProvider
from src.config import Detection, ProximityConfig, RoadDamageConfig
from src.road_region import RoadRegion
from src.proximity import in_corridor


def source_event_times(report: dict, events: list[dict]) -> dict[int, float]:
    """Audit event times against decoded source PTS, including clock fallbacks."""
    if not events or not report.get("event_timing", {}).get("source_pts_frames", 0):
        return {e["frame"]: (e["frame"] - 1) / report["source_fps"] for e in events}
    targets = {e["frame"] for e in events}
    source = cv2.VideoCapture(report["input_path"])
    clock = VideoClock(report["source_fps"])
    times = {}
    try:
        assert source.isOpened(), "Cannot inspect source timestamps"
        for number in range(1, max(targets) + 1):
            ok, _ = source.read()
            assert ok, "Source ended before representative event frame"
            timestamp = clock.timestamp(source.get(cv2.CAP_PROP_POS_MSEC), number)
            if number in targets:
                times[number] = timestamp
    finally:
        source.release()
    return times


def verify(video: Path, report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = Counter(track["class"] for track in report["tracks"].values())
    assert len(report["tracks"]) == report["unique_objects"], "IDs/count mismatch"
    assert sum(report["counts"].values()) == report["unique_objects"], "Category total mismatch"
    for category, count in report["counts"].items():
        assert counts[category] == count, f"Incorrect count: {category}"
    assert sum(track["observations"] for track in report["tracks"].values()) == report["tracked_observations"]
    for track in report["tracks"].values():
        assert track["longest_consecutive_run"] <= track["observations"]
        assert track["observations"] <= track["last_frame"] - track["first_frame"] + 1

    expected_frames = report["frames_processed"]
    targets = {1, expected_frames, max(1, expected_frames // 2),
               max(1, expected_frames // 4), max(1, 3 * expected_frames // 4),
               report.get("peak_active_frame", 1)}
    preview_dir = video.parent / "validation" / video.stem
    road_events = []
    if "road_damage" in report:
        road = report["road_damage"]
        events_path = Path(road["events_path"])
        payload = json.loads(events_path.read_text(encoding="utf-8"))
        road_events = payload["events"]
        assert payload["complete"] == report["complete"], "Road completion mismatch"
        assert payload["unique_potholes"] == road["potholes"] == len(road_events)
        assert len({e["track_id"] for e in road_events}) == len(road_events), "Duplicate road event IDs"
        expected_calls = (expected_frames + road["frame_interval"] - 1) // road["frame_interval"]
        if report["complete"]:
            assert road["performance"]["inferences"] == expected_calls, "Road interval ignored"
        photos = 0
        event_times = source_event_times(report, road_events)
        area = road.get("road_area")
        region = RoadRegion(RoadDamageConfig(road_area=tuple(map(tuple, area)) if area else None,
                            road_area_min_overlap=road.get("road_area_min_overlap", .60)),
                            *report["resolution"])
        for event in road_events:
            assert event["type"] == "pothole"
            assert event["observations"] >= road["confirmation_hits"]
            assert 1 <= event["frame"] <= expected_frames
            assert event["frame"] <= event["last_frame"] <= expected_frames
            assert abs(event["timestamp"] - event_times[event["frame"]]) < 1e-6, "Event/source timestamp mismatch"
            assert event["confidence"] >= road["confidence"]
            assert region.accepts(Detection(event["track_id"], "pothole", event["confidence"],
                                           tuple(event["bbox"]))), "Event outside configured carriageway"
            x1, y1, x2, y2 = event["bbox"]
            left, top, right, bottom = road["roi_pixels"]
            assert left <= x1 < x2 <= right and top <= y1 < y2 <= bottom
            if event["evidence_path"]:
                evidence = events_path.parent / event["evidence_path"]
                assert evidence.is_file(), "Missing road photo"
                image = cv2.imread(str(evidence))
                assert image is not None and list(image.shape[1::-1]) == report["resolution"]
                photos += 1
            targets.update((event["frame"], event["confirmed_frame"], event["last_frame"]))
        assert photos == road["saved_evidence"], "Photo count mismatch"
        if road.get("save_evidence", False):
            assert photos == len(road_events), "Every confirmed event must have a photo"
    if "location" in report:
        location = report["location"]
        assert Path(location["map_path"]).is_file(), "Missing interactive map"
        if "trajectory" in location:
            points = [RoutePoint(**p) for p in location["trajectory"]]
        else:  # Read historical V3 reports without producing new GeoJSON files.
            route = json.loads(Path(location["route_path"]).read_text(encoding="utf-8"))
            points = [RoutePoint(**p) for p in route["properties"]["points"]]
        provider = VideoMetadataLocationProvider(points)
        assert len(points) == location["route_points"]
        assert location["geolocated_potholes"] == len(road_events)
        assert len({e["event_id"] for e in road_events}) == len(road_events)
        for event in road_events:
            assert event["confidence"] >= 0.50, "Low-confidence pothole exported"
            assert event["location_source"] == location["source"]
            point = provider.point_at(event["timestamp"])
            assert abs(event["latitude"] - point.latitude) < 1e-9
            assert abs(event["longitude"] - point.longitude) < 1e-9
            assert event["evidence_image"] == event["evidence_path"]
    prior_check = preview_dir / "verification.json"
    scene = report.get("scene") or {}
    assert not scene.get("errors"), "An optional perception layer failed"
    if "crosswalk" in scene:
        crossing = scene["crosswalk"]
        assert crossing["crosswalks_seen"] == len(crossing["tracks"]), "Crosswalk count mismatch"
        if report["complete"]:
            assert crossing["performance"]["inferences"] == (expected_frames+crossing["frame_interval"]-1)//crossing["frame_interval"]
        for track in crossing["tracks"].values():
            assert track["max_confidence"] >= crossing["confidence"]
            targets.update((track["confirmed_frame"],track["last_frame"]))
    if "proximity" in scene:
        proximity = scene["proximity"]
        profile = proximity["thresholds"]
        config = ProximityConfig(corridor=tuple(map(tuple,proximity["corridor"])))
        trace = [json.loads(line) for line in Path(proximity["trace_path"]).read_text(encoding="utf-8").splitlines()]
        assert len(trace) == expected_frames, "Missing proximity trace frames"
        width,height = report["resolution"]
        for row in trace:
            if row["state"] in {"CAUTION","NEAR"}:
                assert profile is not None, "Uncalibrated profile produced a warning"
                assert row["persistent_frames"] >= proximity["warning_frames"]
                assert row["fresh_samples"] >= 2
                assert str(row["track_id"]) in report["tracks"], "Proximity invented a track ID"
                assert in_corridor(Detection(row["track_id"],"car",1,tuple(row["box"])),config,(height,width))
        targets.update(t["frame"] for t in proximity["transitions"])
    if prior_check.is_file():
        prior = json.loads(prior_check.read_text(encoding="utf-8"))
        # Refresh earlier review frames too, so a re-export leaves no stale previews.
        targets.update(i for i in prior.get("sample_frames", []) if 1 <= i <= expected_frames)
    # Include first appearances of different classes to inspect crowded scenes.
    for category in ("person", "motorcycle", "bicycle", "bus", "truck", "traffic light", "stop sign"):
        class_tracks = [t for t in report["tracks"].values() if t["class"] == category]
        if class_tracks:
            targets.add(min(t["first_frame"] for t in class_tracks))
    preview_dir.mkdir(parents=True, exist_ok=True)
    samples: dict[int, np.ndarray] = {}
    decoded = 0
    capture = cv2.VideoCapture(str(video))
    try:
        assert capture.isOpened(), "Output cannot be opened"
        fps = capture.get(cv2.CAP_PROP_FPS)
        assert abs(fps - report["source_fps"]) < 0.02, "Playback FPS changed"
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            h, w = frame.shape[:2]
            assert [w, h] == report["resolution"], "Output resolution changed"
            if decoded in targets:
                sample_path = preview_dir / f"frame_{decoded:04d}.jpg"
                assert cv2.imwrite(str(sample_path), frame), f"Cannot save {sample_path}"
                # Keep original-resolution samples on disk, small tiles in memory.
                # A contact sheet made from dozens of 4K frames otherwise exhausts RAM.
                if w > 464:
                    samples[decoded] = cv2.resize(frame, (464, round(h * 464 / w)), interpolation=cv2.INTER_AREA)
                else:
                    samples[decoded] = frame
    finally:
        capture.release()
    assert decoded == expected_frames, f"Expected {expected_frames} frames, decoded {decoded}"
    if report["complete"]:
        assert decoded == report["source_total_frames"]

    images = []
    for index, frame in sorted(samples.items()):
        label = np.full((28, frame.shape[1], 3), (22, 24, 24), dtype=np.uint8)
        cv2.putText(label, f"FOTOGRAMA {index:04d} / {(index - 1) / fps:.1f}s", (14, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 240, 240), 1, cv2.LINE_AA)
        images.append(np.vstack((label, frame)))
    if images:
        while len(images) % 3:
            images.append(np.zeros_like(images[0]))
        sheet = np.vstack([np.hstack(images[i:i + 3]) for i in range(0, len(images), 3)])
        assert cv2.imwrite(str(preview_dir / "contact_sheet.jpg"), sheet)

    tracks = report["tracks"]
    most_persistent = sorted(tracks.items(), key=lambda item: item[1]["longest_consecutive_run"], reverse=True)[:5]
    verification = {
        "decoded_frames": decoded,
        "resolution": report["resolution"],
        "playback_fps": fps,
        "duration_seconds": round(decoded / fps, 3),
        "file_size_bytes": video.stat().st_size,
        "unique_ids": len(tracks),
        "tracked_observations": report["tracked_observations"],
        "tracks_with_at_least_30_consecutive_frames": sum(t["longest_consecutive_run"] >= 30 for t in tracks.values()),
        "single_observation_tracks": sum(t["observations"] == 1 for t in tracks.values()),
        "most_persistent_tracks": dict(most_persistent),
        "sample_frames": sorted(samples),
        "checks_passed": True,
        "road_events": len(road_events),
    }
    (preview_dir / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("outputs/urban_vision_output.mp4"))
    args = parser.parse_args()
    print(json.dumps(verify(args.video, args.video.with_suffix(".json")), indent=2))


if __name__ == "__main__":
    main()
