"""Shared image-space matching; each perception layer retains its own ID namespace."""

import math


def geometry(box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w, h = max(0, x2 - x1), max(0, y2 - y1)
    return (x1 + x2) / 2, (y1 + y2) / 2, w * h, math.hypot(w, h)


def overlap(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = geometry(a)[2] + geometry(b)[2] - intersection
    return intersection / union if union else 0.0


def match_boxes(tracks: dict[int, tuple], boxes: list[tuple]) -> dict[int, int]:
    pairs = []
    for track_id, box in tracks.items():
        ax, ay, area_a, diagonal_a = geometry(box)
        for index, candidate in enumerate(boxes):
            bx, by, area_b, diagonal_b = geometry(candidate)
            iou = overlap(box, candidate)
            distance = math.hypot(ax - bx, ay - by) / max(diagonal_a, diagonal_b, 1)
            ratio = area_b / area_a if area_a else 0
            if iou >= .20:
                pairs.append((0, -iou, track_id, index))
            elif distance <= .5 and .5 <= ratio <= 2:
                pairs.append((1, distance, track_id, index))
    matched, used = {}, set()
    for _, _, track_id, index in sorted(pairs):
        if index not in matched and track_id not in used:
            matched[index] = track_id
            used.add(track_id)
    return matched
