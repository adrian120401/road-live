"""Approximate unique-object counts and bounded image-space trajectories."""

from collections import Counter, deque
from dataclasses import dataclass, field

from .config import CLASSES, Detection


@dataclass
class TrackStats:
    class_name: str
    first_frame: int
    last_frame: int
    observations: int = 1
    consecutive_frames: int = 1
    longest_run: int = 1
    gaps: int = 0
    class_changes: int = 0
    votes: deque = field(default_factory=lambda: deque(maxlen=10))
    pending_class: str | None = None
    pending_hits: int = 0
    initial_class: str = ""

    def resolve_class(self, detection: Detection) -> str:
        self.votes.append((detection.class_name, detection.confidence))
        scores: Counter = Counter()
        for name, confidence in self.votes:
            scores[name] += confidence
        winner, score = scores.most_common(1)[0]
        if winner == self.class_name or score <= sum(scores.values()) / 2:
            self.pending_class, self.pending_hits = None, 0
            return self.class_name
        self.pending_hits = self.pending_hits + 1 if self.pending_class == winner else 1
        self.pending_class = winner
        return winner if self.pending_hits >= 3 else self.class_name

    def observe(self, detection: Detection, frame: int) -> None:
        continuous = frame == self.last_frame + 1
        self.consecutive_frames = self.consecutive_frames + 1 if continuous else 1
        self.longest_run = max(self.longest_run, self.consecutive_frames)
        self.gaps += int(not continuous)
        self.class_changes += int(detection.class_name != self.class_name)
        self.last_frame = frame
        self.observations += 1


class Analytics:
    def __init__(self, trail_frames: int = 30) -> None:
        if trail_frames < 2:
            raise ValueError("trail_frames must be at least 2.")
        self.trail_frames = trail_frames
        self.tracks: dict[int, TrackStats] = {}
        self.counts: Counter[str] = Counter({name: 0 for name in CLASSES})
        self.histories: dict[int, deque[tuple[int, tuple[float, float]]]] = {}
        self.tracked_observations = 0
        self.untracked_observations = 0
        self.current_frame = 0
        self.peak_active_tracks = 0
        self.peak_active_frame = 0

    def update(self, detections: list[Detection], frame: int) -> None:
        if frame <= self.current_frame:
            raise ValueError("Analytics requires strictly increasing frame indices.")
        self.current_frame = frame
        frame_ids: set[int] = set()
        for detection in detections:
            if detection.class_name not in self.counts:
                continue
            track_id = detection.track_id
            if track_id is None:
                self.untracked_observations += 1
                continue
            if track_id in frame_ids:
                continue
            frame_ids.add(track_id)
            self.tracked_observations += 1
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackStats(detection.class_name, frame, frame)
                self.tracks[track_id].initial_class = detection.class_name
                self.tracks[track_id].resolve_class(detection)
                self.counts[detection.class_name] += 1
            else:
                self.tracks[track_id].observe(detection, frame)
                track = self.tracks[track_id]
                category = track.resolve_class(detection)
                if category != track.class_name:
                    self.counts[track.class_name] -= 1
                    self.counts[category] += 1
                    track.class_name = category
            history = self.histories.setdefault(track_id, deque(maxlen=self.trail_frames))
            history.append((frame, detection.ground_point))
        if len(frame_ids) > self.peak_active_tracks:
            self.peak_active_tracks = len(frame_ids)
            self.peak_active_frame = frame
        self._expire_history(frame)

    def _expire_history(self, frame: int) -> None:
        cutoff = frame - self.trail_frames + 1
        for track_id in list(self.histories):
            history = self.histories[track_id]
            while history and history[0][0] < cutoff:
                history.popleft()
            if not history:
                del self.histories[track_id]

    def class_for(self, detection: Detection) -> str:
        track = self.tracks.get(detection.track_id)
        return track.class_name if track else detection.class_name

    def report(self) -> dict:
        return {
            "unique_objects": len(self.tracks),
            "counts": dict(self.counts),
            "counting_method": "Unique tracking IDs; confidence-weighted category over last 10 observations, 3-hit confirmation.",
            "counting_limitations": "Lost/reassigned IDs can overcount; ID switches can undercount.",
            "vehicles_definition": "car only",
            "tracked_observations": self.tracked_observations,
            "untracked_observations": self.untracked_observations,
            "peak_active_tracks": self.peak_active_tracks,
            "peak_active_frame": self.peak_active_frame,
            "tracks": {str(track_id): {
                "class": track.class_name,
                "initial_class": track.initial_class,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "observations": track.observations,
                "longest_consecutive_run": track.longest_run,
                "visibility_gaps": track.gaps,
                "class_change_observations": track.class_changes,
            } for track_id, track in self.tracks.items()},
        }
