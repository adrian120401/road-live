"""Bounded wall-time profiling independent from model-specific inference timing."""

from collections import defaultdict, deque
from contextlib import contextmanager
import time

import numpy as np


class StageTimings:
    def __init__(self) -> None:
        self.samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=4096))
        self.totals: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def record(self, name: str, seconds: float) -> None:
        self.samples[name].append(seconds * 1000)
        self.totals[name] += seconds
        self.calls[name] += 1

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - started)

    def report(self) -> dict:
        return {name: {"calls": self.calls[name], "total_seconds": round(self.totals[name], 3),
                       "mean_ms": round(self.totals[name] * 1000 / self.calls[name], 3),
                       "p95_ms": round(float(np.percentile(values, 95)), 3),
                       "p95_window": len(values)} for name, values in self.samples.items()}
