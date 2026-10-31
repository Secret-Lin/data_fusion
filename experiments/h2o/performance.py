"""Wall-time diagnostics for the offline H2O runner."""
from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator


class StageTimings:
    def __init__(self) -> None:
        self.stages: dict[str, dict[str, float | int]] = {}

    def add(self, name: str, elapsed: float) -> None:
        stage = self.stages.setdefault(name, {"seconds": 0.0, "calls": 0})
        stage["seconds"] += elapsed
        stage["calls"] += 1

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def report(self, elapsed: float, frames: int, observations: int) -> dict:
        return {
            "take_wall_s": elapsed,
            "synchronized_frames_per_s": frames / elapsed if elapsed else None,
            "observations_per_s": observations / elapsed if elapsed else None,
            "stages": self.stages,
            "timing_scope": "Caller wall time; RGB-D wait excludes overlapped worker time. "
                            "register includes estimator construction; GPU methods return CPU poses. "
                            "Take time excludes process import and final report writes.",
        }
