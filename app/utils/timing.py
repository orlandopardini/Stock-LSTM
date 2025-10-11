# app/utils/timing.py
from __future__ import annotations
import time
from dataclasses import dataclass

@dataclass
class Stopwatch:
    start: float = 0.0
    end: float = 0.0
    running: bool = False

    def __enter__(self):
        self.start = time.perf_counter()
        self.running = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def stop(self):
        if self.running:
            self.end = time.perf_counter()
            self.running = False
        return self.elapsed

    @property
    def elapsed(self) -> float:
        return (time.perf_counter() - self.start) if self.running else (self.end - self.start)
