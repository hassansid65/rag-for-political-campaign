"""
Latency instrumentation.

Every stage of a turn records into a `Trace`, which is returned on the API
response (`timings_ms`) and aggregated into rolling percentiles exposed on
/health and /metrics. Latency you can't see is latency you can't optimize.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Trace:
    """Per-request stage timings, in milliseconds."""

    name: str = "turn"
    started_at: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            # Repeated labels accumulate (e.g. two embed calls in one turn).
            self.stages[label] = round(self.stages.get(label, 0.0) + elapsed, 2)
            METRICS.observe(f"{self.name}.{label}", elapsed)

    def mark(self, label: str) -> float:
        """Record time-since-start for a milestone (e.g. first token out)."""
        elapsed = (time.perf_counter() - self.started_at) * 1000.0
        self.marks[label] = round(elapsed, 2)
        METRICS.observe(f"{self.name}.mark.{label}", elapsed)
        return elapsed

    def set(self, **kwargs: Any) -> None:
        self.meta.update(kwargs)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000.0, 2)

    def finish(self) -> dict[str, Any]:
        total = self.total_ms
        METRICS.observe(f"{self.name}.total", total)
        payload: dict[str, Any] = dict(self.stages)
        payload.update({f"@{k}": v for k, v in self.marks.items()})
        payload["total"] = total
        return payload


class MetricsRegistry:
    """Thread-safe rolling window of stage latencies for percentile reporting."""

    def __init__(self, window: int = 512) -> None:
        self._window = window
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, key: str, value_ms: float) -> None:
        with self._lock:
            self._samples[key].append(value_ms)

    def incr(self, key: str, by: int = 1) -> None:
        with self._lock:
            self._counters[key] += by

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            items = {k: list(v) for k, v in self._samples.items() if v}
        out: dict[str, dict[str, float]] = {}
        for key, values in items.items():
            ordered = sorted(values)
            out[key] = {
                "count": len(ordered),
                "p50": round(_pct(ordered, 0.50), 2),
                "p95": round(_pct(ordered, 0.95), 2),
                "p99": round(_pct(ordered, 0.99), 2),
                "mean": round(statistics.fmean(ordered), 2),
                "max": round(ordered[-1], 2),
            }
        return out

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._counters.clear()


def _pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    low, high = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    frac = idx - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


METRICS = MetricsRegistry()
