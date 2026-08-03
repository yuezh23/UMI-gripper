"""Thread-safe in-memory buffers and training-compatible nearest-time synchronization."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class ForceSample:
    timestamp_ns: int
    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float


@dataclass(frozen=True)
class ImageSample:
    timestamp_ns: int
    gray: np.ndarray


@dataclass(frozen=True)
class SyncedSample:
    timestamp_ns: int
    force_left: ForceSample
    force_right: ForceSample
    gelsight_left: ImageSample
    gelsight_right: ImageSample


class RealtimeSensorBuffer:
    """Keep recent samples and synchronize on left-force timestamps."""

    STREAMS = ("force_left", "force_right", "gelsight_left", "gelsight_right")

    def __init__(self, retention_sec: float = 8.0):
        if retention_sec <= 0:
            raise ValueError("retention_sec must be positive")
        self.retention_ns = int(retention_sec * 1e9)
        self._streams: dict[str, deque] = {name: deque() for name in self.STREAMS}
        self._lock = Lock()

    def append(self, stream: str, sample: ForceSample | ImageSample) -> None:
        if stream not in self._streams:
            raise KeyError(f"Unknown sensor stream {stream!r}")
        with self._lock:
            queue = self._streams[stream]
            if queue and sample.timestamp_ns < queue[-1].timestamp_ns:
                # Reentrant ROS callbacks can finish out of order. The nearest-
                # timestamp synchronizer uses bisect and therefore requires
                # every stream to remain sorted.
                timestamps = [item.timestamp_ns for item in queue]
                queue.insert(bisect_left(timestamps, sample.timestamp_ns), sample)
            else:
                queue.append(sample)
            newest_ns = queue[-1].timestamp_ns
            cutoff_ns = newest_ns - self.retention_ns
            while queue and queue[0].timestamp_ns < cutoff_ns:
                queue.popleft()

    def clear(self) -> None:
        with self._lock:
            for queue in self._streams.values():
                queue.clear()

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {name: len(queue) for name, queue in self._streams.items()}

    def latest_timestamps(self) -> dict[str, int | None]:
        with self._lock:
            return {
                name: queue[-1].timestamp_ns if queue else None
                for name, queue in self._streams.items()
            }

    def latest_common_timestamp_ns(self) -> int | None:
        """Return the newest timestamp for which all four streams have data."""

        with self._lock:
            timestamps = [
                queue[-1].timestamp_ns
                for queue in self._streams.values()
                if queue
            ]
            if len(timestamps) != len(self.STREAMS):
                return None
            return min(timestamps)

    def snapshot_synced(
        self,
        start_ns: int,
        end_ns: int,
        tolerance_ns: int = 100_000_000,
    ) -> list[SyncedSample]:
        if end_ns <= start_ns:
            return []
        with self._lock:
            streams = {name: list(queue) for name, queue in self._streams.items()}
        if any(not streams[name] for name in self.STREAMS):
            return []

        anchors = [
            sample
            for sample in streams["force_left"]
            if start_ns < sample.timestamp_ns <= end_ns
        ]
        candidates = {
            name: streams[name]
            for name in ("force_right", "gelsight_left", "gelsight_right")
        }
        timestamps = {
            name: [sample.timestamp_ns for sample in values]
            for name, values in candidates.items()
        }

        output = []
        for left_force in anchors:
            selected = {}
            valid = True
            for name, values in candidates.items():
                index = _nearest_index(timestamps[name], left_force.timestamp_ns)
                sample = values[index]
                if abs(sample.timestamp_ns - left_force.timestamp_ns) > tolerance_ns:
                    valid = False
                    break
                selected[name] = sample
            if valid:
                output.append(
                    SyncedSample(
                        timestamp_ns=left_force.timestamp_ns,
                        force_left=left_force,
                        force_right=selected["force_right"],
                        gelsight_left=selected["gelsight_left"],
                        gelsight_right=selected["gelsight_right"],
                    )
                )
        return output


def _nearest_index(timestamps: list[int], target: int) -> int:
    if not timestamps:
        raise ValueError("Cannot find nearest item in an empty sequence")
    index = bisect_left(timestamps, target)
    if index == 0:
        return 0
    if index == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[index - 1]
    after = timestamps[index]
    return index if abs(after - target) < abs(before - target) else index - 1
