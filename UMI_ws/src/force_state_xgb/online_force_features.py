"""Training-compatible baselines and one-second features for synchronized live samples."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import numpy as np

from force_state_common import AXES, SIDES, extract_window
from realtime_sensor_buffer import ForceSample, SyncedSample


@dataclass(frozen=True)
class OnlineBaseline:
    session_start_ns: int
    baseline_end_ns: int
    force_offsets: dict[str, dict[str, float]]
    gelsight_references: dict[str, np.ndarray]
    gelsight_diff_offsets: dict[str, float]


@dataclass(frozen=True)
class OnlineWindow:
    valid: bool
    features: dict[str, float]
    coverage_sec: float
    num_records: int
    reason: str


class OnlineFeatureExtractor:
    def __init__(
        self,
        *,
        baseline_duration_sec: float = 3.0,
        force_baseline_sec: float = 0.75,
        gelsight_baseline_start_sec: float = 0.5,
        pixel_threshold: float = 8.0,
        force_abs_limit: float = 10.0,
        window_sec: float = 1.0,
        min_window_coverage_sec: float = 0.8,
        min_window_records: int = 4,
        min_baseline_records: int = 20,
    ):
        self.baseline_duration_sec = baseline_duration_sec
        self.force_baseline_sec = force_baseline_sec
        self.gelsight_baseline_start_sec = gelsight_baseline_start_sec
        self.pixel_threshold = pixel_threshold
        self.force_abs_limit = force_abs_limit
        self.window_sec = window_sec
        self.min_window_coverage_sec = min_window_coverage_sec
        self.min_window_records = min_window_records
        self.min_baseline_records = min_baseline_records
        self._gelsight_cache: dict[
            str,
            OrderedDict[int, tuple[float, float, float, float, float, float]],
        ] = {side: OrderedDict() for side in SIDES}
        self._gelsight_cache_locks = {side: Lock() for side in SIDES}

    def build_baseline(
        self,
        samples: list[SyncedSample],
        session_start_ns: int,
    ) -> OnlineBaseline:
        for side, cache in self._gelsight_cache.items():
            with self._gelsight_cache_locks[side]:
                cache.clear()
        baseline_end_ns = session_start_ns + int(self.baseline_duration_sec * 1e9)
        use = [
            sample
            for sample in samples
            if session_start_ns < sample.timestamp_ns <= baseline_end_ns
        ]
        if len(use) < self.min_baseline_records:
            raise ValueError(f"Insufficient synchronized baseline records: {len(use)}")

        force_offsets: dict[str, dict[str, float]] = {}
        for side in SIDES:
            early = [
                sample
                for sample in use
                if sample.timestamp_ns
                <= session_start_ns + int(self.force_baseline_sec * 1e9)
            ]
            if not early:
                raise ValueError(f"No {side} force samples in force baseline interval")
            force_offsets[side] = {}
            for axis in AXES:
                values = [
                    float(getattr(getattr(sample, f"force_{side}"), axis))
                    for sample in early
                ]
                force_offsets[side][axis] = float(np.median(values))

        references: dict[str, np.ndarray] = {}
        diff_offsets: dict[str, float] = {}
        gel_start_ns = session_start_ns + int(self.gelsight_baseline_start_sec * 1e9)
        for side in SIDES:
            unique_images: dict[int, np.ndarray] = {}
            for sample in use:
                image_sample = getattr(sample, f"gelsight_{side}")
                # Offline training selects the reference interval using the synchronized
                # left-force anchor time, then deduplicates repeated image paths.
                if gel_start_ns <= sample.timestamp_ns <= baseline_end_ns:
                    unique_images.setdefault(image_sample.timestamp_ns, image_sample.gray)
            if not unique_images:
                raise ValueError(f"No {side} GelSight images in baseline interval")
            shapes = {image.shape for image in unique_images.values()}
            if len(shapes) != 1:
                raise ValueError(f"{side} GelSight image shape changed: {sorted(shapes)}")
            references[side] = np.median(
                np.stack(list(unique_images.values())),
                axis=0,
            ).astype(np.float32)

            weighted_diffs = []
            for sample in use:
                if gel_start_ns <= sample.timestamp_ns <= baseline_end_ns:
                    image = getattr(sample, f"gelsight_{side}").gray
                    weighted_diffs.append(float(np.mean(
                        np.abs(image.astype(np.float32) - references[side])
                    )))
            diff_offsets[side] = float(np.median(weighted_diffs))

        return OnlineBaseline(
            session_start_ns=session_start_ns,
            baseline_end_ns=baseline_end_ns,
            force_offsets=force_offsets,
            gelsight_references=references,
            gelsight_diff_offsets=diff_offsets,
        )

    def precompute_gelsight(
        self,
        side: str,
        timestamp_ns: int,
        gray: np.ndarray,
        baseline: OnlineBaseline,
    ) -> None:
        """Compute one post-baseline image once, outside the inference timer."""

        if side not in SIDES:
            raise ValueError(f"Unknown GelSight side {side!r}")
        self._cached_gelsight_values(side, timestamp_ns, gray, baseline)

    def extract(
        self,
        samples: list[SyncedSample],
        prediction_time_ns: int,
        baseline: OnlineBaseline,
    ) -> OnlineWindow:
        window_start_ns = prediction_time_ns - int(self.window_sec * 1e9)
        use = [
            sample
            for sample in samples
            if window_start_ns < sample.timestamp_ns <= prediction_time_ns
        ]
        if not use:
            return OnlineWindow(False, {}, 0.0, 0, "empty_window")

        times = np.asarray(
            [
                (sample.timestamp_ns - baseline.session_start_ns) / 1e9
                for sample in use
            ],
            dtype=float,
        )
        signals = self._signals(use, baseline)
        prediction_sec = (prediction_time_ns - baseline.session_start_ns) / 1e9
        features, coverage, count = extract_window(
            times,
            signals,
            prediction_sec,
            self.window_sec,
        )
        if count < self.min_window_records:
            return OnlineWindow(False, features, coverage, count, "too_few_records")
        if coverage < self.min_window_coverage_sec:
            return OnlineWindow(False, features, coverage, count, "insufficient_coverage")
        return OnlineWindow(True, features, coverage, count, "ok")

    def _signals(
        self,
        samples: list[SyncedSample],
        baseline: OnlineBaseline,
    ) -> dict[str, np.ndarray]:
        signals: dict[str, np.ndarray] = {}
        for side in SIDES:
            corrected: dict[str, np.ndarray] = {}
            force_samples: list[ForceSample] = [
                getattr(sample, f"force_{side}") for sample in samples
            ]
            for axis in AXES:
                corrected[axis] = np.asarray(
                    [
                        float(getattr(sample, axis)) - baseline.force_offsets[side][axis]
                        for sample in force_samples
                    ],
                    dtype=float,
                )
            force_norm = np.sqrt(
                corrected["fx"] ** 2 + corrected["fy"] ** 2 + corrected["fz"] ** 2
            )
            torque_norm = np.sqrt(
                corrected["tx"] ** 2 + corrected["ty"] ** 2 + corrected["tz"] ** 2
            )
            outlier = (force_norm > self.force_abs_limit) | (
                torque_norm > self.force_abs_limit
            )
            outlier |= np.any(
                np.vstack(
                    [np.abs(corrected[axis]) > self.force_abs_limit for axis in AXES]
                ),
                axis=0,
            )
            for axis in AXES:
                values = corrected[axis].copy()
                values[outlier] = math.nan
                corrected[axis] = values
                signals[f"{side}_force_{axis}"] = values
            force_norm = np.sqrt(
                corrected["fx"] ** 2 + corrected["fy"] ** 2 + corrected["fz"] ** 2
            )
            torque_norm = np.sqrt(
                corrected["tx"] ** 2 + corrected["ty"] ** 2 + corrected["tz"] ** 2
            )
            signals[f"{side}_force_outlier"] = outlier.astype(float)
            signals[f"{side}_force_norm"] = force_norm
            signals[f"{side}_torque_norm"] = torque_norm
            signals[f"{side}_force_norm_delta"] = force_norm
            signals[f"{side}_torque_norm_delta"] = torque_norm

            gel_values = [[], [], [], [], [], []]
            for sample in samples:
                image_sample = getattr(sample, f"gelsight_{side}")
                values = self._cached_gelsight_values(
                    side,
                    image_sample.timestamp_ns,
                    image_sample.gray,
                    baseline,
                )
                for target, value in zip(gel_values, values, strict=True):
                    target.append(value)
            raw_diff = np.asarray(gel_values[0], dtype=float)
            signals[f"{side}_gelsight_diff_mean"] = (
                raw_diff - baseline.gelsight_diff_offsets[side]
            )
            signals[f"{side}_gelsight_raw_diff_mean"] = raw_diff
            signals[f"{side}_gelsight_contact_area"] = np.asarray(gel_values[1])
            signals[f"{side}_gelsight_center_x"] = np.asarray(gel_values[2])
            signals[f"{side}_gelsight_center_y"] = np.asarray(gel_values[3])
            signals[f"{side}_gelsight_diff_max"] = np.asarray(gel_values[4])
            signals[f"{side}_gelsight_diff_p95"] = np.asarray(gel_values[5])
        return signals

    def _cached_gelsight_values(
        self,
        side: str,
        timestamp_ns: int,
        gray: np.ndarray,
        baseline: OnlineBaseline,
    ) -> tuple[float, float, float, float, float, float]:
        cache = self._gelsight_cache[side]
        lock = self._gelsight_cache_locks[side]
        with lock:
            values = cache.get(timestamp_ns)
            if values is not None:
                cache.move_to_end(timestamp_ns)
                return values

        values = _gelsight_values(
            gray,
            baseline.gelsight_references[side],
            self.pixel_threshold,
        )
        with lock:
            cache[timestamp_ns] = values
            cache.move_to_end(timestamp_ns)
            while len(cache) > 256:
                cache.popitem(last=False)
        return values


def _gelsight_values(
    gray: np.ndarray,
    reference: np.ndarray,
    pixel_threshold: float,
) -> tuple[float, float, float, float, float, float]:
    if gray.shape != reference.shape:
        raise ValueError(
            f"GelSight image shape {gray.shape} differs from baseline {reference.shape}"
        )
    diff = np.abs(gray.astype(np.float32) - reference)
    mask = diff >= pixel_threshold
    if np.any(mask):
        ys, xs = np.nonzero(mask)
        center_x = float(np.mean(xs) / max(mask.shape[1] - 1, 1))
        center_y = float(np.mean(ys) / max(mask.shape[0] - 1, 1))
        diff_max = float(np.max(diff))
        diff_p95 = float(np.percentile(diff, 95))
    else:
        center_x = math.nan
        center_y = math.nan
        diff_max = 0.0
        diff_p95 = 0.0
    return (
        float(np.mean(diff)),
        float(np.mean(mask)),
        center_x,
        center_y,
        diff_max,
        diff_p95,
    )
