"""Shared data loading, phase detection, and causal feature utilities."""
"""Shared data loading, phase detection, and causal feature utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


LEVEL_TO_LABEL = {"low": "too_low", "medium": "fine", "high": "too_high"}
LABELS = ("too_low", "fine", "too_high")
SIDES = ("left", "right")
AXES = ("fx", "fy", "fz", "tx", "ty", "tz")
META_COLUMNS = {
    "sample_id", "episode_id", "episode_dir", "jsonl_path", "object_type",
    "force_level", "label", "trial_number", "phase", "prediction_time_sec",
    "window_start_sec", "window_end_sec",
}


@dataclass(frozen=True)
class Episode:
    jsonl_path: Path
    episode_dir: Path
    episode_id: str
    object_type: str
    force_level: str
    label: str
    trial_number: str


@dataclass(frozen=True)
class Phase:
    contact_start_sec: float | None
    evaluation_time_sec: float
    phase: str
    confidence: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}, line {line_number}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _seconds(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(value):
        return math.nan
    return value / 1e9 if abs(value) > 1e12 else value


def relative_times(records: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray([_seconds(row.get("timestamp")) for row in records], dtype=float)
    finite = values[np.isfinite(values)]
    return values - finite[0] if finite.size else values


def infer_episode(path: Path, root: Path) -> Episode:
    parts = path.resolve().relative_to(root.resolve()).parts
    index = next((i for i, part in enumerate(parts) if part.lower() in LEVEL_TO_LABEL), None)
    if index is None or index == 0:
        raise ValueError(f"Cannot infer object and label from {path}")
    level = parts[index].lower()
    trial = parts[index + 1] if index + 1 < len(parts) else "unknown"
    return Episode(
        path.resolve(), path.resolve().parent, path.resolve().parent.name,
        parts[index - 1].lower(), level, LEVEL_TO_LABEL[level], trial,
    )


def discover_episodes(root: Path) -> list[Episode]:
    root = root.resolve()
    return [infer_episode(path, root) for path in sorted(root.rglob("synced_data.jsonl"))]


def schema_union(records: Iterable[dict[str, Any]], limit: int = 100) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {}
    for index, row in enumerate(records):
        if index >= limit:
            break
        result.setdefault("__root__", set()).update(map(str, row.keys()))
        for key, value in row.items():
            if isinstance(value, dict):
                result.setdefault(str(key), set()).update(map(str, value.keys()))
    return {key: sorted(value) for key, value in sorted(result.items())}


def _number(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if np.isfinite(value) else math.nan


def _image_path(episode_dir: Path, row: dict[str, Any]) -> Path | None:
    value = row.get("image", row.get("path"))
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else episode_dir / path


def _gray(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None


def load_signals(
    episode: Episode,
    baseline_sec: float = 0.75,
    pixel_threshold: float = 8.0,
    include_gelsight: bool = True,
    force_abs_limit: float = 10.0,
    gelsight_baseline_start_sec: float = 0.5,
    gelsight_baseline_end_sec: float = 3.0,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    records = read_jsonl(episode.jsonl_path)
    times = relative_times(records)
    baseline_mask = np.isfinite(times) & (times <= baseline_sec)
    signals: dict[str, np.ndarray] = {}
    for side in SIDES:
        raw_values = {}
        for axis in AXES:
            raw_values[axis] = np.asarray([
                _number(row.get(f"force_{side}", {}).get(axis))
                if isinstance(row.get(f"force_{side}"), dict) else math.nan
                for row in records
            ])
        values = {}
        for axis in AXES:
            baseline_values = raw_values[axis][baseline_mask & np.isfinite(raw_values[axis])]
            baseline = float(np.median(baseline_values)) if baseline_values.size else 0.0
            values[axis] = raw_values[axis] - baseline
        force = np.sqrt(values["fx"] ** 2 + values["fy"] ** 2 + values["fz"] ** 2)
        torque = np.sqrt(values["tx"] ** 2 + values["ty"] ** 2 + values["tz"] ** 2)
        outlier = (force > force_abs_limit) | (torque > force_abs_limit)
        outlier |= np.any(np.vstack([np.abs(values[axis]) > force_abs_limit for axis in AXES]), axis=0)
        for axis in AXES:
            values[axis] = values[axis].copy()
            values[axis][outlier] = math.nan
            signals[f"{side}_force_{axis}"] = values[axis]
        force = np.sqrt(values["fx"] ** 2 + values["fy"] ** 2 + values["fz"] ** 2)
        torque = np.sqrt(values["tx"] ** 2 + values["ty"] ** 2 + values["tz"] ** 2)
        signals[f"{side}_force_outlier"] = outlier.astype(float)
        signals[f"{side}_force_norm"] = force
        signals[f"{side}_torque_norm"] = torque
        signals[f"{side}_force_norm_delta"] = force
        signals[f"{side}_torque_norm_delta"] = torque

    if include_gelsight:
        reference_mask = (
            np.isfinite(times)
            & (times >= gelsight_baseline_start_sec)
            & (times <= gelsight_baseline_end_sec)
        )
        for side in SIDES:
            image_cache: dict[Path, np.ndarray | None] = {}
            reference_images: list[np.ndarray] = []
            seen_reference_paths: set[Path] = set()
            for index, row in enumerate(records):
                if not reference_mask[index]:
                    continue
                gel = row.get(f"gelsight_{side}")
                path = _image_path(episode.episode_dir, gel) if isinstance(gel, dict) else None
                if path is None or path in seen_reference_paths:
                    continue
                seen_reference_paths.add(path)
                image_cache[path] = _gray(path)
                if image_cache[path] is not None:
                    reference_images.append(image_cache[path])
            if reference_images:
                shape = reference_images[0].shape
                aligned = [
                    image if image.shape == shape else np.asarray(
                        Image.fromarray(image).resize((shape[1], shape[0])))
                    for image in reference_images
                ]
                reference = np.median(np.stack(aligned), axis=0).astype(np.float32)
            else:
                reference = None
            cache: dict[Path, tuple[float, float, float, float] | None] = {}
            output = [[], [], [], []]
            for row in records:
                gel = row.get(f"gelsight_{side}")
                path = _image_path(episode.episode_dir, gel) if isinstance(gel, dict) else None
                values: tuple[float, float, float, float]
                if path is None:
                    values = (math.nan,) * 4
                elif path in cache:
                    values = cache[path] or (math.nan,) * 4
                else:
                    if path not in image_cache:
                        image_cache[path] = _gray(path)
                    image = image_cache[path]
                    if image is None:
                        cache[path] = None
                        values = (math.nan,) * 4
                    else:
                        if reference is None:
                            reference = image.astype(np.float32)
                        if image.shape != reference.shape:
                            image = np.asarray(Image.fromarray(image).resize(
                                (reference.shape[1], reference.shape[0])))
                        diff = np.abs(image.astype(np.float32) - reference)
                        mask = diff >= pixel_threshold
                        if np.any(mask):
                            ys, xs = np.nonzero(mask)
                            cx = float(np.mean(xs) / max(mask.shape[1] - 1, 1))
                            cy = float(np.mean(ys) / max(mask.shape[0] - 1, 1))
                        else:
                            cx = cy = math.nan
                        values = (float(np.mean(diff)), float(np.mean(mask)), cx, cy)
                        cache[path] = values
                for target, value in zip(output, values):
                    target.append(value)
            raw_diff = np.asarray(output[0], dtype=float)
            offset_values = raw_diff[reference_mask & np.isfinite(raw_diff)]
            offset = float(np.median(offset_values)) if offset_values.size else 0.0
            signals[f"{side}_gelsight_diff_mean"] = raw_diff - offset
            signals[f"{side}_gelsight_raw_diff_mean"] = raw_diff
            for name, value in zip(("contact_area", "center_x", "center_y"), output[1:]):
                signals[f"{side}_gelsight_{name}"] = np.asarray(value, dtype=float)
    return records, times, signals


def _bilateral(signals: dict[str, np.ndarray], left: str, right: str, size: int) -> np.ndarray:
    a = signals.get(left, np.full(size, np.nan))
    b = signals.get(right, np.full(size, np.nan))
    result = np.full(size, np.nan)
    valid = np.isfinite(a) & np.isfinite(b)
    result[valid] = np.minimum(a[valid], b[valid])
    return result


def _either_side(signals: dict[str, np.ndarray], left: str, right: str, size: int) -> np.ndarray:
    a = signals.get(left, np.full(size, np.nan))
    b = signals.get(right, np.full(size, np.nan))
    result = np.full(size, np.nan)
    only_a = np.isfinite(a) & ~np.isfinite(b)
    only_b = ~np.isfinite(a) & np.isfinite(b)
    both = np.isfinite(a) & np.isfinite(b)
    result[only_a] = a[only_a]
    result[only_b] = b[only_b]
    result[both] = np.maximum(a[both], b[both])
    return result


def _rolling_slope(times: np.ndarray, values: np.ndarray, duration: float) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    for index in np.flatnonzero(np.isfinite(times) & np.isfinite(values)):
        start = np.searchsorted(times, times[index] - duration)
        use = np.arange(start, index + 1)
        use = use[np.isfinite(times[use]) & np.isfinite(values[use])]
        if use.size >= 3 and times[use[-1]] > times[use[0]]:
            result[index] = abs(float(np.polyfit(times[use], values[use], 1)[0]))
    return result


def _rolling_median(times: np.ndarray, values: np.ndarray, duration: float) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    for index in np.flatnonzero(np.isfinite(times) & np.isfinite(values)):
        start = np.searchsorted(times, times[index] - duration)
        window = values[start:index + 1]
        window = window[np.isfinite(window)]
        if window.size:
            result[index] = float(np.median(window))
    return result


def detect_phase(
    times: np.ndarray,
    signals: dict[str, np.ndarray],
    grasp_start_sec: float,
    grasp_end_sec: float,
    force_threshold: float = 0.15,
    gelsight_threshold: float = 2.0,
    stable_slope: float = 0.20,
    stable_sec: float = 0.30,
    evaluation_mode: str = "nominal",
    min_settle_after_contact_sec: float = 0.80,
    fallback_evaluation_sec: float = 3.0,
) -> Phase:
    search = np.isfinite(times) & (times >= grasp_start_sec) & (times <= grasp_end_sec)
    if evaluation_mode not in {"adaptive", "nominal", "settled"}:
        raise ValueError("evaluation_mode must be adaptive, nominal, or settled")
    force = _either_side(signals, "left_force_norm_delta", "right_force_norm_delta", len(times))
    gel = _either_side(signals, "left_gelsight_diff_mean", "right_gelsight_diff_mean", len(times))
    contact = search & (
        (np.isfinite(force) & (force >= force_threshold)) |
        (np.isfinite(gel) & (gel >= gelsight_threshold))
    )
    # Keep fallback endpoints exact: the causal window is (end - 1, end].
    # The feature builder checks actual record coverage and rejects sparse trials.
    end_time = float(grasp_end_sec)
    fallback_time = float(fallback_evaluation_sec)
    indices = np.flatnonzero(contact)
    if not indices.size:
        return Phase(None, fallback_time, "no_contact_fallback", 0.5)
    contact_start = float(times[indices[0]])
    if evaluation_mode == "nominal":
        return Phase(contact_start, end_time, "contact_at_nominal_time", 0.7)
    force_stack = np.vstack([signals["left_force_norm_delta"], signals["right_force_norm_delta"]])
    valid_force = np.isfinite(force_stack).any(axis=0)
    average_force = np.full(len(times), np.nan)
    average_force[valid_force] = np.nanmean(force_stack[:, valid_force], axis=0)
    smoothed_force = _rolling_median(times, average_force, stable_sec)
    stable = contact & (times >= contact_start + min_settle_after_contact_sec)
    stable &= _rolling_slope(times, smoothed_force, stable_sec) <= stable_slope
    candidates = np.flatnonzero(stable)
    if candidates.size:
        if evaluation_mode == "adaptive":
            # First stable plateau is the earliest causal point at which the
            # applied force can be assessed, before later lifting or shaking.
            return Phase(contact_start, float(times[candidates[0]]), "adaptive_settled_contact", 0.8)
        return Phase(contact_start, float(times[candidates[-1]]), "settled_contact", 0.8)
    if evaluation_mode == "adaptive":
        return Phase(contact_start, min(contact_start + min_settle_after_contact_sec, end_time),
            "contact_unsettled_fallback", 0.5)
    return Phase(contact_start, end_time, "contact_without_settle", 0.5)
    


def _summary(times: np.ndarray, values: np.ndarray, prefix: str) -> dict[str, float]:
    valid = np.isfinite(times) & np.isfinite(values)
    array = values[valid]
    if not array.size:
        result = {f"{prefix}_{key}": math.nan for key in
                  ("current", "mean", "std", "min", "max", "range", "p95", "slope")}
        result[f"{prefix}_missing_ratio"] = 1.0
        return result
    slope = math.nan
    if array.size >= 3 and times[valid][-1] > times[valid][0]:
        slope = float(np.polyfit(times[valid], array, 1)[0])
    return {
        f"{prefix}_current": float(array[-1]), f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)), f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_range": float(np.max(array) - np.min(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)), f"{prefix}_slope": slope,
        f"{prefix}_missing_ratio": float(1 - np.mean(valid)),
    }


def extract_window(
    times: np.ndarray, signals: dict[str, np.ndarray], prediction_time: float, window_sec: float,
) -> tuple[dict[str, float], float, int]:
    mask = np.isfinite(times) & (times > prediction_time - window_sec) & (times <= prediction_time)
    selected = times[mask]
    if not selected.size:
        return {}, 0.0, 0
    result = {"window_coverage_sec": float(selected[-1] - selected[0]),
              "window_num_records": float(selected.size)}
    for name, values in signals.items():
        result.update(_summary(selected, values[mask], name))
            # extra gelsight temporal features
    for side in SIDES:
        area = signals.get(f"{side}_gelsight_contact_area")
        if area is not None:
            window_area = area[mask]
            valid = np.isfinite(window_area)
            if np.any(valid):
                result[f"{side}_gelsight_contact_ratio"] = float(np.mean(window_area[valid] > 0.01))
                result[f"{side}_gelsight_area_growth"] = float(window_area[valid][-1] - window_area[valid][0])
                result[f"{side}_gelsight_area_stability"] = float(np.std(window_area[valid]))
                result[f"{side}_gelsight_area_mean"] = float(np.mean(window_area[valid]))
                result[f"{side}_gelsight_area_max"] = float(np.max(window_area[valid]))
            else:
                result[f"{side}_gelsight_contact_ratio"] = math.nan
                result[f"{side}_gelsight_area_growth"] = math.nan
                result[f"{side}_gelsight_area_stability"] = math.nan
                result[f"{side}_gelsight_area_mean"] = math.nan
                result[f"{side}_gelsight_area_max"] = math.nan
    for prefix, left, right in (
        ("force_balance", "left_force_norm_delta", "right_force_norm_delta"),
        ("torque_balance", "left_torque_norm_delta", "right_torque_norm_delta"),
        ("gelsight_diff_balance", "left_gelsight_diff_mean", "right_gelsight_diff_mean"),
        ("gelsight_area_balance", "left_gelsight_contact_area", "right_gelsight_contact_area"),
    ):
        for stat in ("current", "mean", "max", "slope"):
            a, b = result.get(f"{left}_{stat}"), result.get(f"{right}_{stat}")
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                result[f"{prefix}_absdiff_{stat}"] = abs(a - b)
                result[f"{prefix}_ratio_{stat}"] = min(abs(a), abs(b)) / max(abs(a), abs(b), 1e-9)
            else:
                result[f"{prefix}_absdiff_{stat}"] = math.nan
                result[f"{prefix}_ratio_{stat}"] = math.nan
    return result, float(selected[-1] - selected[0]), int(selected.size)


def feature_columns(columns: Iterable[str], modality: str) -> list[str]:
    result = []
    for column in columns:
        if column in META_COLUMNS or column.startswith("window_") or "outlier" in column:
            continue
        if (
            "center_x" in column
            or "center_y" in column
        ):
            continue
        force = "force_" in column or "torque_" in column
        gel = "gelsight_" in column
        if modality == "force" and force and not gel:
            result.append(column)
        elif modality == "gelsight" and gel and not force:
            result.append(column)
        elif modality == "combined" and (force or gel):
            result.append(column)
    return sorted(result)

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


LEVEL_TO_LABEL = {"low": "too_low", "medium": "fine", "high": "too_high"}
LABELS = ("too_low", "fine", "too_high")
SIDES = ("left", "right")
AXES = ("fx", "fy", "fz", "tx", "ty", "tz")
META_COLUMNS = {
    "sample_id", "episode_id", "episode_dir", "jsonl_path", "object_type",
    "force_level", "label", "trial_number", "phase", "prediction_time_sec",
    "window_start_sec", "window_end_sec",
}


@dataclass(frozen=True)
class Episode:
    jsonl_path: Path
    episode_dir: Path
    episode_id: str
    object_type: str
    force_level: str
    label: str
    trial_number: str


@dataclass(frozen=True)
class Phase:
    contact_start_sec: float | None
    evaluation_time_sec: float
    phase: str
    confidence: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}, line {line_number}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _seconds(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(value):
        return math.nan
    return value / 1e9 if abs(value) > 1e12 else value


def relative_times(records: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray([_seconds(row.get("timestamp")) for row in records], dtype=float)
    finite = values[np.isfinite(values)]
    return values - finite[0] if finite.size else values


def infer_episode(path: Path, root: Path) -> Episode:
    parts = path.resolve().relative_to(root.resolve()).parts
    index = next((i for i, part in enumerate(parts) if part.lower() in LEVEL_TO_LABEL), None)
    if index is None or index == 0:
        raise ValueError(f"Cannot infer object and label from {path}")
    level = parts[index].lower()
    trial = parts[index + 1] if index + 1 < len(parts) else "unknown"
    return Episode(
        path.resolve(), path.resolve().parent, path.resolve().parent.name,
        parts[index - 1].lower(), level, LEVEL_TO_LABEL[level], trial,
    )


def discover_episodes(root: Path) -> list[Episode]:
    root = root.resolve()
    return [infer_episode(path, root) for path in sorted(root.rglob("synced_data.jsonl"))]


def schema_union(records: Iterable[dict[str, Any]], limit: int = 100) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {}
    for index, row in enumerate(records):
        if index >= limit:
            break
        result.setdefault("__root__", set()).update(map(str, row.keys()))
        for key, value in row.items():
            if isinstance(value, dict):
                result.setdefault(str(key), set()).update(map(str, value.keys()))
    return {key: sorted(value) for key, value in sorted(result.items())}


def _number(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if np.isfinite(value) else math.nan


def _image_path(episode_dir: Path, row: dict[str, Any]) -> Path | None:
    value = row.get("image", row.get("path"))
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.is_absolute() else episode_dir / path


def _gray(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None


def load_signals(
    episode: Episode,
    baseline_sec: float = 0.75,
    pixel_threshold: float = 8.0,
    include_gelsight: bool = True,
    force_abs_limit: float = 10.0,
    gelsight_baseline_start_sec: float = 0.5,
    gelsight_baseline_end_sec: float = 3.0,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    records = read_jsonl(episode.jsonl_path)
    times = relative_times(records)
    baseline_mask = np.isfinite(times) & (times <= baseline_sec)
    signals: dict[str, np.ndarray] = {}
    
    # 1. Force / Torque Signals
    for side in SIDES:
        raw_values = {}
        for axis in AXES:
            raw_values[axis] = np.asarray([
                _number(row.get(f"force_{side}", {}).get(axis))
                if isinstance(row.get(f"force_{side}"), dict) else math.nan
                for row in records
            ])
        values = {}
        for axis in AXES:
            baseline_values = raw_values[axis][baseline_mask & np.isfinite(raw_values[axis])]
            baseline = float(np.median(baseline_values)) if baseline_values.size else 0.0
            values[axis] = raw_values[axis] - baseline
        force = np.sqrt(values["fx"] ** 2 + values["fy"] ** 2 + values["fz"] ** 2)
        torque = np.sqrt(values["tx"] ** 2 + values["ty"] ** 2 + values["tz"] ** 2)
        outlier = (force > force_abs_limit) | (torque > force_abs_limit)
        outlier |= np.any(np.vstack([np.abs(values[axis]) > force_abs_limit for axis in AXES]), axis=0)
        for axis in AXES:
            values[axis] = values[axis].copy()
            values[axis][outlier] = math.nan
            signals[f"{side}_force_{axis}"] = values[axis]
        force = np.sqrt(values["fx"] ** 2 + values["fy"] ** 2 + values["fz"] ** 2)
        torque = np.sqrt(values["tx"] ** 2 + values["ty"] ** 2 + values["tz"] ** 2)
        signals[f"{side}_force_outlier"] = outlier.astype(float)
        signals[f"{side}_force_norm"] = force
        signals[f"{side}_torque_norm"] = torque
        signals[f"{side}_force_norm_delta"] = force
        signals[f"{side}_torque_norm_delta"] = torque

    # 2. GelSight Signals (优化：增加 diff_max 和 diff_p95)
    if include_gelsight:
        reference_mask = (
            np.isfinite(times)
            & (times >= gelsight_baseline_start_sec)
            & (times <= gelsight_baseline_end_sec)
        )
        for side in SIDES:
            image_cache: dict[Path, np.ndarray | None] = {}
            reference_images: list[np.ndarray] = []
            seen_reference_paths: set[Path] = set()
            for index, row in enumerate(records):
                if not reference_mask[index]:
                    continue
                gel = row.get(f"gelsight_{side}")
                path = _image_path(episode.episode_dir, gel) if isinstance(gel, dict) else None
                if path is None or path in seen_reference_paths:
                    continue
                seen_reference_paths.add(path)
                image_cache[path] = _gray(path)
                if image_cache[path] is not None:
                    reference_images.append(image_cache[path])
            if reference_images:
                shape = reference_images[0].shape
                aligned = [
                    image if image.shape == shape else np.asarray(
                        Image.fromarray(image).resize((shape[1], shape[0])))
                    for image in reference_images
                ]
                reference = np.median(np.stack(aligned), axis=0).astype(np.float32)
            else:
                reference = None
                
            cache: dict[Path, tuple[float, float, float, float, float, float] | None] = {}
            output = [[], [], [], [], [], []]  # diff_mean, contact_area, cx, cy, diff_max, diff_p95
            
            for row in records:
                gel = row.get(f"gelsight_{side}")
                path = _image_path(episode.episode_dir, gel) if isinstance(gel, dict) else None
                values: tuple[float, float, float, float, float, float]
                if path is None:
                    values = (math.nan,) * 6
                elif path in cache:
                    values = cache[path] or (math.nan,) * 6
                else:
                    if path not in image_cache:
                        image_cache[path] = _gray(path)
                    image = image_cache[path]
                    if image is None:
                        cache[path] = None
                        values = (math.nan,) * 6
                    else:
                        if reference is None:
                            reference = image.astype(np.float32)
                        if image.shape != reference.shape:
                            image = np.asarray(Image.fromarray(image).resize(
                                (reference.shape[1], reference.shape[0])))
                        diff = np.abs(image.astype(np.float32) - reference)
                        mask = diff >= pixel_threshold
                        if np.any(mask):
                            ys, xs = np.nonzero(mask)
                            cx = float(np.mean(xs) / max(mask.shape[1] - 1, 1))
                            cy = float(np.mean(ys) / max(mask.shape[0] - 1, 1))
                            diff_max = float(np.max(diff))
                            diff_p95 = float(np.percentile(diff, 95))
                        else:
                            cx = cy = math.nan
                            diff_max = 0.0
                            diff_p95 = 0.0
                        values = (float(np.mean(diff)), float(np.mean(mask)), cx, cy, diff_max, diff_p95)
                        cache[path] = values
                for target, value in zip(output, values):
                    target.append(value)
                    
            raw_diff = np.asarray(output[0], dtype=float)
            offset_values = raw_diff[reference_mask & np.isfinite(raw_diff)]
            offset = float(np.median(offset_values)) if offset_values.size else 0.0
            signals[f"{side}_gelsight_diff_mean"] = raw_diff - offset
            signals[f"{side}_gelsight_raw_diff_mean"] = raw_diff
            signals[f"{side}_gelsight_contact_area"] = np.asarray(output[1], dtype=float)
            signals[f"{side}_gelsight_center_x"] = np.asarray(output[2], dtype=float)
            signals[f"{side}_gelsight_center_y"] = np.asarray(output[3], dtype=float)
            signals[f"{side}_gelsight_diff_max"] = np.asarray(output[4], dtype=float)
            signals[f"{side}_gelsight_diff_p95"] = np.asarray(output[5], dtype=float)

    return records, times, signals


def _either_side(signals: dict[str, np.ndarray], left: str, right: str, size: int) -> np.ndarray:
    a = signals.get(left, np.full(size, np.nan))
    b = signals.get(right, np.full(size, np.nan))
    result = np.full(size, np.nan)
    only_a = np.isfinite(a) & ~np.isfinite(b)
    only_b = ~np.isfinite(a) & np.isfinite(b)
    both = np.isfinite(a) & np.isfinite(b)
    result[only_a] = a[only_a]
    result[only_b] = b[only_b]
    result[both] = np.maximum(a[both], b[both])
    return result


def _rolling_slope(times: np.ndarray, values: np.ndarray, duration: float) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    for index in np.flatnonzero(np.isfinite(times) & np.isfinite(values)):
        start = np.searchsorted(times, times[index] - duration)
        use = np.arange(start, index + 1)
        use = use[np.isfinite(times[use]) & np.isfinite(values[use])]
        if use.size >= 3 and times[use[-1]] > times[use[0]]:
            result[index] = abs(float(np.polyfit(times[use], values[use], 1)[0]))
    return result


def _rolling_median(times: np.ndarray, values: np.ndarray, duration: float) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    for index in np.flatnonzero(np.isfinite(times) & np.isfinite(values)):
        start = np.searchsorted(times, times[index] - duration)
        window = values[start:index + 1]
        window = window[np.isfinite(window)]
        if window.size:
            result[index] = float(np.median(window))
    return result


def detect_phase(
    times: np.ndarray,
    signals: dict[str, np.ndarray],
    grasp_start_sec: float,
    grasp_end_sec: float,
    force_threshold: float = 0.15,
    gelsight_threshold: float = 2.0,
    stable_slope: float = 0.20,
    stable_sec: float = 0.30,
    evaluation_mode: str = "nominal",
    min_settle_after_contact_sec: float = 0.80,
    fallback_evaluation_sec: float = 3.0,
) -> Phase:
    search = np.isfinite(times) & (times >= grasp_start_sec) & (times <= grasp_end_sec)
    if evaluation_mode not in {"adaptive", "nominal", "settled"}:
        raise ValueError("evaluation_mode must be adaptive, nominal, or settled")
    force = _either_side(signals, "left_force_norm_delta", "right_force_norm_delta", len(times))
    gel = _either_side(signals, "left_gelsight_diff_mean", "right_gelsight_diff_mean", len(times))
    contact = search & (
        (np.isfinite(force) & (force >= force_threshold)) |
        (np.isfinite(gel) & (gel >= gelsight_threshold))
    )
    end_time = float(grasp_end_sec)
    fallback_time = float(fallback_evaluation_sec)
    indices = np.flatnonzero(contact)
    if not indices.size:
        return Phase(None, fallback_time, "no_contact_fallback", 0.5)
    contact_start = float(times[indices[0]])
    if evaluation_mode == "nominal":
        return Phase(contact_start, end_time, "contact_at_nominal_time", 0.7)

    # 优化点：Phase Detection 结合了 Force 与 GelSight 的稳定性评估
    force_stack = np.vstack([signals["left_force_norm_delta"], signals["right_force_norm_delta"]])
    valid_force = np.isfinite(force_stack).any(axis=0)
    average_force = np.full(len(times), np.nan)
    average_force[valid_force] = np.nanmean(force_stack[:, valid_force], axis=0)
    
    gel_area_stack = np.vstack([
        signals.get("left_gelsight_contact_area", np.full(len(times), np.nan)),
        signals.get("right_gelsight_contact_area", np.full(len(times), np.nan))
    ])
    valid_gel = np.isfinite(gel_area_stack).any(axis=0)
    average_gel_area = np.full(len(times), np.nan)
    average_gel_area[valid_gel] = np.nanmean(gel_area_stack[:, valid_gel], axis=0)

    smoothed_force = _rolling_median(times, average_force, stable_sec)
    smoothed_gel = _rolling_median(times, average_gel_area, stable_sec)

    stable = contact & (times >= contact_start + min_settle_after_contact_sec)
    force_is_stable = _rolling_slope(times, smoothed_force, stable_sec) <= stable_slope
    gel_is_stable = _rolling_slope(times, smoothed_gel, stable_sec) <= 0.05
    
    # 结合双模态平稳判断（若 GelSight 无效则降级仅看 Force）
    stable &= (force_is_stable & (gel_is_stable | ~np.isfinite(smoothed_gel)))

    candidates = np.flatnonzero(stable)
    if candidates.size:
        if evaluation_mode == "adaptive":
            return Phase(contact_start, float(times[candidates[0]]), "adaptive_settled_contact", 0.8)
        return Phase(contact_start, float(times[candidates[-1]]), "settled_contact", 0.8)
    if evaluation_mode == "adaptive":
        return Phase(contact_start, min(contact_start + min_settle_after_contact_sec, end_time),
            "contact_unsettled_fallback", 0.5)
    return Phase(contact_start, end_time, "contact_without_settle", 0.5)


def _summary(times: np.ndarray, values: np.ndarray, prefix: str) -> dict[str, float]:
    valid = np.isfinite(times) & np.isfinite(values)
    array = values[valid]
    if not array.size:
        result = {f"{prefix}_{key}": math.nan for key in
                  ("current", "mean", "std", "min", "max", "range", "p95", "slope")}
        result[f"{prefix}_missing_ratio"] = 1.0
        return result
    slope = math.nan
    if array.size >= 3 and times[valid][-1] > times[valid][0]:
        slope = float(np.polyfit(times[valid], array, 1)[0])
    return {
        f"{prefix}_current": float(array[-1]), f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)), f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_range": float(np.max(array) - np.min(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)), f"{prefix}_slope": slope,
        f"{prefix}_missing_ratio": float(1 - np.mean(valid)),
    }


def extract_window(
    times: np.ndarray, signals: dict[str, np.ndarray], prediction_time: float, window_sec: float,
) -> tuple[dict[str, float], float, int]:
    mask = np.isfinite(times) & (times > prediction_time - window_sec) & (times <= prediction_time)
    selected = times[mask]
    if not selected.size:
        return {}, 0.0, 0
    result = {"window_coverage_sec": float(selected[-1] - selected[0]),
              "window_num_records": float(selected.size)}
    
    for name, values in signals.items():
        result.update(_summary(selected, values[mask], name))

    # 优化点：更丰富且严谨的 GelSight 空间与动态特征提取
    for side in SIDES:
        area = signals.get(f"{side}_gelsight_contact_area")
        if area is not None:
            window_area = area[mask]
            valid = np.isfinite(window_area)
            if np.any(valid):
                result[f"{side}_gelsight_contact_ratio"] = float(np.mean(window_area[valid] > 0.01))
                result[f"{side}_gelsight_area_growth"] = float(window_area[valid][-1] - window_area[valid][0])
                result[f"{side}_gelsight_area_stability"] = float(np.std(window_area[valid]))
                result[f"{side}_gelsight_area_mean"] = float(np.mean(window_area[valid]))
                result[f"{side}_gelsight_area_max"] = float(np.max(window_area[valid]))
            else:
                result[f"{side}_gelsight_contact_ratio"] = math.nan
                result[f"{side}_gelsight_area_growth"] = math.nan
                result[f"{side}_gelsight_area_stability"] = math.nan
                result[f"{side}_gelsight_area_mean"] = math.nan
                result[f"{side}_gelsight_area_max"] = math.nan
                
        # 优化点：计算微滑移速度 (Slip Rate / Centroid Displacement Velocity)
        cx = signals.get(f"{side}_gelsight_center_x")
        cy = signals.get(f"{side}_gelsight_center_y")
        if cx is not None and cy is not None:
            window_cx, window_cy = cx[mask], cy[mask]
            valid_c = np.isfinite(window_cx) & np.isfinite(window_cy)
            if np.sum(valid_c) >= 2:
                dt = selected[valid_c][-1] - selected[valid_c][0]
                dt = max(dt, 1e-3)
                dx = window_cx[valid_c][-1] - window_cx[valid_c][0]
                dy = window_cy[valid_c][-1] - window_cy[valid_c][0]
                result[f"{side}_gelsight_slip_rate"] = float(np.sqrt(dx**2 + dy**2) / dt)
            else:
                result[f"{side}_gelsight_slip_rate"] = math.nan

    # 左右侧对称指标计算
    for prefix, left, right in (
        ("force_balance", "left_force_norm_delta", "right_force_norm_delta"),
        ("torque_balance", "left_torque_norm_delta", "right_torque_norm_delta"),
        ("gelsight_diff_balance", "left_gelsight_diff_mean", "right_gelsight_diff_mean"),
        ("gelsight_area_balance", "left_gelsight_contact_area", "right_gelsight_contact_area"),
    ):
        for stat in ("current", "mean", "max", "slope"):
            a, b = result.get(f"{left}_{stat}"), result.get(f"{right}_{stat}")
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                result[f"{prefix}_absdiff_{stat}"] = abs(a - b)
                result[f"{prefix}_ratio_{stat}"] = min(abs(a), abs(b)) / max(abs(a), abs(b), 1e-9)
            else:
                result[f"{prefix}_absdiff_{stat}"] = math.nan
                result[f"{prefix}_ratio_{stat}"] = math.nan
    return result, float(selected[-1] - selected[0]), int(selected.size)


def feature_columns(columns: Iterable[str], modality: str) -> list[str]:
    """优化点：取消对 center_x / center_y 的硬编码排除，允许更丰富模态加入。"""
    result = []
    for column in columns:
        if column in META_COLUMNS or column.startswith("window_") or "outlier" in column:
            continue
        force = "force_" in column or "torque_" in column
        gel = "gelsight_" in column
        if modality == "force" and force and not gel:
            result.append(column)
        elif modality == "gelsight" and gel and not force:
            result.append(column)
        elif modality == "combined" and (force or gel):
            result.append(column)
    return sorted(result)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)