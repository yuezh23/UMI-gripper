"""Build trial-level features from synchronized UMI JSONL logs.

Expected layout:

data_yue/pilot/plastic_bottle/
    metadata.csv
    low/episodes/episode_xxx/synced_data.jsonl
    medium/episodes/episode_xxx/synced_data.jsonl
    high/episodes/episode_xxx/synced_data.jsonl

Each synced JSONL line should contain:
    {
      "timestamp": 1783927746490132480,
      "force_left": {"stamp": ..., "fx": ..., "fy": ..., "fz": ..., "tx": ..., "ty": ..., "tz": ...},
      "force_right": {...},
      "gelsight_left": {"timestamp": ..., "image": "images/gelsight_left/00000000.png"},
      "gelsight_right": {"timestamp": ..., "image": "images/gelsight_right/00000000.png"}
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


FORCE_COLUMNS = ("fx", "fy", "fz", "tx", "ty", "tz")
FINAL_LABELS = {"no_change", "too_low", "too_high"}
SIDES = ("left", "right")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
    return records


def resolve_path(base_dir: Path, value: Any) -> Path | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base_dir / path


def timestamps_to_seconds(values: Any) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64)
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size == 0:
        return timestamps
    if np.nanmedian(np.abs(finite)) > 1e12:
        return timestamps / 1e9
    return timestamps


def numeric_array(records: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [record.get(field, 0.0) for record in records]
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)


def read_grayscale_image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.uint8)
    except FileNotFoundError:
        return None
    except OSError:
        return None


def summarize_signal(values: np.ndarray, prefix: str, name: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_{name}_mean": 0.0,
            f"{prefix}_{name}_std": 0.0,
            f"{prefix}_{name}_min": 0.0,
            f"{prefix}_{name}_max": 0.0,
            f"{prefix}_{name}_range": 0.0,
        }
    return {
        f"{prefix}_{name}_mean": float(np.mean(values)),
        f"{prefix}_{name}_std": float(np.std(values)),
        f"{prefix}_{name}_min": float(np.min(values)),
        f"{prefix}_{name}_max": float(np.max(values)),
        f"{prefix}_{name}_range": float(np.max(values) - np.min(values)),
    }


def summarize_drop_after_peak(values: np.ndarray, prefix: str, name: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_{name}_drop_after_peak": 0.0,
            f"{prefix}_{name}_drop_ratio": 0.0,
            f"{prefix}_{name}_end_minus_start": 0.0,
        }

    peak = float(np.max(values))
    end = float(values[-1])
    start = float(values[0])
    drop = max(0.0, peak - end)
    return {
        f"{prefix}_{name}_drop_after_peak": drop,
        f"{prefix}_{name}_drop_ratio": drop / max(abs(peak), 1e-6),
        f"{prefix}_{name}_end_minus_start": end - start,
    }


def summarize_velocity(
    timestamps: np.ndarray,
    values: np.ndarray,
    prefix: str,
    name: str,
) -> dict[str, float]:
    if timestamps.size < 2 or values.size < 2:
        return {
            f"{prefix}_{name}_velocity_mean": 0.0,
            f"{prefix}_{name}_velocity_std": 0.0,
            f"{prefix}_{name}_velocity_max_abs": 0.0,
        }

    dt = np.diff(timestamps)
    dv = np.diff(values)
    valid = np.isfinite(dt) & np.isfinite(dv) & (dt > 1e-9)
    if not valid.any():
        return {
            f"{prefix}_{name}_velocity_mean": 0.0,
            f"{prefix}_{name}_velocity_std": 0.0,
            f"{prefix}_{name}_velocity_max_abs": 0.0,
        }

    velocity = dv[valid] / dt[valid]
    return {
        f"{prefix}_{name}_velocity_mean": float(np.mean(velocity)),
        f"{prefix}_{name}_velocity_std": float(np.std(velocity)),
        f"{prefix}_{name}_velocity_max_abs": float(np.max(np.abs(velocity))),
    }


def summarize_half_window(values: np.ndarray, prefix: str, name: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_{name}_first_half_mean": 0.0,
            f"{prefix}_{name}_second_half_mean": 0.0,
            f"{prefix}_{name}_second_to_first_ratio": 0.0,
        }

    split = max(1, values.size // 2)
    first = values[:split]
    second = values[split:] if split < values.size else values[split - 1 :]
    first_mean = float(np.mean(first)) if first.size else 0.0
    second_mean = float(np.mean(second)) if second.size else 0.0
    return {
        f"{prefix}_{name}_first_half_mean": first_mean,
        f"{prefix}_{name}_second_half_mean": second_mean,
        f"{prefix}_{name}_second_to_first_ratio": second_mean / max(abs(first_mean), 1e-6),
    }


def longest_active_run_sec(timestamps: np.ndarray, active: np.ndarray) -> float:
    if timestamps.size == 0 or active.size == 0:
        return 0.0

    timestamps = np.asarray(timestamps, dtype=float)
    active = np.asarray(active, dtype=bool)
    n = min(timestamps.size, active.size)
    timestamps = timestamps[:n]
    active = active[:n]

    best = 0.0
    run_start: float | None = None
    last_active_time: float | None = None

    for timestamp, is_active in zip(timestamps, active):
        if not np.isfinite(timestamp):
            continue
        if is_active:
            if run_start is None:
                run_start = float(timestamp)
            last_active_time = float(timestamp)
        elif run_start is not None and last_active_time is not None:
            best = max(best, last_active_time - run_start)
            run_start = None
            last_active_time = None

    if run_start is not None and last_active_time is not None:
        best = max(best, last_active_time - run_start)
    return float(max(best, 0.0))


def active_features(
    timestamps: np.ndarray,
    values: np.ndarray,
    prefix: str,
    name: str,
    margin: float,
    baseline_points: int,
) -> dict[str, float]:
    if timestamps.size == 0 or values.size == 0:
        return {
            f"{prefix}_{name}_active_ratio": 0.0,
            f"{prefix}_{name}_active_duration_sec": 0.0,
            f"{prefix}_{name}_longest_active_run_sec": 0.0,
        }

    n = min(timestamps.size, values.size)
    timestamps = timestamps[:n]
    values = values[:n]
    baseline_n = max(1, min(baseline_points, n))
    baseline = values[:baseline_n]
    threshold = float(np.mean(baseline) + max(margin, 3.0 * np.std(baseline)))
    active = values > threshold
    duration = max(float(timestamps[-1] - timestamps[0]), 1e-6) if n >= 2 else 0.0
    active_ratio = float(np.mean(active)) if active.size else 0.0
    return {
        f"{prefix}_{name}_active_ratio": active_ratio,
        f"{prefix}_{name}_active_duration_sec": active_ratio * duration,
        f"{prefix}_{name}_longest_active_run_sec": longest_active_run_sec(timestamps, active),
    }


def binary_contact_features(
    timestamps: np.ndarray,
    active: np.ndarray,
    prefix: str,
    name: str,
) -> dict[str, float | int]:
    if timestamps.size == 0 or active.size == 0:
        return {
            f"{prefix}_{name}_contact_detected": 0,
            f"{prefix}_{name}_contact_ratio": 0.0,
            f"{prefix}_{name}_contact_duration_sec": 0.0,
            f"{prefix}_{name}_longest_contact_run_sec": 0.0,
        }

    n = min(timestamps.size, active.size)
    timestamps = timestamps[:n]
    active = np.asarray(active[:n], dtype=bool)
    duration = max(float(timestamps[-1] - timestamps[0]), 1e-6) if n >= 2 else 0.0
    contact_ratio = float(np.mean(active)) if active.size else 0.0
    return {
        f"{prefix}_{name}_contact_detected": int(active.any()),
        f"{prefix}_{name}_contact_ratio": contact_ratio,
        f"{prefix}_{name}_contact_duration_sec": contact_ratio * duration,
        f"{prefix}_{name}_longest_contact_run_sec": longest_active_run_sec(timestamps, active),
    }


def extract_force_features(records: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    if not records:
        return {
            f"{prefix}_num_force_samples": 0,
            f"{prefix}_force_duration_sec": 0.0,
            f"{prefix}_force_sample_rate_hz": 0.0,
        }

    stamps = timestamps_to_seconds([record.get("stamp", np.nan) for record in records])
    valid = np.isfinite(stamps)
    records = [record for record, is_valid in zip(records, valid) if is_valid]
    stamps = stamps[valid]
    if stamps.size == 0:
        return {
            f"{prefix}_num_force_samples": 0,
            f"{prefix}_force_duration_sec": 0.0,
            f"{prefix}_force_sample_rate_hz": 0.0,
        }

    duration = max(float(stamps[-1] - stamps[0]), 1e-6)

    fx = numeric_array(records, "fx")
    fy = numeric_array(records, "fy")
    fz = numeric_array(records, "fz")
    tx = numeric_array(records, "tx")
    ty = numeric_array(records, "ty")
    tz = numeric_array(records, "tz")

    normal_force = np.abs(fz)
    tangential_force = np.sqrt(fx**2 + fy**2)
    force_norm = np.sqrt(fx**2 + fy**2 + fz**2)
    torque_norm = np.sqrt(tx**2 + ty**2 + tz**2)

    force_slope = 0.0
    if normal_force.size >= 2:
        force_slope = float((normal_force[-1] - normal_force[0]) / duration)

    baseline_count = max(3, min(20, len(force_norm) // 10))
    force_baseline = float(np.median(force_norm[:baseline_count])) if force_norm.size else 0.0
    normal_baseline = float(np.median(normal_force[:baseline_count])) if normal_force.size else 0.0
    force_delta = np.abs(force_norm - force_baseline)
    normal_delta = np.abs(normal_force - normal_baseline)

    features: dict[str, float | int] = {
        f"{prefix}_num_force_samples": int(len(records)),
        f"{prefix}_force_duration_sec": duration,
        f"{prefix}_force_sample_rate_hz": float(len(records) / duration),
        f"{prefix}_normal_force_slope": force_slope,
        f"{prefix}_force_norm_baseline": force_baseline,
        f"{prefix}_normal_force_baseline": normal_baseline,
    }
    for name, values in {
        "normal_force": normal_force,
        "tangential_force": tangential_force,
        "force_norm": force_norm,
        "torque_norm": torque_norm,
        "fx": fx,
        "fy": fy,
        "fz": fz,
        "tx": tx,
        "ty": ty,
        "tz": tz,
    }.items():
        features.update(summarize_signal(values, prefix, name))

    for name, values in {
        "normal_force": normal_force,
        "force_norm": force_norm,
        "torque_norm": torque_norm,
    }.items():
        features.update(summarize_drop_after_peak(values, prefix, name))
        features.update(summarize_velocity(stamps, values, prefix, name))
        features.update(summarize_half_window(values, prefix, name))

    for name, values in {
        "normal_force_delta": normal_delta,
        "force_norm_delta": force_delta,
    }.items():
        features.update(summarize_signal(values, prefix, name))
        features.update(summarize_half_window(values, prefix, name))
        features.update(
            active_features(
                timestamps=stamps,
                values=values,
                prefix=prefix,
                name=name,
                margin=0.25,
                baseline_points=20,
            )
        )

    loose_force_contact = force_delta > 0.15
    strict_force_contact = force_delta > 0.25
    loose_normal_contact = normal_delta > 0.15
    strict_normal_contact = normal_delta > 0.25
    features.update(
        binary_contact_features(stamps, loose_force_contact, prefix, "force_norm_delta_loose")
    )
    features.update(
        binary_contact_features(stamps, strict_force_contact, prefix, "force_norm_delta_strict")
    )
    features.update(
        binary_contact_features(stamps, loose_normal_contact, prefix, "normal_force_delta_loose")
    )
    features.update(
        binary_contact_features(stamps, strict_normal_contact, prefix, "normal_force_delta_strict")
    )
    return features


def unique_gelsight_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for record in records:
        key = (record.get("timestamp"), record.get("image"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def gelsight_diff_series(
    records: list[dict[str, Any]],
    trial_dir: Path,
) -> list[tuple[float, float]]:
    records = unique_gelsight_records(records)
    if not records:
        return []

    timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in records])
    image_paths = [resolve_path(trial_dir, record.get("image")) for record in records]
    baseline = None
    series: list[tuple[float, float]] = []

    for timestamp, image_path in zip(timestamps, image_paths):
        if not np.isfinite(timestamp) or image_path is None:
            continue
        img = read_grayscale_image(image_path)
        if img is None:
            continue
        if baseline is None:
            baseline = img
        if img.shape != baseline.shape:
            img = np.asarray(Image.fromarray(img).resize((baseline.shape[1], baseline.shape[0])))
        diff = np.abs(img.astype(np.int16) - baseline.astype(np.int16))
        series.append((float(timestamp), float(np.mean(diff))))
    return series


def force_delta_series(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
    if not records:
        return []

    stamps = timestamps_to_seconds([record.get("stamp", np.nan) for record in records])
    valid = np.isfinite(stamps)
    records = [record for record, is_valid in zip(records, valid) if is_valid]
    stamps = stamps[valid]
    if stamps.size == 0:
        return []

    fx = numeric_array(records, "fx")
    fy = numeric_array(records, "fy")
    fz = numeric_array(records, "fz")
    force_norm = np.sqrt(fx**2 + fy**2 + fz**2)
    baseline_count = max(3, min(20, len(force_norm) // 10))
    baseline = float(np.median(force_norm[:baseline_count]))
    deltas = np.abs(force_norm - baseline)
    return [(float(timestamp), float(delta)) for timestamp, delta in zip(stamps, deltas)]


def force_norm(record: dict[str, Any]) -> float:
    fx = float(record.get("fx", 0.0) or 0.0)
    fy = float(record.get("fy", 0.0) or 0.0)
    fz = float(record.get("fz", 0.0) or 0.0)
    return float(np.sqrt(fx**2 + fy**2 + fz**2))


def stable_contact_features(
    timestamps: np.ndarray,
    active: np.ndarray,
    prefix: str,
    name: str,
    stable_sec: float = 2.0,
) -> dict[str, float | int]:
    features = binary_contact_features(timestamps, active, prefix, name)
    longest_key = f"{prefix}_{name}_longest_contact_run_sec"
    features[f"{prefix}_{name}_stable_{stable_sec:g}s_detected"] = int(
        float(features[longest_key]) >= stable_sec
    )
    return features


def simultaneous_force_contact_features(
    synced: list[dict[str, Any]],
    loose_threshold: float = 0.15,
    strict_threshold: float = 0.25,
    baseline_points: int = 20,
) -> dict[str, float | int]:
    timestamps: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []

    for record in synced:
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        left_record = record.get("force_left")
        right_record = record.get("force_right")
        if (
            np.isfinite(timestamp)
            and isinstance(left_record, dict)
            and isinstance(right_record, dict)
        ):
            timestamps.append(float(timestamp))
            left_values.append(force_norm(left_record))
            right_values.append(force_norm(right_record))

    if not timestamps:
        empty = np.asarray([], dtype=float)
        features: dict[str, float | int] = {}
        for name in ("simultaneous_delta_loose", "simultaneous_delta_strict"):
            features.update(stable_contact_features(empty, empty.astype(bool), "both_force", name))
        return features

    ts = np.asarray(timestamps, dtype=float)
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    baseline_n = max(3, min(baseline_points, len(ts) // 4 if len(ts) >= 4 else len(ts)))
    left_baseline = float(np.median(left[:baseline_n]))
    right_baseline = float(np.median(right[:baseline_n]))
    left_delta = np.abs(left - left_baseline)
    right_delta = np.abs(right - right_baseline)
    min_delta = np.minimum(left_delta, right_delta)
    max_delta = np.maximum(left_delta, right_delta)
    balance_ratio = min_delta / np.maximum(max_delta, 1e-6)

    left_loose = left_delta > loose_threshold
    right_loose = right_delta > loose_threshold
    left_strict = left_delta > strict_threshold
    right_strict = right_delta > strict_threshold
    both_loose = left_loose & right_loose
    both_strict = left_strict & right_strict

    features = {
        "both_force_simultaneous_left_baseline": left_baseline,
        "both_force_simultaneous_right_baseline": right_baseline,
        "both_force_simultaneous_delta_min_mean": float(np.mean(min_delta)),
        "both_force_simultaneous_delta_min_max": float(np.max(min_delta)),
        "both_force_simultaneous_delta_balance_ratio_mean": float(np.mean(balance_ratio)),
        "both_force_simultaneous_delta_balance_ratio_min": float(np.min(balance_ratio)),
        "both_force_simultaneous_loose_one_sided_contact_ratio": float(
            np.mean(left_loose ^ right_loose)
        ),
        "both_force_simultaneous_strict_one_sided_contact_ratio": float(
            np.mean(left_strict ^ right_strict)
        ),
    }
    features.update(
        stable_contact_features(ts, both_loose, "both_force", "simultaneous_delta_loose")
    )
    features.update(
        stable_contact_features(ts, both_strict, "both_force", "simultaneous_delta_strict")
    )
    return features


def mean_gelsight_diff(
    record: dict[str, Any],
    trial_dir: Path,
    baseline: np.ndarray | None,
    image_cache: dict[Path, np.ndarray | None],
) -> tuple[float | None, np.ndarray | None]:
    image_path = resolve_path(trial_dir, record.get("image"))
    if image_path is None:
        return None, baseline

    if image_path not in image_cache:
        image_cache[image_path] = read_grayscale_image(image_path)
    img = image_cache[image_path]
    if img is None:
        return None, baseline

    if baseline is None:
        baseline = img
    if img.shape != baseline.shape:
        img = np.asarray(Image.fromarray(img).resize((baseline.shape[1], baseline.shape[0])))

    diff = np.abs(img.astype(np.int16) - baseline.astype(np.int16))
    return float(np.mean(diff)), baseline


def simultaneous_gelsight_contact_features(
    synced: list[dict[str, Any]],
    trial_dir: Path,
    loose_delta_threshold: float = 0.10,
    strict_delta_threshold: float = 0.25,
) -> dict[str, float | int]:
    timestamps: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []
    image_cache: dict[Path, np.ndarray | None] = {}
    left_baseline: np.ndarray | None = None
    right_baseline: np.ndarray | None = None

    for record in synced:
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        left_record = record.get("gelsight_left")
        right_record = record.get("gelsight_right")
        if (
            not np.isfinite(timestamp)
            or not isinstance(left_record, dict)
            or not isinstance(right_record, dict)
        ):
            continue

        left_diff, left_baseline = mean_gelsight_diff(
            left_record, trial_dir, left_baseline, image_cache
        )
        right_diff, right_baseline = mean_gelsight_diff(
            right_record, trial_dir, right_baseline, image_cache
        )
        if left_diff is None or right_diff is None:
            continue

        timestamps.append(float(timestamp))
        left_values.append(left_diff)
        right_values.append(right_diff)

    if not timestamps:
        empty = np.asarray([], dtype=float)
        features: dict[str, float | int] = {}
        for name in ("simultaneous_diff_loose", "simultaneous_diff_strict"):
            features.update(stable_contact_features(empty, empty.astype(bool), "both_gelsight", name))
        return features

    ts = np.asarray(timestamps, dtype=float)
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    min_delta = np.minimum(left, right)
    max_delta = np.maximum(left, right)
    balance_ratio = min_delta / np.maximum(max_delta, 1e-6)

    left_loose = left > loose_delta_threshold
    right_loose = right > loose_delta_threshold
    left_strict = left > strict_delta_threshold
    right_strict = right > strict_delta_threshold
    both_loose = left_loose & right_loose
    both_strict = left_strict & right_strict

    features = {
        "both_gelsight_simultaneous_diff_min_mean": float(np.mean(min_delta)),
        "both_gelsight_simultaneous_diff_min_max": float(np.max(min_delta)),
        "both_gelsight_simultaneous_diff_balance_ratio_mean": float(np.mean(balance_ratio)),
        "both_gelsight_simultaneous_diff_balance_ratio_min": float(np.min(balance_ratio)),
        "both_gelsight_simultaneous_loose_one_sided_contact_ratio": float(
            np.mean(left_loose ^ right_loose)
        ),
        "both_gelsight_simultaneous_strict_one_sided_contact_ratio": float(
            np.mean(left_strict ^ right_strict)
        ),
    }
    features.update(
        stable_contact_features(ts, both_loose, "both_gelsight", "simultaneous_diff_loose")
    )
    features.update(
        stable_contact_features(ts, both_strict, "both_gelsight", "simultaneous_diff_strict")
    )
    return features


def force_arrays_from_synced(
    synced: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []

    for record in synced:
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        left_record = record.get("force_left")
        right_record = record.get("force_right")
        if (
            np.isfinite(timestamp)
            and isinstance(left_record, dict)
            and isinstance(right_record, dict)
        ):
            timestamps.append(float(timestamp))
            left_values.append(force_norm(left_record))
            right_values.append(force_norm(right_record))

    return (
        np.asarray(timestamps, dtype=float),
        np.asarray(left_values, dtype=float),
        np.asarray(right_values, dtype=float),
    )


def baseline_from_precontact(
    timestamps: np.ndarray,
    values: np.ndarray,
    contact_start: float | None,
    fallback_points: int,
) -> tuple[float, float]:
    if timestamps.size == 0 or values.size == 0:
        return 0.0, 0.0

    if contact_start is not None and np.isfinite(contact_start):
        baseline_values = values[timestamps < float(contact_start)]
        if baseline_values.size >= 3:
            return float(np.median(baseline_values)), float(np.std(baseline_values))

    n = max(1, min(fallback_points, values.size))
    baseline_values = values[:n]
    return float(np.median(baseline_values)), float(np.std(baseline_values))


def stage_mask(
    timestamps: np.ndarray,
    contact_start: float | None,
    start_offset: float,
    end_offset: float,
) -> np.ndarray:
    if timestamps.size == 0 or contact_start is None or not np.isfinite(contact_start):
        return np.zeros(timestamps.shape, dtype=bool)
    relative = timestamps - float(contact_start)
    return (relative >= start_offset) & (relative < end_offset)


def summarize_stage_binary(
    timestamps: np.ndarray,
    active: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    if timestamps.size == 0 or active.size == 0 or not mask.any():
        return {
            f"{prefix}_ratio": 0.0,
            f"{prefix}_duration_sec": 0.0,
            f"{prefix}_longest_run_sec": 0.0,
            f"{prefix}_detected": 0,
        }

    stage_timestamps = timestamps[mask]
    stage_active = np.asarray(active[mask], dtype=bool)
    duration = (
        max(float(stage_timestamps[-1] - stage_timestamps[0]), 1e-6)
        if stage_timestamps.size >= 2
        else 0.0
    )
    ratio = float(np.mean(stage_active)) if stage_active.size else 0.0
    return {
        f"{prefix}_ratio": ratio,
        f"{prefix}_duration_sec": ratio * duration,
        f"{prefix}_longest_run_sec": longest_active_run_sec(stage_timestamps, stage_active),
        f"{prefix}_detected": int(stage_active.any()),
    }


def summarize_stage_continuous(
    values: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if values.size == 0 or not mask.any():
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_end_minus_start": 0.0,
        }

    stage_values = values[mask]
    if stage_values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_end_minus_start": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(stage_values)),
        f"{prefix}_max": float(np.max(stage_values)),
        f"{prefix}_range": float(np.max(stage_values) - np.min(stage_values)),
        f"{prefix}_end_minus_start": float(stage_values[-1] - stage_values[0]),
    }


def safe_ratio(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-6:
        return default
    return float(num / den)


def add_relative_response_features(features: dict[str, Any]) -> None:
    stages = ("establish_0_1s", "grasp_1_3s", "hold_3_5s")
    for prefix in (
        "effective_force",
        "effective_torque",
        "effective_tx",
        "effective_ty",
        "effective_tz",
    ):
        stage_means = {
            stage: float(features.get(f"{prefix}_{stage}_min_delta_mean", 0.0) or 0.0)
            for stage in stages
        }
        peak_mean = max(stage_means.values())
        hold_mean = stage_means["hold_3_5s"]
        grasp_mean = stage_means["grasp_1_3s"]
        establish_mean = stage_means["establish_0_1s"]
        features[f"{prefix}_hold_to_grasp_min_delta_mean_ratio"] = safe_ratio(
            hold_mean, grasp_mean
        )
        features[f"{prefix}_hold_to_establish_min_delta_mean_ratio"] = safe_ratio(
            hold_mean, establish_mean
        )
        features[f"{prefix}_hold_to_peak_min_delta_mean_ratio"] = safe_ratio(
            hold_mean, peak_mean
        )
        features[f"{prefix}_hold_minus_grasp_min_delta_mean"] = hold_mean - grasp_mean
        features[f"{prefix}_hold_minus_establish_min_delta_mean"] = (
            hold_mean - establish_mean
        )

        for contact_kind in ("both_loose_contact", "both_strict_contact"):
            hold_ratio = float(
                features.get(f"{prefix}_hold_3_5s_{contact_kind}_ratio", 0.0) or 0.0
            )
            grasp_ratio = float(
                features.get(f"{prefix}_grasp_1_3s_{contact_kind}_ratio", 0.0) or 0.0
            )
            establish_ratio = float(
                features.get(f"{prefix}_establish_0_1s_{contact_kind}_ratio", 0.0) or 0.0
            )
            peak_ratio = max(hold_ratio, grasp_ratio, establish_ratio)
            features[f"{prefix}_hold_to_grasp_{contact_kind}_ratio"] = safe_ratio(
                hold_ratio, grasp_ratio
            )
            features[f"{prefix}_hold_to_peak_{contact_kind}_ratio"] = safe_ratio(
                hold_ratio, peak_ratio
            )
            features[f"{prefix}_hold_minus_grasp_{contact_kind}_ratio"] = (
                hold_ratio - grasp_ratio
            )

    gel_stage_measures = ("min_diff", "min_area")
    for measure in gel_stage_measures:
        stage_means = {
            stage: float(
                features.get(f"effective_gelsight_{stage}_{measure}_mean", 0.0) or 0.0
            )
            for stage in stages
        }
        peak_mean = max(stage_means.values())
        hold_mean = stage_means["hold_3_5s"]
        grasp_mean = stage_means["grasp_1_3s"]
        establish_mean = stage_means["establish_0_1s"]
        features[f"effective_gelsight_hold_to_grasp_{measure}_mean_ratio"] = safe_ratio(
            hold_mean, grasp_mean
        )
        features[
            f"effective_gelsight_hold_to_establish_{measure}_mean_ratio"
        ] = safe_ratio(hold_mean, establish_mean)
        features[f"effective_gelsight_hold_to_peak_{measure}_mean_ratio"] = safe_ratio(
            hold_mean, peak_mean
        )
        features[f"effective_gelsight_hold_minus_grasp_{measure}_mean"] = (
            hold_mean - grasp_mean
        )

    left_area_max = float(features.get("left_gelsight_contact_area_max", 0.0) or 0.0)
    right_area_max = float(features.get("right_gelsight_contact_area_max", 0.0) or 0.0)
    area_min_max = min(left_area_max, right_area_max)
    area_max_max = max(left_area_max, right_area_max)
    features["gelsight_contact_area_min_max"] = area_min_max
    features["gelsight_contact_area_max_max"] = area_max_max
    features["gelsight_contact_area_min_to_max_ratio"] = safe_ratio(
        area_min_max, area_max_max
    )
    features["gelsight_contact_area_min_max_log1p"] = float(np.log1p(area_min_max))


def wrench_signal_value(record: dict[str, Any], signal_name: str) -> float:
    fx = float(record.get("fx", 0.0) or 0.0)
    fy = float(record.get("fy", 0.0) or 0.0)
    fz = float(record.get("fz", 0.0) or 0.0)
    tx = float(record.get("tx", 0.0) or 0.0)
    ty = float(record.get("ty", 0.0) or 0.0)
    tz = float(record.get("tz", 0.0) or 0.0)

    if signal_name == "force_norm":
        return float(np.sqrt(fx**2 + fy**2 + fz**2))
    if signal_name == "torque_norm":
        return float(np.sqrt(tx**2 + ty**2 + tz**2))
    if signal_name == "tx":
        return tx
    if signal_name == "ty":
        return ty
    if signal_name == "tz":
        return tz
    raise ValueError(f"Unknown wrench signal: {signal_name}")


def wrench_signal_arrays_from_synced(
    synced: list[dict[str, Any]],
    signal_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []

    for record in synced:
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        left_record = record.get("force_left")
        right_record = record.get("force_right")
        if (
            np.isfinite(timestamp)
            and isinstance(left_record, dict)
            and isinstance(right_record, dict)
        ):
            timestamps.append(float(timestamp))
            left_values.append(wrench_signal_value(left_record, signal_name))
            right_values.append(wrench_signal_value(right_record, signal_name))

    return (
        np.asarray(timestamps, dtype=float),
        np.asarray(left_values, dtype=float),
        np.asarray(right_values, dtype=float),
    )


def dynamic_delta_threshold(
    left_noise: float,
    right_noise: float,
    min_margin: float,
    noise_scale: float,
) -> float:
    noise = max(float(left_noise), float(right_noise))
    return float(max(min_margin, noise_scale * noise))


def effective_bilateral_wrench_signal_features(
    full_synced: list[dict[str, Any]],
    windowed_synced: list[dict[str, Any]],
    contact_start: float | None,
    signal_name: str,
    output_prefix: str,
    loose_min_margin: float,
    strict_min_margin: float,
    loose_noise_scale: float = 3.0,
    strict_noise_scale: float = 5.0,
) -> dict[str, float | int]:
    full_ts, full_left, full_right = wrench_signal_arrays_from_synced(full_synced, signal_name)
    ts, left, right = wrench_signal_arrays_from_synced(windowed_synced, signal_name)
    if ts.size == 0:
        return {}

    left_baseline, left_noise = baseline_from_precontact(full_ts, full_left, contact_start, 20)
    right_baseline, right_noise = baseline_from_precontact(full_ts, full_right, contact_start, 20)
    loose_threshold = dynamic_delta_threshold(
        left_noise, right_noise, loose_min_margin, loose_noise_scale
    )
    strict_threshold = dynamic_delta_threshold(
        left_noise, right_noise, strict_min_margin, strict_noise_scale
    )

    left_delta = np.abs(left - left_baseline)
    right_delta = np.abs(right - right_baseline)
    min_delta = np.minimum(left_delta, right_delta)
    max_delta = np.maximum(left_delta, right_delta)
    balance_ratio = min_delta / np.maximum(max_delta, 1e-6)

    left_loose = left_delta > loose_threshold
    right_loose = right_delta > loose_threshold
    left_strict = left_delta > strict_threshold
    right_strict = right_delta > strict_threshold
    both_loose = left_loose & right_loose
    both_strict = left_strict & right_strict
    one_sided_loose = left_loose ^ right_loose
    one_sided_strict = left_strict ^ right_strict

    features: dict[str, float | int] = {
        f"{output_prefix}_left_precontact_baseline": left_baseline,
        f"{output_prefix}_right_precontact_baseline": right_baseline,
        f"{output_prefix}_left_precontact_noise": left_noise,
        f"{output_prefix}_right_precontact_noise": right_noise,
        f"{output_prefix}_loose_threshold": loose_threshold,
        f"{output_prefix}_strict_threshold": strict_threshold,
        f"{output_prefix}_min_delta_mean": float(np.mean(min_delta)),
        f"{output_prefix}_min_delta_max": float(np.max(min_delta)),
        f"{output_prefix}_balance_ratio_mean": float(np.mean(balance_ratio)),
        f"{output_prefix}_balance_ratio_min": float(np.min(balance_ratio)),
    }

    stages = {
        "establish_0_1s": (0.0, 1.0),
        "grasp_1_3s": (1.0, 3.0),
        "hold_3_5s": (3.0, 5.0),
    }
    for stage_name, (start_offset, end_offset) in stages.items():
        mask = stage_mask(ts, contact_start, start_offset, end_offset)
        features.update(
            summarize_stage_binary(
                ts,
                both_loose,
                mask,
                f"{output_prefix}_{stage_name}_both_loose_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                both_strict,
                mask,
                f"{output_prefix}_{stage_name}_both_strict_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                one_sided_loose,
                mask,
                f"{output_prefix}_{stage_name}_one_sided_loose_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                one_sided_strict,
                mask,
                f"{output_prefix}_{stage_name}_one_sided_strict_contact",
            )
        )
        features.update(
            summarize_stage_continuous(
                min_delta,
                mask,
                f"{output_prefix}_{stage_name}_min_delta",
            )
        )
        features.update(
            summarize_stage_continuous(
                balance_ratio,
                mask,
                f"{output_prefix}_{stage_name}_balance_ratio",
            )
        )

    early_mask = stage_mask(ts, contact_start, 0.0, 1.0)
    hold_mask = stage_mask(ts, contact_start, 3.0, 5.0)
    early_min = float(np.mean(min_delta[early_mask])) if early_mask.any() else 0.0
    hold_min = float(np.mean(min_delta[hold_mask])) if hold_mask.any() else 0.0
    hold_loose_ratio = float(np.mean(both_loose[hold_mask])) if hold_mask.any() else 0.0
    hold_strict_ratio = float(np.mean(both_strict[hold_mask])) if hold_mask.any() else 0.0
    hold_loose_run = (
        longest_active_run_sec(ts[hold_mask], both_loose[hold_mask]) if hold_mask.any() else 0.0
    )
    hold_strict_run = (
        longest_active_run_sec(ts[hold_mask], both_strict[hold_mask]) if hold_mask.any() else 0.0
    )
    features[f"{output_prefix}_hold_to_establish_min_delta_ratio"] = hold_min / max(
        abs(early_min), 1e-6
    )
    features[f"{output_prefix}_hold_both_loose_contact_ratio"] = hold_loose_ratio
    features[f"{output_prefix}_hold_both_strict_contact_ratio"] = hold_strict_ratio
    features[f"{output_prefix}_hold_both_loose_longest_run_sec"] = hold_loose_run
    features[f"{output_prefix}_hold_both_strict_longest_run_sec"] = hold_strict_run
    features[f"{output_prefix}_grasp_established_proxy"] = int(
        hold_loose_ratio >= 0.4 or hold_loose_run >= 1.0
    )
    features[f"{output_prefix}_excessive_proxy"] = int(
        hold_strict_ratio >= 0.5 or hold_strict_run >= 1.0
    )
    return features


def effective_grasp_force_features(
    full_synced: list[dict[str, Any]],
    windowed_synced: list[dict[str, Any]],
    contact_start: float | None,
    loose_threshold: float = 0.15,
    strict_threshold: float = 0.25,
) -> dict[str, float | int]:
    full_ts, full_left, full_right = force_arrays_from_synced(full_synced)
    ts, left, right = force_arrays_from_synced(windowed_synced)
    if ts.size == 0:
        return {}

    left_baseline, left_noise = baseline_from_precontact(full_ts, full_left, contact_start, 20)
    right_baseline, right_noise = baseline_from_precontact(full_ts, full_right, contact_start, 20)
    left_delta = np.abs(left - left_baseline)
    right_delta = np.abs(right - right_baseline)
    min_delta = np.minimum(left_delta, right_delta)
    max_delta = np.maximum(left_delta, right_delta)
    balance_ratio = min_delta / np.maximum(max_delta, 1e-6)

    left_loose = left_delta > loose_threshold
    right_loose = right_delta > loose_threshold
    left_strict = left_delta > strict_threshold
    right_strict = right_delta > strict_threshold
    both_loose = left_loose & right_loose
    both_strict = left_strict & right_strict
    one_sided_strict = left_strict ^ right_strict

    features: dict[str, float | int] = {
        "effective_force_left_precontact_baseline": left_baseline,
        "effective_force_right_precontact_baseline": right_baseline,
        "effective_force_left_precontact_noise": left_noise,
        "effective_force_right_precontact_noise": right_noise,
        "effective_force_min_delta_mean": float(np.mean(min_delta)),
        "effective_force_min_delta_max": float(np.max(min_delta)),
        "effective_force_balance_ratio_mean": float(np.mean(balance_ratio)),
        "effective_force_balance_ratio_min": float(np.min(balance_ratio)),
    }

    stages = {
        "establish_0_1s": (0.0, 1.0),
        "grasp_1_3s": (1.0, 3.0),
        "hold_3_5s": (3.0, 5.0),
    }
    for stage_name, (start_offset, end_offset) in stages.items():
        mask = stage_mask(ts, contact_start, start_offset, end_offset)
        features.update(
            summarize_stage_binary(
                ts,
                both_loose,
                mask,
                f"effective_force_{stage_name}_both_loose_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                both_strict,
                mask,
                f"effective_force_{stage_name}_both_strict_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                one_sided_strict,
                mask,
                f"effective_force_{stage_name}_one_sided_strict_contact",
            )
        )
        features.update(
            summarize_stage_continuous(
                min_delta,
                mask,
                f"effective_force_{stage_name}_min_delta",
            )
        )
        features.update(
            summarize_stage_continuous(
                balance_ratio,
                mask,
                f"effective_force_{stage_name}_balance_ratio",
            )
        )

    early_mask = stage_mask(ts, contact_start, 0.0, 1.0)
    hold_mask = stage_mask(ts, contact_start, 3.0, 5.0)
    early_min = float(np.mean(min_delta[early_mask])) if early_mask.any() else 0.0
    hold_min = float(np.mean(min_delta[hold_mask])) if hold_mask.any() else 0.0
    hold_strict_ratio = float(np.mean(both_strict[hold_mask])) if hold_mask.any() else 0.0
    hold_strict_run = (
        longest_active_run_sec(ts[hold_mask], both_strict[hold_mask]) if hold_mask.any() else 0.0
    )
    features["effective_force_hold_to_establish_min_delta_ratio"] = hold_min / max(
        abs(early_min), 1e-6
    )
    features["effective_force_hold_both_strict_contact_ratio"] = hold_strict_ratio
    features["effective_force_hold_both_strict_longest_run_sec"] = hold_strict_run
    features["effective_force_grasp_established_proxy"] = int(
        hold_strict_ratio >= 0.4 or hold_strict_run >= 1.0
    )
    return features


def gelsight_diff_arrays_from_synced(
    synced: list[dict[str, Any]],
    trial_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    left_values: list[float] = []
    right_values: list[float] = []
    image_cache: dict[Path, np.ndarray | None] = {}
    left_baseline: np.ndarray | None = None
    right_baseline: np.ndarray | None = None

    for record in synced:
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        left_record = record.get("gelsight_left")
        right_record = record.get("gelsight_right")
        if (
            not np.isfinite(timestamp)
            or not isinstance(left_record, dict)
            or not isinstance(right_record, dict)
        ):
            continue
        left_diff, left_baseline = mean_gelsight_diff(
            left_record, trial_dir, left_baseline, image_cache
        )
        right_diff, right_baseline = mean_gelsight_diff(
            right_record, trial_dir, right_baseline, image_cache
        )
        if left_diff is None or right_diff is None:
            continue
        timestamps.append(float(timestamp))
        left_values.append(left_diff)
        right_values.append(right_diff)

    return (
        np.asarray(timestamps, dtype=float),
        np.asarray(left_values, dtype=float),
        np.asarray(right_values, dtype=float),
    )


def dynamic_threshold(mean: float, std: float, margin: float, std_scale: float) -> float:
    return float(mean + max(margin, std_scale * std))


def average_gelsight_baseline_image(
    synced: list[dict[str, Any]],
    trial_dir: Path,
    side: str,
    contact_start: float | None,
    max_baseline_frames: int = 5,
) -> np.ndarray | None:
    records = [
        record[f"gelsight_{side}"]
        for record in synced
        if isinstance(record.get(f"gelsight_{side}"), dict)
    ]
    records = unique_gelsight_records(records)
    if not records:
        return None

    timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in records])
    valid_records = [
        (float(timestamp), record)
        for timestamp, record in zip(timestamps, records)
        if np.isfinite(timestamp)
    ]
    if not valid_records:
        return None

    candidates: list[tuple[float, dict[str, Any]]] = []
    if contact_start is not None and np.isfinite(contact_start):
        candidates = [
            (timestamp, record)
            for timestamp, record in valid_records
            if timestamp < float(contact_start)
        ]
    if not candidates:
        candidates = valid_records[:max_baseline_frames]

    selected = candidates[-max_baseline_frames:]
    images: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None
    for _, record in selected:
        image_path = resolve_path(trial_dir, record.get("image"))
        if image_path is None:
            continue
        img = read_grayscale_image(image_path)
        if img is None:
            continue
        if target_shape is None:
            target_shape = img.shape
        elif img.shape != target_shape:
            img = np.asarray(Image.fromarray(img).resize((target_shape[1], target_shape[0])))
        images.append(img.astype(np.float32))

    if not images:
        return None
    return np.mean(np.stack(images, axis=0), axis=0)


def gelsight_diff_area_against_baseline(
    record: dict[str, Any],
    trial_dir: Path,
    baseline: np.ndarray | None,
    image_cache: dict[Path, np.ndarray | None],
    area_pixel_threshold: int,
) -> tuple[float | None, float | None]:
    if baseline is None:
        return None, None

    image_path = resolve_path(trial_dir, record.get("image"))
    if image_path is None:
        return None, None
    if image_path not in image_cache:
        image_cache[image_path] = read_grayscale_image(image_path)
    img = image_cache[image_path]
    if img is None:
        return None, None
    if img.shape != baseline.shape:
        img = np.asarray(Image.fromarray(img).resize((baseline.shape[1], baseline.shape[0])))

    diff = np.abs(img.astype(np.float32) - baseline)
    return float(np.mean(diff)), float(np.sum(diff > area_pixel_threshold))


def gelsight_arrays_from_precontact_baseline(
    synced: list[dict[str, Any]],
    trial_dir: Path,
    left_baseline: np.ndarray | None,
    right_baseline: np.ndarray | None,
    area_pixel_threshold: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    left_diff_values: list[float] = []
    right_diff_values: list[float] = []
    left_area_values: list[float] = []
    right_area_values: list[float] = []
    image_cache: dict[Path, np.ndarray | None] = {}

    for record in synced:
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        left_record = record.get("gelsight_left")
        right_record = record.get("gelsight_right")
        if (
            not np.isfinite(timestamp)
            or not isinstance(left_record, dict)
            or not isinstance(right_record, dict)
        ):
            continue

        left_diff, left_area = gelsight_diff_area_against_baseline(
            left_record, trial_dir, left_baseline, image_cache, area_pixel_threshold
        )
        right_diff, right_area = gelsight_diff_area_against_baseline(
            right_record, trial_dir, right_baseline, image_cache, area_pixel_threshold
        )
        if (
            left_diff is None
            or right_diff is None
            or left_area is None
            or right_area is None
        ):
            continue

        timestamps.append(float(timestamp))
        left_diff_values.append(left_diff)
        right_diff_values.append(right_diff)
        left_area_values.append(left_area)
        right_area_values.append(right_area)

    return (
        np.asarray(timestamps, dtype=float),
        np.asarray(left_diff_values, dtype=float),
        np.asarray(right_diff_values, dtype=float),
        np.asarray(left_area_values, dtype=float),
        np.asarray(right_area_values, dtype=float),
    )


def effective_grasp_gelsight_features(
    full_synced: list[dict[str, Any]],
    windowed_synced: list[dict[str, Any]],
    trial_dir: Path,
    contact_start: float | None,
) -> dict[str, float | int]:
    left_baseline_img = average_gelsight_baseline_image(
        full_synced, trial_dir, "left", contact_start
    )
    right_baseline_img = average_gelsight_baseline_image(
        full_synced, trial_dir, "right", contact_start
    )
    full_ts, full_left, full_right, _, _ = gelsight_arrays_from_precontact_baseline(
        full_synced,
        trial_dir,
        left_baseline_img,
        right_baseline_img,
    )
    ts, left, right, left_area, right_area = gelsight_arrays_from_precontact_baseline(
        windowed_synced,
        trial_dir,
        left_baseline_img,
        right_baseline_img,
    )
    if ts.size == 0:
        return {}

    left_mean, left_std = baseline_from_precontact(full_ts, full_left, contact_start, 5)
    right_mean, right_std = baseline_from_precontact(full_ts, full_right, contact_start, 5)
    left_loose_threshold = dynamic_threshold(left_mean, left_std, margin=0.05, std_scale=2.0)
    right_loose_threshold = dynamic_threshold(right_mean, right_std, margin=0.05, std_scale=2.0)
    left_strict_threshold = dynamic_threshold(left_mean, left_std, margin=0.15, std_scale=3.0)
    right_strict_threshold = dynamic_threshold(right_mean, right_std, margin=0.15, std_scale=3.0)

    min_diff = np.minimum(left, right)
    max_diff = np.maximum(left, right)
    balance_ratio = min_diff / np.maximum(max_diff, 1e-6)
    min_area = np.minimum(left_area, right_area)
    area_balance_ratio = min_area / np.maximum(np.maximum(left_area, right_area), 1e-6)

    left_loose = left > left_loose_threshold
    right_loose = right > right_loose_threshold
    left_strict = left > left_strict_threshold
    right_strict = right > right_strict_threshold
    both_loose = left_loose & right_loose
    both_strict = left_strict & right_strict
    one_sided_loose = left_loose ^ right_loose
    one_sided_strict = left_strict ^ right_strict

    features: dict[str, float | int] = {
        "effective_gelsight_left_precontact_mean": left_mean,
        "effective_gelsight_right_precontact_mean": right_mean,
        "effective_gelsight_left_precontact_std": left_std,
        "effective_gelsight_right_precontact_std": right_std,
        "effective_gelsight_left_loose_threshold": left_loose_threshold,
        "effective_gelsight_right_loose_threshold": right_loose_threshold,
        "effective_gelsight_left_strict_threshold": left_strict_threshold,
        "effective_gelsight_right_strict_threshold": right_strict_threshold,
        "effective_gelsight_min_diff_mean": float(np.mean(min_diff)),
        "effective_gelsight_min_diff_max": float(np.max(min_diff)),
        "effective_gelsight_balance_ratio_mean": float(np.mean(balance_ratio)),
        "effective_gelsight_balance_ratio_min": float(np.min(balance_ratio)),
        "effective_gelsight_min_area_mean": float(np.mean(min_area)),
        "effective_gelsight_min_area_max": float(np.max(min_area)),
        "effective_gelsight_area_balance_ratio_mean": float(np.mean(area_balance_ratio)),
        "effective_gelsight_area_balance_ratio_min": float(np.min(area_balance_ratio)),
    }

    stages = {
        "establish_0_1s": (0.0, 1.0),
        "grasp_1_3s": (1.0, 3.0),
        "hold_3_5s": (3.0, 5.0),
    }
    for stage_name, (start_offset, end_offset) in stages.items():
        mask = stage_mask(ts, contact_start, start_offset, end_offset)
        features.update(
            summarize_stage_binary(
                ts,
                both_loose,
                mask,
                f"effective_gelsight_{stage_name}_both_loose_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                both_strict,
                mask,
                f"effective_gelsight_{stage_name}_both_strict_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                one_sided_loose,
                mask,
                f"effective_gelsight_{stage_name}_one_sided_loose_contact",
            )
        )
        features.update(
            summarize_stage_binary(
                ts,
                one_sided_strict,
                mask,
                f"effective_gelsight_{stage_name}_one_sided_strict_contact",
            )
        )
        features.update(
            summarize_stage_continuous(
                min_diff,
                mask,
                f"effective_gelsight_{stage_name}_min_diff",
            )
        )
        features.update(
            summarize_stage_continuous(
                balance_ratio,
                mask,
                f"effective_gelsight_{stage_name}_balance_ratio",
            )
        )
        features.update(
            summarize_stage_continuous(
                min_area,
                mask,
                f"effective_gelsight_{stage_name}_min_area",
            )
        )
        features.update(
            summarize_stage_continuous(
                area_balance_ratio,
                mask,
                f"effective_gelsight_{stage_name}_area_balance_ratio",
            )
        )

    hold_mask = stage_mask(ts, contact_start, 3.0, 5.0)
    hold_loose_ratio = float(np.mean(both_loose[hold_mask])) if hold_mask.any() else 0.0
    hold_strict_ratio = float(np.mean(both_strict[hold_mask])) if hold_mask.any() else 0.0
    hold_loose_run = (
        longest_active_run_sec(ts[hold_mask], both_loose[hold_mask]) if hold_mask.any() else 0.0
    )
    hold_strict_run = (
        longest_active_run_sec(ts[hold_mask], both_strict[hold_mask]) if hold_mask.any() else 0.0
    )
    hold_min_area = float(np.mean(min_area[hold_mask])) if hold_mask.any() else 0.0
    hold_area_balance = float(np.mean(area_balance_ratio[hold_mask])) if hold_mask.any() else 0.0
    features["effective_gelsight_hold_both_loose_contact_ratio"] = hold_loose_ratio
    features["effective_gelsight_hold_both_strict_contact_ratio"] = hold_strict_ratio
    features["effective_gelsight_hold_both_loose_longest_run_sec"] = hold_loose_run
    features["effective_gelsight_hold_both_strict_longest_run_sec"] = hold_strict_run
    features["effective_gelsight_hold_min_area_mean"] = hold_min_area
    features["effective_gelsight_hold_area_balance_ratio_mean"] = hold_area_balance
    features["effective_gelsight_grasp_established_proxy"] = int(
        hold_loose_ratio >= 0.4 or hold_loose_run >= 1.0
    )
    features["effective_gelsight_excessive_proxy"] = int(
        hold_strict_ratio >= 0.5 or hold_strict_run >= 1.0
    )
    return features


def first_threshold_crossing(
    series: list[tuple[float, float]],
    baseline_points: int,
    margin: float,
) -> float | None:
    if len(series) < baseline_points + 1:
        return None

    values = np.asarray([value for _, value in series], dtype=float)
    baseline = values[:baseline_points]
    threshold = float(np.mean(baseline) + max(margin, 3.0 * np.std(baseline)))

    for timestamp, value in series[baseline_points:]:
        if value > threshold:
            return timestamp
    return None


def detect_contact_start(
    synced: list[dict[str, Any]],
    trial_dir: Path,
    gelsight_margin: float,
    force_margin: float,
) -> tuple[float | None, str, float | None]:
    candidates: list[tuple[float, str]] = []

    for side in SIDES:
        gelsight_records = [
            record[f"gelsight_{side}"]
            for record in synced
            if isinstance(record.get(f"gelsight_{side}"), dict)
        ]
        series = gelsight_diff_series(gelsight_records, trial_dir)
        crossing = first_threshold_crossing(series, baseline_points=5, margin=gelsight_margin)
        if crossing is not None:
            candidates.append((crossing, f"{side}_gelsight"))

    if candidates:
        timestamp, source = min(candidates, key=lambda item: item[0])
        return timestamp, source, None

    for side in SIDES:
        force_records = [
            record[f"force_{side}"]
            for record in synced
            if isinstance(record.get(f"force_{side}"), dict)
        ]
        series = force_delta_series(force_records)
        crossing = first_threshold_crossing(series, baseline_points=20, margin=force_margin)
        if crossing is not None:
            candidates.append((crossing, f"{side}_force"))

    if candidates:
        timestamp, source = min(candidates, key=lambda item: item[0])
        return timestamp, source, None

    timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in synced])
    valid_timestamps = timestamps[np.isfinite(timestamps)]
    fallback = float(valid_timestamps[0]) if valid_timestamps.size else None
    return fallback, "fallback_start", None


def filter_synced_window(
    synced: list[dict[str, Any]],
    start_time: float | None,
    window_sec: float,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    if start_time is None:
        return synced, None, None

    end_time = start_time + window_sec
    windowed: list[dict[str, Any]] = []
    for record in synced:
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        if np.isfinite(timestamp) and start_time <= timestamp <= end_time:
            windowed.append(record)

    if not windowed:
        return synced, start_time, None

    window_timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in windowed])
    valid = window_timestamps[np.isfinite(window_timestamps)]
    actual_duration = float(valid[-1] - valid[0]) if valid.size >= 2 else 0.0
    return windowed, start_time, actual_duration


def extract_gelsight_features(
    records: list[dict[str, Any]],
    trial_dir: Path,
    prefix: str,
    threshold: int,
    max_frames: int | None,
) -> dict[str, float | int]:
    records = unique_gelsight_records(records)
    if not records:
        return {
            f"{prefix}_num_gelsight_frames": 0,
            f"{prefix}_gelsight_duration_sec": 0.0,
            f"{prefix}_gelsight_sample_rate_hz": 0.0,
        }

    timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in records])
    valid = np.isfinite(timestamps)
    records = [record for record, is_valid in zip(records, valid) if is_valid]
    timestamps = timestamps[valid]
    if timestamps.size == 0:
        return {
            f"{prefix}_num_gelsight_frames": 0,
            f"{prefix}_gelsight_duration_sec": 0.0,
            f"{prefix}_gelsight_sample_rate_hz": 0.0,
        }

    if max_frames is not None and len(records) > max_frames:
        indices = np.linspace(0, len(records) - 1, max_frames).round().astype(int)
        records = [records[i] for i in indices]
        timestamps = timestamps[indices]

    duration = max(float(timestamps[-1] - timestamps[0]), 1e-6)
    image_pairs = [
        (float(timestamp), resolve_path(trial_dir, record.get("image")))
        for timestamp, record in zip(timestamps, records)
    ]
    image_pairs = [(timestamp, path) for timestamp, path in image_pairs if path is not None]
    if not image_pairs:
        return {
            f"{prefix}_num_gelsight_frames": 0,
            f"{prefix}_gelsight_duration_sec": duration,
            f"{prefix}_gelsight_sample_rate_hz": 0.0,
        }

    baseline = read_grayscale_image(image_pairs[0][1])
    if baseline is None:
        raise FileNotFoundError(f"Could not read first GelSight image: {image_pairs[0][1]}")

    areas: list[float] = []
    centers: list[tuple[float, float]] = []
    intensities: list[float] = []
    used_timestamps: list[float] = []
    missing_images = 0

    for timestamp, image_path in image_pairs:
        img = read_grayscale_image(image_path)
        if img is None:
            missing_images += 1
            continue
        if img.shape != baseline.shape:
            img = np.asarray(Image.fromarray(img).resize((baseline.shape[1], baseline.shape[0])))

        diff = np.abs(img.astype(np.int16) - baseline.astype(np.int16)).astype(np.uint8)
        mask = diff > threshold
        area = float(np.sum(mask))
        intensity = float(np.mean(diff))

        if area > 0:
            ys, xs = np.where(mask)
            center = (float(np.mean(xs)), float(np.mean(ys)))
        else:
            center = (0.0, 0.0)

        areas.append(area)
        centers.append(center)
        intensities.append(intensity)
        used_timestamps.append(float(timestamp))

    if not areas:
        return {
            f"{prefix}_num_gelsight_frames": 0,
            f"{prefix}_gelsight_duration_sec": duration,
            f"{prefix}_gelsight_sample_rate_hz": 0.0,
            f"{prefix}_missing_gelsight_images": int(missing_images),
        }

    areas_arr = np.asarray(areas, dtype=float)
    centers_arr = np.asarray(centers, dtype=float)
    intensities_arr = np.asarray(intensities, dtype=float)
    used_timestamps_arr = np.asarray(used_timestamps, dtype=float)
    intensity_delta = np.maximum(0.0, intensities_arr - intensities_arr[0])
    contact_shift = np.sqrt(
        (centers_arr[:, 0] - centers_arr[0, 0]) ** 2
        + (centers_arr[:, 1] - centers_arr[0, 1]) ** 2
    )

    contact_step = np.zeros_like(contact_shift)
    if centers_arr.shape[0] >= 2:
        center_delta = np.sqrt(
            np.diff(centers_arr[:, 0]) ** 2 + np.diff(centers_arr[:, 1]) ** 2
        )
        contact_step = np.concatenate([[0.0], center_delta])

    features: dict[str, float | int] = {
        f"{prefix}_num_gelsight_frames": int(len(areas_arr)),
        f"{prefix}_gelsight_duration_sec": duration,
        f"{prefix}_gelsight_sample_rate_hz": float(len(areas_arr) / duration),
        f"{prefix}_missing_gelsight_images": int(missing_images),
        f"{prefix}_contact_area_growth": float(areas_arr[-1] - areas_arr[0]),
        f"{prefix}_contact_area_growth_rate": float((areas_arr[-1] - areas_arr[0]) / duration),
    }
    for name, values in {
        "contact_area": areas_arr,
        "contact_shift": contact_shift,
        "contact_step": contact_step,
        "diff_intensity": intensities_arr,
        "diff_intensity_delta": intensity_delta,
    }.items():
        features.update(summarize_signal(values, prefix, name))
        features.update(summarize_half_window(values, prefix, name))

    for name, values in {
        "contact_area": areas_arr,
        "diff_intensity": intensities_arr,
    }.items():
        features.update(summarize_drop_after_peak(values, prefix, name))
        features.update(summarize_velocity(used_timestamps_arr, values, prefix, name))

    features.update(
        active_features(
            timestamps=used_timestamps_arr,
            values=intensities_arr,
            prefix=prefix,
            name="diff_intensity",
            margin=0.12,
            baseline_points=5,
        )
    )
    features.update(
        binary_contact_features(
            timestamps=used_timestamps_arr,
            active=intensity_delta > 0.10,
            prefix=prefix,
            name="diff_intensity_delta_loose",
        )
    )
    features.update(
        binary_contact_features(
            timestamps=used_timestamps_arr,
            active=intensity_delta > 0.25,
            prefix=prefix,
            name="diff_intensity_delta_strict",
        )
    )
    features.update(
        binary_contact_features(
            timestamps=used_timestamps_arr,
            active=areas_arr > 50,
            prefix=prefix,
            name="contact_area_nontrivial",
        )
    )
    features.update(
        active_features(
            timestamps=used_timestamps_arr,
            values=areas_arr,
            prefix=prefix,
            name="contact_area",
            margin=1.0,
            baseline_points=5,
        )
    )

    features.update(
        summarize_velocity(used_timestamps_arr, contact_shift, prefix, "contact_shift")
    )
    features.update(
        summarize_velocity(used_timestamps_arr, contact_step, prefix, "contact_step")
    )
    return features


def extract_synced_trial_features(
    synced_jsonl: Path,
    trial_dir: Path,
    threshold: int,
    max_frames: int | None,
    contact_window_sec: float,
    gelsight_contact_margin: float,
    force_contact_margin: float,
) -> dict[str, Any]:
    synced = read_jsonl(synced_jsonl)
    if not synced:
        return {"num_synced_samples": 0}
    full_synced = synced

    full_timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in synced])
    full_valid = np.isfinite(full_timestamps)
    full_timestamps = full_timestamps[full_valid]
    full_duration = (
        max(float(full_timestamps[-1] - full_timestamps[0]), 1e-6)
        if full_timestamps.size
        else 0.0
    )

    contact_start, contact_source, _ = detect_contact_start(
        synced=synced,
        trial_dir=trial_dir,
        gelsight_margin=gelsight_contact_margin,
        force_margin=force_contact_margin,
    )
    synced, contact_start, contact_duration = filter_synced_window(
        synced=synced,
        start_time=contact_start,
        window_sec=contact_window_sec,
    )

    timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in synced])
    valid = np.isfinite(timestamps)
    timestamps = timestamps[valid]
    duration = max(float(timestamps[-1] - timestamps[0]), 1e-6) if timestamps.size else 0.0

    features: dict[str, Any] = {
        "full_num_synced_samples": int(len(full_synced)),
        "full_synced_duration_sec": full_duration,
        "num_synced_samples": int(len(synced)),
        "synced_duration_sec": duration,
        "synced_sample_rate_hz": float(len(synced) / duration) if duration > 0 else 0.0,
        "contact_window_sec": float(contact_window_sec),
        "contact_start_sec": float(contact_start) if contact_start is not None else np.nan,
        "contact_start_offset_sec": (
            float(contact_start - full_timestamps[0])
            if contact_start is not None and full_timestamps.size
            else np.nan
        ),
        "contact_window_duration_sec": (
            float(contact_duration) if contact_duration is not None else duration
        ),
        "contact_detection_source": contact_source,
    }

    for side in SIDES:
        force_records = [
            record[f"force_{side}"]
            for record in synced
            if isinstance(record.get(f"force_{side}"), dict)
        ]
        gelsight_records = [
            record[f"gelsight_{side}"]
            for record in synced
            if isinstance(record.get(f"gelsight_{side}"), dict)
        ]
        features.update(extract_force_features(force_records, f"{side}_force"))
        features.update(
            extract_gelsight_features(
                records=gelsight_records,
                trial_dir=trial_dir,
                prefix=f"{side}_gelsight",
                threshold=threshold,
                max_frames=max_frames,
        )
    )

    features.update(
        simultaneous_force_contact_features(
            synced=synced,
            loose_threshold=0.15,
            strict_threshold=0.25,
            baseline_points=20,
        )
    )
    features.update(
        simultaneous_gelsight_contact_features(
            synced=synced,
            trial_dir=trial_dir,
            loose_delta_threshold=0.10,
            strict_delta_threshold=0.25,
        )
    )
    features.update(
        effective_grasp_force_features(
            full_synced=full_synced,
            windowed_synced=synced,
            contact_start=contact_start,
            loose_threshold=0.15,
            strict_threshold=0.25,
        )
    )
    for signal_name, output_prefix, loose_margin, strict_margin in (
        ("torque_norm", "effective_torque", 0.05, 0.15),
        ("tx", "effective_tx", 0.02, 0.08),
        ("ty", "effective_ty", 0.02, 0.08),
        ("tz", "effective_tz", 0.02, 0.08),
    ):
        features.update(
            effective_bilateral_wrench_signal_features(
                full_synced=full_synced,
                windowed_synced=synced,
                contact_start=contact_start,
                signal_name=signal_name,
                output_prefix=output_prefix,
                loose_min_margin=loose_margin,
                strict_min_margin=strict_margin,
            )
        )
    features.update(
        effective_grasp_gelsight_features(
            full_synced=full_synced,
            windowed_synced=synced,
            trial_dir=trial_dir,
            contact_start=contact_start,
        )
    )

    left_force = features.get("left_force_force_norm_mean")
    right_force = features.get("right_force_force_norm_mean")
    if left_force is not None and right_force is not None:
        left = float(left_force)
        right = float(right_force)
        features["force_balance_abs_mean"] = abs(left - right)
        features["force_balance_ratio_mean"] = min(left, right) / max(left, right, 1e-6)

    left_area = features.get("left_gelsight_contact_area_mean")
    right_area = features.get("right_gelsight_contact_area_mean")
    if left_area is not None and right_area is not None:
        left = float(left_area)
        right = float(right_area)
        features["contact_area_balance_abs_mean"] = abs(left - right)
        features["contact_area_balance_ratio_mean"] = min(left, right) / max(left, right, 1e-6)

    left_force_active = features.get("left_force_force_norm_delta_active_ratio")
    right_force_active = features.get("right_force_force_norm_delta_active_ratio")
    if left_force_active is not None and right_force_active is not None:
        left = float(left_force_active)
        right = float(right_force_active)
        features["both_force_active_ratio_approx"] = min(left, right)
        features["force_active_balance_abs"] = abs(left - right)

    left_force_run = features.get("left_force_force_norm_delta_longest_active_run_sec")
    right_force_run = features.get("right_force_force_norm_delta_longest_active_run_sec")
    if left_force_run is not None and right_force_run is not None:
        features["both_force_longest_active_run_sec_approx"] = min(
            float(left_force_run), float(right_force_run)
        )

    left_force_delta_max = features.get("left_force_force_norm_delta_max")
    right_force_delta_max = features.get("right_force_force_norm_delta_max")
    if left_force_delta_max is not None and right_force_delta_max is not None:
        left = float(left_force_delta_max)
        right = float(right_force_delta_max)
        features["both_force_delta_min_max"] = min(left, right)
        features["force_delta_balance_ratio"] = min(left, right) / max(left, right, 1e-6)

    left_force_contact_ratio = features.get("left_force_force_norm_delta_strict_contact_ratio")
    right_force_contact_ratio = features.get("right_force_force_norm_delta_strict_contact_ratio")
    if left_force_contact_ratio is not None and right_force_contact_ratio is not None:
        left = float(left_force_contact_ratio)
        right = float(right_force_contact_ratio)
        features["both_force_contact_ratio_approx"] = min(left, right)
        features["force_contact_ratio_balance_abs"] = abs(left - right)

    left_force_contact_run = features.get(
        "left_force_force_norm_delta_strict_longest_contact_run_sec"
    )
    right_force_contact_run = features.get(
        "right_force_force_norm_delta_strict_longest_contact_run_sec"
    )
    if left_force_contact_run is not None and right_force_contact_run is not None:
        features["both_force_longest_contact_run_sec_approx"] = min(
            float(left_force_contact_run), float(right_force_contact_run)
        )

    left_gel_active = features.get("left_gelsight_diff_intensity_active_ratio")
    right_gel_active = features.get("right_gelsight_diff_intensity_active_ratio")
    if left_gel_active is not None and right_gel_active is not None:
        left = float(left_gel_active)
        right = float(right_gel_active)
        features["both_gelsight_active_ratio_approx"] = min(left, right)
        features["gelsight_active_balance_abs"] = abs(left - right)

    left_gel_run = features.get("left_gelsight_diff_intensity_longest_active_run_sec")
    right_gel_run = features.get("right_gelsight_diff_intensity_longest_active_run_sec")
    if left_gel_run is not None and right_gel_run is not None:
        features["both_gelsight_longest_active_run_sec_approx"] = min(
            float(left_gel_run), float(right_gel_run)
        )

    left_gel_delta_max = features.get("left_gelsight_diff_intensity_delta_max")
    right_gel_delta_max = features.get("right_gelsight_diff_intensity_delta_max")
    if left_gel_delta_max is not None and right_gel_delta_max is not None:
        left = float(left_gel_delta_max)
        right = float(right_gel_delta_max)
        features["both_gelsight_diff_delta_min_max"] = min(left, right)
        features["gelsight_diff_delta_balance_ratio"] = min(left, right) / max(left, right, 1e-6)

    left_gel_contact_ratio = features.get(
        "left_gelsight_diff_intensity_delta_loose_contact_ratio"
    )
    right_gel_contact_ratio = features.get(
        "right_gelsight_diff_intensity_delta_loose_contact_ratio"
    )
    if left_gel_contact_ratio is not None and right_gel_contact_ratio is not None:
        left = float(left_gel_contact_ratio)
        right = float(right_gel_contact_ratio)
        features["both_gelsight_contact_ratio_approx"] = min(left, right)
        features["gelsight_contact_ratio_balance_abs"] = abs(left - right)

    left_gel_contact_run = features.get(
        "left_gelsight_diff_intensity_delta_loose_longest_contact_run_sec"
    )
    right_gel_contact_run = features.get(
        "right_gelsight_diff_intensity_delta_loose_longest_contact_run_sec"
    )
    if left_gel_contact_run is not None and right_gel_contact_run is not None:
        features["both_gelsight_longest_contact_run_sec_approx"] = min(
            float(left_gel_contact_run), float(right_gel_contact_run)
        )

    contact_channels = 0
    for key in (
        "left_force_force_norm_delta_strict_contact_detected",
        "right_force_force_norm_delta_strict_contact_detected",
        "left_gelsight_diff_intensity_delta_loose_contact_detected",
        "right_gelsight_diff_intensity_delta_loose_contact_detected",
    ):
        contact_channels += int(float(features.get(key, 0)))
    features["num_contact_channels"] = contact_channels
    features["weak_contact_score"] = 1.0 - contact_channels / 4.0
    add_relative_response_features(features)

    return features


def validate_metadata(metadata: pd.DataFrame) -> None:
    required = {"trial_id", "trial_dir", "synced_jsonl", "final_instruction"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"metadata.csv missing required columns: {sorted(missing)}")

    bad_labels = set(metadata["final_instruction"].dropna().astype(str)) - FINAL_LABELS
    if bad_labels:
        raise ValueError(
            "final_instruction must be one of "
            f"{sorted(FINAL_LABELS)}; found {sorted(bad_labels)}"
        )


def build_features(
    session_dir: Path,
    output_path: Path | None,
    threshold: int,
    max_frames: int | None,
    contact_window_sec: float,
    gelsight_contact_margin: float,
    force_contact_margin: float,
) -> Path:
    session_dir = session_dir.resolve()
    metadata_path = session_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    validate_metadata(metadata)

    rows: list[dict[str, Any]] = []
    for _, meta in metadata.iterrows():
        row = dict(meta)
        trial_dir = resolve_path(session_dir, meta["trial_dir"])
        if trial_dir is None:
            raise ValueError(f"trial_dir is empty for trial {meta['trial_id']}")

        synced_jsonl = resolve_path(trial_dir, meta["synced_jsonl"])
        if synced_jsonl is None or not synced_jsonl.exists():
            print(f"Warning: missing synced jsonl for {meta['trial_id']}: {synced_jsonl}")
        else:
            row.update(
                extract_synced_trial_features(
                    synced_jsonl=synced_jsonl,
                    trial_dir=trial_dir,
                    threshold=threshold,
                    max_frames=max_frames,
                    contact_window_sec=contact_window_sec,
                    gelsight_contact_margin=gelsight_contact_margin,
                    force_contact_margin=force_contact_margin,
                )
            )
        rows.append(row)

    features = pd.DataFrame(rows)

    if output_path is None:
        output_path = session_dir / "features.csv"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate features.csv from synced UMI force/GelSight JSONL logs."
    )
    parser.add_argument(
        "--session-dir",
        default="data_yue/pilot/plastic_bottle",
        help="Session directory containing metadata.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output features.csv path. Default: <session-dir>/features.csv",
    )
    parser.add_argument(
        "--gelsight-threshold",
        type=int,
        default=8,
        help="Pixel-difference threshold for simple GelSight contact mask.",
    )
    parser.add_argument(
        "--max-gelsight-frames",
        type=int,
        default=200,
        help="Maximum unique GelSight frames sampled per trial. Use 0 for all frames.",
    )
    parser.add_argument(
        "--contact-window-sec",
        type=float,
        default=5.0,
        help="Seconds after detected contact start used for feature extraction.",
    )
    parser.add_argument(
        "--gelsight-contact-margin",
        type=float,
        default=0.25,
        help="Minimum GelSight diff increase above baseline for contact detection.",
    )
    parser.add_argument(
        "--force-contact-margin",
        type=float,
        default=0.25,
        help="Minimum force-norm change above baseline for fallback contact detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frames = None if args.max_gelsight_frames == 0 else args.max_gelsight_frames
    output = build_features(
        session_dir=Path(args.session_dir),
        output_path=Path(args.output) if args.output else None,
        threshold=args.gelsight_threshold,
        max_frames=max_frames,
        contact_window_sec=args.contact_window_sec,
        gelsight_contact_margin=args.gelsight_contact_margin,
        force_contact_margin=args.force_contact_margin,
    )
    print(f"Saved features to: {output}")


if __name__ == "__main__":
    main()
