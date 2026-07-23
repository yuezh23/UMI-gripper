"""Check whether GelSight logs contain useful signal for XGBoost analysis.

The script scans a tree like:

    train/paper_cup/high/01/episodes/episode_xxx/synced_data.jsonl
    train/softball/medium/02/episodes/episode_xxx/synced_data.jsonl
    train/foam_brick/low/03/episodes/episode_xxx/synced_data.jsonl

It extracts GelSight-only trial-level image-difference features, summarizes
availability and class separability, and optionally runs XGBoost cross
validation when xgboost and scikit-learn are installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


SIDES = ("left", "right")
FORCE_LEVELS = ("low", "medium", "high")
LEAKAGE_TOKENS = (
    "timestamp",
    "duration",
    "sample_rate",
    "records",
    "frames_used",
    "missing",
    "jsonl_path",
    "episode_id",
    "num_synced",
)
OBJECT_ALIASES = {
    "paper_cup": "paper_cup",
    "softball": "softball",
    "foam_brick": "foam_brick",
    "bottle_cap": "bottle_cap",
    "marker": "marker",
    "maker": "marker",
    "plastic_bottle": "plastic_bottle",
    "plastic_bottel": "plastic_bottle",
}


@dataclass
class Episode:
    jsonl_path: Path
    trial_dir: Path
    object_type: str
    force_level: str
    episode_id: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def timestamps_to_seconds(values: Any) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64)
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size and np.nanmedian(np.abs(finite)) > 1e12:
        return timestamps / 1e9
    return timestamps


def resolve_path(base_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return base_dir / path


def read_gray(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("L"), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def unique_gelsight_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for record in records:
        image_value = (
            record.get("image")
            or record.get("image_path")
            or record.get("path")
            or record.get("file")
        )
        key = (record.get("timestamp"), image_value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def select_evenly(records: list[dict[str, Any]], max_frames: int | None) -> list[dict[str, Any]]:
    if max_frames is None or len(records) <= max_frames:
        return records
    indices = np.linspace(0, len(records) - 1, max_frames).round().astype(int)
    return [records[int(i)] for i in indices]


def first_existing_image(
    records: list[dict[str, Any]], trial_dir: Path
) -> tuple[Path | None, np.ndarray | None]:
    for record in records:
        image_path = resolve_path(
            trial_dir,
            record.get("image")
            or record.get("image_path")
            or record.get("path")
            or record.get("file"),
        )
        if image_path is None:
            continue
        image = read_gray(image_path)
        if image is not None:
            return image_path, image
    return None, None


def extract_side_features(
    records: list[dict[str, Any]],
    trial_dir: Path,
    side: str,
    max_frames: int | None,
    pixel_threshold: int,
    baseline_frames: int,
) -> dict[str, float | int | str]:
    prefix = f"{side}_gelsight"
    records = select_evenly(unique_gelsight_records(records), max_frames)
    result: dict[str, float | int | str] = {
        f"{prefix}_records": int(len(records)),
        f"{prefix}_frames_used": 0,
        f"{prefix}_missing_images": 0,
        f"{prefix}_missing_ratio": 1.0 if records else 0.0,
        f"{prefix}_duration_sec": 0.0,
        f"{prefix}_sample_rate_hz": 0.0,
    }
    if not records:
        return result

    timestamps = timestamps_to_seconds([record.get("timestamp", np.nan) for record in records])
    valid_ts = timestamps[np.isfinite(timestamps)]
    if valid_ts.size >= 2:
        duration = max(float(valid_ts[-1] - valid_ts[0]), 1e-6)
        result[f"{prefix}_duration_sec"] = duration
        result[f"{prefix}_sample_rate_hz"] = float(len(records) / duration)

    _, first_image = first_existing_image(records, trial_dir)
    if first_image is None:
        return result

    baseline_images: list[np.ndarray] = []
    frame_means: list[float] = []
    frame_stds: list[float] = []
    diffs_mean: list[float] = []
    diffs_p95: list[float] = []
    contact_areas: list[float] = []
    used_timestamps: list[float] = []
    missing = 0

    for record in records:
        image_path = resolve_path(
            trial_dir,
            record.get("image")
            or record.get("image_path")
            or record.get("path")
            or record.get("file"),
        )
        image = read_gray(image_path) if image_path is not None else None
        if image is None:
            missing += 1
            continue
        if image.shape != first_image.shape:
            image = np.asarray(Image.fromarray(image).resize((first_image.shape[1], first_image.shape[0])))
        if len(baseline_images) < baseline_frames:
            baseline_images.append(image.astype(np.float32))

    if not baseline_images:
        result[f"{prefix}_missing_images"] = int(missing)
        result[f"{prefix}_missing_ratio"] = 1.0
        return result

    baseline = np.median(np.stack(baseline_images, axis=0), axis=0)

    for record in records:
        image_path = resolve_path(
            trial_dir,
            record.get("image")
            or record.get("image_path")
            or record.get("path")
            or record.get("file"),
        )
        image = read_gray(image_path) if image_path is not None else None
        if image is None:
            continue
        if image.shape != first_image.shape:
            image = np.asarray(Image.fromarray(image).resize((first_image.shape[1], first_image.shape[0])))
        image_f = image.astype(np.float32)
        diff = np.abs(image_f - baseline)
        frame_means.append(float(np.mean(image_f)))
        frame_stds.append(float(np.std(image_f)))
        diffs_mean.append(float(np.mean(diff)))
        diffs_p95.append(float(np.percentile(diff, 95)))
        contact_areas.append(float(np.sum(diff > pixel_threshold)))
        timestamp = timestamps_to_seconds([record.get("timestamp", np.nan)])[0]
        if np.isfinite(timestamp):
            used_timestamps.append(float(timestamp))

    frames_used = len(diffs_mean)
    result[f"{prefix}_frames_used"] = int(frames_used)
    result[f"{prefix}_missing_images"] = int(len(records) - frames_used)
    result[f"{prefix}_missing_ratio"] = float((len(records) - frames_used) / max(len(records), 1))
    if used_timestamps:
        result[f"{prefix}_first_timestamp_sec"] = float(used_timestamps[0])
        result[f"{prefix}_last_timestamp_sec"] = float(used_timestamps[-1])

    diffs_mean_arr = np.asarray(diffs_mean, dtype=float)
    contact_area_arr = np.asarray(contact_areas, dtype=float)
    baseline_n = max(1, min(baseline_frames, diffs_mean_arr.size))
    noise_mean = float(np.mean(diffs_mean_arr[:baseline_n])) if baseline_n else 0.0
    noise_std = float(np.std(diffs_mean_arr[:baseline_n])) if baseline_n else 0.0
    active_threshold = noise_mean + max(0.25, 3.0 * noise_std)
    active = diffs_mean_arr > active_threshold

    result.update(summarize(np.asarray(frame_means), f"{prefix}_frame_mean"))
    result.update(summarize(np.asarray(frame_stds), f"{prefix}_frame_std"))
    result.update(summarize(diffs_mean_arr, f"{prefix}_diff_mean"))
    result.update(summarize(np.asarray(diffs_p95), f"{prefix}_diff_p95"))
    result.update(summarize(contact_area_arr, f"{prefix}_contact_area"))
    result[f"{prefix}_baseline_noise_mean"] = noise_mean
    result[f"{prefix}_baseline_noise_std"] = noise_std
    result[f"{prefix}_active_threshold"] = active_threshold
    result[f"{prefix}_active_ratio"] = float(np.mean(active)) if active.size else 0.0
    result[f"{prefix}_nonzero_change_ratio"] = float(np.mean(diffs_mean_arr > 1e-6)) if active.size else 0.0
    result[f"{prefix}_max_minus_baseline_noise"] = float(
        np.max(diffs_mean_arr) - noise_mean if diffs_mean_arr.size else 0.0
    )
    result[f"{prefix}_area_max_over_threshold"] = int(
        np.max(contact_area_arr) > 0 if contact_area_arr.size else 0
    )
    return result


def infer_episode(jsonl_path: Path, root: Path) -> Episode:
    rel_parts = [part.lower() for part in jsonl_path.relative_to(root).parts]
    object_type = "unknown"
    force_level = "unknown"
    force_index: int | None = None
    for index, part in enumerate(rel_parts):
        if part in OBJECT_ALIASES:
            object_type = OBJECT_ALIASES[part]
        if part in FORCE_LEVELS:
            force_level = part
            force_index = index
    if object_type == "unknown" and rel_parts:
        object_index = force_index - 1 if force_index is not None and force_index > 0 else 0
        object_name = rel_parts[object_index]
        object_type = OBJECT_ALIASES.get(object_name, object_name)
    return Episode(
        jsonl_path=jsonl_path,
        trial_dir=jsonl_path.parent,
        object_type=object_type,
        force_level=force_level,
        episode_id=jsonl_path.parent.name,
    )


def discover_episodes(root: Path) -> list[Episode]:
    root = root.resolve()
    paths = sorted(root.rglob("synced_data.jsonl"))
    return [infer_episode(path, root) for path in paths]


def extract_episode_features(
    episode: Episode,
    max_frames: int | None,
    pixel_threshold: int,
    baseline_frames: int,
) -> dict[str, Any]:
    records = read_jsonl(episode.jsonl_path)
    row: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "object_type": episode.object_type,
        "force_level": episode.force_level,
        "jsonl_path": str(episode.jsonl_path),
        "num_synced_records": int(len(records)),
    }
    for side in SIDES:
        side_records = [
            record[f"gelsight_{side}"]
            for record in records
            if isinstance(record.get(f"gelsight_{side}"), dict)
        ]
        row.update(
            extract_side_features(
                records=side_records,
                trial_dir=episode.trial_dir,
                side=side,
                max_frames=max_frames,
                pixel_threshold=pixel_threshold,
                baseline_frames=baseline_frames,
            )
        )
    row["both_gelsight_frames_used_min"] = min(
        int(row.get("left_gelsight_frames_used", 0)),
        int(row.get("right_gelsight_frames_used", 0)),
    )
    row["both_gelsight_diff_mean_max_min"] = min(
        float(row.get("left_gelsight_diff_mean_max", 0.0)),
        float(row.get("right_gelsight_diff_mean_max", 0.0)),
    )
    row["both_gelsight_contact_area_max_min"] = min(
        float(row.get("left_gelsight_contact_area_max", 0.0)),
        float(row.get("right_gelsight_contact_area_max", 0.0)),
    )
    row["best_gelsight_diff_mean_max"] = max(
        float(row.get("left_gelsight_diff_mean_max", 0.0)),
        float(row.get("right_gelsight_diff_mean_max", 0.0)),
    )
    row["best_gelsight_active_ratio"] = max(
        float(row.get("left_gelsight_active_ratio", 0.0)),
        float(row.get("right_gelsight_active_ratio", 0.0)),
    )
    return row


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if any(token in lower for token in LEAKAGE_TOKENS):
            continue
        if "gelsight" not in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).std()) > 1e-12:
                cols.append(col)
    return cols


def eta_squared(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = df[target].astype(str)
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        mask = values.notna() & labels.notna()
        x = values[mask].to_numpy(dtype=float)
        y = labels[mask]
        if x.size < 2 or y.nunique() < 2:
            continue
        total_ss = float(np.sum((x - np.mean(x)) ** 2))
        if total_ss <= 1e-12:
            eta = 0.0
        else:
            between = 0.0
            for _, group_values in pd.Series(x).groupby(y.to_numpy()):
                arr = group_values.to_numpy(dtype=float)
                between += len(arr) * float((np.mean(arr) - np.mean(x)) ** 2)
            eta = between / total_ss
        rows.append({"target": target, "feature": feature, "eta_squared": eta})
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False)


def nearest_centroid_cv(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    labels = df[target].astype(str)
    valid = labels.ne("unknown")
    data = df.loc[valid, features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = labels[valid].reset_index(drop=True)
    x = data.to_numpy(dtype=float)
    if x.shape[0] < 3 or labels.nunique() < 2 or labels.value_counts().min() < 2:
        return {"target": target, "model": "nearest_centroid", "accuracy": np.nan, "note": "too few samples"}

    preds: list[str] = []
    truth: list[str] = []
    for i in range(x.shape[0]):
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[i] = False
        train_x = x[train_mask]
        test_x = x[i]
        train_y = labels[train_mask].to_numpy()
        mean = np.mean(train_x, axis=0)
        std = np.std(train_x, axis=0)
        std[std < 1e-9] = 1.0
        train_z = (train_x - mean) / std
        test_z = (test_x - mean) / std
        best_label = None
        best_dist = math.inf
        for label in sorted(set(train_y)):
            centroid = np.mean(train_z[train_y == label], axis=0)
            dist = float(np.linalg.norm(test_z - centroid))
            if dist < best_dist:
                best_dist = dist
                best_label = str(label)
        preds.append(str(best_label))
        truth.append(str(labels.iloc[i]))
    accuracy = float(np.mean(np.asarray(preds) == np.asarray(truth)))
    return {"target": target, "model": "nearest_centroid", "accuracy": accuracy, "note": "fallback_without_xgboost"}


def xgboost_cv(df: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any] | None:
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier
    except ImportError:
        return None

    labels = df[target].astype(str)
    valid = labels.ne("unknown")
    data = df.loc[valid, features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = labels[valid].reset_index(drop=True)
    if labels.nunique() < 2 or labels.value_counts().min() < 2:
        return {"target": target, "model": "xgboost", "accuracy": np.nan, "note": "too few samples"}

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x = data.to_numpy(dtype=float)
    n_splits = int(min(5, labels.value_counts().min()))
    model = XGBClassifier(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob" if labels.nunique() > 2 else "binary:logistic",
        eval_metric="mlogloss" if labels.nunique() > 2 else "logloss",
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
    return {
        "target": target,
        "model": "xgboost",
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "folds": n_splits,
        "note": "gel_features_only",
    }


def write_group_summary(df: pd.DataFrame, output_path: Path) -> None:
    metric_cols = [
        "left_gelsight_frames_used",
        "right_gelsight_frames_used",
        "left_gelsight_missing_ratio",
        "right_gelsight_missing_ratio",
        "left_gelsight_diff_mean_max",
        "right_gelsight_diff_mean_max",
        "left_gelsight_contact_area_max",
        "right_gelsight_contact_area_max",
        "best_gelsight_diff_mean_max",
        "best_gelsight_active_ratio",
    ]
    existing = [col for col in metric_cols if col in df.columns]
    grouped = (
        df.groupby(["object_type", "force_level"], dropna=False)[existing]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.to_csv(output_path, index=False)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    if max_rows is not None:
        df = df.head(max_rows)
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
            )
        else:
            text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in text_df.columns]
    rows = text_df.values.tolist()
    widths = [
        max(len(headers[i]), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        headers[i].ljust(widths[i]) for i in range(len(headers))
    ) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def quality_decision(df: pd.DataFrame, force_eta: pd.DataFrame, object_eta: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    episodes = len(df)
    object_count = int(df["object_type"].nunique()) if "object_type" in df else 0
    force_count = int(df["force_level"].nunique()) if "force_level" in df else 0
    trial_counts = (
        df.groupby(["object_type", "force_level"], dropna=False).size()
        if {"object_type", "force_level"}.issubset(df.columns)
        else pd.Series(dtype=int)
    )
    expected_per_cell = int(trial_counts.max()) if not trial_counts.empty else 20
    expected = object_count * force_count * expected_per_cell
    if episodes < expected:
        notes.append(
            f"- Episodes found: {episodes}; expected around {expected} for "
            f"{object_count} objects x {force_count} forces x {expected_per_cell} episodes."
        )
    else:
        notes.append(
            f"- Episodes found: {episodes}, matching the expected scale "
            f"({object_count} objects x {force_count} forces x {expected_per_cell} episodes)."
        )

    frame_cols = [col for col in ("left_gelsight_frames_used", "right_gelsight_frames_used") if col in df]
    min_frames_median = float(df[frame_cols].min(axis=1).median()) if frame_cols else 0.0
    missing_cols = [col for col in ("left_gelsight_missing_ratio", "right_gelsight_missing_ratio") if col in df]
    missing_median = float(df[missing_cols].max(axis=1).median()) if missing_cols else 1.0
    best_signal = float(df.get("best_gelsight_diff_mean_max", pd.Series([0.0])).median())
    active_ratio = float(df.get("best_gelsight_active_ratio", pd.Series([0.0])).median())
    force_eta_max = float(force_eta["eta_squared"].max()) if not force_eta.empty else 0.0
    object_eta_max = float(object_eta["eta_squared"].max()) if not object_eta.empty else 0.0

    notes.append(f"- Median minimum frames per episode across left/right: {min_frames_median:.1f}.")
    notes.append(f"- Median worst-side missing image ratio: {missing_median:.3f}.")
    notes.append(f"- Median best-side max mean pixel diff: {best_signal:.3f}.")
    notes.append(f"- Median best-side active ratio: {active_ratio:.3f}.")
    notes.append(f"- Best GelSight eta^2 for force_level: {force_eta_max:.3f}.")
    notes.append(f"- Best GelSight eta^2 for object_type: {object_eta_max:.3f}.")

    if min_frames_median < 20 or missing_median > 0.2:
        verdict = "NOT READY: GelSight availability is weak or image paths are missing."
    elif best_signal < 0.5 and active_ratio < 0.05:
        verdict = "WEAK: images are present, but GelSight changes look close to baseline/noise."
    elif force_eta_max >= 0.15 or object_eta_max >= 0.15:
        verdict = "PROMISING: GelSight features show measurable class separation and are worth trying in XGBoost."
    else:
        verdict = "BORDERLINE: GelSight changes exist, but class separation is weak; combine with force features or improve contact/lighting."
    notes.insert(0, f"**Verdict:** {verdict}")
    return notes


def write_markdown_report(
    output_path: Path,
    root: Path,
    df: pd.DataFrame,
    force_eta: pd.DataFrame,
    object_eta: pd.DataFrame,
    cv_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# GelSight XGBoost Precheck",
        "",
        f"Root: `{root}`",
        "",
        "## Decision",
        "",
        *quality_decision(df, force_eta, object_eta),
        "",
        "## Class Counts",
        "",
    ]
    counts = df.groupby(["object_type", "force_level"]).size().reset_index(name="episodes")
    lines.append(markdown_table(counts))
    lines.extend(["", "## Top Force-Level GelSight Features", ""])
    lines.append(
        markdown_table(force_eta, max_rows=15)
        if not force_eta.empty
        else "No usable force-level eta^2 results."
    )
    lines.extend(["", "## Top Object-Type GelSight Features", ""])
    lines.append(
        markdown_table(object_eta, max_rows=15)
        if not object_eta.empty
        else "No usable object-type eta^2 results."
    )
    lines.extend(["", "## Cross-Validation", ""])
    if cv_rows:
        lines.append(markdown_table(pd.DataFrame(cv_rows)))
    else:
        lines.append("No model check could be run.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- XGBoost CV runs only when `xgboost` and `scikit-learn` are installed.",
            "- The fallback nearest-centroid check is not a replacement for XGBoost; it is a quick sanity check for GelSight-only separability.",
            "- If GelSight-only is borderline, try adding force features and using a contact-window crop before training.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="data_yue/train",
        help="Root directory to scan for synced_data.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/gelsight_xgb_precheck",
        help="Directory for features.csv, summaries, and report.md.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=160,
        help="Maximum GelSight frames sampled per side per episode. Use 0 for all frames.",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=8,
        help="Pixel difference threshold for contact-area features.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=5,
        help="Number of initial frames used as GelSight baseline.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress after this many episodes. Use 0 to disable progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_frames = None if args.max_frames == 0 else args.max_frames

    episodes = discover_episodes(root)
    if not episodes:
        raise FileNotFoundError(f"No synced_data.jsonl files found under {root}")

    print(f"Found {len(episodes)} episodes under {root}", flush=True)
    print(
        f"Using max_frames={args.max_frames}, pixel_threshold={args.pixel_threshold}, "
        f"baseline_frames={args.baseline_frames}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        rows.append(
            extract_episode_features(
            episode=episode,
            max_frames=max_frames,
            pixel_threshold=args.pixel_threshold,
            baseline_frames=args.baseline_frames,
        )
        )
        if args.progress_every > 0 and (
            index == 1 or index % args.progress_every == 0 or index == len(episodes)
        ):
            print(
                f"[{index}/{len(episodes)}] {episode.object_type}/{episode.force_level}/"
                f"{episode.episode_id}",
                flush=True,
            )

    print("Computing GelSight separability and model checks...", flush=True)
    df = pd.DataFrame(rows)
    features_path = output_dir / "gelsight_features.csv"
    df.to_csv(features_path, index=False, quoting=csv.QUOTE_MINIMAL)

    feature_cols = numeric_feature_columns(df)
    force_eta = eta_squared(df, "force_level", feature_cols)
    object_eta = eta_squared(df, "object_type", feature_cols)
    force_eta.to_csv(output_dir / "force_level_eta_squared.csv", index=False)
    object_eta.to_csv(output_dir / "object_type_eta_squared.csv", index=False)
    write_group_summary(df, output_dir / "group_summary.csv")

    cv_rows: list[dict[str, Any]] = []
    for target in ("force_level", "object_type"):
        xgb_result = xgboost_cv(df, target, feature_cols)
        cv_rows.append(xgb_result if xgb_result is not None else nearest_centroid_cv(df, target, feature_cols))
    pd.DataFrame(cv_rows).to_csv(output_dir / "model_check.csv", index=False)

    report_path = output_dir / "report.md"
    write_markdown_report(report_path, root, df, force_eta, object_eta, cv_rows)
    print(f"Saved GelSight features: {features_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
