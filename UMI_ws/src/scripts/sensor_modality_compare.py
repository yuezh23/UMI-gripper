"""Compare force sensor and GelSight signal strength on synced UMI episodes.

This script runs a fair modality comparison on the same episodes:

    1. force-only
    2. gelsight-only
    3. force+gelsight

It intentionally excludes timestamps, paths, frame counts, sample rates, and
durations from model features so that the comparison focuses on physical signal
instead of collection-order leakage.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gelsight_xgb_precheck as gel


SIDES = ("left", "right")
FORCE_COLUMNS = ("fx", "fy", "fz", "tx", "ty", "tz")
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


def numeric_array(records: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [record.get(field, 0.0) for record in records]
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)


def summarize_signal(values: np.ndarray, prefix: str, name: str) -> dict[str, float]:
    return gel.summarize(np.asarray(values, dtype=float), f"{prefix}_{name}")


def summarize_half_window(values: np.ndarray, prefix: str, name: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_{name}_first_half_mean": 0.0,
            f"{prefix}_{name}_second_half_mean": 0.0,
            f"{prefix}_{name}_second_minus_first": 0.0,
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
        f"{prefix}_{name}_second_minus_first": second_mean - first_mean,
        f"{prefix}_{name}_second_to_first_ratio": second_mean / max(abs(first_mean), 1e-9),
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
    n = min(timestamps.size, values.size)
    timestamps = timestamps[:n]
    values = values[:n]
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


def active_ratio(values: np.ndarray, baseline_points: int = 20) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    baseline_n = max(1, min(baseline_points, values.size))
    baseline = values[:baseline_n]
    threshold = float(np.mean(baseline) + max(0.02, 3.0 * np.std(baseline)))
    return float(np.mean(values > threshold))


def extract_force_features(records: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        f"{prefix}_records": int(len(records)),
        f"{prefix}_duration_sec": 0.0,
        f"{prefix}_sample_rate_hz": 0.0,
    }
    if not records:
        return result

    stamps = gel.timestamps_to_seconds([record.get("stamp", np.nan) for record in records])
    valid = np.isfinite(stamps)
    records = [record for record, is_valid in zip(records, valid) if is_valid]
    stamps = stamps[valid]
    if stamps.size == 0:
        return result
    duration = max(float(stamps[-1] - stamps[0]), 1e-6)
    result[f"{prefix}_duration_sec"] = duration
    result[f"{prefix}_sample_rate_hz"] = float(len(records) / duration)

    values_by_name: dict[str, np.ndarray] = {}
    for col in FORCE_COLUMNS:
        values_by_name[col] = numeric_array(records, col)
    fx = values_by_name["fx"]
    fy = values_by_name["fy"]
    fz = values_by_name["fz"]
    tx = values_by_name["tx"]
    ty = values_by_name["ty"]
    tz = values_by_name["tz"]
    values_by_name["normal_force_abs"] = np.abs(fz)
    values_by_name["tangential_force"] = np.sqrt(fx**2 + fy**2)
    values_by_name["force_norm"] = np.sqrt(fx**2 + fy**2 + fz**2)
    values_by_name["torque_norm"] = np.sqrt(tx**2 + ty**2 + tz**2)

    for name, values in values_by_name.items():
        result.update(summarize_signal(values, prefix, name))
        result.update(summarize_half_window(values, prefix, name))
        if name in {"force_norm", "normal_force_abs", "torque_norm"}:
            result.update(summarize_velocity(stamps, values, prefix, name))

    for name in ("force_norm", "normal_force_abs", "torque_norm"):
        values = values_by_name[name]
        baseline_n = max(3, min(20, values.size // 10 if values.size >= 10 else values.size))
        baseline = float(np.median(values[:baseline_n])) if baseline_n else 0.0
        delta = np.abs(values - baseline)
        result[f"{prefix}_{name}_baseline"] = baseline
        result.update(summarize_signal(delta, prefix, f"{name}_delta"))
        result.update(summarize_half_window(delta, prefix, f"{name}_delta"))
        result[f"{prefix}_{name}_delta_active_ratio"] = active_ratio(delta)
    return result


def extract_force_episode_features(records: list[dict[str, Any]]) -> dict[str, float | int]:
    features: dict[str, float | int] = {}
    for side in SIDES:
        side_records = [
            record[f"force_{side}"]
            for record in records
            if isinstance(record.get(f"force_{side}"), dict)
        ]
        features.update(extract_force_features(side_records, f"{side}_force"))

    left_delta = float(features.get("left_force_force_norm_delta_max", 0.0))
    right_delta = float(features.get("right_force_force_norm_delta_max", 0.0))
    left_mean = float(features.get("left_force_force_norm_mean", 0.0))
    right_mean = float(features.get("right_force_force_norm_mean", 0.0))
    left_normal = float(features.get("left_force_normal_force_abs_delta_max", 0.0))
    right_normal = float(features.get("right_force_normal_force_abs_delta_max", 0.0))
    left_torque = float(features.get("left_force_torque_norm_delta_max", 0.0))
    right_torque = float(features.get("right_force_torque_norm_delta_max", 0.0))

    features.update(
        {
            "both_force_force_norm_delta_max_min": min(left_delta, right_delta),
            "best_force_force_norm_delta_max": max(left_delta, right_delta),
            "force_force_norm_delta_balance_ratio": min(left_delta, right_delta)
            / max(left_delta, right_delta, 1e-9),
            "force_force_norm_mean_balance_abs": abs(left_mean - right_mean),
            "force_force_norm_mean_balance_ratio": min(left_mean, right_mean)
            / max(left_mean, right_mean, 1e-9),
            "both_force_normal_force_abs_delta_max_min": min(left_normal, right_normal),
            "best_force_normal_force_abs_delta_max": max(left_normal, right_normal),
            "both_force_torque_norm_delta_max_min": min(left_torque, right_torque),
            "best_force_torque_norm_delta_max": max(left_torque, right_torque),
        }
    )
    return features


def extract_all_features(
    episode: gel.Episode,
    max_frames: int | None,
    pixel_threshold: int,
    baseline_frames: int,
) -> dict[str, Any]:
    records = gel.read_jsonl(episode.jsonl_path)
    row = gel.extract_episode_features(
        episode=episode,
        max_frames=max_frames,
        pixel_threshold=pixel_threshold,
        baseline_frames=baseline_frames,
    )
    row.update(extract_force_episode_features(records))
    return row


def is_model_feature(col: str) -> bool:
    lower = col.lower()
    return not any(token in lower for token in LEAKAGE_TOKENS)


def feature_columns(df: pd.DataFrame, modality: str) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if not is_model_feature(col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).std()) <= 1e-12:
            continue
        if modality == "force" and ("_force" in col or col.startswith("force_") or col.startswith("both_force") or col.startswith("best_force")):
            cols.append(col)
        elif modality == "gelsight" and "gelsight" in col:
            cols.append(col)
        elif modality == "combined" and (
            "gelsight" in col
            or "_force" in col
            or col.startswith("force_")
            or col.startswith("both_force")
            or col.startswith("best_force")
        ):
            cols.append(col)
    return cols


def cv_result(df: pd.DataFrame, target: str, features: list[str], modality: str) -> dict[str, Any]:
    if not features:
        return {
            "target": target,
            "modality": modality,
            "model": "none",
            "accuracy": np.nan,
            "accuracy_std": np.nan,
            "folds": np.nan,
            "feature_count": 0,
            "note": "no_features",
        }
    xgb = gel.xgboost_cv(df, target, features)
    result = xgb if xgb is not None else gel.nearest_centroid_cv(df, target, features)
    result = dict(result)
    result["modality"] = modality
    result["feature_count"] = len(features)
    return result


def top_eta(df: pd.DataFrame, target: str, features: list[str], modality: str) -> pd.DataFrame:
    eta = gel.eta_squared(df, target, features)
    if eta.empty:
        return eta
    eta.insert(1, "modality", modality)
    return eta


def modality_summary(cv_df: pd.DataFrame, eta_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in cv_df.iterrows():
        target = row["target"]
        modality = row["modality"]
        eta_subset = eta_df[(eta_df["target"] == target) & (eta_df["modality"] == modality)]
        rows.append(
            {
                "target": target,
                "modality": modality,
                "model": row.get("model", ""),
                "accuracy": row.get("accuracy", np.nan),
                "accuracy_std": row.get("accuracy_std", np.nan),
                "folds": row.get("folds", np.nan),
                "feature_count": row.get("feature_count", 0),
                "best_eta_squared": float(eta_subset["eta_squared"].max()) if not eta_subset.empty else np.nan,
                "note": row.get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def comparison_notes(summary: pd.DataFrame, target: str) -> list[str]:
    rows = summary[summary["target"] == target].set_index("modality")
    notes: list[str] = []
    if not {"force", "gelsight", "combined"}.issubset(rows.index):
        return [f"- {target}: missing one or more modality results."]
    force_acc = float(rows.loc["force", "accuracy"])
    gel_acc = float(rows.loc["gelsight", "accuracy"])
    combined_acc = float(rows.loc["combined", "accuracy"])
    force_eta = float(rows.loc["force", "best_eta_squared"])
    gel_eta = float(rows.loc["gelsight", "best_eta_squared"])
    acc_gap = force_acc - gel_acc
    eta_gap = force_eta - gel_eta
    combined_gain = combined_acc - max(force_acc, gel_acc)
    notes.append(
        f"- {target}: force-only accuracy={force_acc:.3f}, GelSight-only accuracy={gel_acc:.3f}, combined accuracy={combined_acc:.3f}."
    )
    notes.append(
        f"- {target}: best eta^2 force={force_eta:.3f}, GelSight={gel_eta:.3f}; gaps are accuracy={acc_gap:.3f}, eta^2={eta_gap:.3f}."
    )
    notes.append(f"- {target}: combined gain over the better single modality is {combined_gain:.3f}.")
    if gel_acc >= 0.75 * force_acc or gel_eta >= 0.60 * force_eta:
        notes.append(
            f"- {target}: GelSight is not negligible relative to force sensor under these thresholds."
        )
    else:
        notes.append(
            f"- {target}: force sensor appears substantially stronger; inspect confusion and feature quality before claiming parity."
        )
    return notes


def write_report(
    output_path: Path,
    root: Path,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    eta_df: pd.DataFrame,
) -> None:
    counts = df.groupby(["object_type", "force_level"]).size().reset_index(name="episodes")
    lines = [
        "# Sensor Modality Comparison",
        "",
        f"Root: `{root}`",
        "",
        "## Decision Notes",
        "",
        *comparison_notes(summary, "force_level"),
        "",
        "## Class Counts",
        "",
        gel.markdown_table(counts),
        "",
        "## Model Summary",
        "",
        gel.markdown_table(summary),
        "",
        "## Top Eta Squared Features",
        "",
        gel.markdown_table(eta_df.sort_values("eta_squared", ascending=False).head(30)),
        "",
        "## Feature Filtering",
        "",
        "- Model features exclude timestamp, duration, sample-rate, path, record count, frame count, and missing-data columns.",
        "- `combined` uses the union of filtered force and GelSight features.",
        "- If `model` is `nearest_centroid`, install `xgboost` and `scikit-learn` to get XGBoost CV results.",
    ]
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
        default="reports/sensor_modality_compare",
        help="Directory for features and comparison reports.",
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
        help="Pixel difference threshold for GelSight contact-area features.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=5,
        help="Number of initial GelSight frames used as image baseline.",
    )
    parser.add_argument(
        "--targets",
        default="force_level,object_type",
        help="Comma-separated targets to compare. Default: force_level,object_type.",
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
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]

    episodes = gel.discover_episodes(root)
    if not episodes:
        raise FileNotFoundError(f"No synced_data.jsonl files found under {root}")

    print(f"Found {len(episodes)} episodes under {root}", flush=True)
    print(
        f"Using max_frames={args.max_frames}, pixel_threshold={args.pixel_threshold}, "
        f"baseline_frames={args.baseline_frames}, targets={','.join(targets)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        rows.append(
            extract_all_features(
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

    print("Computing modality feature sets, eta^2, and model checks...", flush=True)
    df = pd.DataFrame(rows)
    features_path = output_dir / "sensor_features.csv"
    df.to_csv(features_path, index=False, quoting=csv.QUOTE_MINIMAL)

    cv_rows: list[dict[str, Any]] = []
    eta_frames: list[pd.DataFrame] = []
    for target in targets:
        for modality in ("force", "gelsight", "combined"):
            features = feature_columns(df, modality)
            cv_rows.append(cv_result(df, target, features, modality))
            eta = top_eta(df, target, features, modality)
            if not eta.empty:
                eta_frames.append(eta)

    cv_df = pd.DataFrame(cv_rows)
    eta_df = pd.concat(eta_frames, ignore_index=True) if eta_frames else pd.DataFrame()
    summary = modality_summary(cv_df, eta_df)

    cv_df.to_csv(output_dir / "model_check_by_modality.csv", index=False)
    eta_df.to_csv(output_dir / "eta_squared_by_modality.csv", index=False)
    summary.to_csv(output_dir / "modality_summary.csv", index=False)
    report_path = output_dir / "report.md"
    write_report(report_path, root, df, summary, eta_df)

    print(f"Saved sensor features: {features_path}")
    print(f"Saved modality summary: {output_dir / 'modality_summary.csv'}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
