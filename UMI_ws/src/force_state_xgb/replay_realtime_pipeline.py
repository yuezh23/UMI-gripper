#!/usr/bin/env python3
"""Replay a recorded episode through the live synchronizer and feature extractor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from online_force_features import OnlineFeatureExtractor
from realtime_sensor_buffer import ForceSample, ImageSample, RealtimeSensorBuffer
from xgb_ensemble import XGBForceStateEnsemble


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_episode(episode: Path) -> RealtimeSensorBuffer:
    buffer = RealtimeSensorBuffer(retention_sec=120.0)
    for side in ("left", "right"):
        force_path = episode / f"recording_force_mms101_{side}.jsonl"
        for row in read_jsonl(force_path):
            buffer.append(
                f"force_{side}",
                ForceSample(
                    timestamp_ns=int(row["stamp"]),
                    fx=float(row["fx"]),
                    fy=float(row["fy"]),
                    fz=float(row["fz"]),
                    tx=float(row["tx"]),
                    ty=float(row["ty"]),
                    tz=float(row["tz"]),
                ),
            )
        gel_path = episode / f"recording_gelsight_{side}.jsonl"
        for row in read_jsonl(gel_path):
            with Image.open(episode / row["image"]) as image:
                gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
            buffer.append(
                f"gelsight_{side}",
                ImageSample(timestamp_ns=int(row["timestamp"]), gray=gray),
            )
    return buffer


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--session-offset-sec", type=float, default=0.0)
    parser.add_argument(
        "--prediction-sec",
        type=float,
        default=4.47,
        help="Default avoids the recorder's periodic disk-flush gap in the supplied rate test.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=workspace
        / "artifacts/force_state/leave_one_object_out_top100_strict_v3",
    )
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = args.episode.expanduser().resolve()
    buffer = load_episode(episode)
    latest = buffer.latest_timestamps()
    first_force = read_jsonl(episode / "recording_force_mms101_left.jsonl")[0]
    session_start_ns = int(first_force["stamp"]) + int(args.session_offset_sec * 1e9)
    baseline_end_ns = session_start_ns + 3_000_000_000
    baseline_samples = buffer.snapshot_synced(
        session_start_ns,
        baseline_end_ns,
        tolerance_ns=100_000_000,
    )
    extractor = OnlineFeatureExtractor()
    baseline = extractor.build_baseline(baseline_samples, session_start_ns)

    prediction_ns = session_start_ns + int(args.prediction_sec * 1e9)
    window_samples = buffer.snapshot_synced(
        prediction_ns - 1_000_000_000,
        prediction_ns,
        tolerance_ns=100_000_000,
    )
    window = extractor.extract(window_samples, prediction_ns, baseline)
    print(f"episode={episode}")
    print(f"latest_timestamps={latest}")
    print(f"baseline_records={len(baseline_samples)}")
    print(
        f"window_valid={window.valid} reason={window.reason} "
        f"coverage={window.coverage_sec:.3f}s records={window.num_records} "
        f"features={len(window.features)}"
    )
    if not window.valid or args.skip_model:
        return
    prediction = XGBForceStateEnsemble(args.model_root).predict(window.features)
    print(f"label={prediction.label}")
    print(f"probabilities={prediction.probabilities}")
    print(f"agreement={prediction.model_agreement:.3f}")
    print(f"fold_labels={prediction.fold_labels}")


if __name__ == "__main__":
    main()
