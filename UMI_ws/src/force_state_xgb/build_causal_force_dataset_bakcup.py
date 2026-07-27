"""Step 4: extract features only from the one second ending at t_eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from force_state_common import Episode, extract_window, feature_columns, load_signals, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("reports/force_state/phase_events.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/force_state/causal_windows.csv"))
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--samples-per-episode", type=int, default=1)
    parser.add_argument("--sample-span-sec", type=float, default=0.2)
    parser.add_argument("--pixel-threshold", type=float, default=8.0)
    parser.add_argument("--force-abs-limit", type=float, default=10.0)
    parser.add_argument("--gelsight-baseline-start-sec", type=float, default=0.5)
    parser.add_argument("--gelsight-baseline-end-sec", type=float, default=3.0)
    parser.add_argument("--max-window-outlier-ratio", type=float, default=0)
    parser.add_argument("--include-rejected", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.events)
    rows, skipped = [], []
    for row_index, event in events.iterrows():
        if event["phase"] == "error":
            skipped.append({"episode_id": event["episode_id"], "reason": "phase_error"})
            continue
        if not args.include_rejected and event.get("review_status") == "rejected":
            continue
        path = Path(str(event["jsonl_path"]))
        episode = Episode(path, path.parent, str(event["episode_id"]),
                          str(event["object_type"]), str(event["force_level"]),
                          str(event["label"]), str(event["trial_number"]))
        try:
            _, times, signals = load_signals(
                episode, pixel_threshold=args.pixel_threshold,
                force_abs_limit=args.force_abs_limit,
                gelsight_baseline_start_sec=args.gelsight_baseline_start_sec,
                gelsight_baseline_end_sec=args.gelsight_baseline_end_sec,
            )
        except Exception as exc:
            skipped.append({"episode_id": episode.episode_id,
                            "reason": f"load_error:{type(exc).__name__}"})
            continue
        evaluation = float(event["evaluation_time_sec"])
        if args.samples_per_episode == 1:
            sample_times = [evaluation]
        else:
            sample_times = np.linspace(max(args.window_sec, evaluation - args.sample_span_sec),
                                       evaluation, args.samples_per_episode)
        for sample_index, prediction_time in enumerate(sample_times, 1):
            features, coverage, count = extract_window(times, signals, float(prediction_time),
                                                       args.window_sec)
            if not count or coverage < args.min_coverage:
                skipped.append({"episode_id": episode.episode_id,
                                "reason": f"coverage:{coverage:.3f}"})
                continue
            outlier_ratio = max(
                float(features.get("left_force_outlier_mean", 0.0)),
                float(features.get("right_force_outlier_mean", 0.0)),
            )
            if outlier_ratio > args.max_window_outlier_ratio:
                skipped.append({"episode_id": episode.episode_id,
                                "reason": f"force_outlier:{outlier_ratio:.3f}"})
                continue
            output = {
                "sample_id": f"{episode.episode_id}__{sample_index:02d}",
                "episode_id": episode.episode_id, "episode_dir": str(episode.episode_dir),
                "jsonl_path": str(episode.jsonl_path), "object_type": episode.object_type,
                "force_level": episode.force_level, "label": episode.label,
                "trial_number": episode.trial_number, "phase": str(event["phase"]),
                "prediction_time_sec": float(prediction_time),
                "window_start_sec": float(prediction_time - args.window_sec),
                "window_end_sec": float(prediction_time),
            }
            output.update(features)
            rows.append(output)
        if (row_index + 1) % 25 == 0 or row_index + 1 == len(events):
            print(f"Processed {row_index + 1}/{len(events)}")
    if not rows:
        raise RuntimeError("No feature rows generated")
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    write_json(args.output.with_suffix(".features.json"), {
        "window_sec": args.window_sec, "pixel_threshold": args.pixel_threshold,
        "force_abs_limit": args.force_abs_limit,
        "gelsight_baseline_start_sec": args.gelsight_baseline_start_sec,
        "gelsight_baseline_end_sec": args.gelsight_baseline_end_sec,
        "rows": len(frame), "episodes": int(frame["episode_id"].nunique()),
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
        "force_features": feature_columns(frame.columns, "force"),
        "gelsight_features": feature_columns(frame.columns, "gelsight"),
        "combined_features": feature_columns(frame.columns, "combined"),
        "skipped": skipped,
    })
    print(frame.groupby(["label", "phase"]).size().to_string())
    print(f"Wrote {args.output}; rows={len(frame)}, episodes={frame['episode_id'].nunique()}")


if __name__ == "__main__":
    main()
