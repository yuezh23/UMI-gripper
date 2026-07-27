"""Step 3: search the action recording and use a pre-lift fallback if needed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from force_state_common import detect_phase, discover_episodes, load_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_yue/train"))
    parser.add_argument("--output", type=Path, default=Path("reports/force_state/phase_events.csv"))
    parser.add_argument("--grasp-start-sec", type=float, default=0.0)
    # This is the latest time included in phase search, not a feature window.
    parser.add_argument("--grasp-end-sec", type=float, default=15.0)
    parser.add_argument("--fallback-eval-sec", type=float, default=3.0,
                        help="No-contact fallback uses the window ending here; default is (2, 3].")
    parser.add_argument("--history-sec", type=float, default=1.0)
    parser.add_argument("--force-threshold", type=float, default=0.15)
    parser.add_argument("--gelsight-threshold", type=float, default=0.15)
    parser.add_argument("--stable-slope", type=float, default=0.20)
    parser.add_argument("--stable-sec", type=float, default=1.0)
    parser.add_argument("--pixel-threshold", type=float, default=8.0)
    parser.add_argument("--force-abs-limit", type=float, default=10.0)
    parser.add_argument("--gelsight-baseline-start-sec", type=float, default=0.5)
    parser.add_argument("--gelsight-baseline-end-sec", type=float, default=3.0)
    parser.add_argument("--evaluation-mode", choices=("adaptive", "nominal", "settled"), default="adaptive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    episodes = discover_episodes(args.data_root)
    for index, episode in enumerate(episodes, 1):
        flags = []
        error_detail = ""
        try:
            _, times, signals = load_signals(
                episode, pixel_threshold=args.pixel_threshold,
                force_abs_limit=args.force_abs_limit,
                gelsight_baseline_start_sec=args.gelsight_baseline_start_sec,
                gelsight_baseline_end_sec=args.gelsight_baseline_end_sec,
            )
            phase = detect_phase(times, signals, args.grasp_start_sec, args.grasp_end_sec,
                                 args.force_threshold, args.gelsight_threshold,
                                 args.stable_slope, args.stable_sec, args.evaluation_mode,
                                 fallback_evaluation_sec=args.fallback_eval_sec)
            if phase.evaluation_time_sec < args.history_sec:
                flags.append("insufficient_history")
            left_outlier_ratio = float(np.nanmean(signals["left_force_outlier"]))
            right_outlier_ratio = float(np.nanmean(signals["right_force_outlier"]))
            if max(left_outlier_ratio, right_outlier_ratio) > 0:
                flags.append("force_outlier_present")
        except Exception as exc:
            phase = None
            flags.append(f"error:{type(exc).__name__}")
            error_detail = str(exc)
        rows.append({
            "episode_id": episode.episode_id, "episode_dir": str(episode.episode_dir),
            "jsonl_path": str(episode.jsonl_path), "object_type": episode.object_type,
            "force_level": episode.force_level, "label": episode.label,
            "trial_number": episode.trial_number,
            "contact_start_sec": phase.contact_start_sec if phase else None,
            "evaluation_time_sec": phase.evaluation_time_sec if phase else args.grasp_end_sec,
            "phase": phase.phase if phase else "error",
            "phase_confidence": phase.confidence if phase else 0.0,
            "left_force_outlier_ratio": left_outlier_ratio if phase else None,
            "right_force_outlier_ratio": right_outlier_ratio if phase else None,
            "grasp_start_sec": args.grasp_start_sec, "grasp_end_sec": args.grasp_end_sec,
            "fallback_evaluation_sec": args.fallback_eval_sec,
            "quality_flag": ";".join(flags) if flags else "ok",
            "error_detail": error_detail,
            "review_status": "pending" if flags else "auto_ok",
        })
        if index % 25 == 0 or index == len(episodes):
            print(f"Detected {index}/{len(episodes)}")
    frame = pd.DataFrame(rows).sort_values(["object_type", "force_level", "trial_number"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.groupby(["label", "phase"]).size().to_string())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
