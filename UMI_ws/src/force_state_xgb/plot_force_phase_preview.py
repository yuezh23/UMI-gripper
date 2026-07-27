"""Step 2: plot representative timelines to review grasp phase bounds."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from force_state_common import detect_phase, discover_episodes, load_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_yue/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/force_state/phase_preview"))
    parser.add_argument("--per-object-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grasp-start-sec", type=float, default=3.0)
    parser.add_argument("--grasp-end-sec", type=float, default=6.2)
    parser.add_argument("--force-threshold", type=float, default=0.15)
    parser.add_argument("--gelsight-threshold", type=float, default=0.15)
    parser.add_argument("--stable-slope", type=float, default=0.20)
    parser.add_argument("--stable-sec", type=float, default=1.0)
    parser.add_argument("--pixel-threshold", type=float, default=8.0)
    parser.add_argument("--force-abs-limit", type=float, default=10.0)
    parser.add_argument("--gelsight-baseline-start-sec", type=float, default=0.5)
    parser.add_argument("--gelsight-baseline-end-sec", type=float, default=3.0)
    parser.add_argument("--evaluation-mode", choices=("nominal", "settled", "adaptive"), default="nominal")
    parser.add_argument("--fallback-eval-sec", type=float, default=3.0, help="Fallback evaluation time when no contact is detected.")
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt

    args = parse_args()
    grouped = defaultdict(list)
    for episode in discover_episodes(args.data_root):
        grouped[(episode.object_type, episode.force_level)].append(episode)
    rng, selected = random.Random(args.seed), []
    for key in sorted(grouped):
        rng.shuffle(grouped[key])
        selected.extend(grouped[key][:args.per_object_class])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, episode in enumerate(selected, 1):
        _, times, signals = load_signals(
            episode, pixel_threshold=args.pixel_threshold,
            force_abs_limit=args.force_abs_limit,
            gelsight_baseline_start_sec=args.gelsight_baseline_start_sec,
            gelsight_baseline_end_sec=args.gelsight_baseline_end_sec,
        )
        phase = detect_phase(times, signals, args.grasp_start_sec, args.grasp_end_sec,
                             args.force_threshold, args.gelsight_threshold,
                             args.stable_slope, args.stable_sec, args.evaluation_mode, args.fallback_eval_sec)
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(times, signals["left_force_norm_delta"], label="left force delta")
        axes[0].plot(times, signals["right_force_norm_delta"], label="right force delta")
        axes[1].plot(times, signals["left_gelsight_diff_mean"], label="left GelSight diff")
        axes[1].plot(times, signals["right_gelsight_diff_mean"], label="right GelSight diff")
        axes[0].set_ylabel("Force norm delta")
        axes[1].set_ylabel("GelSight residual difference")
        axes[1].set_xlabel("Seconds")
        for axis in axes:
            axis.axvspan(args.grasp_start_sec, args.grasp_end_sec, color="0.9")
            axis.axvline(phase.evaluation_time_sec, color="red", linestyle="--", label="t_eval")
            if phase.contact_start_sec is not None:
                axis.axvline(phase.contact_start_sec, color="green", linestyle=":", label="contact")
            axis.legend(loc="upper right")
            axis.grid(alpha=0.25)
        figure.suptitle(f"{episode.object_type}/{episode.force_level}/{episode.trial_number} | "
                       f"{phase.phase}, t_eval={phase.evaluation_time_sec:.2f}s")
        figure.tight_layout()
        path = args.output_dir / f"{episode.object_type}__{episode.force_level}__{episode.trial_number}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        print(f"[{index}/{len(selected)}] {path}")


if __name__ == "__main__":
    main()
