"""Step 1: audit all synchronized training episodes and their schema."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from force_state_common import discover_episodes, read_jsonl, relative_times, schema_union, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_yue/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/force_state"))
    return parser.parse_args()


def valid_ratio(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([isinstance(row.get(key), dict) for row in records])) if records else 0.0


def main() -> None:
    args = parse_args()
    episodes = discover_episodes(args.data_root)
    if not episodes:
        raise FileNotFoundError(f"No synced_data.jsonl below {args.data_root}")
    rows, schema = [], {}
    for index, episode in enumerate(episodes, 1):
        records = read_jsonl(episode.jsonl_path)
        times = relative_times(records)
        finite = times[np.isfinite(times)]
        duration = float(finite[-1] - finite[0]) if finite.size >= 2 else 0.0
        ratios = {key: valid_ratio(records, key) for key in
                  ("force_left", "force_right", "gelsight_left", "gelsight_right")}
        flags = []
        if not records:
            flags.append("empty")
        if duration < 10:
            flags.append("short_duration")
        for key, ratio in ratios.items():
            if ratio < 0.95:
                flags.append(f"low_{key}_ratio")
        rows.append({
            "episode_id": episode.episode_id, "episode_dir": str(episode.episode_dir),
            "jsonl_path": str(episode.jsonl_path), "object_type": episode.object_type,
            "force_level": episode.force_level, "label": episode.label,
            "trial_number": episode.trial_number, "duration_sec": duration,
            "num_synced_records": len(records),
            "left_force_valid_ratio": ratios["force_left"],
            "right_force_valid_ratio": ratios["force_right"],
            "left_gelsight_valid_ratio": ratios["gelsight_left"],
            "right_gelsight_valid_ratio": ratios["gelsight_right"],
            "quality_flag": ";".join(flags) if flags else "ok",
        })
        for parent, keys in schema_union(records).items():
            schema.setdefault(parent, set()).update(keys)
        if index % 25 == 0 or index == len(episodes):
            print(f"Inspected {index}/{len(episodes)}")
    output = pd.DataFrame(rows).sort_values(["object_type", "force_level", "trial_number"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "dataset_manifest.csv", index=False)
    write_json(args.output_dir / "schema_summary.json", {
        "episodes_total": len(output),
        "object_counts": dict(Counter(output["object_type"])),
        "label_counts": dict(Counter(output["label"])),
        "quality_counts": dict(Counter(output["quality_flag"])),
        "schema": {key: sorted(value) for key, value in sorted(schema.items())},
    })
    print(output.groupby(["object_type", "label"]).size().to_string())
    print(f"Wrote {args.output_dir / 'dataset_manifest.csv'}")


if __name__ == "__main__":
    main()
