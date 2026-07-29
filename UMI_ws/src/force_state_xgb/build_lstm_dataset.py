"""
Build LSTM datasets using:

Force:
    Raw temporal force sequence from synced_data.jsonl

GelSight:
    Aggregated XGBoost features from causal_windows_v3.csv

Combined:
    Temporal force features + repeated GelSight features

Output:
    X_force:
        (N, T, 12)

    X_gelsight:
        (N, G)

    X_combined:
        (N, T, 12 + G)

where:
    N = number of samples
    T = number of timesteps
    G = number of GelSight features

The event/casual dataset alignment is performed using:

    episode_id
    + evaluation_time_sec ~= prediction_time_sec

No sample_id is required in phase_events.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Labels
# ============================================================

LABELS = (
    "too_low",
    "fine",
    "too_high",
)

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}


# ============================================================
# Raw Force Features
# ============================================================

FORCE_FEATURES = [
    "force_left_fx",
    "force_left_fy",
    "force_left_fz",
    "force_left_tx",
    "force_left_ty",
    "force_left_tz",

    "force_right_fx",
    "force_right_fy",
    "force_right_fz",
    "force_right_tx",
    "force_right_ty",
    "force_right_tz",
]


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=Path(
            "reports/force_state/phase_events.csv"
        ),
        help="phase_events.csv",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "artifacts/force_state/causal_windows_v3.csv"
        ),
        help=(
            "XGBoost causal dataset containing "
            "GelSight features."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "data_yue/train"
        ),
        help=(
            "Root directory containing "
            "episode data."
        ),
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help=(
            "Causal force window length "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=20,
        help=(
            "Number of fixed timesteps "
            "per force sequence."
        ),
    )

    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=1e-3,
        help=(
            "Maximum allowed difference between "
            "evaluation_time_sec and prediction_time_sec."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/force_state/"
            "lstm_force_gelsight_v3.npz"
        ),
    )

    return parser.parse_args()


# ============================================================
# Load synced_data
# ============================================================

def load_synced_data(
    path: Path,
) -> pd.DataFrame:
    """
    Load synced_data JSONL.

    The top-level synced_data timestamp is used
    as the common time axis.

    Timestamp:
        nanoseconds -> seconds
    """

    rows = []
    base_timestamp = None

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            if not line.strip():
                continue

            item = json.loads(line)

            if "timestamp" not in item:
                continue

            raw_timestamp = (
                float(item["timestamp"]) / 1e9
            )

            if base_timestamp is None:
                base_timestamp = raw_timestamp

            row = {
                "timestamp": raw_timestamp - base_timestamp
            }

            left = item.get(
                "force_left",
                {},
            )

            right = item.get(
                "force_right",
                {},
            )

            for axis in (
                "fx",
                "fy",
                "fz",
                "tx",
                "ty",
                "tz",
            ):

                row[
                    f"force_left_{axis}"
                ] = left.get(
                    axis,
                    np.nan,
                )

                row[
                    f"force_right_{axis}"
                ] = right.get(
                    axis,
                    np.nan,
                )

            rows.append(row)

    if not rows:
        raise ValueError(
            f"No valid data found in {path}"
        )

    frame = pd.DataFrame(rows)

    frame = (
        frame
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return frame


# ============================================================
# Find synced_data
# ============================================================

def find_synced_data(
    data_root: Path,
    episode_id: str,
) -> Path | None:

    direct_candidates = [
        data_root
        / episode_id
        / "synced_data.jsonl",

        data_root
        / episode_id
        / "synced_data.json",
    ]

    for path in direct_candidates:

        if path.exists():
            return path

    matches = list(
        data_root.rglob(
            "synced_data.jsonl"
        )
    )

    for path in matches:

        if episode_id in path.parts:
            return path

    matches = list(
        data_root.rglob(
            "synced_data.json"
        )
    )

    for path in matches:

        if episode_id in path.parts:
            return path

    return None


# ============================================================
# Resample force sequence
# ============================================================

def resample_sequence(
    sequence: np.ndarray,
    timesteps: int,
) -> np.ndarray:
    """
    Resample variable-length force sequence
    to fixed T timesteps.
    """

    if sequence.shape[0] == 0:
        raise ValueError(
            "Cannot resample empty sequence."
        )

    if sequence.shape[0] == timesteps:
        return sequence.astype(
            np.float32
        )

    source_indices = np.linspace(
        0,
        sequence.shape[0] - 1,
        timesteps,
    )

    source_indices = np.round(
        source_indices
    ).astype(int)

    return sequence[
        source_indices
    ].astype(np.float32)


# ============================================================
# Extract causal force window
# ============================================================

def extract_causal_window(
    frame: pd.DataFrame,
    evaluation_time: float,
    window_sec: float,
    timesteps: int,
) -> np.ndarray | None:
    """
    Extract:

        [evaluation_time - window_sec,
         evaluation_time]

    using the top-level synced_data timestamp.

    This is strictly causal.
    """

    start_time = (
        evaluation_time
        - window_sec
    )

    window = frame[
        (frame["timestamp"] >= start_time)
        & (
            frame["timestamp"]
            <= evaluation_time
        )
    ].copy()

    if window.empty:
        return None

    values = (
        window[FORCE_FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Fill missing force values inside the window.
    # --------------------------------------------------------

    values_frame = pd.DataFrame(
        values
    )

    values_frame = (
        values_frame
        .ffill()
        .bfill()
    )

    values = values_frame.to_numpy(
        dtype=np.float32
    )

    # Completely missing features -> 0
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Fixed temporal length
    # --------------------------------------------------------

    values = resample_sequence(
        values,
        timesteps,
    )

    return values


# ============================================================
# Identify GelSight features
# ============================================================

def find_gelsight_features(
    columns: list[str],
) -> list[str]:
    """
    Identify GelSight-only features from
    the XGBoost causal dataset.

    Excludes:
        force features
        metadata
        balance features involving force
    """

    metadata = {
        "sample_id",
        "episode_id",
        "episode_dir",
        "jsonl_path",
        "object_type",
        "force_level",
        "label",
        "trial_number",
        "phase",
        "prediction_time_sec",
        "window_start_sec",
        "window_end_sec",
        "window_coverage_sec",
        "window_num_records",
    }

    gelsight_features = []

    for column in columns:

        if column in metadata:
            continue

        if not column.startswith(
            "left_gelsight_"
        ) and not column.startswith(
            "right_gelsight_"
        ):
            continue

        gelsight_features.append(
            column
        )

    return sorted(
        gelsight_features
    )


# ============================================================
# Create alignment key
# ============================================================

def make_sample_key(
    episode_id: str,
    timestamp: float,
) -> tuple[str, float]:

    return (
        str(episode_id),
        round(
            float(timestamp),
            6,
        ),
    )


# ============================================================
# Match event to causal dataset
# ============================================================

def match_causal_row(
    causal: pd.DataFrame,
    episode_id: str,
    evaluation_time: float,
    tolerance: float,
) -> pd.Series | None:
    """
    Match:

        events.evaluation_time_sec

    to:

        causal.prediction_time_sec

    within the same episode.
    """

    candidates = causal[
        causal["episode_id"].astype(str)
        == str(episode_id)
    ]

    if candidates.empty:
        return None

    differences = (
        candidates["prediction_time_sec"]
        - evaluation_time
    ).abs()

    best_index = differences.idxmin()

    best_difference = float(
        differences.loc[best_index]
    )

    if best_difference > tolerance:
        return None

    return causal.loc[best_index]


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    # ========================================================
    # Load events
    # ========================================================

    print(
        "Loading events..."
    )

    events = pd.read_csv(
        args.events
    )

    required_event_columns = {
        "episode_id",
        "object_type",
        "label",
        "evaluation_time_sec",
    }

    missing_events = (
        required_event_columns
        - set(events.columns)
    )

    if missing_events:

        raise ValueError(
            "phase_events.csv is missing "
            f"columns: {sorted(missing_events)}"
        )

    # ========================================================
    # Load XGBoost causal dataset
    # ========================================================

    print(
        "Loading XGBoost dataset..."
    )

    causal = pd.read_csv(
        args.dataset
    )

    required_causal_columns = {
        "sample_id",
        "episode_id",
        "object_type",
        "label",
        "prediction_time_sec",
    }

    missing_causal = (
        required_causal_columns
        - set(causal.columns)
    )

    if missing_causal:

        raise ValueError(
            "causal dataset is missing "
            f"columns: {sorted(missing_causal)}"
        )

    # ========================================================
    # Convert time columns
    # ========================================================

    events[
        "evaluation_time_sec"
    ] = pd.to_numeric(
        events[
            "evaluation_time_sec"
        ],
        errors="coerce",
    )

    causal[
        "prediction_time_sec"
    ] = pd.to_numeric(
        causal[
            "prediction_time_sec"
        ],
        errors="coerce",
    )

    events = events.dropna(
        subset=[
            "evaluation_time_sec"
        ]
    )

    causal = causal.dropna(
        subset=[
            "prediction_time_sec"
        ]
    )

    # ========================================================
    # Find GelSight features
    # ========================================================

    gelsight_features = (
        find_gelsight_features(
            causal.columns.tolist()
        )
    )

    if not gelsight_features:

        raise ValueError(
            "No GelSight features found "
            "in causal dataset."
        )

    print(
        f"GelSight features found: "
        f"{len(gelsight_features)}"
    )

    # ========================================================
    # Check GelSight feature values
    # ========================================================

    for column in gelsight_features:

        causal[column] = pd.to_numeric(
            causal[column],
            errors="coerce",
        )

    # ========================================================
    # Build lookup by episode
    # ========================================================

    causal_by_episode = {
        episode_id: group.copy()
        for episode_id, group
        in causal.groupby(
            causal["episode_id"].astype(str)
        )
    }

    # ========================================================
    # Storage
    # ========================================================

    X_force = []
    X_gelsight = []

    y = []

    episode_ids = []
    object_types = []
    sample_ids = []
    evaluation_times = []

    # ========================================================
    # Cache raw synced_data
    # ========================================================

    episode_cache: dict[
        str,
        pd.DataFrame | None
    ] = {}

    skipped_no_data = 0
    skipped_no_window = 0
    skipped_no_causal_match = 0

    # ========================================================
    # Process each event
    # ========================================================

    for index, event in events.iterrows():

        episode_id = str(
            event["episode_id"]
        )

        object_type = str(
            event["object_type"]
        )

        label = str(
            event["label"]
        )

        evaluation_time = float(
            event["evaluation_time_sec"]
        )

        # ----------------------------------------------------
        # Label check
        # ----------------------------------------------------

        if label not in LABEL_TO_INDEX:

            print(
                "[WARNING] Unknown label "
                f"{label} at row {index}"
            )

            continue

        # ----------------------------------------------------
        # Load synced_data
        # ----------------------------------------------------

        if episode_id not in episode_cache:

            synced_path = find_synced_data(
                args.data_root,
                episode_id,
            )

            if synced_path is None:

                print(
                    "[WARNING] Cannot find "
                    f"synced_data for episode "
                    f"{episode_id}"
                )

                episode_cache[
                    episode_id
                ] = None

            else:

                try:

                    episode_cache[
                        episode_id
                    ] = load_synced_data(
                        synced_path
                    )

                except Exception as exc:

                    print(
                        "[WARNING] Failed to load "
                        f"{synced_path}: {exc}"
                    )

                    episode_cache[
                        episode_id
                    ] = None

        data = episode_cache[
            episode_id
        ]

        if data is None:

            skipped_no_data += 1
            continue

        # ----------------------------------------------------
        # Force temporal window
        # ----------------------------------------------------

        force_sequence = (
            extract_causal_window(
                frame=data,
                evaluation_time=evaluation_time,
                window_sec=args.window_sec,
                timesteps=args.timesteps,
            )
        )

        if force_sequence is None:

            print(
                "[WARNING] Empty force causal "
                f"window for episode "
                f"{episode_id} @ "
                f"{evaluation_time:.6f}"
            )

            skipped_no_window += 1
            continue

        # ----------------------------------------------------
        # Match XGBoost causal sample
        # ----------------------------------------------------

        episode_causal = (
            causal_by_episode.get(
                episode_id
            )
        )

        if episode_causal is None:

            print(
                "[WARNING] No causal dataset "
                f"entry for episode "
                f"{episode_id}"
            )

            skipped_no_causal_match += 1
            continue

        causal_row = match_causal_row(
            causal=episode_causal,
            episode_id=episode_id,
            evaluation_time=evaluation_time,
            tolerance=args.time_tolerance,
        )

        if causal_row is None:

            print(
                "[WARNING] No causal match "
                f"for episode {episode_id} "
                f"@ {evaluation_time:.6f}"
            )

            skipped_no_causal_match += 1
            continue

        # ----------------------------------------------------
        # GelSight features
        # ----------------------------------------------------

        gelsight_vector = (
            causal_row[
                gelsight_features
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        # ----------------------------------------------------
        # Handle missing GelSight values
        # ----------------------------------------------------

        gelsight_vector = np.nan_to_num(
            gelsight_vector,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ----------------------------------------------------
        # Save sample
        # ----------------------------------------------------

        X_force.append(
            force_sequence
        )

        X_gelsight.append(
            gelsight_vector
        )

        y.append(
            LABEL_TO_INDEX[label]
        )

        episode_ids.append(
            episode_id
        )

        object_types.append(
            object_type
        )

        sample_ids.append(
            str(
                causal_row["sample_id"]
            )
        )

        evaluation_times.append(
            evaluation_time
        )

    # ========================================================
    # Check result
    # ========================================================

    if not X_force:

        raise RuntimeError(
            "No valid LSTM samples were created."
        )

    # ========================================================
    # Convert arrays
    # ========================================================

    X_force = np.stack(
        X_force
    ).astype(
        np.float32
    )

    X_gelsight = np.stack(
        X_gelsight
    ).astype(
        np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    episode_ids = np.asarray(
        episode_ids
    )

    object_types = np.asarray(
        object_types
    )

    sample_ids = np.asarray(
        sample_ids
    )

    evaluation_times = np.asarray(
        evaluation_times,
        dtype=np.float64,
    )

    # ========================================================
    # Combined representation
    #
    # Repeat the same GelSight vector at every
    # temporal timestep.
    #
    # Force:
    #     (N, T, 12)
    #
    # GelSight:
    #     (N, G)
    #
    # Repeated GelSight:
    #     (N, T, G)
    #
    # Combined:
    #     (N, T, 12 + G)
    # ========================================================

    N = X_force.shape[0]
    T = X_force.shape[1]
    G = X_gelsight.shape[1]

    X_gelsight_temporal = np.repeat(
        X_gelsight[:, None, :],
        T,
        axis=1,
    )

    X_combined = np.concatenate(
        [
            X_force,
            X_gelsight_temporal,
        ],
        axis=2,
    ).astype(
        np.float32
    )

    # ========================================================
    # Output
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,

        # Force temporal sequence
        X_force=X_force,

        # Static/aggregated GelSight features
        X_gelsight=X_gelsight,

        # Force + GelSight temporal input
        X_combined=X_combined,

        # Labels
        y=y,

        # Metadata
        episode_ids=episode_ids,
        object_types=object_types,
        sample_ids=sample_ids,
        evaluation_times=evaluation_times,

        # Feature names
        force_feature_names=np.asarray(
            FORCE_FEATURES
        ),

        gelsight_feature_names=np.asarray(
            gelsight_features
        ),

        combined_feature_names=np.asarray(
            FORCE_FEATURES
            + gelsight_features
        ),

        labels=np.asarray(
            LABELS
        ),
    )

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "LSTM Force + GelSight dataset created"
    )

    print(
        "=" * 70
    )

    print(
        f"Output: {args.output}"
    )

    print()

    print(
        f"Samples: {N}"
    )

    print(
        f"Timesteps: {T}"
    )

    print(
        f"Force features: {X_force.shape[-1]}"
    )

    print(
        f"GelSight features: {G}"
    )

    print(
        f"Combined features: "
        f"{X_combined.shape[-1]}"
    )

    print()

    print(
        f"X_force shape: "
        f"{X_force.shape}"
    )

    print(
        f"X_gelsight shape: "
        f"{X_gelsight.shape}"
    )

    print(
        f"X_combined shape: "
        f"{X_combined.shape}"
    )

    print(
        f"y shape: "
        f"{y.shape}"
    )

    print()

    print(
        "Skipped samples:"
    )

    print(
        f"  no synced_data: "
        f"{skipped_no_data}"
    )

    print(
        f"  empty force window: "
        f"{skipped_no_window}"
    )

    print(
        f"  no causal match: "
        f"{skipped_no_causal_match}"
    )

    print()

    print(
        "Label distribution:"
    )

    counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    for label_index, count in counts.items():

        print(
            f"  {LABELS[label_index]}: "
            f"{count}"
        )

    print()

    # ========================================================
    # Alignment quality
    # ========================================================

    print(
        "Alignment check:"
    )

    print(
        f"  phase events: "
        f"{len(events)}"
    )

    print(
        f"  causal dataset: "
        f"{len(causal)}"
    )

    print(
        f"  final samples: "
        f"{N}"
    )

    print()

    print(
        "Feature structure:"
    )

    print(
        "  Force:"
        " synced_data temporal sequence"
    )

    print(
        "  GelSight:"
        " XGBoost causal features"
    )

    print(
        "  Combined:"
        " Force sequence + repeated GelSight features"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()