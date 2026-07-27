"""Step 5: train force-only, GelSight-only, and combined XGBoost models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from force_state_common import LABELS, feature_columns, write_json


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/force_state/causal_windows.csv"),
    )
    parser.add_argument(
        "--split",
        choices=("stratified_episode", "leave_one_object_out"),
        default="stratified_episode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/force_state/models"),
    )
    parser.add_argument(
        "--modalities",
        default="force,gelsight,combined",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
    )
    return parser.parse_args()


def split_episode_ids(
    frame: pd.DataFrame,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[set[str], set[str], set[str]]:
    episodes = frame[
        ["episode_id", "object_type", "label"]
    ].drop_duplicates("episode_id")

    rng = np.random.default_rng(seed)
    train, val, test = set(), set(), set()

    for _, group in episodes.groupby(
        ["object_type", "label"],
        sort=True,
    ):
        ids = group["episode_id"].astype(str).to_numpy()
        rng.shuffle(ids)

        n_test = (
            max(1, round(len(ids) * test_fraction))
            if test_fraction and len(ids) >= 3
            else 0
        )

        n_val = (
            max(1, round(len(ids) * val_fraction))
            if len(ids) - n_test >= 2
            else 0
        )

        test.update(ids[:n_test])
        val.update(ids[n_test:n_test + n_val])
        train.update(ids[n_test + n_val:])

    return train, val, test


def folds(
    frame: pd.DataFrame,
    args: argparse.Namespace,
):
    if args.split == "stratified_episode":
        yield (
            "within_object",
            *split_episode_ids(
                frame,
                args.seed,
                args.val_fraction,
                args.test_fraction,
            ),
        )
        return

    # Leave-One-Object-Out:
    # 6 objects are used for train/validation,
    # 1 object is completely held out for testing.
    for index, object_name in enumerate(
        sorted(frame["object_type"].unique())
    ):
        test = set(
            frame.loc[
                frame["object_type"] == object_name,
                "episode_id",
            ].astype(str)
        )

        remaining = frame[
            ~frame["episode_id"].astype(str).isin(test)
        ]

        train, val, _ = split_episode_ids(
            remaining,
            args.seed + index,
            args.val_fraction,
            0,
        )

        yield (
            f"test_{object_name}",
            train,
            val,
            test,
        )


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): clean(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [clean(v) for v in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return (
            float(value)
            if np.isfinite(value)
            else None
        )

    return value


def train(
    frame: pd.DataFrame,
    columns: list[str],
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    output: Path,
    modality: str,
    args: argparse.Namespace,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from xgboost import XGBClassifier

    parts = [
        frame[
            frame["episode_id"].astype(str).isin(ids)
        ].copy()
        for ids in (train_ids, val_ids, test_ids)
    ]

    train_set, val_set, test_set = parts
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Stage 1:
    # Determine usable candidate features using TRAINING DATA ONLY
    # ============================================================

    candidate_features = []

    for column in columns:
        values = pd.to_numeric(
            train_set[column],
            errors="coerce",
        )

        if (
            values.notna().mean() >= 0.10
            and values.nunique(dropna=True) > 1
        ):
            candidate_features.append(column)

    if not candidate_features:
        raise ValueError(
            f"No usable {modality} features"
        )

    print(
        f"{modality}: candidate features = "
        f"{len(candidate_features)}"
    )

    def matrix_from_columns(
        data: pd.DataFrame,
        feature_list: list[str],
    ) -> np.ndarray:
        return (
            data[feature_list]
            .apply(pd.to_numeric, errors="coerce")
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .to_numpy(float)
        )

    # ============================================================
    # Stage 2:
    # Feature selection using TRAINING DATA ONLY
    #
    # IMPORTANT:
    # Validation and test objects are NOT used here.
    # ============================================================

    selection_matrix = matrix_from_columns(
        train_set,
        candidate_features,
    )

    selection_y = (
        train_set["label"]
        .map(LABEL_TO_INDEX)
        .to_numpy(int)
    )

    selection_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
        missing=np.nan,
        reg_alpha=1,
        reg_lambda=2,
    )

    selection_model.fit(
        selection_matrix,
        selection_y,
        verbose=False,
    )

    importance = pd.DataFrame(
        {
            "feature": candidate_features,
            "importance": (
                selection_model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    top_k = min(
        args.top_k,
        len(importance),
    )

    usable = (
        importance
        .head(top_k)["feature"]
        .tolist()
    )

    print(
        f"{modality}: selected top-{len(usable)} features"
    )

    # Save the feature-selection result for this LOO fold.
    importance.to_csv(
        output / "feature_selection_importance.csv",
        index=False,
    )

    pd.DataFrame(
        {"feature": usable}
    ).to_csv(
        output / "selected_top_features.csv",
        index=False,
    )

    # ============================================================
    # Stage 3:
    # Build matrices using ONLY selected Top-K features
    # ============================================================

    def matrix(data: pd.DataFrame) -> np.ndarray:
        return (
            data[usable]
            .apply(pd.to_numeric, errors="coerce")
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .to_numpy(float)
        )

    train_matrix = matrix(train_set)
    val_matrix = matrix(val_set)
    test_matrix = matrix(test_set)

    print(
        f"{modality}: train NaN ratio = "
        f"{np.isnan(train_matrix).mean():.3f}"
    )

    y = [
        part["label"]
        .map(LABEL_TO_INDEX)
        .to_numpy(int)
        for part in parts
    ]

    # ============================================================
    # Stage 4:
    # Final XGBoost training
    # ============================================================

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
        missing=np.nan,
        reg_alpha=1,
        reg_lambda=2,
    )

    model.fit(
        train_matrix,
        y[0],
        eval_set=[(val_matrix, y[1])],
        verbose=False,
    )

    probabilities = model.predict_proba(
        test_matrix
    )

    prediction = np.argmax(
        probabilities,
        axis=1,
    )

    train_prediction = model.predict(
        train_matrix
    )

    val_prediction = model.predict(
        val_matrix
    )

    train_accuracy = accuracy_score(
        y[0],
        train_prediction,
    )

    validation_accuracy = accuracy_score(
        y[1],
        val_prediction,
    )

    test_accuracy = accuracy_score(
        y[2],
        prediction,
    )

    metrics = clean(
        {
            "modality": modality,
            "accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "macro_f1": f1_score(
                y[2],
                prediction,
                labels=range(3),
                average="macro",
            ),
            "classification_report": classification_report(
                y[2],
                prediction,
                labels=range(3),
                target_names=LABELS,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y[2],
                prediction,
                labels=range(3),
            ).tolist(),
            "train_episodes": len(train_ids),
            "validation_episodes": len(val_ids),
            "test_episodes": len(test_ids),
            "candidate_features": len(candidate_features),
            "features": len(usable),
        }
    )

    model.save_model(
        output / "model.json"
    )

    write_json(
        output / "metrics.json",
        metrics,
    )

    write_json(
        output / "model_metadata.json",
        clean(
            {
                "labels": list(LABELS),
                "modality": modality,
                "feature_columns": usable,

                # Keep window_sec at the original value.
                "window_sec": 1.0,

                "pixel_threshold": 8.0,
                "top_k": args.top_k,
                "candidate_features": len(
                    candidate_features
                ),
                "parameters": model.get_params(),
            }
        ),
    )

    predictions = test_set[
        [
            "sample_id",
            "episode_id",
            "object_type",
            "label",
        ]
    ].copy()

    predictions["prediction"] = [
        LABELS[value]
        for value in prediction
    ]

    for index, label in enumerate(LABELS):
        predictions[f"prob_{label}"] = (
            probabilities[:, index]
        )

    predictions.to_csv(
        output / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "feature": usable,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    ).to_csv(
        output / "feature_importance.csv",
        index=False,
    )

    return metrics


def main() -> None:
    args = parse_args()

    frame = pd.read_csv(
        args.dataset
    )

    modalities = [
        value.strip()
        for value in args.modalities.split(",")
        if value.strip()
    ]

    summary = []

    for (
        fold_name,
        train_ids,
        val_ids,
        test_ids,
    ) in folds(frame, args):

        # Check episode-level leakage.
        if (
            train_ids & val_ids
            or train_ids & test_ids
            or val_ids & test_ids
        ):
            raise RuntimeError(
                f"Episode leakage in {fold_name}"
            )

        fold_dir = (
            args.output_dir / fold_name
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        split_rows = (
            [
                {
                    "episode_id": x,
                    "split": name,
                }
                for x in sorted(ids)
            ]
            for name, ids in (
                ("train", train_ids),
                ("validation", val_ids),
                ("test", test_ids),
            )
        )

        pd.DataFrame(
            [
                row
                for group in split_rows
                for row in group
            ]
        ).to_csv(
            fold_dir / "episode_split.csv",
            index=False,
        )

        for modality in modalities:

            metrics = train(
                frame,
                feature_columns(
                    frame.columns,
                    modality,
                ),
                train_ids,
                val_ids,
                test_ids,
                fold_dir / modality,
                modality,
                args,
            )

            metrics["fold"] = fold_name
            summary.append(metrics)

            print(
                f"{fold_name}/{modality}: "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"train={metrics['train_accuracy']:.3f}, "
                f"val={metrics['validation_accuracy']:.3f}, "
                f"test={metrics['accuracy']:.3f}, "
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        args.output_dir
        / "metrics_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            clean(summary),
            handle,
            indent=2,
        )

    pd.DataFrame(
        [
            {
                key: row[key]
                for key in (
                    "fold",
                    "modality",
                    "accuracy",
                    "macro_f1",
                    "test_episodes",
                    "features",
                )
            }
            for row in summary
        ]
    ).to_csv(
        args.output_dir
        / "metrics_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()