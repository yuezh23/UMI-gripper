"""Load and average the seven combined leave-one-object-out XGBoost models."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_LABELS = ("too_low", "fine", "too_high")


@dataclass(frozen=True)
class FoldSpec:
    name: str
    model_path: Path
    metadata_path: Path
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class EnsemblePrediction:
    label: str
    probabilities: dict[str, float]
    confidence: float
    model_agreement: float
    fold_labels: tuple[str, ...]


def discover_fold_specs(model_root: Path) -> list[FoldSpec]:
    """Validate metadata without importing xgboost."""

    root = Path(model_root).expanduser().resolve()
    specs: list[FoldSpec] = []
    for combined_dir in sorted(root.glob("test_*/combined")):
        model_path = combined_dir / "model.json"
        metadata_path = combined_dir / "model_metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            continue
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        labels = tuple(metadata.get("labels", ()))
        if labels != EXPECTED_LABELS:
            raise ValueError(
                f"{metadata_path}: expected labels {EXPECTED_LABELS}, got {labels}"
            )
        if metadata.get("modality") != "combined":
            raise ValueError(f"{metadata_path}: modality must be combined")
        if float(metadata.get("window_sec", -1)) != 1.0:
            raise ValueError(f"{metadata_path}: window_sec must be 1.0")
        columns = tuple(map(str, metadata.get("feature_columns", ())))
        if len(columns) != 100 or len(set(columns)) != 100:
            raise ValueError(
                f"{metadata_path}: expected 100 unique selected features, got {len(columns)}"
            )
        specs.append(
            FoldSpec(
                name=combined_dir.parent.name,
                model_path=model_path,
                metadata_path=metadata_path,
                feature_columns=columns,
            )
        )
    if len(specs) != 7:
        raise ValueError(f"Expected 7 combined LOO folds under {root}, found {len(specs)}")
    return specs


class XGBForceStateEnsemble:
    """Seven-fold soft-voting ensemble with fold-specific Top-100 features."""

    def __init__(
        self,
        model_root: Path,
        *,
        n_jobs: int = 1,
        fold_workers: int = 7,
    ):
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least 1")
        if fold_workers < 1:
            raise ValueError("fold_workers must be at least 1")
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is required for realtime inference. Install the model-compatible "
                "version (the current artifacts were saved by XGBoost 3.3.0)."
            ) from exc

        self.specs = discover_fold_specs(model_root)
        self.fold_workers = min(fold_workers, len(self.specs))
        self.models: list[Any] = []
        for spec in self.specs:
            model = XGBClassifier(n_jobs=n_jobs)
            model.load_model(spec.model_path)
            # Loading restores the training-time booster configuration. Reset
            # the runtime thread count: one-row inference is faster and causes
            # much less ROS callback contention with a single thread per fold.
            model.set_params(n_jobs=n_jobs)
            self.models.append(model)

    def predict(self, features: dict[str, float]) -> EnsemblePrediction:
        def predict_fold(spec: FoldSpec, model: Any) -> np.ndarray:
            matrix = np.asarray(
                [[features.get(name, np.nan) for name in spec.feature_columns]],
                dtype=float,
            )
            probabilities = np.asarray(model.predict_proba(matrix)[0], dtype=float)
            if probabilities.shape != (len(EXPECTED_LABELS),):
                raise RuntimeError(
                    f"{spec.name}: expected 3 probabilities, got {probabilities.shape}"
                )
            return probabilities

        with ThreadPoolExecutor(max_workers=self.fold_workers) as executor:
            fold_probabilities = list(
                executor.map(predict_fold, self.specs, self.models)
            )
        fold_label_indices = [
            int(np.argmax(probabilities))
            for probabilities in fold_probabilities
        ]

        mean_probability = np.mean(np.stack(fold_probabilities), axis=0)
        label_index = int(np.argmax(mean_probability))
        agreement = float(np.mean(np.asarray(fold_label_indices) == label_index))
        return EnsemblePrediction(
            label=EXPECTED_LABELS[label_index],
            probabilities={
                label: float(mean_probability[index])
                for index, label in enumerate(EXPECTED_LABELS)
            },
            confidence=float(mean_probability[label_index]),
            model_agreement=agreement,
            fold_labels=tuple(EXPECTED_LABELS[index] for index in fold_label_indices),
        )
