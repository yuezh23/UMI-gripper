"""Deployment-time calibration for the realtime three-class XGBoost output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Mapping


LABELS = ("too_low", "fine", "too_high")


@dataclass(frozen=True)
class CalibratedDecision:
    label: str
    raw_label: str
    too_low_probability_median: float
    history_count: int
    history_size: int
    too_low_probability_threshold: float
    ready: bool


class ProbabilityDecisionCalibrator:
    """Preserve raw too-high decisions and calibrate only too-low versus fine."""

    def __init__(
        self,
        *,
        too_low_probability_threshold: float = 0.20,
        history_size: int = 3,
    ) -> None:
        if not 0.0 <= too_low_probability_threshold <= 1.0:
            raise ValueError("too_low_probability_threshold must be in [0, 1]")
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.too_low_probability_threshold = too_low_probability_threshold
        self.history_size = history_size
        self._too_low_history: deque[float] = deque(maxlen=history_size)

    def reset(self) -> None:
        self._too_low_history.clear()

    def update(
        self,
        raw_label: str,
        probabilities: Mapping[str, float],
    ) -> CalibratedDecision:
        if raw_label not in LABELS:
            raise ValueError(f"Unknown raw label {raw_label!r}")
        missing = [label for label in LABELS if label not in probabilities]
        if missing:
            raise ValueError(f"Missing probabilities for: {', '.join(missing)}")
        too_low_probability = float(probabilities["too_low"])
        if not isfinite(too_low_probability) or not 0.0 <= too_low_probability <= 1.0:
            raise ValueError(
                f"Invalid too_low probability {too_low_probability!r}"
            )

        self._too_low_history.append(too_low_probability)
        median_probability = float(median(self._too_low_history))

        # The too-high path intentionally remains the model's original argmax.
        if raw_label == "too_high":
            label = "too_high"
        elif median_probability >= self.too_low_probability_threshold:
            label = "too_low"
        else:
            label = "fine"

        history_count = len(self._too_low_history)
        return CalibratedDecision(
            label=label,
            raw_label=raw_label,
            too_low_probability_median=median_probability,
            history_count=history_count,
            history_size=self.history_size,
            too_low_probability_threshold=self.too_low_probability_threshold,
            # Too-high needs no calibration history. Low/fine waits for a full
            # history before a future motion controller may consume it.
            ready=raw_label == "too_high" or history_count >= self.history_size,
        )

