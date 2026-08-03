from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from force_state_decision import ProbabilityDecisionCalibrator
from online_force_features import OnlineFeatureExtractor
from realtime_sensor_buffer import (
    ForceSample,
    ImageSample,
    RealtimeSensorBuffer,
    SyncedSample,
)
from xgb_ensemble import EXPECTED_LABELS, discover_fold_specs


MODEL_ROOT = (
    Path(__file__).resolve().parents[3]
    / "artifacts/force_state/leave_one_object_out_top100_strict_v3"
)


def force(timestamp_ns: int, value: float = 0.0) -> ForceSample:
    return ForceSample(timestamp_ns, value, 0.0, 0.0, 0.0, 0.0, 0.0)


def probabilities(too_low: float, fine: float, too_high: float) -> dict[str, float]:
    return {"too_low": too_low, "fine": fine, "too_high": too_high}


class DecisionCalibratorTest(unittest.TestCase):
    def test_calibrates_only_too_low_versus_fine(self) -> None:
        calibrator = ProbabilityDecisionCalibrator(
            too_low_probability_threshold=0.20,
            history_size=3,
        )
        first = calibrator.update("fine", probabilities(0.10, 0.60, 0.30))
        second = calibrator.update("fine", probabilities(0.30, 0.50, 0.20))
        third = calibrator.update("fine", probabilities(0.40, 0.45, 0.15))

        self.assertEqual(first.label, "fine")
        self.assertFalse(first.ready)
        self.assertEqual(second.label, "too_low")
        self.assertFalse(second.ready)
        self.assertEqual(third.label, "too_low")
        self.assertTrue(third.ready)
        self.assertAlmostEqual(third.too_low_probability_median, 0.30)

    def test_raw_too_high_is_never_overridden(self) -> None:
        calibrator = ProbabilityDecisionCalibrator()
        decision = calibrator.update(
            "too_high",
            probabilities(0.90, 0.05, 0.05),
        )
        self.assertEqual(decision.label, "too_high")
        self.assertTrue(decision.ready)

    def test_reset_discards_previous_probability_history(self) -> None:
        calibrator = ProbabilityDecisionCalibrator(history_size=3)
        for _ in range(3):
            calibrator.update("fine", probabilities(0.40, 0.50, 0.10))
        calibrator.reset()
        decision = calibrator.update("fine", probabilities(0.10, 0.80, 0.10))
        self.assertEqual(decision.label, "fine")
        self.assertEqual(decision.history_count, 1)
        self.assertFalse(decision.ready)


class ModelMetadataTest(unittest.TestCase):
    def test_discovers_seven_combined_folds(self) -> None:
        specs = discover_fold_specs(MODEL_ROOT)
        self.assertEqual(len(specs), 7)
        self.assertTrue(all(len(spec.feature_columns) == 100 for spec in specs))

    def test_rejects_wrong_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            combined = root / "test_bad/combined"
            combined.mkdir(parents=True)
            (combined / "model.json").write_text("{}", encoding="utf-8")
            (combined / "model_metadata.json").write_text(
                json.dumps(
                    {
                        "labels": list(EXPECTED_LABELS),
                        "modality": "combined",
                        "window_sec": 1.0,
                        "feature_columns": ["duplicate"] * 100,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "100 unique"):
                discover_fold_specs(root)


class SynchronizerTest(unittest.TestCase):
    def test_latest_common_timestamp_uses_slowest_stream(self) -> None:
        buffer = RealtimeSensorBuffer(retention_sec=5.0)
        gray = np.zeros((4, 4), dtype=np.uint8)
        self.assertIsNone(buffer.latest_common_timestamp_ns())
        buffer.append("force_left", force(1_200_000_000))
        buffer.append("force_right", force(1_180_000_000))
        buffer.append("gelsight_left", ImageSample(1_150_000_000, gray))
        buffer.append("gelsight_right", ImageSample(1_190_000_000, gray))
        self.assertEqual(buffer.latest_common_timestamp_ns(), 1_150_000_000)

    def test_nearest_neighbor_and_tolerance(self) -> None:
        buffer = RealtimeSensorBuffer(retention_sec=5.0)
        gray = np.zeros((4, 4), dtype=np.uint8)
        for timestamp in (1_000_000_000, 1_020_000_000, 1_200_000_000):
            buffer.append("force_left", force(timestamp))
        buffer.append("force_right", force(1_010_000_000))
        buffer.append("force_right", force(1_210_000_000))
        buffer.append("gelsight_left", ImageSample(1_000_000_000, gray))
        buffer.append("gelsight_left", ImageSample(1_210_000_000, gray))
        buffer.append("gelsight_right", ImageSample(1_005_000_000, gray))
        buffer.append("gelsight_right", ImageSample(1_205_000_000, gray))

        synced = buffer.snapshot_synced(
            900_000_000,
            1_250_000_000,
            tolerance_ns=100_000_000,
        )
        self.assertEqual([sample.timestamp_ns for sample in synced], [
            1_000_000_000,
            1_020_000_000,
            1_200_000_000,
        ])
        self.assertEqual(synced[1].force_right.timestamp_ns, 1_010_000_000)

    def test_out_of_order_callbacks_remain_sorted(self) -> None:
        buffer = RealtimeSensorBuffer(retention_sec=5.0)
        gray = np.zeros((4, 4), dtype=np.uint8)
        for timestamp in (1_200_000_000, 1_000_000_000, 1_100_000_000):
            buffer.append("force_left", force(timestamp))
        for stream in ("force_right",):
            for timestamp in (1_000_000_000, 1_100_000_000, 1_200_000_000):
                buffer.append(stream, force(timestamp))
        for stream in ("gelsight_left", "gelsight_right"):
            for timestamp in (1_000_000_000, 1_100_000_000, 1_200_000_000):
                buffer.append(stream, ImageSample(timestamp, gray))

        synced = buffer.snapshot_synced(900_000_000, 1_250_000_000)
        self.assertEqual(
            [sample.timestamp_ns for sample in synced],
            [1_000_000_000, 1_100_000_000, 1_200_000_000],
        )


class OnlineFeatureTest(unittest.TestCase):
    def test_builds_baseline_and_full_window(self) -> None:
        start_ns = 10_000_000_000
        samples = []
        for index in range(211):
            timestamp = start_ns + index * 20_000_000  # 50 Hz, 4.2 seconds
            relative_sec = (timestamp - start_ns) / 1e9
            baseline_force = 1.0
            measured_force = baseline_force if relative_sec <= 3.0 else 2.0
            gray_value = 0 if relative_sec <= 3.0 else 10
            gray = np.full((8, 8), gray_value, dtype=np.uint8)
            samples.append(
                SyncedSample(
                    timestamp_ns=timestamp,
                    force_left=force(timestamp, measured_force),
                    force_right=force(timestamp, measured_force),
                    gelsight_left=ImageSample(timestamp, gray),
                    gelsight_right=ImageSample(timestamp, gray),
                )
            )

        extractor = OnlineFeatureExtractor(
            min_window_coverage_sec=0.9,
            min_window_records=20,
            min_baseline_records=20,
        )
        baseline = extractor.build_baseline(samples, start_ns)
        result = extractor.extract(
            samples,
            prediction_time_ns=start_ns + 4_100_000_000,
            baseline=baseline,
        )
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(result.coverage_sec, 0.9)
        self.assertGreaterEqual(result.num_records, 45)
        self.assertAlmostEqual(result.features["left_force_fx_current"], 1.0)
        self.assertAlmostEqual(result.features["right_force_fx_mean"], 1.0)
        self.assertAlmostEqual(
            result.features["left_gelsight_raw_diff_mean_current"],
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
