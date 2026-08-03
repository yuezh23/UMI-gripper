#!/usr/bin/env python3
"""ROS 2 node for triggered baseline collection and live force-state inference."""

from __future__ import annotations

import argparse
import json
import logging
import time
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image as PILImage

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from force_state_decision import ProbabilityDecisionCalibrator
from online_force_features import OnlineBaseline, OnlineFeatureExtractor
from realtime_sensor_buffer import ForceSample, ImageSample, RealtimeSensorBuffer
from xgb_ensemble import XGBForceStateEnsemble


class RuntimePhase(str, Enum):
    IDLE = "idle"
    BASELINE = "baseline"
    WINDOW_FILL = "window_fill"
    PREDICTING = "predicting"
    FAULT = "fault"


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ros_image_to_gray(message: Image) -> np.ndarray:
    """Match the recorder PNG -> PIL.convert('L') training conversion."""

    encoding = message.encoding.lower()
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3}.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported GelSight encoding {message.encoding!r}")
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    visible_bytes = width * channels
    if step < visible_bytes:
        raise ValueError(f"Invalid Image.step={step} for {width=} and {encoding=}")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError(
            f"Image contains {raw.size} bytes; expected {height * step}"
        )
    rows = raw.reshape(height, step)[:, :visible_bytes]
    if encoding == "mono8":
        return rows.reshape(height, width).copy()
    color = rows.reshape(height, width, channels)
    if encoding == "bgr8":
        color = color[:, :, ::-1]
    return np.asarray(PILImage.fromarray(color, mode="RGB").convert("L"), dtype=np.uint8)


class RealtimeForceStateNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("realtime_force_state")
        self.args = args
        self.buffer = RealtimeSensorBuffer(retention_sec=args.retention_sec)
        self.extractor = OnlineFeatureExtractor(
            baseline_duration_sec=args.baseline_duration_sec,
            force_baseline_sec=args.force_baseline_sec,
            gelsight_baseline_start_sec=args.gelsight_baseline_start_sec,
            pixel_threshold=args.pixel_threshold,
            force_abs_limit=args.force_abs_limit,
            window_sec=args.window_sec,
            min_window_coverage_sec=args.min_window_coverage_sec,
            min_window_records=args.min_window_records,
            min_baseline_records=args.min_baseline_records,
        )
        self.decision_calibrator = ProbabilityDecisionCalibrator(
            too_low_probability_threshold=args.too_low_probability_threshold,
            history_size=args.decision_history_size,
        )
        self.get_logger().info(f"Loading XGBoost ensemble from {args.model_root}")
        self.ensemble = XGBForceStateEnsemble(
            args.model_root,
            n_jobs=args.xgb_n_jobs,
            fold_workers=args.xgb_fold_workers,
        )
        self.get_logger().info(f"Loaded {len(self.ensemble.models)} combined folds")

        self.phase = RuntimePhase.IDLE
        self.session_start_ns: int | None = None
        self.baseline: OnlineBaseline | None = None
        self.last_prediction_ns: int | None = None
        self.last_prediction_payload: dict[str, Any] | None = None
        self.last_error = ""
        self._processing_lock = Lock()

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        force_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Image conversion and model inference must not starve the high-rate
        # force callbacks. The buffers are thread-safe, while the service and
        # state-machine timer remain mutually exclusive.
        self.sensor_callbacks = {
            stream: MutuallyExclusiveCallbackGroup()
            for stream in (
                "gelsight_left",
                "gelsight_right",
                "force_left",
                "force_right",
            )
        }
        self.runtime_callbacks = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            Image,
            args.gelsight_left_topic,
            lambda msg: self._image_callback("gelsight_left", msg),
            image_qos,
            callback_group=self.sensor_callbacks["gelsight_left"],
        )
        self.create_subscription(
            Image,
            args.gelsight_right_topic,
            lambda msg: self._image_callback("gelsight_right", msg),
            image_qos,
            callback_group=self.sensor_callbacks["gelsight_right"],
        )
        self.create_subscription(
            WrenchStamped,
            args.force_left_topic,
            lambda msg: self._force_callback("force_left", msg),
            force_qos,
            callback_group=self.sensor_callbacks["force_left"],
        )
        self.create_subscription(
            WrenchStamped,
            args.force_right_topic,
            lambda msg: self._force_callback("force_right", msg),
            force_qos,
            callback_group=self.sensor_callbacks["force_right"],
        )

        self.result_publisher = self.create_publisher(String, args.result_topic, 10)
        self.start_service = self.create_service(
            Trigger,
            args.start_service,
            self._start,
            callback_group=self.runtime_callbacks,
        )
        self.timer = self.create_timer(
            1.0 / args.inference_hz,
            self._tick,
            callback_group=self.runtime_callbacks,
        )
        self.get_logger().info(
            f"Ready for sensors. Start service: {args.start_service}; "
            f"result topic: {args.result_topic}"
        )

    def _force_callback(self, stream: str, message: WrenchStamped) -> None:
        self.buffer.append(
            stream,
            ForceSample(
                timestamp_ns=stamp_to_ns(message.header.stamp),
                fx=float(message.wrench.force.x),
                fy=float(message.wrench.force.y),
                fz=float(message.wrench.force.z),
                tx=float(message.wrench.torque.x),
                ty=float(message.wrench.torque.y),
                tz=float(message.wrench.torque.z),
            ),
        )

    def _image_callback(self, stream: str, message: Image) -> None:
        try:
            gray = ros_image_to_gray(message)
            sample = ImageSample(
                timestamp_ns=stamp_to_ns(message.header.stamp),
                gray=gray,
            )
            self.buffer.append(
                stream,
                sample,
            )
            baseline = self.baseline
            if baseline is not None:
                self.extractor.precompute_gelsight(
                    stream.removeprefix("gelsight_"),
                    sample.timestamp_ns,
                    sample.gray,
                    baseline,
                )
        except Exception as exc:
            self.get_logger().error(
                f"{stream} conversion failed: {type(exc).__name__}: {exc}",
                throttle_duration_sec=2.0,
            )

    def _start(self, _request: Trigger.Request, response: Trigger.Response):
        if self.phase not in {RuntimePhase.IDLE, RuntimePhase.FAULT}:
            response.success = False
            response.message = f"Cannot start while phase={self.phase.value}"
            return response
        health = self._sensor_health(self._now_ns())
        if not health["ready"]:
            response.success = False
            response.message = f"Sensors are not ready: {health['ages_sec']}"
            return response

        self.buffer.clear()
        self.session_start_ns = self._now_ns()
        self.baseline = None
        self.last_prediction_ns = None
        self.last_prediction_payload = None
        self.decision_calibrator.reset()
        self.last_error = ""
        self.phase = RuntimePhase.BASELINE
        response.success = True
        response.message = (
            f"Baseline started for {self.args.baseline_duration_sec:.1f} seconds"
        )
        self.get_logger().info(response.message)
        return response

    def _tick(self) -> None:
        if not self._processing_lock.acquire(blocking=False):
            return
        try:
            now_ns = self._now_ns()
            if self.phase is RuntimePhase.BASELINE:
                self._maybe_finish_baseline(now_ns)
            if self.phase in {RuntimePhase.WINDOW_FILL, RuntimePhase.PREDICTING}:
                self._maybe_predict(now_ns)
            else:
                self._publish_status(now_ns)
        except Exception as exc:
            if not rclpy.ok():
                return
            self.phase = RuntimePhase.FAULT
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"Realtime inference fault: {self.last_error}")
            self._publish_status(self._now_ns())
        finally:
            self._processing_lock.release()

    def _maybe_finish_baseline(self, now_ns: int) -> None:
        assert self.session_start_ns is not None
        baseline_end_ns = self.session_start_ns + int(
            self.args.baseline_duration_sec * 1e9
        )
        common_data_ns = self.buffer.latest_common_timestamp_ns()
        if common_data_ns is None or common_data_ns < baseline_end_ns:
            self._publish_status(
                now_ns,
                extra={
                    "waiting_for_sensor_timestamp_ns": baseline_end_ns,
                    "latest_common_sensor_timestamp_ns": common_data_ns,
                },
            )
            return
        samples = self.buffer.snapshot_synced(
            self.session_start_ns,
            baseline_end_ns,
            self.args.sync_tolerance_ns,
        )
        self.baseline = self.extractor.build_baseline(samples, self.session_start_ns)
        self.phase = RuntimePhase.WINDOW_FILL
        self.get_logger().info(
            f"Baseline ready from {len(samples)} synchronized records; filling 1 s window"
        )
        self._publish_status(now_ns)

    def _maybe_predict(self, now_ns: int) -> None:
        assert self.baseline is not None
        prediction_ns = self.buffer.latest_common_timestamp_ns()
        if prediction_ns is None:
            self.phase = RuntimePhase.WINDOW_FILL
            self._publish_status(now_ns)
            return
        if prediction_ns - self.baseline.baseline_end_ns < int(
            self.args.window_sec * 1e9
        ):
            self.phase = RuntimePhase.WINDOW_FILL
            self._publish_status(now_ns)
            return
        window_start_ns = prediction_ns - int(self.args.window_sec * 1e9)
        snapshot_started_s = time.perf_counter()
        samples = self.buffer.snapshot_synced(
            window_start_ns,
            prediction_ns,
            self.args.sync_tolerance_ns,
        )
        snapshot_finished_s = time.perf_counter()
        window = self.extractor.extract(samples, prediction_ns, self.baseline)
        features_finished_s = time.perf_counter()
        timing_ms = {
            "snapshot": (snapshot_finished_s - snapshot_started_s) * 1000.0,
            "features": (features_finished_s - snapshot_finished_s) * 1000.0,
        }
        if not window.valid:
            self.phase = RuntimePhase.WINDOW_FILL
            self._publish_status(
                now_ns,
                extra={
                    "window_valid": False,
                    "window_reason": window.reason,
                    "window_coverage_sec": window.coverage_sec,
                    "window_num_records": window.num_records,
                    "prediction_timestamp_ns": prediction_ns,
                    "prediction_age_sec": max((now_ns - prediction_ns) / 1e9, 0.0),
                    "new_prediction": False,
                    "timing_ms": timing_ms,
                },
            )
            return

        new_prediction = prediction_ns != self.last_prediction_ns
        if new_prediction:
            model_started_s = time.perf_counter()
            prediction = self.ensemble.predict(window.features)
            timing_ms["model"] = (time.perf_counter() - model_started_s) * 1000.0
            decision = self.decision_calibrator.update(
                prediction.label,
                prediction.probabilities,
            )
            self.last_prediction_ns = prediction_ns
            self.last_prediction_payload = {
                "label": decision.label,
                "raw_label": prediction.label,
                "probabilities": prediction.probabilities,
                "confidence": prediction.probabilities[decision.label],
                "raw_confidence": prediction.confidence,
                "model_agreement": prediction.model_agreement,
                "fold_labels": list(prediction.fold_labels),
                "decision_ready": decision.ready,
                "decision_rule": {
                    "too_high": "raw_model_label",
                    "too_low_probability_median": (
                        decision.too_low_probability_median
                    ),
                    "too_low_probability_threshold": (
                        decision.too_low_probability_threshold
                    ),
                    "history_count": decision.history_count,
                    "history_size": decision.history_size,
                },
            }
        assert self.last_prediction_payload is not None
        self.phase = RuntimePhase.PREDICTING
        self._publish_status(
            now_ns,
            extra={
                "window_valid": True,
                "window_reason": "ok",
                "window_coverage_sec": window.coverage_sec,
                "window_num_records": window.num_records,
                "prediction_timestamp_ns": prediction_ns,
                "prediction_age_sec": max((now_ns - prediction_ns) / 1e9, 0.0),
                "new_prediction": new_prediction,
                "timing_ms": timing_ms,
                **self.last_prediction_payload,
            },
        )

    def _publish_status(
        self,
        now_ns: int,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        # Feature extraction and seven-fold inference take measurable time.
        # Publish and evaluate health against the actual completion time.
        now_ns = self._now_ns()
        health = self._sensor_health(now_ns)
        remaining = 0.0
        if self.phase is RuntimePhase.BASELINE and self.session_start_ns is not None:
            elapsed = (now_ns - self.session_start_ns) / 1e9
            remaining = max(self.args.baseline_duration_sec - elapsed, 0.0)
        payload: dict[str, Any] = {
            "timestamp_ns": now_ns,
            "phase": self.phase.value,
            "valid": self.phase is RuntimePhase.PREDICTING and health["ready"],
            "baseline_ready": self.baseline is not None,
            "baseline_remaining_sec": remaining,
            "sensor_ready": health["ready"],
            "sensor_ages_sec": health["ages_sec"],
            "sensor_counts": self.buffer.counts(),
            "error": self.last_error or None,
        }
        if extra:
            payload.update(extra)
        prediction_timestamp_ns = payload.get("prediction_timestamp_ns")
        if prediction_timestamp_ns is not None:
            prediction_age_sec = max(
                (now_ns - int(prediction_timestamp_ns)) / 1e9,
                0.0,
            )
            payload["prediction_age_sec"] = prediction_age_sec
            if prediction_age_sec > self.args.max_prediction_age_sec:
                payload["valid"] = False
                payload["stale_prediction"] = True
            else:
                payload["stale_prediction"] = False
        message = String()
        message.data = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        self.result_publisher.publish(message)

    def _sensor_health(self, now_ns: int) -> dict[str, Any]:
        timestamps = self.buffer.latest_timestamps()
        ages = {
            name: (
                None
                if timestamp is None
                else max((now_ns - timestamp) / 1e9, 0.0)
            )
            for name, timestamp in timestamps.items()
        }
        ready = all(
            age is not None and age <= self.args.sensor_timeout_sec
            for age in ages.values()
        )
        return {"ready": ready, "ages_sec": ages}

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=workspace
        / "artifacts/force_state/leave_one_object_out_top100_strict_v3",
    )
    parser.add_argument("--force-left-topic", default="/force_torque/left")
    parser.add_argument("--force-right-topic", default="/force_torque/right")
    parser.add_argument("--gelsight-left-topic", default="/gelsight/left/image_raw")
    parser.add_argument("--gelsight-right-topic", default="/gelsight/right/image_raw")
    parser.add_argument(
        "--start-service",
        default="/force_state/right_gripper/start",
    )
    parser.add_argument(
        "--result-topic",
        default="/force_state/right_gripper/result",
    )
    parser.add_argument("--baseline-duration-sec", type=float, default=3.0)
    parser.add_argument("--force-baseline-sec", type=float, default=0.75)
    parser.add_argument("--gelsight-baseline-start-sec", type=float, default=0.5)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--inference-hz", type=float, default=3.0)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--xgb-fold-workers", type=int, default=7)
    parser.add_argument("--sync-tolerance-sec", type=float, default=0.1)
    # The current F/T driver occasionally delivers a burst about 0.8 s late.
    # Window validity remains strict; this timeout only reports stream liveness.
    parser.add_argument("--sensor-timeout-sec", type=float, default=1.25)
    parser.add_argument("--max-prediction-age-sec", type=float, default=1.5)
    parser.add_argument(
        "--too-low-probability-threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument("--decision-history-size", type=int, default=3)
    parser.add_argument("--retention-sec", type=float, default=8.0)
    parser.add_argument("--pixel-threshold", type=float, default=8.0)
    parser.add_argument("--force-abs-limit", type=float, default=10.0)
    # Match build_causal_force_dataset.py, which trained only on coverage >= 0.80.
    parser.add_argument("--min-window-coverage-sec", type=float, default=0.8)
    # The strict-v3 training dataset contains accepted windows with 4 records.
    parser.add_argument("--min-window-records", type=int, default=4)
    parser.add_argument("--min-baseline-records", type=int, default=20)
    parsed, ros_args = parser.parse_known_args(argv)
    if parsed.inference_hz <= 0:
        parser.error("--inference-hz must be positive")
    if parsed.xgb_n_jobs < 1:
        parser.error("--xgb-n-jobs must be at least 1")
    if parsed.xgb_fold_workers < 1:
        parser.error("--xgb-fold-workers must be at least 1")
    if parsed.max_prediction_age_sec <= 0:
        parser.error("--max-prediction-age-sec must be positive")
    if not 0.0 <= parsed.too_low_probability_threshold <= 1.0:
        parser.error("--too-low-probability-threshold must be in [0, 1]")
    if parsed.decision_history_size < 1:
        parser.error("--decision-history-size must be at least 1")
    parsed.sync_tolerance_ns = int(parsed.sync_tolerance_sec * 1e9)
    return parsed, ros_args


def main(argv=None) -> None:
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = RealtimeForceStateNode(args)
    # Four ordered sensor groups plus one runtime group can all make progress.
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
