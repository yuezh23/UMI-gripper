# Right-gripper realtime force-state inference

This stage performs sensor buffering, training-compatible feature extraction,
and seven-fold XGBoost inference. It does **not** command either gripper.

The live sequence is:

```text
start service
  -> collect a complete 3 s sensor-timestamp baseline
  -> fill a 1 s causal window
  -> publish rolling too_low / fine / too_high predictions
```

The F/T driver currently delivers some messages in bursts. Consequently, a
complete three seconds of sensor-timestamp data can take more than three
seconds of wall-clock time to arrive. Downstream gripper control must wait for
`baseline_ready=true`; it must not rely on a fixed wall-clock sleep.

## 1. Build once

```bash
cd /home/ubuntu/lerobot_trossen/UMI-gripper-yue/UMI_ws
./build.sh
```

The repository virtual environment must contain the model-compatible runtime:

```bash
cd /home/ubuntu/lerobot_trossen
.venv/bin/python -c "import xgboost; print(xgboost.__version__)"
```

The copied strict-v3 models were saved by XGBoost 3.3.0.

## 2. Start the two sensor drivers

Terminal A:

```bash
cd /home/ubuntu/lerobot_trossen/UMI-gripper-yue/UMI_ws
./gelsight.sh
```

Terminal B:

```bash
cd /home/ubuntu/lerobot_trossen/UMI-gripper-yue/UMI_ws
./force_sensor.sh
```

Use the raw image topics. `/gelsight/right` is not a published topic.

Optional checks:

```bash
cd /home/ubuntu/lerobot_trossen/UMI-gripper-yue/UMI_ws
source install/setup.bash
ros2 topic hz /gelsight/right/image_raw
ros2 topic hz /force_torque/right
```

`ros2 topic hz` can under-report the raw GelSight stream on this machine.
The recorder test measured about 13 images/s and about 50 F/T messages/s.

## 3. Start realtime inference

Terminal C:

```bash
cd /home/ubuntu/lerobot_trossen/UMI-gripper-yue/UMI_ws
./realtime_force_state.sh
```

Expected startup messages include:

```text
Loaded 7 combined folds
Ready for sensors
```

## 4. Trigger a stationary baseline

For this standalone test, leave the gripper stationary and unloaded throughout
the baseline. Terminal D:

```bash
cd /home/ubuntu/lerobot_trossen/UMI-gripper-yue/UMI_ws
source install/setup.bash
ros2 service call /force_state/right_gripper/start std_srvs/srv/Trigger '{}'
```

The service should return `success=True`. The inference terminal will later
print:

```text
Baseline ready from ... synchronized records; filling 1 s window
```

## 5. Inspect the complete result

Read one result:

```bash
cd /home/ubuntu/lerobot_trossen/UMI-gripper-yue/UMI_ws
source install/setup.bash
ros2 topic echo /force_state/right_gripper/result \
  std_msgs/msg/String --once --full-length
```

Read continuously:

```bash
ros2 topic echo /force_state/right_gripper/result \
  std_msgs/msg/String --full-length
```

A prediction may be consumed by a future gripper controller only when all of
these conditions are true:

```text
phase == "predicting"
valid == true
window_valid == true
sensor_ready == true
stale_prediction == false
new_prediction == true
decision_ready == true
```

Important fields:

- `label`: calibrated deployment result: `too_low`, `fine`, or `too_high`.
- `raw_label`: original seven-fold argmax result before calibration.
- `probabilities`: seven-fold soft-vote probabilities.
- `confidence`: probability of the calibrated `label`.
- `raw_confidence`: probability of `raw_label`.
- `model_agreement`: fraction of folds agreeing with `raw_label`; it is kept
  as a diagnostic and does not gate the calibrated `too_low` result.
- `decision_ready`: immediately true for an original `too_high`; low/fine
  becomes true after three valid new predictions have filled the median filter.
- `decision_rule`: calibration threshold, current median, and history count.
- `window_coverage_sec`: accepted only when at least 0.80 s.
- `prediction_age_sec`: accepted only when no more than 1.5 s.
- `new_prediction`: false when the same sensor window is republished.
- `timing_ms`: diagnostic time for synchronization, features, and the model.

Deployment calibration changes only the `too_low` versus `fine` boundary:

```text
if raw_label == "too_high":
    label = "too_high"
elif median(last 3 valid-new P(too_low)) >= 0.20:
    label = "too_low"
else:
    label = "fine"
```

The original `too_high` decision is intentionally unchanged. Empty gripper,
missed grasp, and insufficient grasp force are all operational `too_low`
states, so they continue closing rather than being rejected as no-contact.

The 2026-07-31 empty-gripper live check produced valid 0.98--0.99 s windows
with roughly 75--101 synchronized records and prediction ages around
0.28--0.58 s. The stable output was `too_low`, as expected with no contact.

## 6. Stop the test

Press `Ctrl+C` in the inference terminal, then stop the sensor terminals when
they are no longer needed.

## Trossen integration

The Trossen teleoperation bridge is implemented in
`../../../right_gripper_slow_close`. Its controller:

1. `SPACE` starts opening only the right follower gripper.
2. When the measured right gripper is confirmed fully open, call the baseline
   start service.
3. Holds fully open until `baseline_ready=true` and at least three seconds have
   elapsed.
4. Waits for the first fresh valid prediction.
5. Applies a small right-gripper-only correction only for a fresh valid result:
   `fine` holds, `too_low` closes slightly, and `too_high` opens slightly.
6. Clamps every target to the measured absolute limits and makes `ESC` hold the
   current measured position immediately.

See `../../../right_gripper_slow_close/README.md` for the four-terminal
dry-run and hardware test commands.
