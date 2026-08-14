# UMI Gripper Real-Time Force Feedback

This version extends the offline XGBoost pipeline in [README.md](https://github.com/yuezh23/UMI-gripper/blob/main/README.md)
with real-time ROS 2 inference and right-gripper force feedback for a Trossen
leader/follower system.

The right follower arm continues to follow the right leader's position and
orientation. Only the right follower gripper opening is overridden. The
real-time node maintains a rolling one-second window of GelSight and MMS101
data, extracts the same features used during training, and combines the
predictions of the seven leave-one-object-out XGBoost models.

```text
too_low  -> tighten the right gripper slightly
fine     -> keep the current opening
too_high -> loosen the right gripper slightly
```

The controller requires repeated consistent decisions before changing the
opening, which reduces oscillation near class boundaries. The first accepted
`fine` state can also apply a small extra tightening offset to improve the
holding margin.

## Repository layout

```text
lerobot_trossen/
├── right_gripper_slow_close/        # Trossen controller and experiment recorders
└── UMI-gripper/
    ├── README.md                     # offline collection and training guide
    ├── README_REALTIME.md            # this guide
    └── UMI_ws/
        ├── src/
        │   ├── force_state_xgb/      # offline and real-time XGBoost pipeline
        │   ├── mms101_driver/        # force/torque ROS 2 driver
        │   ├── ros2_gelsight_package/ # GelSight ROS 2 publisher
        │   └── sensor_framework/     # sensor synchronization and recording
        ├── artifacts/force_state/
        │   ├── causal_windows_v3.csv
        │   └── leave_one_object_out_top100_strict_v3/ # main real-time models
        ├── build.sh
        ├── gelsight.sh
        ├── force_sensor.sh
        ├── record.sh
        └── realtime_force_state.sh
```

Generated `build/`, `install/`, and `log/` directories are not source files.
Raw datasets and experiment recordings are also kept outside Git.

## Clone the two repositories

Use the
[`umi-gripper-force-feedback` branch of `lerobot_trossen`](https://github.com/yuezh23/lerobot_trossen/tree/umi-gripper-force-feedback),
then clone UMI Gripper inside it using the exact destination name shown below:

```bash
cd ~
git clone --branch umi-gripper-force-feedback --single-branch \
  https://github.com/yuezh23/lerobot_trossen.git

cd ~/lerobot_trossen
git clone --branch realtime-force-state --single-branch \
  https://github.com/yuezh23/UMI-gripper.git
```

The required result is:

```text
~/lerobot_trossen/UMI-gripper/UMI_ws
```

Follow the main Trossen README to install `uv` and create its `.venv`, then
install the XGBoost pipeline dependencies into the same environment:

```bash
cd ~/lerobot_trossen
uv sync
uv pip install --python .venv/bin/python \
  -r UMI-gripper/UMI_ws/src/force_state_xgb/requirements.txt
```

Build the UMI ROS 2 packages once before starting the experiment:

```bash
cd ~/lerobot_trossen/UMI-gripper/UMI_ws
source /opt/ros/jazzy/setup.bash
source build.sh
```

## Run real-time force feedback

Use separate terminals and start them in the following order.

### Terminal A — GelSight cameras

```bash
cd ~/lerobot_trossen/UMI-gripper/UMI_ws
source build.sh
source gelsight.sh
```

### Terminal B — MMS101 force/torque sensors

```bash
cd ~/lerobot_trossen/UMI-gripper/UMI_ws
source build.sh
source force_sensor.sh
```

### Terminal C — rolling-window XGBoost inference

```bash
cd ~/lerobot_trossen/UMI-gripper/UMI_ws
./realtime_force_state.sh
```

The node loads
`artifacts/force_state/leave_one_object_out_top100_strict_v3`, subscribes to
the two GelSight and two force/torque streams, and publishes JSON results on:

```text
/force_state/right_gripper/result
```

The Trossen controller starts baseline collection through:

```text
/force_state/right_gripper/start
```

Wait until Terminal C reports that the model and sensors are ready.

### Terminal D — right-gripper controller and three-camera recorder

Replace the object name and trial number before each run:

```bash
cd ~/lerobot_trossen

OBJECT_NAME=soft_ball2
TRIAL_NUMBER=01

mkdir -p exp_data/right_gripper_hold/${OBJECT_NAME}/${TRIAL_NUMBER}
test ! -e exp_data/right_gripper_hold/${OBJECT_NAME}/${TRIAL_NUMBER}/lerobot

./right_gripper_slow_close/record-slow-close.sh \
  --dataset.root=exp_data/right_gripper_hold/${OBJECT_NAME}/${TRIAL_NUMBER}/lerobot \
  --dataset.repo_id=local/right-gripper-hold \
  --dataset.single_task="Grasp and lift a ${OBJECT_NAME} with the right follower arm" \
  2>&1 | tee \
  exp_data/right_gripper_hold/${OBJECT_NAME}/${TRIAL_NUMBER}/controller.log
```

The `lerobot` directory must not already exist. Select a new trial number if
the `test` command fails.

### Terminal E — optional raw UMI sensor recording

```bash
cd ~/lerobot_trossen/UMI-gripper/UMI_ws

OBJECT_NAME=soft_ball2
TRIAL_NUMBER=01

./record.sh 1 90 \
  ~/lerobot_trossen/exp_data/right_gripper_hold/${OBJECT_NAME}/${TRIAL_NUMBER}/sensors
```

Terminal E records immediately. Start it only after Terminal D is ready, then
press `SPACE` in Terminal D within one or two seconds.

## Experiment controls

```text
SPACE -> fully open -> collect an unloaded baseline -> grasp slowly
too_low -> tighten slightly
fine -> hold the current opening
too_high -> loosen slightly
E -> fully open, finalize the recording, and exit
ESC -> emergency hold and finalize the recording
```

Move the leader arms to a safe grasping pose before pressing `SPACE`. Keep the
fully open right gripper unloaded and stationary during baseline collection.
After the controller reaches `fine`, lift and hold the object, return it to the
table, and press `E`. Use `ESC` only for an emergency; do not use `Ctrl+C` as
the normal end of a recorded trial.

For non-recording control, alternative tight-to-loose modes, fixed-rate scans,
configuration parameters, and the saved-data layout, see the
[`right_gripper_slow_close` guide](https://github.com/yuezh23/lerobot_trossen/blob/umi-gripper-force-feedback/right_gripper_slow_close/README.md).

## Real-time data flow

```text
GelSight + MMS101 ROS 2 topics
        -> synchronized rolling one-second buffer
        -> force and tactile feature extraction
        -> seven-fold combined XGBoost ensemble
        -> too_low / fine / too_high result
        -> right follower gripper adjustment
```

Only the right gripper is controlled by this feedback loop. The left side is
not used by the experiment controller.
