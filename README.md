# UMI Gripper Force-State Learning

This repository contains the offline data collection, feature extraction,
training, and evaluation pipeline for a three-class gripper force-state
classifier. It combines two fingertip GelSight cameras with two MMS101
force/torque sensors and classifies each one-second sensor window as:

| Dataset directory | Model label | Meaning |
| --- | --- | --- |
| `low` | `too_low` | The grasp is too loose. |
| `medium` | `fine` | The grasp force is appropriate. |
| `high` | `too_high` | The grasp is too tight. |

The pipeline can train force-only, GelSight-only, and combined XGBoost models.
It supports episode-level train/validation/test splits and leave-one-object-out
evaluation so that recordings from one episode cannot leak across splits.

> This README covers data collection and offline model training only. For
> real-time inference and Trossen-arm integration, see
> [README.md](https://github.com/yuezh23/UMI-gripper/blob/realtime-force-state/README.md).

## Repository layout

```text
UMI-gripper/
└── UMI_ws/
    ├── src/
    │   ├── force_state_xgb/          # feature, training, evaluation, and real-time code
    │   ├── mms101_driver/            # MMS101 ROS 2 driver
    │   ├── ros2_gelsight_package/    # GelSight ROS 2 publisher
    │   ├── sensor_framework/         # timestamp synchronization and recording
    │   └── gopro_driver/             # GoPro support
    ├── exp_data/train/               # labeled raw recordings (not committed to Git)
    ├── reports/force_state/           # data audits, phase plots, and reviewed events
    ├── artifacts/force_state/         # feature tables, models, predictions, and metrics
    ├── build.sh                       # build and source the ROS 2 workspace
    ├── gelsight.sh                    # start the two GelSight publishers
    ├── force_sensor.sh                # start the two MMS101 drivers
    ├── record.sh                      # record synchronized raw sensor episodes
    └── realtime_force_state.sh        # real-time entry point; described separately
```

`build/`, `install/`, and `log/` are generated ROS 2 files. The raw training
dataset and generated reports are ignored by Git because they can be large.
The repository keeps the reviewed v3 feature table and the main trained model.

## Training-data format

The current dataset contains seven objects, three force levels, and 20 trials
per object/level combination: 420 episodes in total.

```text
UMI_ws/exp_data/train/<object>/<label>/<trial>/
└── episodes/<episode_id>/
    ├── episode_info.json
    ├── metadata.json
    ├── recording_force_mms101_left.jsonl
    ├── recording_force_mms101_right.jsonl
    ├── recording_gelsight_left.jsonl
    ├── recording_gelsight_right.jsonl
    ├── synced_data.jsonl
    └── images/
        ├── gelsight_left/
        └── gelsight_right/
```

The `left` and `right` names refer to the two gripper fingertips. Each MMS101
record contains `fx`, `fy`, `fz`, `tx`, `ty`, and `tz`.

## Collect training data

Connect the two GelSight cameras and two MMS101 sensors, then use three
terminals. These commands save the raw recordings under `~/exp_data/train`.

### Terminal A — GelSight cameras

```bash
cd UMI-gripper/UMI_ws
source build.sh
source gelsight.sh
```

### Terminal B — MMS101 force/torque sensors

```bash
cd UMI-gripper/UMI_ws
source build.sh
source force_sensor.sh
```

### Terminal C — synchronized recorder

Each command records one 15-second episode. Replace the object name and trial
number before recording.

```bash
cd UMI-gripper/UMI_ws
source build.sh
./record.sh 1 15 ~/exp_data/train/{object name}/{trial number}
```

Repeat Terminal C with a new trial number for every recording. The training
pipeline still requires a `low`, `medium`, or `high` directory between the
object name and trial number. In the simplified command, include that label in
`{object name}`; for example:

```bash
./record.sh 1 15 ~/exp_data/train/paper_cup/medium/01
```

Keep the grasp timing consistent and record only the intended condition:

- `low`: the object is not held securely or there is little/no contact.
- `medium`: the object can be held without excessive deformation.
- `high`: the grasp is clearly tighter than necessary.

Before a full collection, verify the active topics:

```bash
ros2 topic list | grep -E 'gelsight|force_torque'
ros2 topic hz /gelsight/right/image_raw/compressed
ros2 topic hz /force_torque/right
```

## Install the model dependencies

Run all offline commands from `UMI_ws` so the relative paths resolve correctly.

```bash
cd ~/UMI-gripper-yue/UMI_ws
python -m pip install -r src/force_state_xgb/requirements.txt
```

## Build and evaluate the model

### 1. Audit the synchronized recordings

```bash
python src/force_state_xgb/inspect_synced_train.py \
  --data-root ~/exp_data/train \
  --output-dir reports/force_state
```

For the complete original dataset,
`reports/force_state/schema_summary.json` should report 420 episodes and 140
episodes per class.

### 2. Preview the grasp phases

```bash
python src/force_state_xgb/plot_force_phase_preview.py \
  --data-root ~/exp_data/train \
  --output-dir reports/force_state/phase_preview_corrected \
  --grasp-start-sec 0.0 \
  --grasp-end-sec 15.0 \
  --fallback-eval-sec 3.0 \
  --force-abs-limit 10.0 \
  --gelsight-baseline-start-sec 0.5 \
  --gelsight-baseline-end-sec 3.0 \
  --evaluation-mode adaptive
```

Review the generated plots before creating the feature dataset. The red line
is the evaluation time `t_eval`; the model will use only the one-second window
ending at that point.

### 3. Detect and review one evaluation time per episode

```bash
python src/force_state_xgb/detect_force_phase.py \
  --data-root ~/exp_data/train \
  --grasp-start-sec 0.0 \
  --grasp-end-sec 15.0 \
  --fallback-eval-sec 3.0 \
  --force-abs-limit 10.0 \
  --gelsight-baseline-start-sec 0.5 \
  --gelsight-baseline-end-sec 3.0 \
  --evaluation-mode adaptive \
  --output reports/force_state/phase_events.csv
```

Inspect `phase_events.csv`. Correct `evaluation_time_sec` when necessary and
set `review_status` to `rejected` for unusable recordings or windows containing
force/torque outliers. A no-contact fallback is valid for many `too_low`
examples.

### 4. Build the reviewed one-second feature dataset

```bash
python src/force_state_xgb/build_causal_force_dataset.py \
  --events reports/force_state/phase_events.csv \
  --window-sec 1.0 \
  --samples-per-episode 1 \
  --force-abs-limit 10.0 \
  --gelsight-baseline-start-sec 0.5 \
  --gelsight-baseline-end-sec 3.0 \
  --max-window-outlier-ratio 0.0 \
  --output artifacts/force_state/causal_windows_v3.csv
```

The committed strict v3 table contains 306 accepted episodes. The exact count
will change if the raw data or manual review decisions change. To build a
diagnostic table that also contains rejected rows, add `--include-rejected`
and write it to a different output such as `causal_windows_all_v3.csv`.

### 5. Train the main leave-one-object-out model

```bash
python src/force_state_xgb/train_force_state_xgb.py \
  --dataset artifacts/force_state/causal_windows_v3.csv \
  --split leave_one_object_out \
  --top-k 100 \
  --output-dir artifacts/force_state/leave_one_object_out_top100_strict_v3
```

Feature selection is performed separately using only each fold's training
data. The command trains force-only, GelSight-only, and combined models for
every held-out object.

For an optional within-object evaluation:

```bash
python src/force_state_xgb/train_force_state_xgb.py \
  --dataset artifacts/force_state/causal_windows_v3.csv \
  --split stratified_episode \
  --top-k 100 \
  --output-dir artifacts/force_state/within_object_top100_strict_v3
```

## Important outputs

- `causal_windows_v3.csv`: reviewed one-second samples used by the main model.
- `causal_windows_v3.features.json`: feature schema and dataset statistics.
- `leave_one_object_out_top100_strict_v3/metrics_summary.*`: fold-level results.
- `leave_one_object_out_top100_strict_v3/test_<object>/<modality>/model.json`:
  trained XGBoost models.
- `model_metadata.json`: label order, selected features, window length, and
  model parameters.
- `test_predictions.csv`, `feature_importance.csv`, and confusion matrices:
  files used for error analysis and reporting.
