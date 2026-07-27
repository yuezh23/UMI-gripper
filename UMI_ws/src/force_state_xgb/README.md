# UMI Force-State XGBoost Pipeline

This directory is self-contained. Copy it to:

```text
/home/ktakahashi/UMI-gripper-yue/UMI_ws/src/force_state_xgb
```

Run commands from `/home/ktakahashi/UMI-gripper-yue/UMI_ws` so the default
data path resolves to `data_yue/train`.

## Expected layout

```text
data_yue/train/
  chips_can/
    high/
      19/
        episodes/
          episode_.../
            synced_data.jsonl
            images/...
```

The directory labels are mapped as follows:

```text
low -> too_low
medium -> fine
high -> too_high
```

## Setup

```bash
cd /home/ktakahashi/UMI-gripper-yue/UMI_ws
python -m pip install -r src/force_state_xgb/requirements.txt
```

## Step 1: audit the synchronized data

```bash
python src/force_state_xgb/inspect_synced_train.py \
  --data-root data_yue/train \
  --output-dir reports/force_state
```

Check that `episodes_total` is 420 and each label has 140 episodes:

```bash
cat reports/force_state/schema_summary.json
```

## Step 2: preview corrected force phases

The first run uses deliberately broad defaults. It creates one plot for each
object/force-level pair (21 plots for seven objects).

```bash
python src/force_state_xgb/plot_force_phase_preview.py \
  --data-root data_yue/train \
  --output-dir reports/force_state/phase_preview_corrected \
  --grasp-start-sec 0.0 \
  --grasp-end-sec 15.0 \
  --fallback-eval-sec 3.0 \
  --force-abs-limit 10.0 \
  --gelsight-baseline-start-sec 0.5 \
  --gelsight-baseline-end-sec 3.0 \
  --evaluation-mode adaptive
```

Open `reports/force_state/phase_preview_corrected/*.png`. The force plot must
no longer contain very large spikes, and the GelSight residual should be near
zero before contact. The shaded `0--15 s` range is the complete search range,
not a feature interval. In adaptive mode, the red line is the first stable
post-contact time. If no contact is found, it is fixed at 3.0 s and the model
uses only `(2, 3]`, preventing later lifting, shaking, placing, and release
data from becoming model inputs. Times are relative to the first synced record.

## Step 3: detect one evaluation time per episode

Use the reviewed bounds from Step 2:

```bash
python src/force_state_xgb/detect_force_phase.py \
  --data-root data_yue/train \
  --grasp-start-sec 0.0 \
  --grasp-end-sec 15.0 \
  --fallback-eval-sec 3.0 \
  --force-abs-limit 10.0 \
  --gelsight-baseline-start-sec 0.5 \
  --gelsight-baseline-end-sec 3.0 \
  --evaluation-mode adaptive \
  --output reports/force_state/phase_events.csv
```

`no_contact_fallback` is valid, especially for `too_low`; it uses the fixed
`2--3 s` feature window. `contact_unsettled_at_contact` means contact exists
but no stable plateau was found, so the first contact time is used to avoid
using later action phases.
Review `quality_flag=force_outlier_present`; do not train a window containing
an F/T outlier. Set `review_status=rejected` only for unusable episodes;
manually corrected `evaluation_time_sec` values are accepted by Step 4.

## Step 4: build one-second causal windows

The first baseline creates exactly one sample per episode:

```bash
python src/force_state_xgb/build_causal_force_dataset.py \
  --events reports/force_state/phase_events.csv \
  --window-sec 1.0 \
  --samples-per-episode 1 \
  --force-abs-limit 10.0 \
  --gelsight-baseline-start-sec 0.5 \
  --gelsight-baseline-end-sec 3.0 \
  --max-window-outlier-ratio 0.0 \
  --output artifacts/force_state/causal_windows.csv
```

Only `(t_eval - 1.0 s, t_eval]` enters the features. Later lifting, shaking,
placing, and release data are never used as model inputs.

## Step 5a: within-object evaluation

```bash
python src/force_state_xgb/train_force_state_xgb.py \
  --dataset artifacts/force_state/causal_windows.csv \
  --split stratified_episode \
  --output-dir artifacts/force_state/within_object
```

## Step 5b: leave-one-object-out evaluation

```bash
python src/force_state_xgb/train_force_state_xgb.py \
  --dataset artifacts/force_state/causal_windows.csv \
  --split leave_one_object_out \
  --output-dir artifacts/force_state/leave_one_object_out
```

Both modes split complete `episode_id` values. Windows from the same episode
cannot cross train, validation, and test partitions. Each run trains
force-only, GelSight-only, and combined models.

## Important parameters

- `--grasp-start-sec`, `--grasp-end-sec`: complete time range searched for a
  contact and stable post-contact point. The one-second causal feature window
  remains the only model input.
- `--fallback-eval-sec`: no-contact fallback endpoint. The default `3.0`
  gives the pre-lift window `(2, 3]`.
- `--force-abs-limit`: physical plausibility limit. Frames above it are not
  used as F/T features. `10.0` is a provisional value justified by the normal
  curves; replace it with the sensor's documented calibrated limit.
- `--gelsight-baseline-start-sec`, `--gelsight-baseline-end-sec`: pre-grasp
  interval used to form a median GelSight reference image.
- `--force-threshold`: one-sided force-norm change needed to mark contact.
- `--gelsight-threshold`: one-sided GelSight residual needed to mark contact.
- `--pixel-threshold`: per-pixel difference used for contact-area features.
- `--evaluation-mode adaptive` (default): searches the complete 15-second
  recording and chooses the first stable post-contact point. If no contact is
  found, `t_eval` is `--fallback-eval-sec`, giving the default `(2, 3]`
  fallback window. `nominal` always uses the endpoint; `settled` is an
  optional diagnostic mode.

Thresholds are initial engineering defaults, not final scientific constants.
Keep the chosen values with the experiment artifacts and report sensitivity to
them.
