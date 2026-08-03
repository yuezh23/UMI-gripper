#!/usr/bin/env bash

set -eo pipefail

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${workspace_dir}/../.." && pwd)"
python_bin="${REALTIME_FORCE_STATE_PYTHON:-${repo_root}/.venv/bin/python}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${workspace_dir}/log/realtime_force_state}"
mkdir -p "${ROS_LOG_DIR}"

source "${workspace_dir}/install/setup.bash"

exec "${python_bin}" \
    "${workspace_dir}/src/force_state_xgb/realtime_force_state_node.py" \
    --model-root \
    "${workspace_dir}/artifacts/force_state/leave_one_object_out_top100_strict_v3" \
    "$@"
