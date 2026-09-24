#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
# ONE-TIME container setup. Run on a COMPUTE node — `apptainer exec` needs a user namespace
# (or SUID apptainer), which login nodes on many clusters block:
#   export APPTAINER_DOCKER_USERNAME='$oauthtoken'; export APPTAINER_DOCKER_PASSWORD=<NGC_API_KEY>
#   srun --account=<acct> --partition=<part> --gres=gpu:1 --time=01:00:00 bash slurm/00_setup_container.sh
set -euo pipefail
source "$(dirname "$0")/lib/container_env.sh"
# Setup must PERSIST pip installs into the overlay, so mount it read-write here
# (apptx mounts it read-only by default; see lib/container_env.sh).
export ROD_OVERLAY_MODE=rw
mkdir -p "$(dirname "$SIF")"

# 1) Pull the PUBLIC container → .sif under $ROD_ROOT (needs NGC creds above).
[ -f "$SIF" ] || apptainer pull "$SIF" docker://nvcr.io/nvidia/nemo-rl:v0.6.0

# 2) Persistent overlay for pip-installs missing from the read-only image.
#    (If `apptainer overlay create` is unavailable: mkdir -p a dir and set ROD_OVERLAY to it.)
[ -e "$ROD_OVERLAY" ] || apptainer overlay create --size 8192 "$ROD_OVERLAY"

# 3) Install nvidia-modelopt (required by megatron.bridge.AutoBridge) and its import-time
#    dependency pulp, both missing from the image,
#    into BOTH the driver venv AND the Megatron WORKER venv (the worker runs the bridge at train
#    time). --no-deps is CRITICAL: modelopt pins an older nvidia-cudnn-cu12 (9.10) and would
#    DOWNGRADE the image's 9.19.0.56 that nemo_rl needs; modelopt doesn't use cudnn at import.
#    (pyarrow/wandb + transformer_engine are already in the image.)
for V in /opt/nemo_rl_venv "$CTR_MEG_VENV"; do
  apptx "$V/bin/pip install --no-cache-dir --no-deps nvidia-modelopt pulp"
done

# 4) Verify imports + the base SFT config that configs/default.yaml inherits.
apptx "/opt/nemo_rl_venv/bin/python -c 'import nemo_rl, pyarrow, wandb; print(\"DRIVER IMPORTS OK\")'"
apptx "$CTR_MEG_VENV/bin/python -c 'from megatron.bridge import AutoBridge; print(\"AUTOBRIDGE OK\")'"   # worker venv (has TE)
apptx "test -f /opt/nemo-rl/examples/configs/sft.yaml && echo BASE_CFG_OK || echo 'WARN: base sft.yaml missing — fix defaults: in configs/default.yaml'"
echo "Setup done. If DRIVER IMPORTS OK + AUTOBRIDGE OK + BASE_CFG_OK printed → stage the data next."
