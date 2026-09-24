#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=rod-convert
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
##SBATCH --account=<YOUR_ACCOUNT>        # set for your cluster, then remove one '#'
##SBATCH --partition=<YOUR_GPU_PARTITION> # set for your cluster, then remove one '#'
#SBATCH --output=%x-%j.out
# Convert a Megatron checkpoint → HF (local save). TRAIN_LOG_DIR is the run directory
# ($ROD_LOG_DIR/<run>); converts its latest checkpoint unless STEP is set. ROD_MODEL_DIR must
# point at an HF model of the same architecture (config + tokenizer source).
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/lib/container_env.sh"
TRAIN_LOG_DIR=${TRAIN_LOG_DIR:?set TRAIN_LOG_DIR to the finished run dir (contains checkpoints/step_*)}
STEP=${STEP:-$(ls -d "$TRAIN_LOG_DIR"/checkpoints/step_* | sed 's#.*/step_##' | sort -n | tail -1)}
test -d "$TRAIN_LOG_DIR/checkpoints/step_${STEP}" || { echo "no checkpoint step_${STEP} in $TRAIN_LOG_DIR/checkpoints"; exit 1; }
OUT=$ROD_SCRATCH_DIR/exported_hf/$(basename "$TRAIN_LOG_DIR")-step_${STEP}
apptx "env -u RANK -u LOCAL_RANK -u WORLD_SIZE \
   $CTR_MEG_VENV/bin/torchrun --standalone --nproc_per_node=1 \
     $REPO/src/rod/utils/convert_mcore_to_hf.py \
     --hf-model $ROD_MODEL_DIR \
     --megatron-dir $TRAIN_LOG_DIR/checkpoints/step_${STEP}/policy/weights \
     --scratch-dir $OUT --tp 1 --pp 1 --ep 1"
echo "HF model at: $OUT/hf"
