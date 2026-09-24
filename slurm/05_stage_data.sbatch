#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=rod-stage-data
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
##SBATCH --account=<YOUR_ACCOUNT>        # set for your cluster, then remove one '#'
##SBATCH --partition=<YOUR_GPU_PARTITION> # set for your cluster, then remove one '#'
#SBATCH --output=%x-%j.out
# Usage: DOMAIN=legal|german sbatch slurm/05_stage_data.sbatch  (see slurm/stage_data.sh)
# Replay pool (selective diversity-aware download) + adaptation corpus + packed blocks corpus,
# as an UNATTENDED batch job — don't run in an interactive srun
# (it dies with the allocation). HF downloads resume if partial. GPU sits idle but
# apptx's `--nv` needs one allocated.
# Submit from the repo root (or `export REPO=...`) so ${REPO:-$SLURM_SUBMIT_DIR} resolves. Preview first: run stage_replay_pool with --dry-run.
set -euo pipefail
bash "${REPO:-$SLURM_SUBMIT_DIR}/slurm/stage_data.sh"
