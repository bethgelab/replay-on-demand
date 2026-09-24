#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=rod-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=16:00:00
##SBATCH --account=<YOUR_ACCOUNT>        # set for your cluster, then remove one '#'
##SBATCH --partition=<YOUR_GPU_PARTITION> # set for your cluster, then remove one '#'
#SBATCH --output=%x-%j.out
# Continued pretraining with RoD, or a plain-CPT baseline / specialist (rod.enable_selection: false).
# RoD runs assume 30_precompute has populated $ROD_CACHE. Logs, checkpoints and the selection trace
# go to $ROD_LOG_DIR/<run>/, with <run> = <setting>-<config name> (or $ROD_RUN_NAME); relaunching
# the same run resumes from its latest checkpoint. Set --time / --mem for your cluster.
# Usage: sbatch slurm/40_train.sbatch configs/experiments/<setting>/<run>.yaml [key=value ...]
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/lib/container_env.sh"
EXP=${1:?usage: sbatch slurm/40_train.sbatch configs/experiments/<setting>/<run>.yaml [key=value ...]}
case "$EXP" in /*) ;; *) EXP=$REPO/$EXP ;; esac
shift
export MY_JOB_NAME=${ROD_RUN_NAME:-$(basename "$(dirname "$EXP")")-$(basename "$EXP" .yaml)}
# ray_head.sh starts a single-node Ray cluster, so run on this node; the experiment configs record
# the node counts of the paper runs (cluster.num_nodes). Later overrides on the command line win.
apptx "bash $REPO/slurm/lib/ray_head.sh $REPO/src/rod/run_rod.py $EXP cluster.num_nodes=1 $*"
