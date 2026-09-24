#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=rod-precompute
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
##SBATCH --account=<YOUR_ACCOUNT>        # set for your cluster, then remove one '#'
##SBATCH --partition=<YOUR_GPU_PARTITION> # set for your cluster, then remove one '#'
#SBATCH --output=%x-%j.out
# Reference-loss precompute: two forward-only passes → one content_hash-keyed ref-loss cache.
# The cache directory $ROD_CACHE is WIPED first, so use a separate ROD_CACHE per setting.
# Usage: sbatch slurm/30_precompute.sbatch configs/experiments/<setting>/precompute.yaml [key=value ...]
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/lib/container_env.sh"
EXP=${1:?usage: sbatch slurm/30_precompute.sbatch configs/experiments/<setting>/precompute.yaml}
case "$EXP" in /*) ;; *) EXP=$REPO/$EXP ;; esac
shift
export MY_JOB_NAME=${ROD_RUN_NAME:-$(basename "$(dirname "$EXP")")-precompute}
# adaptation domain of pass B = data_mixer.adapt_domain of the config (override with ADAPT_DOMAIN)
ADAPT_DOMAIN=${ADAPT_DOMAIN:-$(awk '/^data_mixer:/{f=1;next} /^[^ #]/{f=0} f&&$1=="adapt_domain:"{print $2;exit}' "$EXP")}
: "${ADAPT_DOMAIN:?could not read data_mixer.adapt_domain from $EXP}"
rm -rf "$ROD_CACHE" && mkdir -p "$ROD_CACHE"
# Both passes run on a single-node Ray cluster (ray_head.sh); the configs record the node counts
# of the paper runs.

# Pass A — REPLAY ref = frozen base, general slice → replay_rho = current - base
apptx "export ROD_PRECOMPUTE_OUTPUT_CACHE=$ROD_CACHE \
        ROD_PRECOMPUTE_REFERENCE_MODEL=$ROD_MODEL_DIR \
        ROD_PRECOMPUTE_ADAPT_DOMAIN=general; \
   bash $REPO/slurm/lib/ray_head.sh $REPO/src/rod/run_precompute.py $EXP cluster.num_nodes=1 $*"

# Pass B — ADAPT ref = adaptation specialist, adaptation slice → adapt_rho
apptx "export ROD_PRECOMPUTE_OUTPUT_CACHE=$ROD_CACHE \
        ROD_PRECOMPUTE_REFERENCE_MODEL=$ROD_SPECIALIST_DIR \
        ROD_PRECOMPUTE_ADAPT_DOMAIN=$ADAPT_DOMAIN; \
   bash $REPO/slurm/lib/ray_head.sh $REPO/src/rod/run_precompute.py $EXP cluster.num_nodes=1 $*"

apptx "N=\$(find $ROD_CACHE -maxdepth 1 -name '*.parquet' | wc -l); echo \"ref_loss cache parquet files: \$N\"; test \$N -gt 0"
