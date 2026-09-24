#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
# Sourced by every sbatch script. Defines the Apptainer exec wrapper `apptx` and the
# env consumed by the configs via ${oc.env:...}. Apptainer passes host env into the
# container by default, so exporting here is enough (PYTHONPATH is set inside apptx).
#
# EDIT: set ROD_ROOT to a directory on a shared filesystem visible from all compute nodes.
# It holds the .sif, models, blocks, cache and logs.
export ROD_ROOT=${ROD_ROOT:?set ROD_ROOT to your project dir on a shared filesystem, e.g. export ROD_ROOT=/path/to/rod}
export SIF=${SIF:-$ROD_ROOT/containers/nemo-rl-v0.6.0.sif}   # the pulled .sif (00_setup_container.sh)
# Persistent Apptainer overlay holding pip-installs that aren't in the read-only image
# (e.g. nvidia-modelopt, required by megatron.bridge.AutoBridge). apptx adds --overlay
# automatically once this file exists. Create it with: apptainer overlay create --size 8192 $ROD_OVERLAY
export ROD_OVERLAY=${ROD_OVERLAY:-$ROD_ROOT/overlay.img}
# REPO is auto-detected as this repo's root — it can live anywhere (home is fine; code is tiny).
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"

# --- data / model / cache / log locations (consumed via ${oc.env:...}) ---
export ROD_MODEL_DIR=${ROD_MODEL_DIR:-$ROD_ROOT/models/nemotron-nano-12b-v2-base}
export ROD_SPECIALIST_DIR=${ROD_SPECIALIST_DIR:-$ROD_ROOT/models/legal-specialist}
export ROD_BLOCKS_DIR=${ROD_BLOCKS_DIR:-$ROD_ROOT/data/blocks}
export ROD_CACHE=${ROD_CACHE:-$ROD_ROOT/cache/ref_loss}
export ROD_SCRATCH_DIR=${ROD_SCRATCH_DIR:-$ROD_ROOT/scratch}
export ROD_LOG_DIR=${ROD_LOG_DIR:-$ROD_ROOT/logs}
export ROD_PP=${ROD_PP:-2}                                # pipeline parallel (12B on 80GB GPUs)
export DEFAULT_PARAMS_FILE=$REPO/configs/default.yaml       # read by MasterConfig.from_overrides

# --- logging / auth (wandb + HF run online; compute nodes need internet access) ---
export WANDB_PROJECT=${WANDB_PROJECT:-rod}
export MY_JOB_NAME=${SLURM_JOB_NAME:-rod-run}
export HF_HOME=${HF_HOME:-$ROD_ROOT/hf_home}
# export WANDB_API_KEY=...   # set in your shell / SLURM secrets (optionally WANDB_ENTITY too)
# export HF_TOKEN=...        # needed for the gated Nemotron datasets

# container-internal paths (public image layout: /opt/nemo-rl)
export CTR_PYTHONPATH="$REPO/src:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
# NeMo-RL runs Megatron in a DEDICATED worker venv that carries the compiled stack
# (transformer_engine, etc.) — the base /opt/nemo_rl_venv (driver) does NOT. Standalone
# megatron/bridge tools (mcore→HF convert) must use THIS python:
export CTR_MEG_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker

mkdir -p "$ROD_CACHE" "$ROD_SCRATCH_DIR" "$ROD_LOG_DIR" "$HF_HOME"

# Optional node-local scratch (export ROD_NODE_SCRATCH=<dir> if your cluster has one): it is bound
# into the container, and the JIT / dataset caches are kept there, off the shared filesystem.
if [ -n "${ROD_NODE_SCRATCH:-}" ]; then
  mkdir -p "$ROD_NODE_SCRATCH/cache"
  export TRITON_CACHE_DIR=$ROD_NODE_SCRATCH/cache/triton TORCHINDUCTOR_CACHE_DIR=$ROD_NODE_SCRATCH/cache/inductor
  export CUDA_CACHE_PATH=$ROD_NODE_SCRATCH/cache/cuda HF_DATASETS_CACHE=$ROD_NODE_SCRATCH/cache/hf_datasets
fi

# apptx "<cmd>"  — run a command inside the container: GPUs (--nv), ROD_ROOT + repo bound, PYTHONPATH set.
# Binds ROD_ROOT always, and the repo too (unless it already lives under ROD_ROOT — avoids a redundant bind).
apptx() {
  local binds="--bind $ROD_ROOT:$ROD_ROOT"
  case "$REPO" in "$ROD_ROOT"/*) : ;; *) binds="$binds --bind $REPO:$REPO" ;; esac
  [ -n "${ROD_NODE_SCRATCH:-}" ] && [ -d "$ROD_NODE_SCRATCH" ] && binds="$binds --bind $ROD_NODE_SCRATCH"
  # Mount the overlay READ-ONLY by default: a :rw overlay takes an EXCLUSIVE lock, so
  # concurrent jobs collide with "overlay ... currently in use by another process". The
  # overlay supplies pip packages READ at runtime (nvidia-modelopt etc.); pairing :ro with
  # --writable-tmpfs gives each job its OWN ephemeral writable layer for container-root paths
  # (/tmp, __pycache__), while the pip packages stay shareable :ro. Setup
  # (00_setup_container.sh), which must PERSIST installs into the overlay, exports
  # ROD_OVERLAY_MODE=rw (no tmpfs → writes land in the overlay image).
  local ov=""
  if [ -n "${ROD_OVERLAY:-}" ] && [ -e "${ROD_OVERLAY:-}" ]; then
    local _mode="${ROD_OVERLAY_MODE:-ro}"
    ov="--overlay $ROD_OVERLAY:$_mode"
    [ "$_mode" = "ro" ] && ov="$ov --writable-tmpfs"
  fi
  # Megatron JIT-compiles its C++ dataset helpers via `make -C .../megatron/core/datasets`,
  # but on some filesystems (e.g. fuse-overlayfs) chdir into the container's copy fails with
  # EPERM and the worker dies. If a pre-built writable copy exists, bind it over the container
  # path so chdir works and `make` is a no-op. Create it once (only if you hit that error) with
  #   apptx "cp -r /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/datasets \
  #          $ROD_ROOT/megatron_core_datasets && make -C $ROD_ROOT/megatron_core_datasets"
  local mds="${ROD_MEGATRON_DATASETS:-$ROD_ROOT/megatron_core_datasets}"
  [ -d "$mds" ] && binds="$binds --bind $mds:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/datasets"
  # --pid: run the container in a PRIVATE PID namespace. Ray daemonizes its raylet+workers via
  # setsid, so without this they can escape SLURM's process tracking and keep holding GPU memory
  # after the job ends. With --pid, the container's bash is PID 1 of the namespace; when it exits
  # (or SLURM kills it), the kernel reaps ALL processes in the namespace. Also makes `ray stop`
  # safe (it only sees this job's namespace, never co-located jobs).
  apptainer exec --pid --nv $ov $binds "$SIF" \
    bash -lc "export PYTHONPATH=$CTR_PYTHONPATH:\${PYTHONPATH:-}; $*"
}
