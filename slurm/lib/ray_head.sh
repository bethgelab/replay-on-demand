#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
# Single-node Ray head launcher (no multi-node worker fan-out; SLURM --time owns wall-clock).
# Runs INSIDE the container (invoked via `apptx`). Args: <python_script> <config_yaml>.
set -euo pipefail
PYTHON_SCRIPT="$1"
CONFIG="$2"
shift 2 || true          # remaining args ($@) are hydra-style overrides (a.b=c) forwarded to the script
# Default Ray's GPU count to what SLURM actually allocated (matches --gres), so a
# partial-node allocation works; overridable via NUM_GPUS. Without this, Ray tries
# to start with 8 and errors ("start raylet with 8 GPU, but CUDA_VISIBLE_DEVICES
# contains ['0','1']") on anything smaller than a full node.
if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
  else
    NUM_GPUS=8
  fi
fi
PY=/opt/nemo_rl_venv/bin/python
NODE_IP=$(hostname -I | awk '{print $1}')

# Per-job Ray ports (GCS port + worker-port band derived from the SLURM job id), so several jobs
# sharing a node never connect to each other's Ray. /tmp/ray is per job via --writable-tmpfs.
JOB=${SLURM_JOB_ID:-0}
RAY_PORT="${RAY_PORT:-$(( 6400 + JOB % 400 ))}"
WMIN=$(( 20000 + (JOB % 380) * 100 ))
WMAX=$(( WMIN + 99 ))

# Cap the object store: by default Ray sizes it from node RAM, which on large-memory nodes can
# claim hundreds of GB of /dev/shm charged to the job's cgroup and trigger OOM kills. Our objects
# are tiny (per-step candidate batches), so 32GB is ample and a modest --mem request suffices.
ray start --head --node-ip-address="$NODE_IP" --port="$RAY_PORT" \
    --min-worker-port="$WMIN" --max-worker-port="$WMAX" \
    --num-gpus="$NUM_GPUS" --num-cpus="${SLURM_CPUS_PER_TASK:-$(nproc)}" \
    --object-store-memory=32000000000 \
    --disable-usage-stats --include-dashboard=false

# Wait until all local GPUs register with Ray (usually seconds on one node). Connect to THIS
# job's head explicitly (not address='auto', which could latch onto a co-located job's Ray).
for _ in $(seq 1 60); do
  G=$("$PY" -c "import ray; ray.init(address='$NODE_IP:$RAY_PORT'); print(int(ray.cluster_resources().get('GPU',0))); ray.shutdown()" 2>/dev/null || echo 0)
  [ "$G" -ge "$NUM_GPUS" ] && break
  sleep 5
done

set +e
"$PY" "$PYTHON_SCRIPT" --config "$CONFIG" "$@"
EXIT=$?
set -e
# Stop Ray to free the GPUs. Under apptainer --pid (container_env.sh) the container has its own
# PID namespace, so this only affects this job, and the namespace teardown reaps anything left.
ray stop --force >/dev/null 2>&1 || true
exit "$EXIT"
