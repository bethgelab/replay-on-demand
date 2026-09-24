#!/bin/bash
# Copyright (c) 2026 The RoD Authors.
# SPDX-License-Identifier: Apache-2.0
# stage_data.sh — stage the replay pool and the adaptation corpus, and build the packed
# blocks corpus for one adaptation domain. Data work only: CPU + internet access, NO GPU.
#
#   DOMAIN=legal|german   adaptation domain (default: legal)
#   BASE_MODEL=<hf id>    base model to download into $ROD_MODEL_DIR; its tokenizer packs the
#                         blocks (default: nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base; set
#                         BASE_MODEL= to use a model already in $ROD_MODEL_DIR)
#   EOD_ID=<id>           end-of-document token (default: 2, as for the Nemotron models; set
#                         EOD_ID= for Qwen3.5, which then uses the tokenizer's eos token)
#
# Both domains share the replay pool: a selective, diversity-aware subsample of the
# Nemotron-Pretraining family (NOT a full download — CC-v2 alone is 10.3 TB), staged by
# rod.data.stage_replay_pool with one dir per origin and shards spread evenly across each
# source folder. The build uses the arguments of the paper corpora (identical for the Nemotron
# and Qwen3.5 models apart from the tokenizer); 20% of the replay documents and a 1% holdout:
#
#   legal   full legal corpus (no caps); replay origins grouped into their 4 capabilities
#           (--origin-map); replay blocks subsampled to the size of the legal corpus
#           (--balance-replay), so the corpus is the ~50:50 candidate pool and the configs train
#           with data_mixer.replay_share: 1.0.
#   german  25% of the German documents; every origin capped at 180k blocks with a holdout
#           floor of 1000 blocks; replay origins kept per source (--capability-map). Not
#           balanced here: the configs subsample replay at training time (replay_share ≈ 0.53).
#           The German corpus is not downloaded by this script; stage it first (README, "Data").
#
# ---- RUNBOOK (on a compute node — apptainer is often unavailable on login nodes) -
#   1. export ROD_ROOT / HF_TOKEN / WANDB_API_KEY; source slurm/lib/container_env.sh.
#   2. PREVIEW the replay download plan (no downloads) and eyeball per-origin shard counts:
#        apptx "python -m rod.data.stage_replay_pool --out $ROD_SCRATCH_DIR/raw/general \
#               --capability-map-out $REPO/configs/replay_capability_map.json --dry-run"
#   3. Run this script on a compute node, preferably as a batch job
#      (DOMAIN=legal sbatch slurm/05_stage_data.sbatch).
#   4. Verify the tail: per-origin train=/holdout= table, and
#      "blocks corpus: N total (adapt=..., replay=...)".
#   5. Train the specialist, then run both precompute passes, e.g. for legal:
#        sbatch slurm/40_train.sbatch configs/experiments/nemotron12b-legal/specialist.yaml
#        sbatch slurm/30_precompute.sbatch configs/experiments/nemotron12b-legal/precompute.yaml
# ---------------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/lib/container_env.sh"

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
DOMAIN=${DOMAIN:-legal}
BASE_MODEL=${BASE_MODEL-nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base}
EOD_ID=${EOD_ID-2}
RAW=$ROD_SCRATCH_DIR/raw
GEN=$RAW/general
CAPMAP=$REPO/configs/replay_capability_map.json
mkdir -p "$GEN"

case "$DOMAIN" in
  legal)  DOMAIN_ARGS="--origin-map $CAPMAP --balance-replay" ;;
  german) DOMAIN_ARGS="--adapt-doc-fraction 0.25 --max-blocks-per-origin 180000 \
                       --holdout-floor-per-origin 1000 --capability-map $CAPMAP"
          ls "$RAW"/german/*/*.parquet >/dev/null 2>&1 || {
            echo "no German corpus under $RAW/german/<source>/*.parquet — stage it first (README, \"Data\")"; exit 1; } ;;
  *)      echo "DOMAIN must be legal or german, got '$DOMAIN'"; exit 1 ;;
esac

# --- 0) Base model (initialization + replay reference; its tokenizer packs the blocks below).
#        The adaptation reference (the specialist) is trained, not downloaded.
if [ -n "$BASE_MODEL" ]; then
  apptx "python -c \"from huggingface_hub import snapshot_download; \
    snapshot_download('$BASE_MODEL', local_dir='$ROD_MODEL_DIR')\""
fi

# --- 1) Replay pool — selective, diversity-aware (spread shards; naming-agnostic).
#        Edit STAGING_SPEC in src/rod/data/stage_replay_pool.py to change origins/shards.
apptx "python -m rod.data.stage_replay_pool --out $GEN --capability-map-out $CAPMAP"

# --- 2) Legal corpus (the German corpus is staged separately, see above).
if [ "$DOMAIN" = legal ]; then
  apptx "python -c \"from huggingface_hub import snapshot_download; \
    snapshot_download('nvidia/Nemotron-Pretraining-Legal-v1', repo_type='dataset', \
    local_dir='$RAW/legal', allow_patterns=['**/*.parquet'])\""
fi

# --- 3) Build the blocks corpus with the paper arguments of this domain -------------
apptx "python -m rod.data.rod_pretrain_blocks \
    --adapt-domain $DOMAIN \
    --adapt-glob '$RAW/$DOMAIN/**/*.parquet' \
    --replay-glob '$GEN/**/*.parquet' \
    --out $ROD_BLOCKS_DIR \
    --tokenizer $ROD_MODEL_DIR \
    --block-size 4096 ${EOD_ID:+--eod-id $EOD_ID} \
    --val-holdout-fraction 0.01 \
    --replay-doc-fraction 0.20 \
    $DOMAIN_ARGS \
    --shard-rows 50000 --seed 42"

echo "Blocks corpus ($DOMAIN) -> $ROD_BLOCKS_DIR (replay sources under $GEN; map $CAPMAP)"
