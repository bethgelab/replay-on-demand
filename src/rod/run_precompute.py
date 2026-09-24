# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (c) 2026 The RoD Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from NVIDIA NeMo-RL v0.6.0, the batch preparation in nemo_rl/algorithms/sft.py; modified for RoD.

"""Precompute per-sample reference losses via NeMo-RL's policy worker.

Mirrors NeMo-RL's SFT entrypoint but:
- Loads a *reference* model as the policy (no training).
- Uses the blocks data path with ``replay_share=0.0`` and a configurable
  ``adapt_domain`` so the mixer outputs one domain slice at a time.
- Drives a custom one-pass loop over every block, calling NeMo-RL's forward-only
  ``policy.get_logprobs`` and reducing the logprobs to the per-sequence loss with
  the same function the trainer uses (``rod.patches.rod_loss.compute_target_loss``).
- Writes (content_hash, ref_loss) parquet shards to the configured cache directory.

Typical invocation (see ``slurm/30_precompute.sbatch``), once per reference::

    ROD_PRECOMPUTE_ADAPT_DOMAIN=general \\
    ROD_PRECOMPUTE_REFERENCE_MODEL=/path/to/base_model \\
    ROD_PRECOMPUTE_OUTPUT_CACHE=/path/to/ref_loss_cache \\
        bash slurm/lib/ray_head.sh src/rod/run_precompute.py <experiment.yaml>

The output parquets live at ``<output-cache>/refloss_<split>_<random>.parquet``.
Subsequent precompute passes write into the same directory with different
content hashes; the training-time mixer reads everything in the directory and
joins by content_hash, so multi-pass output composes cleanly.

Inherits the codebase's parallelism contract from ``sft.setup()`` — TP/PP/EP/DP
all configurable via the YAML's ``policy.megatron_cfg`` block. Scales identically
to 30B / 35B without code changes.
"""

from __future__ import annotations

import argparse
import logging
import os
import pprint
import secrets

import torch

from nemo_rl.utils.config import register_omegaconf_resolvers
from rod.configs.common import build_config
from rod.utils import get_job_name

logger = logging.getLogger(__name__)

# Same config schema as the training runs; the ``rod`` block is not used here (the
# output cache comes from --output-cache / ROD_PRECOMPUTE_OUTPUT_CACHE).
from rod.configs.rod import MasterConfig


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI flags + env-var fallbacks.

    The Ray launcher (``slurm/lib/ray_head.sh``) forwards the config path plus
    hydra-style overrides, so precompute-specific knobs are passed via
    ``ROD_PRECOMPUTE_*`` env vars, which the sbatch script sets
    right before invoking the launcher. CLI flags still work for direct
    invocation (e.g. running this script by hand in a dev container).
    """
    p = argparse.ArgumentParser(description="Precompute ref_loss via NeMo-RL policy worker")
    p.add_argument("--config", type=str, required=True, help="Path to the YAML config.")
    p.add_argument(
        "--adapt-domain", type=str, default=None,
        help="Block domain to score this pass (e.g. 'general' for replay blocks, 'legal' or 'german' for adaptation blocks)."
             " Falls back to ROD_PRECOMPUTE_ADAPT_DOMAIN env var.",
    )
    p.add_argument(
        "--reference-model", type=str, default=None,
        help="Path to the reference model checkpoint. Falls back to "
             "ROD_PRECOMPUTE_REFERENCE_MODEL env var.",
    )
    p.add_argument(
        "--output-cache", type=str, default=None,
        help="Directory where this pass writes its parquet shards. Falls back "
             "to ROD_PRECOMPUTE_OUTPUT_CACHE env var.",
    )
    p.add_argument(
        "--max-batches", type=int, default=-1,
        help="Cap the number of microbatches to process (debug). -1 = full dataset.",
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args, extra = p.parse_known_args()

    # Env-var fallbacks (the sbatch scripts pass these knobs as env vars).
    args.adapt_domain = args.adapt_domain or os.environ.get("ROD_PRECOMPUTE_ADAPT_DOMAIN")
    args.reference_model = args.reference_model or os.environ.get("ROD_PRECOMPUTE_REFERENCE_MODEL")
    args.output_cache = args.output_cache or os.environ.get("ROD_PRECOMPUTE_OUTPUT_CACHE")

    missing = [
        name for name, val in [
            ("--adapt-domain / ROD_PRECOMPUTE_ADAPT_DOMAIN", args.adapt_domain),
            ("--reference-model / ROD_PRECOMPUTE_REFERENCE_MODEL", args.reference_model),
            ("--output-cache / ROD_PRECOMPUTE_OUTPUT_CACHE", args.output_cache),
        ]
        if not val
    ]
    if missing:
        raise SystemExit(
            "run_precompute.py: missing required arg(s) — pass each via CLI flag OR env var:\n  - "
            + "\n  - ".join(missing)
        )

    return args, extra


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Bake the CLI overrides into the loaded config dict before sft.setup().

    Why bake here instead of via OmegaConf interpolation: simpler reasoning
    when reading the YAML.
    """
    cfg.setdefault("data_mixer", {})
    cfg["data_mixer"]["adapt_domain"] = args.adapt_domain
    cfg["data_mixer"]["replay_share"] = 0.0  # always single-slice for precompute

    # During precompute, the data mixer must NOT attempt to join ref_loss from
    # an existing cache — we're computing it. Drop ref_loss_cache_path so the
    # mixer skips the join entirely.
    cfg.setdefault("rod", {})
    cfg["rod"]["ref_loss_cache_path"] = None

    # Swap the policy to the reference model — but ONLY the weight path. Leave
    # ``tokenizer.name`` pointed at whatever the YAML already says (typically
    # the base model dir). Two reasons:
    #
    #   1. If the reference model is a Megatron Distributed Checkpoint (DCP)
    #      directory, it only contains sharded weights, not the HF tokenizer
    #      files. AutoTokenizer.from_pretrained() then crashes with
    #      "Couldn't instantiate the backend tokenizer" because tokenizer.json
    #      and tokenizer_config.json aren't there.
    #
    #   2. Even when the reference IS an HF dir, its tokenizer is identical to
    #      the base model's — the adaptation specialist is produced by continued
    #      pre-training of the base model, which doesn't change the vocab. So
    #      using the base tokenizer for both passes is correct by construction.
    cfg["policy"]["model_name"] = args.reference_model

    # Each precompute pass writes its own pair of mixed JSONLs into a
    # pass-specific scratch location so concurrent passes don't trample each
    # other (and so we don't have to clean up after).
    pass_tag = args.adapt_domain
    # Writable scratch for the per-pass mixed JSONLs.
    _scratch = os.environ.get("ROD_SCRATCH_DIR", "/tmp/rod")
    cfg["data"]["train_data_path"] = f"{_scratch}/precompute_{pass_tag}/train.jsonl"
    cfg["data"]["val_data_path"] = f"{_scratch}/precompute_{pass_tag}/val.jsonl"

    # ---- Precompute efficiency (this path is eval-only) --------------------
    # Offloading the optimizer to CPU during the logprob forward frees GPU memory
    # (precompute never steps the optimizer). Default ON; ROD_PRECOMPUTE_OFFLOAD=false
    # disables it. Applied post-validation, pre-sft_setup, so no schema warning.
    _off = os.environ.get("ROD_PRECOMPUTE_OFFLOAD", "true").strip().lower()
    cfg["policy"]["offload_optimizer_for_logprob"] = _off not in ("0", "false", "no", "off")

    # Override val_micro_batch_size for THIS precompute pass only (env-gated) —
    # decouples the precompute microbatch from the training-time val. Unset → use
    # the config value.
    _vmbs = os.environ.get("ROD_PRECOMPUTE_VAL_MBS", "").strip()
    if _vmbs:
        cfg.setdefault("sft", {})["val_micro_batch_size"] = int(_vmbs)

    # policy.get_logprobs microbatches the forward by `policy.logprob_batch_size`
    # (it has no per-call mbs argument). The config value applies unless
    # ROD_LOGPROB_BATCH_SIZE (or the val-mbs override above) is set.
    _lbs = os.environ.get("ROD_LOGPROB_BATCH_SIZE", "").strip() or _vmbs
    if _lbs:
        cfg["policy"]["logprob_batch_size"] = int(_lbs)
    cfg["policy"].setdefault("logprob_batch_size", 8)

    # Optional: route the logprob gather through NeMo-RL's chunked path
    # (ChunkedDistributedLogprob), which chunks the fp32 log-softmax over the
    # sequence dim and avoids the full-[mbs,T,V] fp32 memory spike. `logprob_chunk_size` /
    # `defer_fp32_logits` are real v0.6.0 megatron_cfg knobs read via .get();
    # env-gated + default-off.
    _mcfg = cfg["policy"].setdefault("megatron_cfg", {})
    _chunk = os.environ.get("ROD_LOGPROB_CHUNK_SIZE", "").strip()
    if _chunk:
        _mcfg["logprob_chunk_size"] = int(_chunk)
    if os.environ.get("ROD_DEFER_FP32_LOGITS", "").strip().lower() in ("1", "true", "yes"):
        _mcfg["defer_fp32_logits"] = True
    return cfg


def main() -> None:
    args, cli_overrides = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        force=True,
    )

    logger.info("=" * 72)
    logger.info("RoD precompute pass")
    logger.info("  Config:         %s", args.config)
    logger.info("  Reference model: %s", args.reference_model)
    logger.info("  Adapt domain:   %s", args.adapt_domain)
    logger.info("  Output cache:   %s", args.output_cache)
    logger.info("=" * 72)

    register_omegaconf_resolvers()
    cfg = build_config(MasterConfig, args.config, cli_overrides)

    cfg = _apply_cli_overrides(cfg, args)

    # Patches: only the collate patch is needed for precompute (it threads
    # origin_tag / content_hash / origin into the batch).
    from rod.patches import apply_rod_collate
    apply_rod_collate()

    from nemo_rl.algorithms.sft import setup as sft_setup
    from nemo_rl.algorithms.utils import get_tokenizer, maybe_pad_last_batch
    from nemo_rl.data.llm_message_utils import (
        add_loss_mask_to_message_log,
        batched_message_log_to_flat_message,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.logger import get_next_experiment_dir
    from rod.data.rod_dataset import build_train_and_val_datasets
    from rod.data.rod_pretrain_blocks import mix_and_write_pretrain
    import ray

    # Log/checkpoint dirs (same convention as the training run, even though
    # precompute does not save checkpoints).
    job_name = get_job_name()
    log_dir_root = os.path.join(cfg["logger"]["log_dir"], job_name)
    cfg["logger"]["log_dir"] = get_next_experiment_dir(log_dir_root)
    if cfg["logger"].get("tensorboard_enabled"):
        cfg["logger"]["tensorboard"]["log_dir"] = os.path.join(log_dir_root, "tensorboard")
    # Disable checkpointing entirely for precompute — we don't update weights.
    cfg["checkpointing"]["enabled"] = False

    init_ray()
    tokenizer = get_tokenizer(cfg["policy"]["tokenizer"])

    # The corpus is a pre-packed blocks parquet. Score a single domain slice this
    # pass (replay_share=0 keeps only adapt_domain's blocks) — e.g.
    # adapt_domain=general vs the base ref, adapt_domain=legal vs the specialist ref.
    # Precompute scores the train and holdout blocks of that slice. The train
    # dataloader can drop a final partial batch; the training-time join tolerates
    # that small missing fraction (data_mixer.allow_missing_fraction).
    logger.info("Preparing data: blocks mixer, single slice=%s (replay_share=0.0)", args.adapt_domain)
    mix_and_write_pretrain(
        blocks_path=cfg["data_mixer"]["corpus_path"],
        train_output=cfg["data"]["train_data_path"],
        val_output=cfg["data"]["val_data_path"],
        adapt_domain=cfg["data_mixer"]["adapt_domain"],
        replay_share=float(cfg["data_mixer"]["replay_share"]),  # 0.0 -> single slice
        ref_loss_cache_path=None,  # we are COMPUTING ref_loss this pass
        allow_missing_fraction=0.0,
        seed=cfg["data_mixer"].get("seed", 42),
        # no max_val_samples_per_origin: score EVERY holdout block, so the capped val set of
        # any training run is a subset of the cache
    )

    logger.info("Building datasets ...")
    train_dataset, val_dataset = build_train_and_val_datasets(cfg, tokenizer)

    logger.info("Resolved config:\n%s", pprint.pformat(cfg))

    # Stand the policy up via sft.setup(). The returned NLL loss fn is unused:
    # precompute only runs forward-only logprob passes.
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        _nll_loss_fn,
        rl_logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = sft_setup(cfg, tokenizer, train_dataset, val_dataset)

    # ------------------------------------------------------------------
    # Custom one-pass loop over train + val splits.
    #
    # NeMo-RL's native policy.get_logprobs inference route (the same one DPO uses
    # to precompute its frozen KL-reference logprobs) runs a no_grad + forward_only
    # pass, DP-sharded and gathered to the driver — no backward, no optimizer step.
    # We reduce the returned per-token logprobs to per-sequence NLL with the SAME
    # compute_target_loss the trainer uses (byte-identical ref_loss by
    # construction), then buffer + write shards on the driver.
    # ------------------------------------------------------------------
    # get_logprobs splits each DP shard into microbatches of logprob_batch_size, so
    # batches are padded to a multiple of dp_size * logprob_batch_size.
    lp_mbs = master_config["policy"]["logprob_batch_size"]
    from rod.patches.rod_loss import compute_target_loss

    def _write_refloss_shard(rows: list, label: str) -> None:
        if not rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        os.makedirs(args.output_cache, exist_ok=True)
        schema = pa.schema([
            ("content_hash", pa.string()), ("origin", pa.string()),
            ("origin_tag", pa.int32()), ("ref_loss", pa.float32()),
            ("n_response_tokens", pa.int32()),
        ])
        path = os.path.join(args.output_cache, f"refloss_{label}_{secrets.token_hex(6)}.parquet")
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        logger.info("[%s] wrote %d ref_loss rows -> %s", label, len(rows), path)

    def _run_one_dataloader(dataloader, label: str) -> int:
        if dataloader is None:
            logger.info("[%s] no dataloader, skipping", label)
            return 0
        total = n_rows = 0
        rows_buffer: list = []
        policy.prepare_for_training()
        for batch_idx, batch in enumerate(dataloader):
            if 0 < args.max_batches <= batch_idx:
                logger.info("[%s] hit --max-batches=%d cap", label, args.max_batches)
                break

            # Populate per-message `token_loss_mask` (assistant tokens = 1,
            # others = 0). Mirrors `sft.py:267` — without this call,
            # `batched_message_log_to_flat_message` can't produce the
            # `token_loss_mask` field that we rely on as the response-token
            # mask for the per-sequence NLL aggregation.
            add_loss_mask_to_message_log(
                batch["message_log"],
                roles_to_train_on=["assistant"],
            )

            cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                make_sequence_length_divisible_by=master_config["policy"][
                    "make_sequence_length_divisible_by"
                ],
            )

            # Build the minimum BatchedDataDict that `maybe_pad_last_batch`
            # knows how to extend. The padder ONLY pads a hard-coded set of
            # fields (input_ids, input_lengths, token_mask, sample_mask,
            # reference_policy_logprobs) — any other field is silently left at
            # the original length, and the downstream ``shard_by_batch_size``
            # asserts that all fields share a single length. So:
            #   1. Build with the known-padded fields only.
            #   2. Pad to dp_size × logprob_batch_size.
            #   3. Manually extend our RoD-specific extras
            #      (origin_tag, content_hash, origin) to the new length.
            data = BatchedDataDict({
                "input_ids": cat_and_padded["token_ids"],
                "input_lengths": input_lengths,
                "token_mask": cat_and_padded["token_loss_mask"],
                "sample_mask": batch["loss_multiplier"],
            })
            data.update(cat_and_padded.get_multimodal_dict(as_tensors=False))

            dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
            orig_size = data.size
            data = maybe_pad_last_batch(data, dp_size, lp_mbs)
            target_size = data.size
            pad_count = target_size - orig_size  # 0 if no padding was needed

            # origin_tag is an int64 tensor; pad with zeros (the padding rows
            # have sample_mask=0 so the tag is irrelevant — they are skipped
            # when rows are written below).
            if "origin_tag" in batch:
                origin_tag = batch["origin_tag"]
                if pad_count > 0:
                    origin_tag = torch.cat([
                        origin_tag,
                        torch.zeros(pad_count, dtype=origin_tag.dtype, device=origin_tag.device),
                    ])
                data["origin_tag"] = origin_tag

            def _pad_list(values, n: int) -> list:
                values = list(values)
                # Padding rows have sample_mask=0 and are skipped when rows
                # are written, so empty strings are inert.
                return values + [""] * max(0, n - len(values))

            if "content_hash" in batch:
                data["content_hash"] = _pad_list(batch["content_hash"], target_size)
            if "origin" in batch:
                data["origin"] = _pad_list(batch["origin"], target_size)

            # get_logprobs returns per-token logprobs [B, S] (position 0 == 0.0) gathered
            # to the driver; microbatching is by policy.logprob_batch_size (set in
            # _apply_cli_overrides). Hand it the MINIMAL dict (input_ids + input_lengths) —
            # it needs nothing else, and passing our string metadata (content_hash/origin)
            # through the DP shard is wasteful + a serialization risk. Row order is
            # preserved (no reorder unless sequence_packing/dynamic_batching — both off),
            # so the driver's token_mask / metadata line up with logprobs[i] by index.
            lp_data = BatchedDataDict({
                "input_ids": data["input_ids"],
                "input_lengths": data["input_lengths"],
            })
            logprobs = policy.get_logprobs(lp_data)["logprobs"]       # [B, S], CPU
            # Reduce to per-sequence token-mean NLL with the SAME function the trainer
            # uses: the next-token logprob of token j sits at logprobs[:, 1:][:, j-1], and
            # token_mask[:, 1:] aligns with it.
            next_token_logprobs = logprobs[:, 1:]                     # [B, S-1]
            token_mask = data["token_mask"][:, 1:]                    # [B, S-1]
            sample_mask = data["sample_mask"]                          # [B]
            ref_loss = compute_target_loss(
                next_token_logprobs, token_mask, sample_mask
            )                                                          # [B]
            n_resp = (token_mask * sample_mask.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
            content_hashes = data.get("content_hash", [""] * data.size)
            origins = data.get("origin", ["unknown"] * data.size)
            origin_tags = data["origin_tag"]
            for i in range(data.size):
                if sample_mask[i].item() == 0:
                    continue  # padding row (maybe_pad_last_batch) — skip
                rows_buffer.append({
                    "content_hash": content_hashes[i] if i < len(content_hashes) else "",
                    "origin": origins[i] if i < len(origins) else "unknown",
                    "origin_tag": int(origin_tags[i].item()),
                    "ref_loss": float(ref_loss[i].item()),
                    "n_response_tokens": int(n_resp[i].item()),
                })
            # Periodic flush: bound crash-loss + driver memory on a long pass
            # (~1M rows). Each flush writes one shard.
            if len(rows_buffer) >= 50_000:
                n_rows += len(rows_buffer)
                _write_refloss_shard(rows_buffer, label)
                rows_buffer = []

            total += orig_size
            if batch_idx % 50 == 0:
                logger.info("[%s] Processed batches=%d samples=%d", label, batch_idx + 1, total)

        # Write the remaining buffered rows as a final shard.
        n_rows += len(rows_buffer)
        _write_refloss_shard(rows_buffer, label)
        logger.info("[%s] DONE — %d samples scored (%d ref_loss rows written)", label, total, n_rows)
        return total

    try:
        n_train = _run_one_dataloader(train_dataloader, "train")
        n_val = _run_one_dataloader(val_dataloader, "val")
        logger.info("Total samples scored: train=%d val=%d", n_train, n_val)
        logger.info("Parquet shards written to: %s", args.output_cache)
    finally:
        logger.info("Shutting down policy ...")
        try:
            policy.shutdown()
        except Exception as e:
            logger.warning("Policy shutdown failed (non-fatal): %s", e, exc_info=True)

        if ray.is_initialized():
            logger.info("Shutting down Ray ...")
            ray.shutdown()


if __name__ == "__main__":
    main()
