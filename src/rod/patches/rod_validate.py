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
# Adapted from NVIDIA NeMo-RL v0.6.0, the validation loop in nemo_rl/algorithms/sft.py (validate); modified for RoD.

"""Runtime monkey patch for ``nemo_rl.algorithms.sft.validate``.

The upstream SFT validate aggregates the headline ``val_loss`` assuming the
loss fn is TOKEN_LEVEL — it multiplies each batch's loss by ``num_valid_tokens``
and divides the total by total tokens (NeMo-RL's ``algorithms/sft.py``). For our
SEQUENCE_LEVEL :class:`RHOScoringLossFn`, that mis-scales the reported number.

This patch replaces the aggregation with **sequence-weighted** averaging
(``num_valid_samples`` weights), as NeMo-RL's DPO validation does. Everything
else about ``sft.validate`` is preserved (timing, the policy eval/train mode
toggle, the warn-on-no-batches branch).

It is called by the RoD training loop (:mod:`rod.patches.rod_sft_train`) with
``loss_fn=RHOScoringLossFn``. The per-sample tensors that loss fn emits are
excluded from the scalar metrics by the numeric-only filter below; instead,
the per-sample target loss is regrouped on the driver into exact per-origin
validation losses (``loss_by_origin_<origin>/target_loss``). These per-origin
losses are the inputs to the validation-loss forgetting metric.

The per-corpus losses and rho means are formed from sums and counts accumulated over
all batches; the validation loss is a sequence-weighted average over batches.

Call :func:`apply` once at process startup, before
``from nemo_rl.algorithms.sft import sft_train``.
"""

from __future__ import annotations

import logging
import warnings

log = logging.getLogger(__name__)

_APPLIED = False


def _metric_safe(origin: str) -> str:
    """Origin name usable inside a metric key."""
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in str(origin))


def _patched_validate():
    """Build a validate() with the same signature as NeMo-RL's, forked from its body."""
    import torch

    def patched_validate(
        policy,
        val_dataloader,
        tokenizer,
        loss_fn,
        step: int,
        master_config,
        val_batches: int,
        val_batch_size: int,
        val_mbs: int,
    ):
        """Same signature as NeMo-RL v0.6.0's ``sft.validate`` (sft_train calls it by keyword)."""
        if val_dataloader is None:
            print("  ⚠️ No validation dataloader provided, skipping validation")
            return {}, {}

        # Lazy imports — defer until apply() has had a chance to land.
        from nemo_rl.data.llm_message_utils import (
            add_loss_mask_to_message_log,
            batched_message_log_to_flat_message,
        )
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        from nemo_rl.algorithms.utils import maybe_pad_last_batch
        from nemo_rl.utils.timer import Timer

        # RHOScoringLossFn emits the per-sample target loss (PER_SAMPLE_PREFIX) and
        # per-corpus sums and counts (SUM_PREFIX) into all_mb_metrics. The driver
        # groups the per-sample losses by origin for exact per-origin val losses, and
        # sums the corpus sums over microbatches, DP ranks and batches before dividing.
        from collections import defaultdict
        from rod.patches.rod_scoring_loss import PER_SAMPLE_PREFIX, SUM_PREFIX, corpus_means

        timer = Timer()
        # Collect per-batch (loss, weight) pairs where weight = num_valid_samples.
        # Also accumulate every other metric the loss fn returns, weighted the same way.
        batch_metrics: list[dict] = []
        batch_weights: list[float] = []
        corpus_sums: dict[str, float] = defaultdict(float)
        # per_origin[origin] = running {t: target-loss sum, r: ref-loss sum, n: count}
        per_origin: dict[str, dict] = defaultdict(lambda: {"t": 0.0, "r": 0.0, "n": 0})

        with timer.time("total_validation_time"):
            print(f"▶ Starting validation at step {step}...")
            policy.prepare_for_training()
            for batch_idx, val_batch in enumerate(val_dataloader):
                # Tag per-message loss masks (assistant tokens = 1, others = 0)
                # BEFORE flattening — ``batched_message_log_to_flat_message``
                # reads ``token_loss_mask`` from each message dict and would
                # raise KeyError without this step. Mirrors sft.validate line
                # 267 in the upstream NeMo-RL v0.6.0.
                add_loss_mask_to_message_log(
                    val_batch["message_log"],
                    roles_to_train_on=["assistant"],
                )

                cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                    val_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                # Build the minimum BatchedDataDict ``maybe_pad_last_batch`` knows
                # how to extend. The padder ONLY pads a hard-coded allowlist
                # (input_ids, input_lengths, token_mask, sample_mask,
                # reference_policy_logprobs); ``ref_loss``/``origin_tag`` aren't
                # on it, so adding them before the padder would leave them at
                # the original length and the downstream ``shard_by_batch_size``
                # would raise "Batch sizes are not the same" on partial batches.
                # Pattern mirrors ``run_precompute.py`` — pad first, then thread
                # the extra fields at the post-padding size.
                val_data = BatchedDataDict({
                    "input_ids": cat_and_padded["token_ids"],
                    "input_lengths": input_lengths,
                    "token_mask": cat_and_padded["token_loss_mask"],
                    "sample_mask": val_batch["loss_multiplier"],
                })
                val_data.update(cat_and_padded.get_multimodal_dict(as_tensors=False))

                orig_size = val_data.size
                # Pad EVERY batch up to a multiple of dp_size × val_mbs — not just a
                # partial last batch. A FULL batch whose size isn't divisible by
                # dp_size × val_mbs still shards to a per-rank remainder that
                # make_microbatch_iterator rejects ("Data dict size (2) is not a
                # multiple of microbatch size (8)"), e.g. 16 rows / dp8 = 2, 2 % 8 ≠ 0.
                # maybe_pad_last_batch is a no-op when the batch is already aligned.
                dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
                val_data = maybe_pad_last_batch(val_data, dp_size, val_mbs)
                target_size = val_data.size
                pad_count = target_size - orig_size

                # Pad ref_loss (float tensor) with zeros — padding rows have
                # sample_mask=0 so the loss fn ignores them.
                if "ref_loss" in val_batch:
                    rl = val_batch["ref_loss"]
                    if pad_count > 0:
                        rl = torch.cat([
                            rl,
                            torch.zeros(pad_count, dtype=rl.dtype, device=rl.device),
                        ])
                    val_data["ref_loss"] = rl

                # Pad origin_tag (int64 tensor) — sample_mask=0 on padded rows.
                if "origin_tag" in val_batch:
                    ot = val_batch["origin_tag"]
                    if pad_count > 0:
                        ot = torch.cat([
                            ot,
                            torch.zeros(pad_count, dtype=ot.dtype, device=ot.device),
                        ])
                    val_data["origin_tag"] = ot

                val_results = policy.train(
                    val_data, loss_fn, eval_mode=True, gbs=val_data.size, mbs=val_mbs,
                )

                if len(val_results["all_mb_metrics"]) == 0:
                    warnings.warn(
                        "No validation metrics were collected for this batch."
                    )
                    continue

                # Weight by num_valid_samples (sequence-level normalization).
                num_valid_samples = float(
                    val_data["sample_mask"].sum().item()
                )

                # ``val_results["all_mb_metrics"]`` maps each metric name to its list of
                # per-microbatch (and per-DP-rank) values. As in NeMo-RL's training loop,
                # scalar entries are fractions of a globally normalized value and are
                # summed; the global batch sizes are averaged.
                mb_metrics = val_results["all_mb_metrics"]
                merged: dict[str, float] = {}
                for k, vs in mb_metrics.items():
                    numeric_vs = [float(v) for v in vs if isinstance(v, (int, float))]
                    if not numeric_vs or k.startswith(PER_SAMPLE_PREFIX):
                        continue
                    if k.startswith(SUM_PREFIX):
                        corpus_sums[k[len(SUM_PREFIX):]] += sum(numeric_vs)
                    elif k in ("global_valid_seqs", "global_valid_toks"):
                        merged[k] = sum(numeric_vs) / len(numeric_vs)
                    else:
                        merged[k] = sum(numeric_vs)

                # Headline ``val_loss`` — ``val_results["loss"]`` is already
                # globally aggregated across DP + microbatches. Use it
                # directly rather than re-averaging via mb_metrics.
                if "loss" in val_results:
                    merged["val_loss"] = float(val_results["loss"])

                # --- Per-origin accumulation (driver-side, dilution-free) ----
                # Per-sample target_loss comes back from the loss fn in val_data
                # order (gbs == data.size → the concatenated per-sample tensors
                # are in input order). Align it with the driver-side origin /
                # ref_loss for the real (un-padded) rows and bucket by origin.
                ps_key = PER_SAMPLE_PREFIX + "target_loss"
                origins = val_batch.get("origin")
                if ps_key in mb_metrics and origins is not None:
                    tl_list = mb_metrics[ps_key]
                    try:
                        target_ps = torch.cat(
                            [t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                             for t in tl_list],
                            dim=0,
                        )
                    except Exception:  # pragma: no cover - defensive
                        target_ps = None
                    if target_ps is not None and target_ps.shape[0] >= orig_size:
                        rl = val_data.get("ref_loss")
                        sm = val_data["sample_mask"]
                        n_rows = min(orig_size, len(origins))
                        for i in range(n_rows):
                            if float(sm[i]) <= 0:
                                continue  # padding row
                            acc = per_origin[str(origins[i])]
                            acc["t"] += float(target_ps[i])
                            if rl is not None:
                                acc["r"] += float(rl[i])
                            acc["n"] += 1

                batch_metrics.append(merged)
                batch_weights.append(num_valid_samples)

                if val_batches > 0 and batch_idx >= val_batches - 1:
                    break

        policy.prepare_for_training()

        timing_metrics = timer.get_timing_metrics(reduction_op="sum")

        if not batch_metrics:
            warnings.warn("No validation metrics were collected.")
            return {}, timing_metrics

        total_weight = sum(batch_weights) or 1.0
        # Sequence-weighted average of every metric.
        agg: dict[str, float] = {}
        for metrics_dict, w in zip(batch_metrics, batch_weights):
            for k, v in metrics_dict.items():
                agg.setdefault(k, 0.0)
                agg[k] += v * w / total_weight

        agg.update(corpus_means(corpus_sums))

        # Per-origin val loss: exact means over all processed val samples of each
        # origin, logged as ``validation/loss_by_origin_<origin>/<series>`` (prefix
        # added by the caller).
        for origin, a in per_origin.items():
            if a["n"] == 0:
                continue
            base = f"loss_by_origin_{_metric_safe(origin)}/"
            agg[f"{base}target_loss"] = a["t"] / a["n"]
            agg[f"{base}ref_loss"] = a["r"] / a["n"]
            agg[f"{base}n"] = float(a["n"])

        print("\n📊 Validation Results:")
        if "val_loss" in agg:
            print(f"    • Validation loss: {agg['val_loss']:.4f}")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

        timer.reset()
        return agg, timing_metrics

    return patched_validate


def apply() -> None:
    """Replace ``nemo_rl.algorithms.sft.validate`` with the patched version.

    Idempotent: subsequent calls are no-ops. Must run before any
    ``from nemo_rl.algorithms.sft import sft_train`` so the patched
    ``validate`` is the one ``sft_train`` calls internally.
    """
    global _APPLIED
    if _APPLIED:
        return

    from nemo_rl.algorithms import sft as _sft_mod

    _sft_mod.validate = _patched_validate()

    _APPLIED = True
    log.info("Patched nemo_rl.algorithms.sft.validate (sequence-weighted val_loss aggregation)")
