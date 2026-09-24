# Copyright (c) 2026 The RoD Authors.
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

"""``RHOScoringLossFn`` — eval-only loss fn used for validation.

Validation runs NeMo-RL's ``policy.train(..., eval_mode=True)`` with this loss fn
(see :mod:`rod.patches.rod_validate`). It:

- never contributes gradient (it only runs under ``eval_mode=True``, and its
  returned loss is detached);
- computes the per-sequence loss with :func:`rod.patches.rod_loss.compute_target_loss`,
  the same formula used by the reference-loss precompute and the per-step scoring
  pass, and the corresponding ``rho = target_loss − ref_loss``;
- returns per-corpus sums and counts (target loss, reference loss, rho, replay samples
  with rho > 0), from which the validation driver forms exact means with
  :func:`corpus_means`;
- emits the per-sample target loss under the ``_per_sample_target_loss`` key so the
  validation driver can compute exact per-origin validation losses.

The per-step *training* selection does not use this class: the scoring pass
(:mod:`rod.patches.rod_score_pass`) computes losses directly from
``policy.get_logprobs`` on the driver.

The returned loss is the DP-global masked-mean target loss over valid samples
(the same number :class:`NLLLossFn` would produce on this batch); it is surfaced
as the headline ``val_loss``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from rod.patches.rod_loss import (
    ADAPT_TAG,
    REPLAY_TAG,
    LossInputType,
    LossType,
    _is_model_parallel_src_rank,
    _masked_mean,
    compute_target_loss,
)

log = logging.getLogger(__name__)


# Prefix for per-sample tensors placed into ``all_mb_metrics``; the validation
# driver concatenates these (rather than aggregating them like scalar metrics).
PER_SAMPLE_PREFIX = "_per_sample_"
# Prefix for the per-corpus sums and counts (summed by the driver, see corpus_means).
SUM_PREFIX = "_sum_"

# logged metric -> (numerator, denominator) among the SUM_PREFIX entries
CORPUS_MEANS = {
    "loss/target_loss_adapt": ("target_loss_adapt", "n_adapt"),
    "loss/target_loss_replay": ("target_loss_replay", "n_replay"),
    "loss/ref_loss_adapt": ("ref_loss_adapt", "n_adapt"),
    "loss/ref_loss_replay": ("ref_loss_replay", "n_replay"),
    "rho/rho_mean_adapt": ("rho_adapt", "n_adapt"),
    "rho/rho_mean_replay": ("rho_replay", "n_replay"),
    "rho/frac_rho_replay_above_zero": ("n_rho_replay_above_zero", "n_replay"),
}


def corpus_means(sums: dict[str, float]) -> dict[str, float]:
    """Per-corpus means from the summed ``SUM_PREFIX`` entries (keys without the prefix)."""
    return {k: sums.get(num, 0.0) / max(sums.get(den, 0.0), 1.0) for k, (num, den) in CORPUS_MEANS.items()}


class RHOScoringLossFn:
    """Eval-only, SEQUENCE_LEVEL loss fn: per-sample loss + per-corpus sums for rho metrics."""

    # NeMo-RL ``LossFunction`` Protocol attributes.
    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGPROB

    def __call__(
        self,
        next_token_logprobs: Tensor,
        data: dict[str, Tensor],
        global_valid_seqs: Tensor,
        global_valid_toks: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute per-sample loss + aggregate metrics for one microbatch."""
        # NeMo-RL ships a full-length token mask; shift to match next_token_logprobs.
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]            # [B]
        ref_loss = data.get("ref_loss")              # [B] from the cache; None in plain CPT runs
        origin_tag = data["origin_tag"]              # [B] int64

        loss_per_seq = compute_target_loss(next_token_logprobs, token_mask, sample_mask)  # [B]

        # Plain continual pre-training (no reference cache): there is no ``ref_loss``.
        # Default it to zeros so rho == target_loss (unused) while the held-out
        # target loss we log per origin is computed normally.
        if ref_loss is None:
            ref_loss = torch.zeros_like(loss_per_seq)

        rho = loss_per_seq - ref_loss  # [B]

        with torch.no_grad():
            headline = _masked_mean(loss_per_seq.detach(), sample_mask, global_norm=global_valid_seqs)
            metrics: dict[str, Any] = {
                "loss": float(headline.item()),
                "num_valid_samples": float(sample_mask.sum().item()),
                "global_valid_seqs": float(global_valid_seqs.item()),
            }
            # Rank-local sums and the detached per-sample loss, emitted ONLY from the
            # model-parallel source rank so ranks holding the same data (TP/EP) never
            # count it twice. The driver sums them over microbatches, DP ranks and batches.
            if _is_model_parallel_src_rank():
                sums = self._corpus_sums(loss_per_seq, rho, ref_loss, sample_mask, origin_tag)
                metrics.update({f"{SUM_PREFIX}{k}": v for k, v in sums.items()})
                metrics[f"{PER_SAMPLE_PREFIX}target_loss"] = (
                    loss_per_seq.detach().to("cpu", dtype=torch.float32)
                )
        return headline, metrics

    @staticmethod
    def _corpus_sums(
        loss_per_seq: Tensor,
        rho: Tensor,
        ref_loss: Tensor,
        sample_mask: Tensor,
        origin_tag: Tensor,
    ) -> dict[str, float]:
        """Per-corpus sums and counts over this microbatch's valid samples."""
        is_adapt = (origin_tag == ADAPT_TAG).to(loss_per_seq.dtype) * sample_mask
        is_replay = (origin_tag == REPLAY_TAG).to(loss_per_seq.dtype) * sample_mask
        return {
            "target_loss_adapt": float((loss_per_seq * is_adapt).sum()),
            "target_loss_replay": float((loss_per_seq * is_replay).sum()),
            "ref_loss_adapt": float((ref_loss * is_adapt).sum()),
            "ref_loss_replay": float((ref_loss * is_replay).sum()),
            "rho_adapt": float((rho * is_adapt).sum()),
            "rho_replay": float((rho * is_replay).sum()),
            "n_rho_replay_above_zero": float(((rho > 0).to(rho.dtype) * is_replay).sum()),
            "n_adapt": float(is_adapt.sum()),
            "n_replay": float(is_replay.sum()),
        }
