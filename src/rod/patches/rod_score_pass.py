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

"""Per-step scoring pass: RHO scores for the whole candidate pool.

:func:`policy_score` is called by the RoD training loop
(:mod:`rod.patches.rod_sft_train`) once per step, before selection:

1. One forward-only pass of the CURRENT model over the candidate pool via
   NeMo-RL's ``policy.get_logprobs`` (no autograd, no optimizer step), gathered
   to the driver.
2. Per-sequence loss with :func:`rod.patches.rod_loss.compute_target_loss` — the
   same formula the reference-loss precompute used.
3. ``rho = target_loss − ref_loss`` with the cached reference loss of each
   candidate (the adaptation specialist's loss for adaptation candidates, the
   base model's loss for replay candidates).

Cost: one extra forward pass over the candidate pool per training step.
"""

from __future__ import annotations

import logging
from typing import Any

from torch import Tensor

from rod.patches.rod_loss import _masked_mean, compute_target_loss

log = logging.getLogger(__name__)


def policy_score(policy: Any, candidate_batch: Any) -> dict[str, Any]:
    """Score the candidate pool with the CURRENT (target) model; return per-sample rho.

    Requires ``candidate_batch`` (a :class:`BatchedDataDict` of size
    ``policy.train_global_batch_size``, i.e. the candidate pool) to carry
    ``input_ids``, ``input_lengths``, ``token_mask``, ``sample_mask``, ``ref_loss``
    and ``origin_tag`` (plumbed by the collate patch).
    """
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    cfg_gbs = int(policy.cfg["train_global_batch_size"])
    if candidate_batch.size != cfg_gbs:
        raise ValueError(
            f"policy_score: candidate_batch.size ({candidate_batch.size}) must equal the "
            f"policy's train_global_batch_size ({cfg_gbs})."
        )
    for field in ("input_ids", "input_lengths", "token_mask", "sample_mask", "ref_loss", "origin_tag"):
        if field not in candidate_batch:
            raise RuntimeError(
                f"policy_score: candidate_batch missing '{field}' — check the collate patch."
            )

    # get_logprobs needs only input_ids + input_lengths (attention/position built internally);
    # pass the minimal dict, keep the rest driver-side for the reduction. Row order is preserved
    # (no sequence_packing / dynamic_batching), so logprobs[i] aligns with candidate row i.
    # Microbatching is controlled by policy.logprob_batch_size.
    lp_data = BatchedDataDict({
        "input_ids": candidate_batch["input_ids"],
        "input_lengths": candidate_batch["input_lengths"],
    })
    logprobs = policy.get_logprobs(lp_data)["logprobs"]          # [GBS, S], gathered to driver

    next_token_logprobs = logprobs[:, 1:]                        # [GBS, S-1]
    token_mask = candidate_batch["token_mask"][:, 1:]            # [GBS, S-1]
    sample_mask = candidate_batch["sample_mask"]                 # [GBS]
    ref_loss = candidate_batch["ref_loss"]                       # [GBS]
    origin_tag = candidate_batch["origin_tag"]                   # [GBS]

    target_loss = compute_target_loss(next_token_logprobs, token_mask, sample_mask)  # [GBS]
    rho = target_loss - ref_loss                                 # [GBS]

    return {
        "per_sample_rho": rho,
        "per_sample_target_loss": target_loss,
        "per_sample_origin_tag": origin_tag,
        "per_sample_sample_mask": sample_mask,
        "aggregate_metrics": _score_aggregate_metrics(
            target_loss, rho, ref_loss, sample_mask, origin_tag
        ),
    }


def _score_aggregate_metrics(
    target_loss: Tensor,
    rho: Tensor,
    ref_loss: Tensor,
    sample_mask: Tensor,
    origin_tag: Tensor,
) -> dict[str, float]:
    """Pool-level scalars of the scoring pass (per-corpus means are part of the selection
    metrics). The pool loss is logged as ``score/candidate_loss`` so it does not overwrite
    the training loss of the selected batch (logged as ``loss``)."""
    n_valid = sample_mask.sum()
    return {
        "score/candidate_loss": float(_masked_mean(target_loss, sample_mask, n_valid).item()),
        "score/num_valid_candidates": float(n_valid.item()),
    }
