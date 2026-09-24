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

"""Shared loss utilities for RoD.

- :func:`compute_target_loss` — the per-sequence, token-normalized language-modeling
  loss ``ℓ_θ(x)``. It is the single source of truth for the loss formula: the
  reference-loss precompute (``run_precompute.py``), the per-step scoring pass
  (:mod:`rod.patches.rod_score_pass`) and validation
  (:class:`rod.patches.rod_scoring_loss.RHOScoringLossFn`) all call it, so reference
  and current-model losses are computed identically and their difference
  ``rho = ℓ_θ(x) − ℓ_ref(x)`` is well defined.
- ``ADAPT_TAG`` / ``REPLAY_TAG`` — the origin-tag convention (0 = adaptation,
  1 = replay) shared with the data pipeline.
- Small distributed helpers used by the validation loss fn.
"""

from __future__ import annotations

import enum
import logging

import torch
from torch import Tensor

log = logging.getLogger(__name__)


# NeMo-RL's LossType / LossInputType enums — imported eagerly so loss fns can set
# class attrs at definition time (NeMo-RL's worker reads them). When nemo_rl
# isn't installed (unit tests outside the container), fall back to local enums
# with identical members.
try:  # pragma: no cover - production path
    from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
except ImportError:  # pragma: no cover - test/dev fallback
    class LossType(enum.Enum):  # type: ignore[no-redef]
        TOKEN_LEVEL = "token_level"
        SEQUENCE_LEVEL = "sequence_level"

    class LossInputType(enum.Enum):  # type: ignore[no-redef]
        LOGIT = "logit"
        LOGPROB = "logprob"
        DISTILLATION = "distillation"
        DRAFT = "draft"


# Origin-tag conventions. Keep in lockstep with the data pipeline (rod_pretrain_blocks).
ADAPT_TAG = 0
REPLAY_TAG = 1


def compute_target_loss(
    next_token_logprobs: Tensor,
    token_mask: Tensor,
    sample_mask: Tensor,
) -> Tensor:
    """Per-sequence token-normalized NLL: ``ℓ(x) = (1/|T(x)|) Σ_{t ∈ T(x)} −log p(x_t | x_<t)``.

    Args:
        next_token_logprobs: ``[B, T-1]`` log-prob of the true next token (already
            shifted, as NeMo-RL returns them).
        token_mask: ``[B, T-1]`` 1 on loss-bearing tokens (all tokens for packed
            pre-training blocks), shifted to match ``next_token_logprobs``.
        sample_mask: ``[B]`` 1 for valid samples, 0 for padding samples.

    Returns:
        ``[B]`` per-sequence loss. Padding samples contribute zero.
    """
    mask = token_mask * sample_mask.unsqueeze(-1)
    nll_per_seq = -(next_token_logprobs * mask).sum(dim=-1)  # [B]
    # Clamp the denominator to 1 so padding samples don't produce inf/nan.
    n_tokens = mask.sum(dim=-1).clamp(min=1.0)  # [B]
    return (nll_per_seq / n_tokens) * sample_mask


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _is_model_parallel_src_rank() -> bool:
    """True if this rank should emit the per-sample losses and per-corpus sums.

    Validation carries per-sample losses and per-corpus sums off the workers through
    ``all_mb_metrics`` (see :class:`RHOScoringLossFn`), which the driver concatenates
    or sums across microbatches and DP ranks.

    NeMo-RL v0.6.0 collects ``all_mb_metrics`` from **one representative per DP
    shard** (coord-0 on every replicated axis — TP, PP, CP, EP), so TP/EP do not
    duplicate these entries. Emitting them only on the canonical source
    rank (``tp_rank == 0`` and ``ep_rank == 0``) is therefore a no-op today, but it
    makes the "collected once per DP shard" invariant explicit and skips wasted
    device→CPU copies on ranks whose result is discarded. Under pure DP every rank
    has ``tp_rank == ep_rank == 0`` → returns ``True`` everywhere.

    Context parallelism (CP>1) is unsupported (``run_rod.py`` rejects it): it shards
    tokens within a sequence, so the per-sequence loss would be partial.

    Returns ``True`` when ``parallel_state`` is unavailable (single process / unit
    tests) so nothing is dropped off-cluster.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    try:
        from megatron.core import parallel_state
    except Exception:
        return True
    # tp_rank: tensor-parallel index. ep_rank: expert-parallel index. Both default
    # to 0 (i.e. "this rank emits") if the accessor is absent.
    try:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
    except Exception:
        tp_rank = 0
    try:
        ep_rank = parallel_state.get_expert_model_parallel_rank()
    except Exception:
        ep_rank = 0
    return tp_rank == 0 and ep_rank == 0


def _masked_mean(values: Tensor, mask: Tensor, global_norm: Tensor | float) -> Tensor:
    """Masked mean with a *global* normalization factor.

    Mirrors NeMo-RL's ``masked_mean`` used in NLLLossFn:
    ``sum(values * mask) / global_norm``. The global_norm is expected to be the
    DP-summed denominator so the per-microbatch contribution accumulates
    correctly when NeMo-RL averages across DP at the end.
    """
    numer = (values * mask).sum()
    denom = global_norm if isinstance(global_norm, Tensor) else torch.tensor(
        global_norm, dtype=values.dtype, device=values.device
    )
    # Clamp for safety; if denom is 0 the whole metric is meaningless anyway.
    return numer / denom.clamp(min=1.0)
