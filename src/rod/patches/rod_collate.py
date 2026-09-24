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

"""Runtime monkey patch for ``nemo_rl.data.collate_fn.rl_collate_fn``.

The stock collate drops fields it doesn't know about. This patch threads the
RoD per-sample fields through to the batch:

- per-example ``ref_loss`` (float) into ``data["ref_loss"]`` as a
  ``[B]``-shaped float32 tensor;
- per-example ``origin_tag`` (int, 0=adapt / 1=replay) into
  ``data["origin_tag"]`` as a ``[B]``-shaped int64 tensor;
- ``content_hash`` and ``origin`` (strings) as plain Python lists.

``origin_tag`` is required on every datum, and ``ref_loss`` on every datum once any
has it (a partial batch would silently corrupt rho scoring); both fail loudly.
Missing ``content_hash`` / ``origin`` default to ``""`` / ``"unknown"``.

Call :func:`apply` once at process startup, before NeMo-RL builds its SFT
dataloader (i.e. before ``nemo_rl.algorithms.sft.setup``).
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch

log = logging.getLogger(__name__)

_APPLIED = False


def _make_patched(original):
    @functools.wraps(original)
    def patched_rl_collate_fn(data_batch, *args, **kwargs):
        # Pull our per-example fields BEFORE forwarding to the original collate
        # (the original doesn't know about ref_loss / origin_tag / origin /
        # content_hash and would silently drop them).
        ref_losses: list[Any] = []
        origin_tags: list[Any] = []
        origins: list[Any] = []
        content_hashes: list[Any] = []
        for datum in data_batch:
            ref_losses.append(datum.get("ref_loss"))
            origin_tags.append(datum.get("origin_tag"))
            origins.append(datum.get("origin"))
            content_hashes.append(datum.get("content_hash"))

        data = original(data_batch, *args, **kwargs)

        # ref_loss is OPTIONAL — present for training (joined from cache),
        # absent for precompute (where we're computing it). When all entries
        # are None, skip plumbing it through.
        if any(r is not None for r in ref_losses):
            if any(r is None for r in ref_losses):
                raise RuntimeError(
                    "RoD collate: ref_loss missing on at least one datum. "
                    "Every row must have ref_loss attached by the data mixer (content-hash join). "
                    "If this fires, the mixer's hard-fail policy was bypassed."
                )
            data["ref_loss"] = torch.tensor(ref_losses, dtype=torch.float32)
        if any(t is None for t in origin_tags):
            raise RuntimeError(
                "RoD collate: origin_tag missing on at least one datum. "
                "Every row must have origin_tag set by RHOPreprocessor."
            )
        data["origin_tag"] = torch.tensor(origin_tags, dtype=torch.long)

        # content_hash and origin are strings — can't be tensors. Pass each
        # through as a Python list. NeMo-RL's BatchedDataDict carries non-tensor
        # values alongside tensors. The driver-side selection metrics consume
        # ``origin`` for per-sub-origin (within-corpus) breakdowns; precompute
        # parquet writers consume it for per-origin diagnostic columns.
        data["content_hash"] = [h if h is not None else "" for h in content_hashes]
        data["origin"] = [o if o is not None else "unknown" for o in origins]
        return data

    return patched_rl_collate_fn


def apply() -> None:
    """Replace ``nemo_rl.data.collate_fn.rl_collate_fn`` with the patched version.

    Also re-binds the locally-captured symbol in ``nemo_rl.algorithms.sft`` if
    that module has already been imported.

    Idempotent: subsequent calls are no-ops.
    """
    global _APPLIED
    if _APPLIED:
        return

    import sys

    from nemo_rl.data import collate_fn as _collate_mod

    original = _collate_mod.rl_collate_fn
    patched = _make_patched(original)
    _collate_mod.rl_collate_fn = patched

    _sft_mod = sys.modules.get("nemo_rl.algorithms.sft")
    if _sft_mod is not None and hasattr(_sft_mod, "rl_collate_fn"):
        _sft_mod.rl_collate_fn = patched
        log.info(
            "Re-bound nemo_rl.algorithms.sft.rl_collate_fn (module was loaded before patch)"
        )

    _APPLIED = True
    log.info(
        "Patched nemo_rl.data.collate_fn.rl_collate_fn "
        "(ref_loss + origin_tag + origin + content_hash plumbing)"
    )
