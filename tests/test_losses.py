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

"""Unit tests for the shared loss formula, the scoring-pass metrics and the validation
loss fn (no GPU / no nemo_rl).

Run: pytest tests/test_losses.py
"""
import math

import torch

from rod.patches import rod_scoring_loss
from rod.patches.rod_loss import (
    ADAPT_TAG,
    REPLAY_TAG,
    _is_model_parallel_src_rank,
    compute_target_loss,
)
from rod.patches.rod_score_pass import _score_aggregate_metrics
from rod.patches.rod_scoring_loss import PER_SAMPLE_PREFIX, SUM_PREFIX, RHOScoringLossFn, corpus_means


def test_compute_target_loss_is_token_mean_nll():
    # logprobs of the true next tokens for 2 sequences of 3 loss-bearing positions.
    lp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]])
    mask = torch.ones_like(lp)
    loss = compute_target_loss(lp, mask, torch.ones(2))
    assert torch.allclose(loss, torch.tensor([2.0, 0.5]))


def test_compute_target_loss_respects_token_and_sample_masks():
    lp = torch.tensor([[-1.0, -3.0, -100.0], [-2.0, -2.0, -2.0]])
    token_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])   # 3rd token of row 0 is not loss-bearing
    sample_mask = torch.tensor([1.0, 0.0])                          # row 1 is padding
    loss = compute_target_loss(lp, token_mask, sample_mask)
    assert torch.allclose(loss, torch.tensor([2.0, 0.0]))
    assert torch.isfinite(loss).all()


def test_src_rank_true_without_distributed():
    assert _is_model_parallel_src_rank() is True


def test_score_metrics_do_not_shadow_training_loss():
    target = torch.tensor([2.0, 3.0, 1.0, 1.5])
    ref = torch.tensor([1.0, 1.0, 1.0, 1.0])
    tags = torch.tensor([ADAPT_TAG, ADAPT_TAG, REPLAY_TAG, REPLAY_TAG])
    m = _score_aggregate_metrics(target, target - ref, ref, torch.ones(4), tags)
    assert set(m) == {"score/candidate_loss", "score/num_valid_candidates"}   # no train-step keys
    assert math.isclose(m["score/candidate_loss"], 1.875)


def _tiny_batch(B=4, T=5):
    """Minimal microbatch for RHOScoringLossFn: loss 0.5 everywhere, ref 0.25 on replay."""
    logprobs = torch.full((B, T - 1), -0.5)
    data = {
        "token_mask": torch.ones(B, T),       # full length; the loss fn shifts it
        "sample_mask": torch.ones(B),
        "ref_loss": torch.tensor([0.0, 0.0, 0.25, 0.25]),
        "origin_tag": torch.tensor([ADAPT_TAG, ADAPT_TAG, REPLAY_TAG, REPLAY_TAG]),
    }
    return logprobs, data, torch.tensor(float(B)), torch.tensor(float(B * (T - 1)))


def _sums(metrics):
    return {k[len(SUM_PREFIX):]: v for k, v in metrics.items() if k.startswith(SUM_PREFIX)}


def test_validation_loss_fn_emits_per_sample_loss_and_corpus_sums():
    fn = RHOScoringLossFn()
    logprobs, data, gvs, gvt = _tiny_batch()
    loss, metrics = fn(logprobs, data, gvs, gvt)
    assert math.isclose(float(loss), 0.5)
    assert [k for k in metrics if k.startswith(PER_SAMPLE_PREFIX)] == [f"{PER_SAMPLE_PREFIX}target_loss"]
    assert torch.allclose(metrics[f"{PER_SAMPLE_PREFIX}target_loss"], torch.full((4,), 0.5))
    means = corpus_means(_sums(metrics))
    assert math.isclose(means["rho/rho_mean_adapt"], 0.5)
    assert math.isclose(means["rho/rho_mean_replay"], 0.25)
    assert math.isclose(means["rho/frac_rho_replay_above_zero"], 1.0)


def test_corpus_sums_add_up_across_ranks():
    """Summing the per-rank sums over DP shards (as the validation driver does) gives the
    means over the full batch, whatever the split."""
    fn = RHOScoringLossFn()
    logprobs, data, gvs, gvt = _tiny_batch()
    logprobs = logprobs * torch.tensor([1.0, 2.0, 3.0, 4.0])[:, None]      # distinct losses
    _, full = fn(logprobs, data, gvs, gvt)
    total = {}
    for rows in ([0, 2, 3], [1]):                                             # uneven "DP ranks"
        part = {k: v[rows] for k, v in data.items()}
        _, m = fn(logprobs[rows], part, gvs, gvt)
        for k, v in _sums(m).items():
            total[k] = total.get(k, 0.0) + v
    for k, v in corpus_means(_sums(full)).items():
        assert math.isclose(corpus_means(total)[k], v), k


def test_validation_loss_fn_without_reference_cache():
    fn = RHOScoringLossFn()
    logprobs, data, gvs, gvt = _tiny_batch()
    data.pop("ref_loss")                       # plain continual pre-training
    loss, metrics = fn(logprobs, data, gvs, gvt)
    assert math.isclose(float(loss), 0.5)
    assert math.isclose(corpus_means(_sums(metrics))["loss/ref_loss_replay"], 0.0)


def test_validation_loss_fn_silent_on_non_source_rank(monkeypatch):
    """Non-source model-parallel ranks (same data as the source rank) emit neither
    per-sample tensors nor corpus sums, so nothing is counted twice."""
    monkeypatch.setattr(rod_scoring_loss, "_is_model_parallel_src_rank", lambda: False)
    fn = RHOScoringLossFn()
    logprobs, data, gvs, gvt = _tiny_batch()
    _, metrics = fn(logprobs, data, gvs, gvt)
    assert not any(k.startswith((PER_SAMPLE_PREFIX, SUM_PREFIX)) for k in metrics)
    assert "loss" in metrics
