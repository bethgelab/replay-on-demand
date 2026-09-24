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

"""Unit tests for RoD's driver-side candidate selection (no GPU / no nemo_rl).

Run: pytest tests/test_selection.py
"""
import pytest
import torch

from rod.patches.rod_sft_train import _compute_selection_metrics, _select_candidates

ADAPT, REPLAY = 0, 1


def _pool():
    # 4 adaptation + 4 replay candidates; replay rho values are interleaved with adapt ones.
    rho = torch.tensor([0.9, 0.7, 0.2, 0.1, 0.8, 0.6, 0.05, 0.0])
    tags = torch.tensor([ADAPT] * 4 + [REPLAY] * 4)
    return rho, tags, torch.ones(8)


def test_top_k_is_one_joint_ranking_without_quota():
    rho, tags, mask = _pool()
    idx = _select_candidates(rho=rho, sample_mask=mask, k=4, strategy="top_k", origin_tag=tags)
    assert set(idx.tolist()) == {0, 4, 1, 5}          # the 4 highest rho across BOTH corpora
    # the replay share of the batch emerges from the ranking (here 2 of 4)
    assert int((tags[idx] == REPLAY).sum()) == 2


def test_top_k_all_adaptation_when_replay_rho_is_low():
    rho = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0])   # e.g. step 0: θ_t = θ_0 → ρ_R = 0
    tags = torch.tensor([ADAPT] * 4 + [REPLAY] * 4)
    idx = _select_candidates(rho=rho, sample_mask=torch.ones(8), k=4, strategy="top_k", origin_tag=tags)
    assert (tags[idx] == ADAPT).all()


def test_padding_is_never_selected_and_k_is_capped():
    rho = torch.tensor([5.0, 0.1, 0.2, 9.0])
    mask = torch.tensor([0.0, 1.0, 1.0, 0.0])        # the two largest scores are padding
    idx = _select_candidates(rho=rho, sample_mask=mask, k=4, strategy="top_k")
    assert sorted(idx.tolist()) == [1, 2]
    assert _select_candidates(rho=rho, sample_mask=torch.zeros(4), k=2, strategy="top_k") is None


def test_fixed_ratio_takes_exact_share_uniformly_within_corpora():
    torch.manual_seed(0)
    rho, tags, mask = _pool()
    idx = _select_candidates(rho=rho, sample_mask=mask, k=4, strategy="fixed_ratio",
                             origin_tag=tags, fixed_replay_ratio=0.25)
    assert len(idx) == 4 and len(set(idx.tolist())) == 4
    assert int((tags[idx] == REPLAY).sum()) == 1 and int((tags[idx] == ADAPT).sum()) == 3


def test_fixed_ratio_backfills_when_a_corpus_runs_short():
    rho = torch.zeros(5)
    tags = torch.tensor([ADAPT, REPLAY, REPLAY, REPLAY, REPLAY])
    idx = _select_candidates(rho=rho, sample_mask=torch.ones(5), k=4, strategy="fixed_ratio",
                             origin_tag=tags, fixed_replay_ratio=0.25)
    assert int((tags[idx] == ADAPT).sum()) == 1 and int((tags[idx] == REPLAY).sum()) == 3


def test_unknown_strategy_raises():
    rho, tags, mask = _pool()
    with pytest.raises(ValueError):
        _select_candidates(rho=rho, sample_mask=mask, k=2, strategy="not_a_strategy", origin_tag=tags)


def test_selection_metrics_report_replay_share_and_consistent_losses():
    rho, tags, mask = _pool()
    target = rho + 1.0                                # so ref_loss = target - rho = 1.0 everywhere
    idx = _select_candidates(rho=rho, sample_mask=mask, k=4, strategy="top_k", origin_tag=tags)
    m = _compute_selection_metrics(rho=rho, target_loss=target, origin_tag=tags, sample_mask=mask,
                                   selected_idx=idx, step=1, rich_log_period=1,
                                   origin=["a"] * 4 + ["r"] * 4)
    assert m["selection/n_selected_replay"] == 2 and m["selection/k_effective"] == 4
    assert abs(m["loss/ref_loss_adapt"] - 1.0) < 1e-6 and abs(m["loss/ref_loss_replay"] - 1.0) < 1e-6
    assert m["selection_sanity/rho_mean_selected"] > m["selection_sanity/rho_mean_dropped"]
    assert m["selection_by_origin/replay/r/n_selected"] == 2


def test_fixed_share_rho_strategies():
    """Figure-5a ladder: RHO within the replay stream only, then within both streams."""
    rho, tags, mask = _pool()   # adapt rho 0.9 0.7 0.2 0.1 | replay rho 0.8 0.6 0.05 0.0
    for strategy, want_replay in (("replay_only_rho", {4}), ("fixed_ratio_rho", {4})):
        torch.manual_seed(0)
        idx = _select_candidates(rho=rho, sample_mask=mask, k=4, strategy=strategy,
                                 origin_tag=tags, fixed_replay_ratio=0.25)
        replay_sel = {i for i in idx.tolist() if tags[i] == REPLAY}
        assert replay_sel == want_replay                       # highest-rho replay candidate
        assert int((tags[idx] == ADAPT).sum()) == 3
    idx = _select_candidates(rho=rho, sample_mask=mask, k=4, strategy="fixed_ratio_rho",
                             origin_tag=tags, fixed_replay_ratio=0.25)
    assert {i for i in idx.tolist() if tags[i] == ADAPT} == {0, 1, 2}   # top-3 adapt by rho
