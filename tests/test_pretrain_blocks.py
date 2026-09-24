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

"""Unit tests for the block-ingestion data path (no GPU / no nemo_rl / no real tokenizer).

Run: pytest tests/test_pretrain_blocks.py
Covers: document packing, per-subset origin derivation, origin_map grouping, coarse
fallback, deterministic content hashes, per-domain caps, deterministic holdout split,
replay balancing, the per-origin validation cap, the reference-loss cache join, the
adapt/replay selection of the blocks mixer, and curriculum transfer from selection traces.
"""
import collections
import glob
import hashlib
import json
import os
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rod.data import rod_pretrain_blocks as rpb

DOC = " ".join(f"w{i}" for i in range(60))


def _tid(w):
    return int(hashlib.md5(w.encode()).hexdigest()[:4], 16) % 900 + 10


class _FakeTok:
    """Deterministic whitespace tokenizer standing in for the HF tokenizer."""

    eos_token_id = 2
    bos_token_id = 1

    def __call__(self, batch, add_special_tokens=False):
        return {"input_ids": [[_tid(w) for w in t.split()] for t in batch]}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Raw parquet corpora: legal (2 subsets) + general (2 text subsets + 1 without text)."""
    monkeypatch.setitem(
        __import__("sys").modules, "transformers",
        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeTok())),
    )

    def write(sub, n, with_text=True):
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        col = "text" if with_text else "rel_path"
        pq.write_table(pa.table({col: [DOC] * n}), d / "part.parquet")

    write("legal/CaseHOLD", 200)
    write("legal/eCFR", 200)
    write("general/Nemotron-CC-v2", 200)
    write("general/Nemotron-CC-Math-v1", 200)
    write("general/Nemotron-Code-Metadata", 2, with_text=False)
    return types.SimpleNamespace(
        root=tmp_path,
        adapt_glob=str(tmp_path / "legal/**/*.parquet"),
        replay_glob=str(tmp_path / "general/**/*.parquet"),
    )


def _build(corpus, out, **kw):
    kw.setdefault("val_holdout_fraction", 0.0)
    rpb.build_blocks_corpus(adapt_glob=corpus.adapt_glob, replay_glob=corpus.replay_glob,
                            out_path=str(out), tokenizer_path="fake", block_size=16, eod_id=2,
                            seed=7, **kw)
    rows = []
    for f in sorted(glob.glob(os.path.join(str(out), "*.parquet"))):
        rows.extend(pq.read_table(f).to_pylist())
    return rows


# --- pure helpers -------------------------------------------------------------------------

def test_pack_token_streams_eod_and_drop_partial():
    blocks = list(rpb.pack_token_streams([[5, 6, 7], [8, 9]], block_size=3, eod_id=0))
    # stream = 5 6 7 0 8 9 0 → two full blocks, trailing partial block dropped
    assert blocks == [[5, 6, 7], [0, 8, 9]]
    blocks = list(rpb.pack_token_streams([[5, 6, 7], [8, 9]], block_size=3, eod_id=0,
                                         drop_last_partial=False))
    assert blocks[-1] == [0]


def test_clean_origin():
    assert rpb._clean_origin("Nemotron-CC-Math-v1") == "math_v1"
    assert rpb._clean_origin("CaseHOLD") == "casehold"


def test_block_content_hash_depends_on_origin_and_tokens():
    h = rpb.block_content_hash([1, 2, 3], "a")
    assert h == rpb.block_content_hash([1, 2, 3], "a")
    assert h != rpb.block_content_hash([1, 2, 3], "b")
    assert h != rpb.block_content_hash([1, 2, 4], "a")


def test_split_holdout_per_origin_deterministic_and_floored():
    blocks = [{"domain": "d", "origin": o, "content_hash": f"{o}{i}"}
              for o, n in (("big", 1000), ("small", 10)) for i in range(n)]
    tr1, ho1 = rpb.split_holdout_per_origin(blocks, 0.01, seed=42, floor=5)
    tr2, ho2 = rpb.split_holdout_per_origin(list(reversed(blocks)), 0.01, seed=42, floor=5)
    assert {b["content_hash"] for b in ho1} == {b["content_hash"] for b in ho2}  # order-independent
    by = collections.Counter(b["origin"] for b in ho1)
    assert by["big"] == 10          # ceil(1% of 1000)
    assert by["small"] == 2         # floor 5, but capped at 20% of a small origin
    assert len(tr1) + len(ho1) == len(blocks)


def test_balance_replay_matches_adapt_size_and_is_deterministic():
    rows = ([{"origin_tag": 0, "origin": "a", "content_hash": f"a{i}"} for i in range(2000)] +
            [{"origin_tag": 1, "origin": o, "content_hash": f"{o}{i}"} for o in ("r1", "r2") for i in range(4000)])
    out = rpb.balance_replay(rows, seed=42)
    assert sum(r["origin_tag"] == 0 for r in out) == 2000                     # all adaptation kept
    n_rep = collections.Counter(r["origin"] for r in out if r["origin_tag"] == 1)
    assert abs(sum(n_rep.values()) - 2000) < 150                              # ~ adaptation size
    assert abs(n_rep["r1"] - n_rep["r2"]) < 150                               # proportions preserved
    assert [r["content_hash"] for r in rpb.balance_replay(list(reversed(rows)), seed=42)][::-1] == \
        [r["content_hash"] for r in out]                                      # order-independent
    small = rows[:2100]                                                       # replay already <= adapt
    assert rpb.balance_replay(small) == small


def test_cap_per_origin_keeps_a_fixed_subset():
    rows = [{"origin": o, "content_hash": f"{o}{i}"} for o, n in (("big", 50), ("small", 3)) for i in range(n)]
    out = rpb.cap_per_origin(rows, 10)
    assert collections.Counter(r["origin"] for r in out) == {"big": 10, "small": 3}
    assert {r["content_hash"] for r in rpb.cap_per_origin(list(reversed(rows)), 10)} == \
        {r["content_hash"] for r in out}
    assert rpb.cap_per_origin(rows, None) == rows


# --- corpus build -------------------------------------------------------------------------

def test_per_subset_origins_and_caps(corpus):
    rows = _build(corpus, corpus.root / "out", max_blocks_adapt=30, max_blocks_replay=80)
    by_domain = collections.Counter(r["domain"] for r in rows)
    assert by_domain == {"legal": 30, "general": 80}
    replay_origins = {r["origin"] for r in rows if r["domain"] == "general"}
    assert {"v2", "math_v1"} <= replay_origins
    assert "code_metadata" not in replay_origins            # subset without text is skipped
    assert all(r["origin_tag"] == (1 if r["domain"] == "general" else 0) for r in rows)
    assert all(r["content_hash"] == rpb.block_content_hash(r["token_ids"], r["origin"]) for r in rows)
    assert all(len(r["token_ids"]) == 16 for r in rows)


def test_origin_map_and_coarse_mode(corpus):
    rows = _build(corpus, corpus.root / "out2",
                  origin_map={"Nemotron-CC-v2": "web", "Nemotron-CC-Math-v1": "web"})
    assert {r["origin"] for r in rows if r["domain"] == "general"} == {"web"}
    rows = _build(corpus, corpus.root / "out3", per_subset_origin=False)
    assert {r["origin"] for r in rows} == {"legal", "general"}


# --- reference-loss cache + mixer -----------------------------------------------------------

def _write_cache(cache_dir, rows, ref=lambda r: 1.0 + r["origin_tag"]):
    cache_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(
        [{"content_hash": r["content_hash"], "ref_loss": ref(r)} for r in rows]),
        cache_dir / "refloss_train_x.parquet")


def test_join_ref_loss_attaches_and_fails_on_miss():
    rows = [{"content_hash": "a"}, {"content_hash": "b"}]
    out = rpb.join_ref_loss([dict(r) for r in rows], {"a": 1.5, "b": 2.5})
    assert [r["ref_loss"] for r in out] == [1.5, 2.5]
    with pytest.raises(RuntimeError):
        rpb.join_ref_loss([dict(r) for r in rows], {"a": 1.5}, allow_missing_fraction=0.0)
    kept = rpb.join_ref_loss([dict(r) for r in rows], {"a": 1.5}, allow_missing_fraction=0.6)
    assert [r["content_hash"] for r in kept] == ["a"]          # misses are dropped, never zero-filled


def test_mixer_selects_domains_and_joins_cache(corpus, tmp_path):
    blocks = tmp_path / "blocks"
    rows = _build(corpus, blocks, val_holdout_fraction=0.1)
    _write_cache(tmp_path / "cache", rows)

    def mix(replay_share, full_val_holdout):
        rpb.mix_and_write_pretrain(
            blocks_path=str(blocks), train_output=str(tmp_path / "train.jsonl"),
            val_output=str(tmp_path / "val.jsonl"), adapt_domain="legal",
            replay_share=replay_share, ref_loss_cache_path=str(tmp_path / "cache"),
            full_val_holdout=full_val_holdout)
        read = lambda f: [json.loads(l) for l in open(tmp_path / f)]
        return read("train.jsonl"), read("val.jsonl")

    # Training: all adaptation blocks + all replay blocks, ref_loss joined on every row.
    train, val = mix(1.0, True)
    assert {r["origin_tag"] for r in train} == {0, 1}
    assert all(r["ref_loss"] == 1.0 + r["origin_tag"] for r in train + val)
    assert {r["origin_tag"] for r in val} == {0, 1}
    # Precompute slice (replay_share=0): only the named domain's blocks.
    train, val = mix(0.0, False)
    assert {r["origin_tag"] for r in train} == {0} and {r["origin_tag"] for r in val} == {0}
    # Fixed-replay subsample keeps all adaptation blocks, a fraction of replay blocks,
    # and still validates on the FULL holdout of both domains.
    n_replay_all = sum(1 for r in rows if r["origin_tag"] == 1 and r["split"] == "train")
    train, val = mix(0.25, True)
    assert sum(1 for r in train if r["origin_tag"] == 1) < n_replay_all
    assert sum(1 for r in train if r["origin_tag"] == 0) == sum(
        1 for r in rows if r["origin_tag"] == 0 and r["split"] == "train")
    assert {r["origin_tag"] for r in val} == {0, 1}
    # Per-origin validation cap.
    rpb.mix_and_write_pretrain(
        blocks_path=str(blocks), train_output=str(tmp_path / "train.jsonl"),
        val_output=str(tmp_path / "val.jsonl"), adapt_domain="legal", replay_share=1.0,
        ref_loss_cache_path=str(tmp_path / "cache"), full_val_holdout=True, max_val_samples_per_origin=1)
    val = [json.loads(l) for l in open(tmp_path / "val.jsonl")]
    assert max(collections.Counter(r["origin"] for r in val).values()) == 1


# --- curriculum transfer --------------------------------------------------------------------

def _trace(path, steps):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "selected_samples.jsonl", "w") as fh:
        for step, hashes in steps:
            for h in hashes:
                fh.write(json.dumps({"step": step, "content_hash": h, "origin_tag": 0, "rho": 0.0}) + "\n")


def test_curriculum_replays_trace_in_order_with_repeats(tmp_path):
    rows = [{"content_hash": h, "token_ids": [i]} for i, h in enumerate("abcd")]
    # phase 1: steps 1, 2, then a resume that redoes step 2; phase 2: its own step numbering
    _trace(tmp_path / "conv", [(1, "ab"), (2, "cd"), (2, "ca")])
    _trace(tmp_path / "anneal", [(1, "aa")])
    out = rpb.build_curriculum(rows, [str(tmp_path / "conv"), str(tmp_path / "anneal")], batch_size=2)
    assert [r["content_hash"] for r in out] == list("abcaaa")
    with pytest.raises(ValueError):
        rpb.build_curriculum(rows, [str(tmp_path / "conv")], batch_size=3)     # batch != recorded k
    _trace(tmp_path / "bad", [(1, "az")])
    with pytest.raises(RuntimeError):
        rpb.build_curriculum(rows, [str(tmp_path / "bad")])                   # unknown block
