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

"""Pre-training block ingestion for RoD.

Turns raw-text pre-training corpora into fixed-length token blocks for continual
*pre-training* with RHO replay selection, and joins each block with its cached
reference loss.

Why blocks (not raw docs):
  - Each fixed ``block_size`` block is ONE RHO "sample" → preserves a single
    ``origin_tag`` per block and gives a clean per-block full-sequence LM loss,
    while packing many short docs efficiently (no per-doc padding waste).
  - Packing is **within-origin** so adapt(legal)/replay(general) never mix in a
    block → the joint RHO selector still sees a clean domain split.

Why store pre-tokenized ``token_ids`` (not text):
  - Packing must hit *exactly* ``block_size`` tokens with EOD separators, and a
    tokenize→detokenize→retokenize round-trip is NOT guaranteed identical. Storing
    token_ids (and hashing them) guarantees the precompute pass and the training
    pass score byte-identical content → the ``content_hash`` ref-loss join is exact.

Determinism: documents are packed in a fixed, seeded order so ``content_hash`` is
reproducible across the precompute run and the training run (required for the join).

The heavy lifting (``pack_token_streams``, ``block_content_hash``, holdout split)
is pure-Python and unit-testable without a tokenizer or torch. Tokenization + IO
(``build_blocks``) lazily import transformers/datasets so this module stays
importable in light environments.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
from array import array
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

# origin_tag convention: 0 = adaptation, 1 = replay.
ADAPT_TAG = 0
REPLAY_TAG = 1


# ---------------------------------------------------------------------------
# Pure logic (no tokenizer / torch) — unit-testable
# ---------------------------------------------------------------------------


def pack_token_streams(
    docs: Iterable[list[int]],
    block_size: int,
    eod_id: int,
    *,
    drop_last_partial: bool = True,
) -> Iterator[list[int]]:
    """Pack a stream of tokenized documents into fixed ``block_size`` blocks.

    An ``eod_id`` token is appended after every document (standard pre-training
    document separator; the model learns to predict it, so it stays loss-bearing).
    The concatenated stream is then chunked into ``block_size``-token blocks.

    The trailing partial block is dropped when ``drop_last_partial`` (default) so
    every emitted block is exactly ``block_size`` tokens — this mirrors Megatron's
    pre-training packer and keeps all blocks uniform. The dropped remainder is at
    most ``block_size - 1`` tokens per origin (negligible). Set
    ``drop_last_partial=False`` to keep a shorter final block instead.

    Yields lists of ints (length == block_size, except possibly the last when
    ``drop_last_partial=False``).
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    buf: list[int] = []
    for doc in docs:
        buf.extend(doc)
        buf.append(eod_id)
        # Emit as many full blocks as the buffer currently allows. Keep the
        # tail in ``buf`` so it continues filling from the next document.
        while len(buf) >= block_size:
            yield buf[:block_size]
            del buf[:block_size]

    if buf and not drop_last_partial:
        yield buf


def block_content_hash(token_ids: list[int], origin: str) -> str:
    """SHA256 over canonical ``{origin, token_ids}`` — the ref-loss cache key.

    Sorted-keys, compact-separators JSON over the block's token_ids + origin, so
    precompute and training join exactly.
    """
    payload = json.dumps(
        {"origin": origin, "token_ids": token_ids},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_holdout_per_origin(
    blocks: list[dict],
    holdout_fraction: float,
    *,
    seed: int = 42,
    floor: int = 0,
    group_keys: tuple[str, ...] = ("domain", "origin"),
) -> tuple[list[dict], list[dict]]:
    """Per-origin deterministic (train, holdout) split with a minimum holdout floor.

    Applied within each ``group_keys`` group (default per (domain, origin)), so every
    origin gets a stable held-out slice even when small. For each group of n blocks the
    holdout size is ``max(ceil(fraction*n), min(floor, n))``, capped at 20% of n (and
    at n-1), taken as the blocks with the lowest seeded hash of their ``content_hash``,
    so the split does not depend on block order. ``content_hash``, hence the ref-loss
    join, is untouched; this only assigns ``split``.
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")

    from collections import defaultdict

    def hash01(h: str) -> float:
        digest = hashlib.sha256(f"{seed}:{h}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for b in blocks:
        groups[tuple(b.get(k) for k in group_keys)].append(b)

    train: list[dict] = []
    holdout: list[dict] = []
    for _key, gb in groups.items():
        gb_sorted = sorted(gb, key=lambda b: hash01(b["content_hash"]))
        n = len(gb_sorted)
        k = max(int(math.ceil(holdout_fraction * n)), min(floor, n))
        # Never hold out more than ~20% of an origin: otherwise the floor would
        # cannibalize SMALL origins (e.g. leave ~1 train block), erasing their
        # contribution to the candidate pool. Large origins still get up to `floor`.
        k = min(k, max(1, int(0.2 * n)))
        if k >= n:  # keep >=1 train block if possible
            k = max(0, n - 1)
        holdout.extend(gb_sorted[:k])
        train.extend(gb_sorted[k:])
    return train, holdout


# ---------------------------------------------------------------------------
# IO + tokenization (lazy imports) — runs on the cluster with the real tokenizer
# ---------------------------------------------------------------------------


def _salted_hash01(salt: str, content_hash: str) -> float:
    """Stable unit-interval hash of a block, independent of the holdout bucketing."""
    digest = hashlib.sha256(f"{salt}:{content_hash}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def balance_replay(rows: list[dict], *, seed: int = 42) -> list[dict]:
    """Keep every adaptation block and a uniform, seeded subset of the replay blocks of
    about the same size (train and holdout alike), so the corpus itself is the ~50:50
    candidate pool and training uses ``data_mixer.replay_share: 1.0``. The subset is a
    Bernoulli draw per block, so it preserves per-origin proportions and within-origin
    diversity."""
    n_adapt = sum(1 for r in rows if r["origin_tag"] == ADAPT_TAG)
    n_replay = len(rows) - n_adapt
    if n_replay <= n_adapt:
        return rows
    p = n_adapt / n_replay
    kept = [r for r in rows
            if r["origin_tag"] == ADAPT_TAG or _salted_hash01(f"pool:{seed}", r["content_hash"]) < p]
    logger.info("balance_replay: replay %d -> %d blocks (p=%.4f), adapt %d",
                n_replay, len(kept) - n_adapt, p, n_adapt)
    return kept


def cap_per_origin(rows: list[dict], cap: int | None, *, seed: int = 42) -> list[dict]:
    """Deterministically keep at most ``cap`` rows per ``origin`` (the lowest salted hashes,
    so the choice does not depend on row order)."""
    if not cap:
        return rows
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["origin"], []).append(r)
    out: list[dict] = []
    for origin in sorted(by):
        group = by[origin]
        if len(group) > cap:
            group = sorted(group, key=lambda r: _salted_hash01(f"val:{seed}", r["content_hash"]))[:cap]
        out.extend(group)
    return out


def _glob_parquet(parquet_glob: str) -> list[str]:
    """Sorted list of parquet files matching ``parquet_glob`` (recursive)."""
    import glob

    files = sorted(glob.glob(parquet_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files matched {parquet_glob!r}")
    return files


def _glob_root(parquet_glob: str) -> str:
    """Literal (wildcard-free) prefix directory of a glob.

    ``/data/general/**/*.parquet`` -> ``/data/general``; used to derive each
    file's *subset* as its first path component under this root.
    """
    parts = parquet_glob.split(os.sep)
    root_parts: list[str] = []
    for p in parts:
        if any(c in p for c in "*?["):
            break
        root_parts.append(p)
    return os.sep.join(root_parts) or os.sep


def _clean_origin(name: str) -> str:
    """Normalise a subset directory name into a compact, metric-safe origin id.

    Strips the verbose ``Nemotron[-Pretraining|-CC]-`` prefixes and lowercases /
    underscores the rest, so ``Nemotron-CC-Math-v1`` -> ``math_v1``,
    ``CaseHOLD`` -> ``casehold``. These become the per-origin metric suffixes
    (``rho_by_origin/<corpus>/<origin>/...``, ``validation/loss_by_origin_<origin>/...``),
    so they must be filename-safe.
    """
    import re

    s = name.strip()
    for pre in ("Nemotron-Pretraining-Legal-", "Nemotron-Pretraining-",
                "Nemotron-CC-", "Nemotron-"):
        if s.lower().startswith(pre.lower()):
            s = s[len(pre):]
            break
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "unknown"


def discover_subset_globs(
    parquet_glob: str,
    coarse_origin: str,
    *,
    per_subset: bool = True,
    origin_map: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Group a domain's parquet files by *subset* → ``{origin: [files]}``.

    The subset is a file's first path component under the glob's literal root
    (standard HF ``<root>/<config>/*.parquet`` layout). Each subset maps to a
    fine-grained ``origin`` (via ``origin_map`` override, else ``_clean_origin``),
    which is what enables per-capability rho/selection/loss tracking downstream.

    Falls back to a single ``coarse_origin`` group when ``per_subset`` is False or
    the layout is flat (files sit directly under the root, no subset subdirs).
    """
    files = _glob_parquet(parquet_glob)
    if not per_subset:
        return {coarse_origin: files}

    root = _glob_root(parquet_glob)
    groups: dict[str, list[str]] = {}
    for f in files:
        parts = os.path.relpath(f, root).split(os.sep)
        subset = parts[0] if len(parts) > 1 else None
        if subset is None:  # flat layout — no subdir to name a subset
            origin = coarse_origin
        else:
            origin = (origin_map or {}).get(subset) or _clean_origin(subset)
        groups.setdefault(origin, []).append(f)

    if len(groups) == 1 and coarse_origin in groups:
        logger.warning(
            "no subset subdirectories under %r (flat layout?); using coarse origin %r. "
            "Organise the corpus as <root>/<subset>/*.parquet for per-subset granularity.",
            root, coarse_origin)
    else:
        logger.info("domain %r -> %d subset origins: %s", coarse_origin, len(groups),
                    {k: len(v) for k, v in sorted(groups.items())})
    return groups


def _iter_text_column(files: list[str], text_column: str = "text",
                      sample_fraction: float = 1.0, seed: int = 42) -> Iterator[str]:
    """Yield the text field from a list of parquet files (memory-light).

    When ``sample_fraction`` < 1.0, keep each row with that probability (seeded
    Bernoulli). This is a uniform, PROPORTIONAL subsample spread across ALL
    shards and rows — so within-origin diversity (languages, sub-topics, time)
    is preserved, unlike keeping a few whole shards. Used to right-size an
    over-staged replay pool BEFORE tokenization (bounds tokenize time + peak
    memory) while still representing the pretraining distribution.
    """
    import random

    import pyarrow.parquet as pq

    rng = random.Random(seed) if sample_fraction < 1.0 else None
    for f in files:
        pf = pq.ParquetFile(f)
        # Skip shards without the text column (e.g. Nemotron-Code-Metadata, which
        # ships repo/commit_id/rel_path pointers, not usable text). Warn so a
        # glob-all ingestion is robust without hand-curating per-subset globs.
        if text_column not in pf.schema.names:
            logger.warning("skipping %s: no %r column (cols=%s)", f, text_column, pf.schema.names)
            continue
        for batch in pf.iter_batches(columns=[text_column], batch_size=1024):
            for v in batch.column(text_column).to_pylist():
                if v and (rng is None or rng.random() < sample_fraction):
                    yield v


def build_blocks(
    *,
    parquet_glob: str | None = None,
    parquet_files: list[str] | None = None,
    origin: str,
    origin_tag: int,
    tokenizer,
    block_size: int,
    eod_id: int,
    text_column: str = "text",
    max_docs: int | None = None,
    sample_fraction: float = 1.0,
    sample_seed: int = 42,
    add_bos: bool = False,
    tokenize_batch_size: int = 1000,
) -> list[dict]:
    """Tokenize a corpus and pack it into block rows for one origin.

    Pass either ``parquet_glob`` (globbed here) or a pre-resolved ``parquet_files``
    list (used by ``build_blocks_corpus`` to pack one subset at a time under a
    fine-grained ``origin``).

    Returns a list of row dicts: ``{token_ids, origin, origin_tag, content_hash, n_bytes}``.
    Documents are tokenized in batches with the HF fast tokenizer (``add_special_tokens
    =False``; EOD is added when packing, BOS optionally via ``add_bos``).
    """
    if parquet_files is None:
        if parquet_glob is None:
            raise ValueError("build_blocks: pass either parquet_glob or parquet_files")
        parquet_files = _glob_parquet(parquet_glob)

    bos = ([tokenizer.bos_token_id]
           if (add_bos and tokenizer is not None and tokenizer.bos_token_id is not None) else [])

    def doc_token_lists() -> Iterator[list[int]]:
        texts = _iter_text_column(parquet_files, text_column,
                                  sample_fraction=sample_fraction, seed=sample_seed)
        if max_docs is not None:
            texts = itertools.islice(texts, max_docs)
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= tokenize_batch_size:
                for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                    yield (bos + ids) if bos else ids
                batch = []
        if batch:
            for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                yield (bos + ids) if bos else ids

    rows: list[dict] = []
    has_decode = tokenizer is not None and hasattr(tokenizer, "decode")
    for block in pack_token_streams(doc_token_lists(), block_size, eod_id):
        # Store token_ids as a C int32 array, not a Python list: a list of 4096
        # boxed ints is ~150 KB/block, so a full uncapped corpus (~1.2M blocks)
        # is >100 GB of RAM and OOM-kills the staging job. array("i") is ~16 KB/
        # block (~10x smaller). The content_hash is computed from the plain list
        # BEFORE compaction so the join key is byte-identical to the read-back
        # value (parquet round-trips the column as list<int64> -> Python list).
        #
        # ``n_bytes`` = UTF-8 byte length of the block's decoded text, kept as block
        # metadata for byte-normalized (bits-per-byte) analyses; training does not use
        # it, and it is not part of content_hash (the join key is token_ids + origin).
        n_bytes = None
        if has_decode:
            try:
                n_bytes = len(tokenizer.decode(block).encode("utf-8"))
            except Exception:
                n_bytes = None
        rows.append(
            {
                "token_ids": array("i", block),
                "origin": origin,
                "origin_tag": origin_tag,
                "content_hash": block_content_hash(block, origin),
                "n_bytes": n_bytes,
            }
        )
    logger.info("origin=%s: packed %d blocks of %d tokens", origin, len(rows), block_size)
    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d rows -> %s", len(rows), path)


# ---------------------------------------------------------------------------
# Ingestion (one-time): raw-text parquets -> a packed-blocks "corpus" parquet
# ---------------------------------------------------------------------------
#
# Tokenization happens ONCE here (it's the expensive step). The resulting blocks
# parquet is the "corpus" — both the precompute passes and the training run
# read it via mix_and_write_pretrain (no re-tokenization).


def build_blocks_corpus(
    *,
    adapt_glob: str,
    replay_glob: str,
    out_path: str,
    tokenizer_path: str,
    block_size: int = 4096,
    eod_id: int | None = None,
    val_holdout_fraction: float = 0.01,
    max_docs_adapt: int | None = None,
    max_docs_replay: int | None = None,
    max_blocks_per_domain: int | None = None,
    max_blocks_adapt: int | None = None,
    max_blocks_replay: int | None = None,
    max_blocks_per_origin: int | None = None,
    per_subset_origin: bool = True,
    origin_map: dict[str, str] | None = None,
    capability_map: dict[str, str] | None = None,
    holdout_floor_per_origin: int = 0,
    shard_rows: int = 50000,
    text_column: str = "text",
    add_bos: bool = False,
    replay_doc_fraction: float = 1.0,
    adapt_doc_fraction: float = 1.0,
    adapt_domain: str = "legal",
    replay_domain: str = "general",
    balance_replay_to_adapt: bool = False,
    seed: int = 42,
) -> str:
    """Tokenize + pack the adaptation and replay corpora into a
    SHARDED blocks parquet directory (``out_path`` is a dir), tagged with
    ``domain`` (``adapt_domain`` / ``replay_domain``, e.g. legal/general; drives
    adapt/replay filtering), a fine-grained
    ``origin`` (the source *subset* — enables per-capability rho/selection/loss
    tracking), ``origin_tag`` (0/1), a deterministic ``content_hash``, and a
    ``split`` (train/holdout).

    Per-subset origin (``per_subset_origin=True``, default): each block's
    ``origin`` is its source subset (``<root>/<subset>/*.parquet``), normalised via
    ``origin_map`` (subset→origin overrides, for grouping) else ``_clean_origin``.
    Packing stays within-subset so every block has ONE origin. Set
    ``per_subset_origin=False`` for coarse origins (origin==domain).

    Caps (all deterministic, seeded, applied post-packing):
      - ``max_blocks_per_origin``: subsample EACH subset (guarantees no single
        subset dominates → stable per-origin stats).
      - ``max_blocks_adapt`` / ``max_blocks_replay``: PER-DOMAIN totals AFTER the
        per-origin caps — set an ASYMMETRIC adapt:replay ratio (e.g. 100k adapt /
        200k replay). Each falls back to ``max_blocks_per_domain`` when unset.
      - ``max_blocks_per_domain``: default cap applied to both domains unless the
        per-domain values above override it.

    ``balance_replay_to_adapt`` subsamples the replay blocks to the adaptation size after the
    holdout split (see :func:`balance_replay`).
    """
    import random
    from collections import Counter

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eod = eod_id if eod_id is not None else tok.eos_token_id
    if eod is None:
        raise ValueError("tokenizer has no eos_token_id; pass --eod-id explicitly")
    logger.info("tokenizer=%s block_size=%d eod_id=%d per_subset_origin=%s",
                tokenizer_path, block_size, eod, per_subset_origin)

    def _cap(rows: list[dict], name: str, cap: int | None) -> list[dict]:
        if not cap or len(rows) <= cap:
            return rows
        # Per-(seed,name) rng so each cap is reproducible and independent of the
        # sizes/order of the others (a stable int seed derived via sha256).
        rng = random.Random(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16))
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        keep = sorted(idx[:cap])
        logger.info("capping %s: %d -> %d blocks", name, len(rows), cap)
        return [rows[i] for i in keep]

    def _build_domain(glob: str, domain: str, origin_tag: int, max_docs: int | None,
                      max_blocks_domain: int | None, doc_fraction: float = 1.0) -> list[dict]:
        groups = discover_subset_globs(glob, domain, per_subset=per_subset_origin, origin_map=origin_map)
        rows: list[dict] = []
        for origin, files in sorted(groups.items()):
            r = build_blocks(parquet_files=files, origin=origin, origin_tag=origin_tag,
                             tokenizer=tok, block_size=block_size, eod_id=eod,
                             text_column=text_column, max_docs=max_docs,
                             sample_fraction=doc_fraction, sample_seed=seed, add_bos=add_bos)
            r = _cap(r, f"origin:{origin}", max_blocks_per_origin)
            cap_tag = (capability_map or {}).get(origin, origin)
            for row in r:
                row["domain"] = domain
                row["capability"] = cap_tag
            rows.extend(r)
        return _cap(rows, f"domain:{domain}", max_blocks_domain)

    # Per-domain caps override the shared default (enables asymmetric ratios, e.g.
    # 100k adapt / 200k replay → replay ~67% of the pool).
    adapt_cap = max_blocks_adapt if max_blocks_adapt is not None else max_blocks_per_domain
    replay_cap = max_blocks_replay if max_blocks_replay is not None else max_blocks_per_domain
    adapt_rows = _build_domain(adapt_glob, adapt_domain, ADAPT_TAG, max_docs_adapt, adapt_cap,
                          doc_fraction=adapt_doc_fraction)
    # Each side is document-subsampled (seeded, across all shards, so within-origin diversity is
    # kept) BEFORE tokenization via replay_doc_fraction / adapt_doc_fraction.
    replay_rows = _build_domain(replay_glob, replay_domain, REPLAY_TAG, max_docs_replay, replay_cap,
                            doc_fraction=replay_doc_fraction)

    # split_holdout_per_origin groups by (domain, origin), so splitting the combined
    # list is the same as splitting each group separately.
    all_rows = adapt_rows + replay_rows
    train, holdout = split_holdout_per_origin(
        all_rows, val_holdout_fraction, seed=seed, floor=holdout_floor_per_origin
    )
    for r in train:
        r["split"] = "train"
    for r in holdout:
        r["split"] = "holdout"
    all_rows = train + holdout
    if balance_replay_to_adapt:
        all_rows = balance_replay(all_rows, seed=seed)

    _write_blocks_parquet(all_rows, out_path, shard_rows=shard_rows)

    # Per-origin train/holdout breakdown — this is what powers the per-capability
    # forgetting→sample→train analysis, so surface it at ingestion time.
    tr_by = Counter((r["domain"], r["origin"]) for r in all_rows if r["split"] == "train")
    ho_by = Counter((r["domain"], r["origin"]) for r in all_rows if r["split"] == "holdout")
    origins = sorted(set(tr_by) | set(ho_by))
    cap_by = {(r["domain"], r["origin"]): r.get("capability", r["origin"]) for r in all_rows}
    for dom, org in origins:
        logger.info("  origin %-26s [%-7s cap=%-18s]: train=%d holdout=%d",
                    org, dom, cap_by.get((dom, org), org), tr_by[(dom, org)], ho_by[(dom, org)])
    n_adapt = sum(1 for r in all_rows if r["origin_tag"] == ADAPT_TAG)
    logger.info("blocks corpus: %d total (adapt=%d, replay=%d) across %d origins -> %s",
                len(all_rows), n_adapt, len(all_rows) - n_adapt, len(origins), out_path)
    return out_path


def _write_blocks_parquet(rows: list[dict], out_dir: str, shard_rows: int = 50000) -> None:
    """Write blocks as SHARDED parquet files under ``out_dir`` (``part_00000.parquet`` …).

    Sharding is REQUIRED: a single parquet with a ``list<int64>`` ``token_ids``
    column overflows pyarrow's int32 list offsets once total elements exceed
    ~2.1B (≈ 512k blocks × 4096) — ``read_table`` then raises "List index
    overflow". Each shard stays well under that and reads cleanly;
    ``_read_block_rows`` globs the directory.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    n_shards = 0
    for i in range(0, len(rows), shard_rows):
        chunk = rows[i:i + shard_rows]
        table = pa.Table.from_pylist([
            {
                # array("i") -> plain list for pyarrow inference (transient, one
                # shard at a time, freed after write — bounded peak).
                "token_ids": list(r["token_ids"]),
                "origin": r["origin"],
                "origin_tag": int(r["origin_tag"]),
                "domain": r["domain"],
                "capability": r.get("capability", r["origin"]),
                "n_bytes": r.get("n_bytes"),
                "content_hash": r["content_hash"],
                "split": r["split"],
            }
            for r in chunk
        ])
        pq.write_table(table, os.path.join(out_dir, f"part_{i // shard_rows:05d}.parquet"))
        n_shards += 1
    logger.info("wrote %d blocks across %d shard(s) (shard_rows=%d) -> %s",
                len(rows), n_shards, shard_rows, out_dir)


def _read_block_rows(blocks_path: str, max_rows: int | None = None) -> list[dict]:
    import glob

    import pyarrow.parquet as pq

    if os.path.isdir(blocks_path):
        files = sorted(glob.glob(os.path.join(blocks_path, "*.parquet")))
    else:
        files = [blocks_path]
    if not files:
        raise FileNotFoundError(f"no blocks parquet found at {blocks_path!r}")
    rows: list[dict] = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
        # max_rows: early-stop after enough blocks (quick debug runs via ROD_MAX_BLOCKS).
        # Default None -> read everything (unchanged for real runs).
        if max_rows is not None and len(rows) >= max_rows:
            rows = rows[:max_rows]
            logger.info("_read_block_rows: capped to %d blocks (max_rows)", max_rows)
            break
    return rows


# ---------------------------------------------------------------------------
# Reference-loss cache: load the precomputed (content_hash -> ref_loss) parquets and join
# ---------------------------------------------------------------------------


def load_ref_loss_cache(cache_path: str) -> dict[str, float]:
    """Load the precomputed reference-loss cache into an in-memory dict.

    Reads every ``*.parquet`` file in ``cache_path`` (written by ``run_precompute.py``,
    one or more shards per reference pass) and merges them keyed by ``content_hash``.
    The adaptation pass (specialist over adapt blocks) and the replay pass (base model
    over replay blocks) write into the same directory; their hashes never collide
    because ``origin`` is part of the hash.
    """
    import pyarrow.parquet as pq

    if not os.path.isdir(cache_path):
        raise FileNotFoundError(
            f"ref_loss_cache_path={cache_path} not found or not a directory. "
            "Run run_precompute.py to produce it before training."
        )
    parquet_files = sorted(
        os.path.join(cache_path, f) for f in os.listdir(cache_path) if f.endswith(".parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found under {cache_path}. Run run_precompute.py to produce them."
        )

    cache: dict[str, float] = {}
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["content_hash", "ref_loss"])
        hashes = table.column("content_hash").to_pylist()
        losses = table.column("ref_loss").to_pylist()
        for h, ell in zip(hashes, losses):
            if h in cache and cache[h] != ell:
                logger.warning(
                    "Duplicate content_hash %s with differing ref_loss values "
                    "(%.6f vs %.6f) across cache files; keeping the first.",
                    h, cache[h], ell,
                )
            else:
                cache[h] = ell
        logger.info("Loaded %d ref_loss rows from %s", len(hashes), pf)
    logger.info("ref_loss cache: %d unique content_hashes total", len(cache))
    return cache


def join_ref_loss(
    rows: list[dict],
    cache: dict[str, float],
    allow_missing_fraction: float = 0.0,
) -> list[dict]:
    """Annotate each block row with its cached ``ref_loss``; hard-fail on misses.

    Every block row carries the ``content_hash`` assigned at ingestion. Rows whose hash
    is missing from the cache are *dropped* (never zero-filled); if the missing fraction
    exceeds ``allow_missing_fraction`` (default 0.0) this raises instead, since a miss
    means the precompute and training corpora diverged.
    """
    out: list[dict] = []
    missing_samples: list[str] = []
    missing_count = 0
    for row in rows:
        h = row["content_hash"]
        if h not in cache:
            missing_count += 1
            if len(missing_samples) < 10:
                missing_samples.append(h)
            continue
        row["ref_loss"] = float(cache[h])
        out.append(row)

    if missing_count > 0:
        total = len(rows)
        missing_frac = missing_count / max(total, 1)
        msg = (
            f"ref_loss cache miss: {missing_count} / {total} rows "
            f"({100 * missing_frac:.2f}%). First missing hashes: {missing_samples}"
        )
        if missing_frac > allow_missing_fraction:
            raise RuntimeError(
                msg + f"\nExceeded allow_missing_fraction={allow_missing_fraction}. "
                "Re-run run_precompute.py against the current blocks corpus."
            )
        logger.warning(msg + " (within allow_missing_fraction; dropping missing rows)")
    return out


# ---------------------------------------------------------------------------
# Curriculum transfer: replay a recorded RoD selection stream as a fixed training order
# ---------------------------------------------------------------------------


def read_selection_trace(trace_dir: str, batch_size: int | None = None) -> list[list[str]]:
    """Per-step selected ``content_hash`` lists from one run's selection trace, in step order.

    Reads every ``*.jsonl`` under ``trace_dir`` (rows written by the training loop's
    selected-sample trace: ``{step, content_hash, ...}``). A step can appear more than
    once if the run was resumed from an earlier checkpoint and redid it; the redo is
    always written last. With ``batch_size`` known, each step keeps its last
    ``batch_size`` selections (robust to partial writes); without it, a step seen again
    after other steps restarts that step's list.
    """
    import glob

    files = sorted(glob.glob(os.path.join(trace_dir, "**", "*.jsonl"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no selection trace (*.jsonl) under {trace_dir!r}")
    by_step: dict[int, list[str]] = {}
    last_step = None
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                step = int(r["step"])
                if batch_size is None and step != last_step:
                    by_step[step] = []          # new block for this step (replaces any earlier one)
                by_step.setdefault(step, []).append(r["content_hash"])
                last_step = step
    if batch_size is not None:
        by_step = {s: hs[-batch_size:] for s, hs in by_step.items()}
    return [by_step[s] for s in sorted(by_step)]


def build_curriculum(rows: list[dict], trace_dirs: list[str], batch_size: int | None = None) -> list[dict]:
    """Training rows in the exact order a (smaller) RoD run selected them.

    ``trace_dirs`` are the source run's selection traces in phase order (e.g. the
    constant-LR phase, then the anneal). Every recorded step contributes its selected
    blocks in order, with repeats, resolved by ``content_hash`` against ``rows`` (the
    target model's blocks corpus; source and target must share the tokenizer so the
    hashes match). Trained with ``data.shuffle: false`` and a global batch equal to the
    source run's ``top_k``, each training step then consumes exactly one recorded batch.
    """
    by_hash = {r["content_hash"]: r for r in rows}
    out: list[dict] = []
    missing = 0
    for d in trace_dirs:
        for step_hashes in read_selection_trace(d, batch_size):
            if batch_size is not None and len(step_hashes) != batch_size:
                raise ValueError(
                    f"curriculum step in {d!r} has {len(step_hashes)} samples but the global batch "
                    f"size is {batch_size}; set policy.train_global_batch_size to the source run's top_k."
                )
            for h in step_hashes:
                r = by_hash.get(h)
                if r is None:
                    missing += 1
                    continue
                out.append(r)
    if missing:
        raise RuntimeError(
            f"{missing} curriculum samples not found in the blocks corpus — the source and target "
            "runs must use the same tokenizer and blocks corpus."
        )
    logger.info("curriculum: %d training rows from %d trace dir(s)", len(out), len(trace_dirs))
    return out


# ---------------------------------------------------------------------------
# Mixer (per-run, fast): blocks parquet -> train/val JSONL with ref_loss joined
# ---------------------------------------------------------------------------


def mix_and_write_pretrain(
    *,
    blocks_path: str,
    train_output: str,
    val_output: str,
    adapt_domain: str,
    replay_share: float = 1.0,
    ref_loss_cache_path: str | None = None,
    allow_missing_fraction: float = 0.0,
    full_val_holdout: bool = False,
    seed: int = 42,
    max_blocks: int | None = None,
    curriculum_trace_dirs: list[str] | None = None,
    curriculum_batch_size: int | None = None,
    max_val_samples_per_origin: int | None = None,
) -> None:
    """Select block rows by domain, split train/val, join ref_loss by
    ``content_hash``, write JSONL.

    With ``curriculum_trace_dirs`` the training rows are instead the recorded selection
    stream of another RoD run, in order (see :func:`build_curriculum`); validation is
    unchanged. ``max_val_samples_per_origin`` caps the validation rows of every origin
    (:func:`cap_per_origin`), so each source contributes a bounded, fixed slice.

    Domain selection (adapt/replay semantics):
      - ``replay_share <= 0`` (precompute single-slice): keep ONLY the
        ``adapt_domain`` blocks (the slice this precompute pass scores).
      - ``replay_share > 0`` (training): keep ALL ``adapt_domain`` (adapt)
        blocks + a ``replay_share`` fraction of the other domain (replay).
    ``origin_tag`` is preserved from the blocks (0 = adapt, 1 = replay),
    independent of which domain is the adapt slice.
    """
    import random

    rows = _read_block_rows(blocks_path, max_rows=max_blocks)
    rng = random.Random(seed)

    if replay_share <= 0.0:
        kept = [r for r in rows if r["domain"] == adapt_domain]
    else:
        adapt = [r for r in rows if r["domain"] == adapt_domain]
        other = [r for r in rows if r["domain"] != adapt_domain]
        if replay_share < 1.0 and other:
            k = int(round(len(other) * replay_share))
            other = rng.sample(other, k) if 0 < k < len(other) else (other if k >= len(other) else [])
        kept = adapt + other

    train_rows = [r for r in kept if r.get("split", "train") == "train"]
    if curriculum_trace_dirs:
        train_rows = build_curriculum(rows, list(curriculum_trace_dirs), curriculum_batch_size)
    # full_val_holdout: val on the FULL holdout (all origins, BOTH domains) regardless of the
    # training replay_share subsample, so forgetting (replay val loss) is measured on the SAME
    # held-out set across every arm. RHO uses replay_share=1.0 (already full); the fixed-mix
    # vanilla baselines (replay_share<1) would otherwise val on a subsampled replay holdout and
    # not be comparable. Precompute (single-slice) leaves this False → adapt-slice holdout only.
    val_pool = rows if full_val_holdout else kept
    val_rows = cap_per_origin([r for r in val_pool if r.get("split") == "holdout"],
                              max_val_samples_per_origin, seed=seed)
    logger.info("mix_and_write_pretrain: adapt_domain=%s replay_share=%.3f -> train=%d val=%d (of %d blocks)",
                adapt_domain, replay_share, len(train_rows), len(val_rows), len(rows))
    # Under a read cap (max_blocks, debug runs), the sampled slice may contain NO holdout blocks
    # (holdout is a small hash-selected fraction that can miss the first shards) → an empty val.jsonl
    # crashes the HF `datasets` loader with SchemaInferenceError. Val is irrelevant in debug
    # runs, so fall back to a few train rows as a schema-valid placeholder. Cap-only: real runs (no
    # max_blocks) keep the true holdout and never hit this.
    if not val_rows and max_blocks is not None and train_rows:
        val_rows = train_rows[: min(64, len(train_rows))]
        logger.warning("mix_and_write_pretrain: no holdout under max_blocks cap; using %d train rows as placeholder val",
                       len(val_rows))

    if ref_loss_cache_path:
        cache = load_ref_loss_cache(ref_loss_cache_path)
        train_rows = join_ref_loss(train_rows, cache, allow_missing_fraction)
        val_rows = join_ref_loss(val_rows, cache, allow_missing_fraction)

    # Shuffle the val set (seeded rng → deterministic AND identical across arms) so a fixed
    # val_batches window samples ALL origins and BOTH domains. Without this, val_rows is ordered
    # adapt-then-replay and origin-grouped, so the val loader — which scores only the first
    # val_batches × val_global_batch_size rows, unshuffled — would never reach replay or any
    # origin past the first.
    random.Random(seed).shuffle(val_rows)   # own rng, so the order does not depend on replay_share
    _write_block_jsonl(train_rows, train_output)
    _write_block_jsonl(val_rows, val_output)


def _write_block_jsonl(rows: list[dict], path: str) -> None:
    """Write the fields RHOPreprocessor consumes, plus the capability / n_bytes metadata."""
    keep = ("token_ids", "origin", "origin_tag", "content_hash", "ref_loss", "capability", "n_bytes")
    out = [{k: r[k] for k in keep if k in r} for r in rows]
    write_jsonl(out, path)


if __name__ == "__main__":  # one-time ingestion CLI
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the packed-blocks pretraining corpus.")
    ap.add_argument("--adapt-glob", required=True, help="Parquet glob for the adaptation corpus.")
    ap.add_argument("--replay-glob", required=True, help="Parquet glob for the replay corpus.")
    ap.add_argument("--out", required=True, help="Output blocks DIRECTORY (sharded parquet).")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer dir/id (trust_remote_code).")
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--eod-id", type=int, default=None, help="EOD token id (default: tokenizer.eos_token_id).")
    ap.add_argument("--val-holdout-fraction", type=float, default=0.01)
    ap.add_argument("--max-docs-adapt", type=int, default=None)
    ap.add_argument("--max-docs-replay", type=int, default=None)
    ap.add_argument("--max-blocks-per-domain", type=int, default=None,
                    help="Default per-domain cap applied to BOTH domains unless overridden below.")
    ap.add_argument("--max-blocks-adapt", type=int, default=None,
                    help="Per-domain cap for the adaptation side; overrides --max-blocks-per-domain.")
    ap.add_argument("--max-blocks-replay", type=int, default=None,
                    help="Per-domain cap for the replay side; overrides --max-blocks-per-domain. "
                         "Set adapt/replay asymmetrically, e.g. 100000/200000.")
    ap.add_argument("--max-blocks-per-origin", type=int, default=None,
                    help="Deterministically cap blocks per subset origin (stable per-capability stats).")
    ap.add_argument("--no-per-subset-origin", dest="per_subset_origin", action="store_false",
                    help="Disable per-subset origins; tag every block with the coarse domain.")
    ap.add_argument("--origin-map", default=None,
                    help="JSON file mapping {subset_dir_name: origin} to override/group subset origins.")
    ap.add_argument("--capability-map", default=None,
                    help="JSON file mapping {origin: capability}, stored as the blocks' `capability` "
                         "tag (metadata for per-capability analyses).")
    ap.add_argument("--holdout-floor-per-origin", type=int, default=0,
                    help="Minimum held-out blocks PER origin (in addition to --val-holdout-fraction) so small "
                         "capabilities still get a stable forgetting curve.")
    ap.add_argument("--shard-rows", type=int, default=50000, help="Blocks per output parquet shard.")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--add-bos", action="store_true")
    ap.add_argument("--replay-doc-fraction", type=float, default=1.0,
                    help="Keep this fraction of GENERAL docs (seeded Bernoulli, spread across ALL "
                         "shards → proportional + diverse) BEFORE tokenizing. Right-sizes an "
                         "over-staged replay pool without collapsing within-origin diversity. "
                         "The adaptation side is controlled separately by --adapt-doc-fraction.")
    ap.add_argument("--adapt-doc-fraction", type=float, default=1.0,
                    help="Like --replay-doc-fraction but for the ADAPT side (--adapt-glob): seeded "
                         "Bernoulli over ALL adapt shards BEFORE tokenizing. Right-sizes an over-staged "
                         "adapt pool without collapsing within-origin diversity.")
    ap.add_argument("--adapt-domain", default="legal",
                    help="Domain label for the adapt side (--adapt-glob). Default 'legal'; set e.g. "
                         "'german' for other adaptation domains (flows to blocks 'domain', the mixer's "
                         "adapt_domain, and per-domain metrics).")
    ap.add_argument("--replay-domain", default="general",
                    help="Domain label for the replay side (--replay-glob). Default 'general'.")
    ap.add_argument("--balance-replay", action="store_true",
                    help="Subsample the replay blocks to the adaptation size, so the corpus is the "
                         "~50:50 candidate pool (train with data_mixer.replay_share: 1.0).")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    def _load_json_map(path, what):
        if not path:
            return None
        with open(path) as fh:
            m = json.load(fh)
        logger.info("loaded %s with %d entries from %s", what, len(m), path)
        return m

    origin_map = _load_json_map(a.origin_map, "origin_map (subset->origin)")
    capability_map = _load_json_map(a.capability_map, "capability_map (origin->capability)")
    build_blocks_corpus(
        adapt_glob=a.adapt_glob, replay_glob=a.replay_glob, out_path=a.out,
        tokenizer_path=a.tokenizer, block_size=a.block_size, eod_id=a.eod_id,
        val_holdout_fraction=a.val_holdout_fraction, max_docs_adapt=a.max_docs_adapt,
        max_docs_replay=a.max_docs_replay, max_blocks_per_domain=a.max_blocks_per_domain,
        max_blocks_adapt=a.max_blocks_adapt, max_blocks_replay=a.max_blocks_replay,
        max_blocks_per_origin=a.max_blocks_per_origin, per_subset_origin=a.per_subset_origin,
        origin_map=origin_map, capability_map=capability_map,
        holdout_floor_per_origin=a.holdout_floor_per_origin,
        shard_rows=a.shard_rows, text_column=a.text_column,
        add_bos=a.add_bos, replay_doc_fraction=a.replay_doc_fraction,
        adapt_doc_fraction=a.adapt_doc_fraction, adapt_domain=a.adapt_domain,
        replay_domain=a.replay_domain, balance_replay_to_adapt=a.balance_replay, seed=a.seed,
    )
