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

"""Selective, diversity-aware staging of the replay pool.

Downloads a SMALL, representative subsample of the Nemotron-Pretraining family
into one directory PER ORIGIN under ``--out`` (the blocks builder derives
``origin`` = dir name). The generated capability map either groups the origins into
their capabilities (``--origin-map``) or tags each block with its capability
(``--capability-map``). Design goals:

  * BREADTH — one origin per source folder, spanning every available capability
    (a missing capability is a blind spot the selector can't protect).
  * WITHIN-CAPABILITY DIVERSITY — shards are picked EVENLY SPREAD across each
    folder's shard range (not just shard 0), to hedge against ordering bias and
    cover sub-modes; the heterogeneous Specialized subsets are ROUTED to their
    natural capabilities (Math-Textbooks→reasoning_math, Wiki-Rewrite→world_
    knowledge, Scientific-Coding→code, …) rather than collapsed into one bucket.
  * NAMING-AGNOSTIC — parquet filenames are discovered at runtime via
    ``list_repo_files``, so we never hardcode ``part_000000`` vs
    ``train-00000-of-*`` etc.

Run with ``--dry-run`` first to print exactly which shards WOULD be pulled.
Document subsampling and replay balancing are applied later by the blocks builder
(``rod_pretrain_blocks``); this module only controls which raw shards land on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

# origin dir | repo | prefix (folder within repo; "" = whole repo) | capability | n_shards
# Capabilities: world_knowledge, reasoning_math, multilingual, code. Multiple origins
# per capability = within-capability diversity.
STAGING_SPEC = [
    # --- world knowledge (web + synthetic QA + wiki/retrieval) ---
    ("wk_web",         "nvidia/Nemotron-CC-v2",                      "High-Quality/",                              "world_knowledge", 2),
    ("wk_web_synth",   "nvidia/Nemotron-CC-v2",                      "High-Quality-Synthetic/",                    "world_knowledge", 1),
    ("wk_qa",          "nvidia/Nemotron-CC-v2",                      "Diverse-QA/",                                "world_knowledge", 1),
    ("wk_wiki",        "nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Wiki-Rewrite/",         "world_knowledge", 1),
    ("wk_rqa",         "nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-RQA/",                  "world_knowledge", 1),
    # --- reasoning & math (CC-Math high-quality tier + specialized math/reasoning) ---
    ("math_web",       "nvidia/Nemotron-CC-Math-v1",                 "4plus/",                                     "reasoning_math",  2),
    ("math_textbooks", "nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Math-Textbooks/",       "reasoning_math",  1),
    ("reasoning_stem", "nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-STEM-SFT/",             "reasoning_math",  1),
    ("reasoning_infb", "nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-InfiniByte-Reasoning/", "reasoning_math",  1),
    # --- multilingual (translations into 15 languages, kept as one origin) ---
    ("multilingual",   "nvidia/Nemotron-CC-v2",                      "Translated-Diverse-QA/",                     "multilingual",    3),
    # --- code + SFT-style pretraining data ---
    ("code_sci",       "nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Scientific-Coding/",    "code",            1),
    # NB: Code-v1/v2 "Nemotron-Code-Metadata" is repo/commit POINTERS (no `text` column → 0 blocks);
    # only "Synthetic-Code" carries usable code text, so we target that folder explicitly.
    ("code_synth",     "nvidia/Nemotron-Pretraining-Code-v2",        "Synthetic-Code/",                            "code",            2),
    ("code_sft",       "nvidia/Nemotron-Pretraining-SFT-v1",         "Nemotron-SFT-Code/",                         "code",            1),
    ("knowledge_sft",  "nvidia/Nemotron-Pretraining-SFT-v1",         "Nemotron-SFT-General/",                      "world_knowledge", 1),
    ("math_sft",       "nvidia/Nemotron-Pretraining-SFT-v1",         "Nemotron-SFT-MATH/",                         "reasoning_math",  1),
]


def pick_spread(files: list[str], n: int) -> list[str]:
    """Pick ``n`` files EVENLY SPREAD across the sorted list (indices 0 … len-1).

    Spreading (vs taking the first ``n``) hedges against shard ordering bias
    (e.g. shards grouped by language/source/time) → better sub-mode coverage.
    """
    files = sorted(files)
    if n <= 0 or not files:
        return []
    if n >= len(files):
        return files
    step = len(files) / n
    return [files[min(len(files) - 1, int(i * step))] for i in range(n)]


def build_plan(spec, *, shards_scale: float = 1.0):
    """Resolve each origin's shard list via list_repo_files (no downloads)."""
    from huggingface_hub import list_repo_files

    plan = []
    for origin, repo, prefix, cap, n_shards in spec:
        allf = list_repo_files(repo, repo_type="dataset")
        pq = [f for f in allf if f.endswith(".parquet") and f.startswith(prefix)]
        n = max(1, int(round(n_shards * shards_scale)))
        chosen = pick_spread(pq, n)
        plan.append({"origin": origin, "repo": repo, "prefix": prefix, "capability": cap,
                     "n_available": len(pq), "chosen": chosen})
        logger.info("origin=%-16s cap=%-15s %-45s %4d parquet -> pull %d: %s",
                    origin, cap, f"{repo}:{prefix}", len(pq), len(chosen), chosen)
    return plan


def stage(out_dir: str, plan, *, dry_run: bool = False) -> None:
    from huggingface_hub import snapshot_download

    for e in plan:
        if not e["chosen"]:
            logger.warning("origin=%s: no parquet matched prefix %r — skipping", e["origin"], e["prefix"])
            continue
        if dry_run:
            continue
        snapshot_download(e["repo"], repo_type="dataset",
                          local_dir=os.path.join(out_dir, e["origin"]),
                          allow_patterns=e["chosen"])


def write_capability_map(spec, path: str) -> None:
    m = {origin: cap for (origin, repo, prefix, cap, n) in spec}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(m, fh, indent=2)
        fh.write("\n")
    logger.info("wrote capability_map (%d origins) -> %s", len(m), path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Selective, diversity-aware staging of the replay pool.")
    ap.add_argument("--out", required=True, help="Output dir; one subdir per origin.")
    ap.add_argument("--capability-map-out", required=True, help="Where to write the origin->capability JSON.")
    ap.add_argument("--shards-scale", type=float, default=1.0, help="Scale every origin's n_shards (grow the pool).")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; download nothing.")
    a = ap.parse_args()

    plan = build_plan(STAGING_SPEC, shards_scale=a.shards_scale)
    stage(a.out, plan, dry_run=a.dry_run)
    write_capability_map(STAGING_SPEC, a.capability_map_out)

    print("\n=== staging plan (%s) ===" % ("DRY RUN" if a.dry_run else "downloaded"))
    for e in plan:
        print(f"  {e['origin']:16s} cap={e['capability']:16s} {len(e['chosen'])} of {e['n_available']:4d} shard(s)  <- {e['repo']}:{e['prefix']}")
