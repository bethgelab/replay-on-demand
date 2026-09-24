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

"""Dataset preparation for RoD training.

Two steps:

  1. :func:`prepare_data` — runs the blocks mixer
     (:func:`rod.data.rod_pretrain_blocks.mix_and_write_pretrain`) to produce
     train/val JSONL with ``ref_loss`` joined from the precomputed cache.
  2. :func:`build_train_and_val_datasets` — loads the JSONL, applies
     :class:`RHOPreprocessor`, and wraps the result for NeMo-RL's SFT data path.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def prepare_data(cfg: dict) -> None:
    """Run the blocks mixer to produce train/val JSONL files.

    Takes the ALREADY-MERGED, resolved config dict (default.yaml + experiment
    + CLI overrides), not a YAML path.

    Reads the ``data_mixer`` block — a ``corpus_path`` pointing at the packed
    blocks corpus (built by ``rod_pretrain_blocks``), an ``adapt_domain``
    naming the adaptation slice, and a ``replay_share`` knob. Output JSONL files
    are written to ``data.train_data_path`` / ``data.val_data_path``.

    The reference-loss cache is read from ``rod.ref_loss_cache_path`` and joined
    into every row by content hash. The join hard-fails on any missing row unless
    ``data_mixer.allow_missing_fraction`` is set. If the cache path is ``None``
    (plain continual pre-training, e.g. the adaptation specialist), the join step
    is skipped entirely.
    """
    from .rod_pretrain_blocks import mix_and_write_pretrain

    data_mixer_cfg = cfg.get("data_mixer")
    if not data_mixer_cfg or not data_mixer_cfg.get("corpus_path"):
        raise ValueError("data_mixer.corpus_path (the packed blocks corpus) is required.")
    adapt_domain = data_mixer_cfg.get("adapt_domain")
    if not adapt_domain:
        raise ValueError(
            "data_mixer.adapt_domain is required. Set it to the `domain` value of the "
            "adaptation blocks (e.g. 'legal' or 'german')."
        )
    rod_cfg = cfg.get("rod") or {}

    # Curriculum transfer: train on another RoD run's recorded selection stream, in order.
    # The file order IS the curriculum, so shuffling must be off, and the stream replaces
    # per-step selection, so selection must be off too.
    trace_dirs = rod_cfg.get("curriculum_trace_dirs")
    if trace_dirs:
        if cfg.get("data", {}).get("shuffle", True):
            raise ValueError("rod.curriculum_trace_dirs requires data.shuffle: false.")
        if rod_cfg.get("enable_selection", True):
            raise ValueError("rod.curriculum_trace_dirs requires rod.enable_selection: false.")

    logger.info("Running RoD blocks mixer")
    # ROD_MAX_BLOCKS: opt-in cap on blocks READ from the corpus, for quick debug runs
    # (data prep drops from ~1h to seconds). Unset -> read everything (default; real
    # runs unaffected). Pair with replay_share=1.0 so the cap is robust to shard/domain
    # ordering (all read blocks kept, no empty-train risk).
    _max_blocks_env = os.environ.get("ROD_MAX_BLOCKS")
    _max_blocks = int(_max_blocks_env) if _max_blocks_env else None
    if _max_blocks:
        logger.warning("ROD_MAX_BLOCKS=%d set — capping corpus read (debug only)", _max_blocks)
    mix_and_write_pretrain(
        blocks_path=data_mixer_cfg["corpus_path"],
        train_output=cfg["data"]["train_data_path"],
        val_output=cfg["data"]["val_data_path"],
        adapt_domain=adapt_domain,
        replay_share=float(data_mixer_cfg.get("replay_share", 1.0)),
        ref_loss_cache_path=rod_cfg.get("ref_loss_cache_path"),
        allow_missing_fraction=float(data_mixer_cfg.get("allow_missing_fraction", 0.0)),
        # TRAINING always validates on the full held-out set (adapt + replay), so
        # forgetting is comparable across arms regardless of the train-time
        # replay_share (fixed-replay baselines).
        full_val_holdout=True,
        seed=data_mixer_cfg.get("seed", 42),
        max_blocks=_max_blocks,
        curriculum_trace_dirs=trace_dirs,
        curriculum_batch_size=int(cfg["policy"]["train_global_batch_size"]) if trace_dirs else None,
        max_val_samples_per_origin=data_mixer_cfg.get("max_val_samples_per_origin"),
    )


# ---------------------------------------------------------------------------
# Preprocessor — produces the DatumSpec NeMo-RL's SFT dataset expects
# ---------------------------------------------------------------------------


class RHOPreprocessor:
    """Preprocessor callback for ``AllTaskProcessedDataset``.

    Converts one packed block row (``token_ids`` + metadata, written by
    ``rod_pretrain_blocks``) into the per-sample dict NeMo-RL's SFT data path
    expects. The block is emitted as a single ``assistant``-role message, so
    ``add_loss_mask_to_message_log(roles_to_train_on=["assistant"])`` yields an
    all-ones loss mask → full-sequence LM loss (no chat template, no truncation:
    blocks are pre-bounded to ``max_total_sequence_length``).

    RoD-specific fields, threaded into the batch by the ``rod_collate`` patch:

    - ``ref_loss``: float, the cached reference loss (absent in plain CPT runs).
    - ``origin_tag``: int, 0 = adaptation corpus, 1 = replay corpus.
    - ``origin``: string sub-source name (per-origin logging and analysis).
    - ``content_hash``: the block's cache key (selection trace).
    """

    def __call__(
        self,
        datum_dict: dict,
        task_data_spec,
        tokenizer,
        max_seq_length: int,
        idx: int,
    ) -> dict:
        import torch

        token_ids = torch.tensor(datum_dict["token_ids"], dtype=torch.long)
        message_log = [{"role": "assistant", "token_ids": token_ids, "content": ""}]
        out = {
            "message_log": message_log,
            "length": int(token_ids.shape[0]),
            "origin": str(datum_dict.get("origin", "unknown")),
            "origin_tag": int(datum_dict.get("origin_tag", 0)),
            "content_hash": str(datum_dict.get("content_hash", "")),
            "extra_env_info": None,
            "loss_multiplier": 1.0,
            "idx": idx,
        }
        if "ref_loss" in datum_dict and datum_dict["ref_loss"] is not None:
            out["ref_loss"] = float(datum_dict["ref_loss"])
        return out


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def build_train_and_val_datasets(cfg, tokenizer):
    """Load the block JSONL files and wrap them for NeMo-RL.

    Assumes :func:`prepare_data` (or the precompute script) has already written
    the JSONL files. Returns ``(train_dataset, val_dataset)`` shaped to satisfy
    ``sft.setup()``'s expectations.
    """
    from datasets import load_dataset
    from nemo_rl.data.datasets import AllTaskProcessedDataset
    from nemo_rl.data.interfaces import TaskDataSpec

    # data_mixer is not a NeMo-RL config block; pop before NeMo-RL setup() sees the config.
    cfg.pop("data_mixer", None)

    max_seq_length = cfg["policy"]["max_total_sequence_length"]
    train_data_path = cfg["data"]["train_data_path"]
    val_data_path = cfg["data"].get("val_data_path")

    logger.info("Loading block dataset from %s", train_data_path)
    train_dataset_raw = load_dataset("json", data_files=train_data_path, split="train")
    val_dataset_raw = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset_raw = load_dataset("json", data_files=val_data_path, split="train")
    logger.info(
        "Blocks: train=%d val=%d (max_total_sequence_length=%d)",
        len(train_dataset_raw), len(val_dataset_raw) if val_dataset_raw is not None else 0,
        max_seq_length,
    )

    # A default TaskDataSpec (prompt_file=None → .prompt = None) applies no
    # task-specific prompt template.
    task_spec = TaskDataSpec(task_name="rod")

    train_dataset = AllTaskProcessedDataset(
        train_dataset_raw,
        tokenizer,
        default_task_data_spec=task_spec,
        task_data_processors=RHOPreprocessor(),
        max_seq_length=max_seq_length,
    )
    val_dataset = None
    if val_dataset_raw is not None:
        val_dataset = AllTaskProcessedDataset(
            val_dataset_raw,
            tokenizer,
            default_task_data_spec=task_spec,
            task_data_processors=RHOPreprocessor(),
            max_seq_length=max_seq_length,
        )
    return train_dataset, val_dataset
