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

"""RoD master config.

Layers a RoD-specific config block on top of the standard NeMo-RL
master config (Policy / Data / Logger / Cluster / Checkpointing).

The trainer base is NeMo-RL's SFT algorithm (`nemo_rl.algorithms.sft`); we compose
a `MasterConfig` that satisfies SFT's expectations plus our extra `rod` block.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, NotRequired, TypedDict

import nemo_rl.algorithms.sft as nemo_sft
from nemo_rl.utils.config import load_config
from omegaconf import DictConfig, OmegaConf

from ..common import (
    CheckpointingConfig,
    ClusterConfig,
    DataConfig,
    LoggerConfig,
    PolicyConfig,
    verifiable,
)

log = logging.getLogger(__name__)


@verifiable
class SFTConfig(nemo_sft.SFTConfig):
    """Pass-through wrapper for NeMo-RL's SFTConfig.

    `sft.setup()` reads `master_config["sft"]` for things like `seed`,
    `max_num_steps`, `val_period`, etc. We keep this block alongside the
    `rod` block so SFT machinery sees what it expects.
    """

    pass


@verifiable
class RoDConfig(TypedDict, total=False):
    """RoD-specific knobs (the ``rod:`` block of the YAML config).

    The candidate pool size is ``policy.train_global_batch_size`` (= m·k); ``top_k``
    is the trained batch size k, so the candidate multiplier is m = pool / top_k.
    """

    # Selection
    top_k: int                                        # trained batch size k (required if enable_selection)
    # top_k = RoD; fixed_ratio / replay_only_rho / fixed_ratio_rho = fixed-share ablations
    selection_strategy: Literal["top_k", "fixed_ratio", "replay_only_rho", "fixed_ratio_rho"]
    fixed_replay_ratio: NotRequired[float]           # fixed-share strategies: replay fraction of the k slots
    # False = plain continual pre-training on the full batch (no scoring/selection, no
    # reference-loss cache needed): the adaptation specialist and fixed-replay baselines.
    enable_selection: bool

    # Logging
    rich_log_period: int                  # cadence of the rho-distribution metrics
    log_selected_idx: bool                # write the selected-sample trace (JSONL)
    log_selected_idx_period: int
    log_selected_idx_path: NotRequired[str]  # defaults to <log_dir>/selected_samples.jsonl

    # Curriculum transfer: selection traces of another RoD run (in phase order) whose
    # selected-sample stream is replayed as a fixed training order. Requires
    # enable_selection: false and data.shuffle: false.
    curriculum_trace_dirs: NotRequired[list[str]]

    # Directory of reference-loss *.parquet shards written by run_precompute.py.
    # null skips the ref-loss join (plain CPT runs set it explicitly; default.yaml sets a path).
    ref_loss_cache_path: NotRequired[str]


@verifiable
class MasterConfig(TypedDict):
    """Full RoD master config.

    The `sft` block satisfies NeMo-RL's `sft.setup()` expectations; the
    `rod` block holds the method's settings and `data_mixer` the data settings.
    """

    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    rod: RoDConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    data_mixer: NotRequired[dict[str, Any]]

    @classmethod
    def get_default_params_dict(cls) -> DictConfig:
        """Load the default params from ``$DEFAULT_PARAMS_FILE``.

        ``slurm/lib/container_env.sh`` exports DEFAULT_PARAMS_FILE as
        ``configs/default.yaml``; experiment YAMLs are merged on top of it.
        """
        default_params_file = os.environ["DEFAULT_PARAMS_FILE"]
        cfg = load_config(default_params_file)
        log.info(
            "Loaded default RoD params from %s:\n%s",
            default_params_file,
            OmegaConf.to_yaml(cfg, resolve=False),
        )
        return cfg

    @classmethod
    def from_overrides(cls, overrides: dict[Any, Any]) -> "MasterConfig":
        """Merge user overrides onto defaults and validate.

        Args:
            overrides: Partial config dict with only the fields to override.

        Returns:
            Fully merged + validated MasterConfig (as a plain dict — the
            ``verifiable`` machinery model_dumps for downstream dict indexing).
        """
        defaults = OmegaConf.create(cls.get_default_params_dict())
        merged = OmegaConf.to_container(OmegaConf.merge(defaults, overrides), resolve=True)
        return cls.validate(merged)  # type: ignore[arg-type]
