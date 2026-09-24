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

"""RoD training entry point.

Thin wrapper around NeMo-RL's SFT trainer that:

1. Loads + merges YAML config (``configs/default.yaml`` + experiment YAML + CLI overrides).
2. Applies our runtime patches:
   - ``rod_collate``: threads ref_loss / origin_tag / content_hash
     into each batch.
   - ``rod_validate``: sequence-weighted val_loss aggregation + per-origin
     validation losses.
   - ``rod_sft_train``: replaces sft_train with the score → select → train
     loop. Selection happens on the driver where the rho of the full candidate
     pool is visible.
3. Calls ``sft.setup()`` to build the policy / cluster / dataloaders.
4. Instantiates ``RHOScoringLossFn`` for validation; the train pass uses the
   stock ``NLLLossFn`` returned by setup.
5. Hands everything to the patched ``sft.sft_train()`` plus our sample-
   trace writer.

The Megatron backend (TP/PP/EP) and Ray orchestration are unchanged — they
come from ``sft.setup()`` for free.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pprint

from nemo_rl.utils.config import register_omegaconf_resolvers
from rod.configs.common import build_config
from rod.configs.rod import MasterConfig
from rod.data.rod_dataset import build_train_and_val_datasets, prepare_data
from rod.utils import get_job_name

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="RoD training (NeMo-RL SFT base)")
    p.add_argument("--config", type=str, required=True, help="Path to the RoD YAML config.")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_known_args()


# ---------------------------------------------------------------------------
# Selected-sample trace writer
# ---------------------------------------------------------------------------


class SelectedSampleTraceWriter:
    """Append-only JSONL trace of SELECTED training samples (local file).

    Writes ONE ROW PER (step, selected sample):
        {step, content_hash, origin, origin_tag, rho, target_loss}
    ``content_hash`` is the stable per-block key that joins back to the blocks
    corpus (origin / domain / token_ids) — so selected replay can be categorized
    post-hoc by sub-domain, or by content (detokenize token_ids → classify, e.g.
    into capability categories). Records ALL selected rows (adapt + replay); filter
    by ``origin_tag`` (0=adapt, 1=replay) / ``origin`` downstream.

    Called by the training loop at the selection hook (it has content_hash / origin /
    origin_tag / rho for the full candidate batch). Selection runs single-process
    on the driver, so there is no multi-rank duplication. The JSONL lives under the
    experiment log dir; optionally `wandb.log_artifact` it in post-processing.
    """

    def __init__(self, path: str, enabled: bool, period: int = 1) -> None:
        self.path = path
        self.enabled = enabled
        self.period = period
        self._n = 0
        if self.enabled:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # Open in append so resume-after-crash doesn't clobber the log.
            self._fh = open(path, "a", buffering=1)
            logger.info("Selected-sample trace: appending to %s", path)
        else:
            self._fh = None

    def write_selected(self, step: int, *, content_hash, origin, origin_tag, rho,
                       target_loss=None) -> None:
        """Append one row per selected sample. All inputs are aligned, length == n_selected."""
        if not self.enabled or self._fh is None:
            return
        if self.period > 1 and step % self.period != 0:
            return
        for i in range(len(content_hash)):
            row = {
                "step": int(step),
                "content_hash": content_hash[i],
                "origin": origin[i] if origin is not None else None,
                "origin_tag": int(origin_tag[i]),
                "rho": float(rho[i]),
            }
            if target_loss is not None:
                row["target_loss"] = float(target_loss[i])
            self._fh.write(json.dumps(row) + "\n")
            self._n += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            logger.info("Selected-sample trace: wrote %d rows -> %s", self._n, self.path)


def main() -> None:
    args, cli_overrides = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        force=True,
    )

    logger.info("=" * 72)
    logger.info("RoD Training (NeMo-RL SFT base)")
    logger.info("Config: %s", args.config)
    logger.info("=" * 72)

    # ------------------------------------------------------------------
    # Build config
    # ------------------------------------------------------------------
    register_omegaconf_resolvers()
    cfg = build_config(MasterConfig, args.config, cli_overrides)
    if int(cfg["policy"]["megatron_cfg"].get("context_parallel_size", 1)) > 1:
        raise ValueError("context parallelism is not supported: per-sequence losses need whole sequences")

    # ------------------------------------------------------------------
    # Apply the patches before importing sft_train / setup below, so those
    # names bind to the patched functions.
    # ------------------------------------------------------------------
    from rod.patches import (
        RHOScoringLossFn,
        apply_rod_collate,
        apply_rod_sft_train,
        apply_rod_validate,
        set_rod_sft_train_runtime,
    )
    apply_rod_collate()         # SFT collate — threads ref_loss / origin_tag
    apply_rod_validate()        # sequence-weighted val_loss aggregation
    apply_rod_sft_train()    # driver-side two-pass: score → top-K → train

    # ------------------------------------------------------------------
    # NeMo-RL imports (deferred so Ray is not imported at module level)
    # ------------------------------------------------------------------
    from nemo_rl.algorithms.sft import setup as sft_setup, sft_train
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.logger import get_next_experiment_dir

    import ray

    # ------------------------------------------------------------------
    # Log / checkpoint dirs: each launch logs to a new <run>/exp_<n>; checkpoints
    # live in <run>/checkpoints, so relaunching the same run name resumes it.
    # ------------------------------------------------------------------
    job_name = get_job_name()
    log_dir_root = os.path.join(cfg["logger"]["log_dir"], job_name)
    cfg["logger"]["log_dir"] = get_next_experiment_dir(log_dir_root)
    cfg["checkpointing"]["checkpoint_dir"] = os.path.join(log_dir_root, "checkpoints")
    if cfg["logger"].get("tensorboard_enabled"):
        cfg["logger"]["tensorboard"]["log_dir"] = os.path.join(log_dir_root, "tensorboard")
    logger.info("Log directory:        %s", cfg["logger"]["log_dir"])
    logger.info("Checkpoint directory: %s", cfg["checkpointing"]["checkpoint_dir"])

    # Default sample-trace path under the experiment log dir.
    if not cfg["rod"].get("log_selected_idx_path"):
        cfg["rod"]["log_selected_idx_path"] = os.path.join(
            cfg["logger"]["log_dir"], "selected_samples.jsonl"
        )

    # ------------------------------------------------------------------
    # Ray init + tokenizer + data prep
    # ------------------------------------------------------------------
    init_ray()
    tokenizer = get_tokenizer(cfg["policy"]["tokenizer"])

    logger.info("Preparing data ...")
    prepare_data(cfg)  # pass the MERGED cfg (corpus_path/train_data_path live in default.yaml)
    logger.info("Building datasets ...")
    train_dataset, val_dataset = build_train_and_val_datasets(cfg, tokenizer)

    logger.info("Final config:\n%s", pprint.pformat(cfg))

    # ------------------------------------------------------------------
    # SFT setup — returns the standard 9-tuple. The returned NLLLossFn is the
    # train-pass loss (plain LM loss on the selected samples); RHOScoringLossFn
    # is instantiated separately for validation.
    # ------------------------------------------------------------------
    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        _nll_loss_fn,
        rl_logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = sft_setup(cfg, tokenizer, train_dataset, val_dataset)

    # Two loss functions:
    #   - scoring_loss_fn (RHOScoringLossFn): used by validation (per-origin
    #     losses and rho metrics). The per-step scoring pass computes rho
    #     directly from policy.get_logprobs (see rod_score_pass).
    #   - training_loss_fn (NeMo-RL's stock NLLLossFn): the train pass on the
    #     k selected samples — plain LM loss, no RHO logic inside the loss fn.
    scoring_loss_fn = RHOScoringLossFn()
    logger.info(
        "Validation loss fn: %s | Train loss fn: %s (stock from sft.setup)",
        type(scoring_loss_fn).__name__,
        type(_nll_loss_fn).__name__,
    )

    # ------------------------------------------------------------------
    # Selected-sample trace writer — created BEFORE set_runtime so the training
    # loop can pull it from the runtime slot and record selected samples at the
    # selection hook. Written to a local JSONL file.
    # ------------------------------------------------------------------
    sample_trace = SelectedSampleTraceWriter(
        path=master_config["rod"]["log_selected_idx_path"],
        enabled=bool(master_config["rod"].get("log_selected_idx", False)),
        period=int(master_config["rod"].get("log_selected_idx_period", 1)),
    )

    # Stash the validation loss fn + RoD cfg + sample-trace writer in the training
    # loop's runtime slot — the patched sft_train reads them at iter time.
    set_rod_sft_train_runtime(
        scoring_loss_fn=scoring_loss_fn,
        rod_cfg=master_config["rod"],
        training_loss_fn=_nll_loss_fn,
        sample_trace=sample_trace,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting RoD training ...")
    try:
        # The patched sft_train ignores the loss_fn positional arg and reads
        # its loss fns from the runtime slot set above; _nll_loss_fn is passed
        # only to keep the upstream sft_train signature.
        sft_train(
            policy,
            train_dataloader,
            val_dataloader,
            tokenizer,
            _nll_loss_fn,
            master_config,
            rl_logger,
            checkpointer,
            sft_save_state,
        )
    finally:
        sample_trace.close()
    logger.info("RoD training complete.")

    logger.info("Shutting down policy workers ...")
    try:
        policy.shutdown()
    except Exception as e:
        logger.warning("Policy shutdown failed (non-fatal): %s", e, exc_info=True)

    if ray.is_initialized():
        logger.info("Shutting down Ray ...")
        ray.shutdown()


if __name__ == "__main__":
    main()
