# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (c) 2026 The RoD Authors.
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
#
# Adapted from NVIDIA NeMo-RL v0.6.0, the training loop in nemo_rl/algorithms/sft.py (sft_train); modified for RoD.

"""RoD training loop: score → select → train, per step.

Replaces :func:`nemo_rl.algorithms.sft.sft_train` with a version that, at every
training step:

1. **Scores** the candidate pool (``policy.train_global_batch_size`` = m·k
   sequences) with the CURRENT model: one forward-only pass via
   :func:`rod.patches.rod_score_pass.policy_score`, returning per-sample
   ``rho = ℓ_θ(x) − ℓ_ref(x)`` on the driver. The reference loss is the cached loss
   of the adaptation specialist for adaptation candidates and of the pretrained
   base model for replay candidates.
2. **Selects** k candidates on the driver with :func:`_select_candidates`:

   - ``top_k`` (RoD): one joint top-k over the pooled rho of adaptation and
     replay candidates, with no per-corpus quota — the replay share of the
     trained batch emerges from the competition.
   - Fixed-share ablations (``fixed_ratio``, ``replay_only_rho``,
     ``fixed_ratio_rho``): a fixed fraction of the k slots goes to replay and the
     rest to adaptation; within each corpus the slots are filled uniformly at
     random or by top-rho, depending on the strategy.

   Padding samples are always masked out.
3. **Trains** with NeMo-RL's stock NLL loss on the k selected samples only
   (``Policy.train`` with ``gbs=k``): backward runs on k samples, not m·k.

With ``rod.enable_selection=false`` the loop skips scoring and selection and
trains on the full batch — this is used for plain continual pre-training (the
no-replay / fixed-replay baselines and the adaptation specialist).

Per step, selection metrics are logged on the driver where the full rho vector
is visible (per-corpus rho/loss means, selected-vs-dropped rho, per-origin
selection counts and rho percentiles), and the selected samples can be written
to a JSONL trace (``rod.log_selected_idx``).

Cost: one extra forward-only pass over the candidate pool per step, in addition
to the forward-backward pass over the k selected samples.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

log = logging.getLogger(__name__)

_APPLIED = False


def _patched_sft_train_builder():
    """Build the RoD ``sft_train`` that replaces the upstream one.

    The returned function has the same signature as the upstream
    :func:`nemo_rl.algorithms.sft.sft_train`. The extra inputs the
    driver-side selection needs (the validation loss fn and the ``rod``
    config) are pulled from module-level slots set via :func:`set_runtime`
    just before calling sft_train.
    """
    # Lazy imports so this module is importable in tests/dev without NeMo-RL.
    import numpy as np
    from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
    from nemo_rl.data.llm_message_utils import (
        add_loss_mask_to_message_log,
        batched_message_log_to_flat_message,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.utils.nsys import maybe_gpu_profile_step
    from nemo_rl.utils.timer import TimeoutChecker, Timer

    from rod.patches.rod_score_pass import policy_score

    def patched_sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,                          # unused: the loss fns come from set_runtime()
        master_config,
        logger,
        checkpointer,
        sft_save_state,
    ) -> None:
        """Score → select → train, per iteration.

        ``loss_fn`` (the positional arg of the upstream signature) is ignored.
        Two loss fns are used instead:
        - ``scoring_loss_fn``: :class:`RHOScoringLossFn`, used for validation
          (set via :func:`set_runtime`);
        - ``training_loss_fn``: NeMo-RL's stock NLLLossFn for the train pass on
          the selected samples.
        """
        if _RUNTIME["scoring_loss_fn"] is None:
            raise RuntimeError(
                "rod_sft_train: scoring_loss_fn not set. Call "
                "set_runtime(scoring_loss_fn=..., rod_cfg=...) before "
                "invoking sft_train."
            )
        scoring_loss_fn = _RUNTIME["scoring_loss_fn"]
        sample_trace = _RUNTIME.get("sample_trace")  # optional selected-sample trace writer
        rod_cfg = _RUNTIME["rod_cfg"] or {}
        # Plain (no-RHO) mode: when False, skip the ref_loss requirement + score
        # + selection and train the FULL candidate batch with the LM loss. Used for
        # the adaptation specialist and the plain continual-pretraining baselines.
        enable_selection = bool(rod_cfg.get("enable_selection", True))
        # k = trained batch size; the candidate pool is policy.train_global_batch_size
        # (= m·k), so the candidate multiplier m is pool size / top_k.
        pool_size = int(master_config["policy"]["train_global_batch_size"])
        if enable_selection and rod_cfg.get("top_k") is None:
            raise ValueError("rod.top_k (the trained batch size k) is required when rod.enable_selection=true.")
        top_k = int(rod_cfg.get("top_k") or pool_size)
        rich_log_period = int(rod_cfg.get("rich_log_period", 50))
        # "top_k" = RoD joint selection; fixed_ratio / replay_only_rho / fixed_ratio_rho =
        # fixed-share ablations (see _select_candidates).
        selection_strategy = str(rod_cfg.get("selection_strategy", "top_k"))
        fixed_replay_ratio = float(rod_cfg.get("fixed_replay_ratio", 0.25))
        # One-shot startup line so the run log makes the active selection
        # configuration explicit.
        if enable_selection:
            log.info(
                "RoD selection: strategy=%s | top_k=%d | candidate pool=%d (m=%.2f) | fixed_replay_ratio=%s",
                selection_strategy, top_k, pool_size, pool_size / max(top_k, 1),
                fixed_replay_ratio if selection_strategy != "top_k" else "n/a",
            )
        else:
            log.info("RoD selection disabled: plain continued pretraining on batches of %d", pool_size)
        # The training loss fn defaults to NeMo-RL's stock NLLLossFn.
        training_loss_fn = _RUNTIME["training_loss_fn"] or NLLLossFn()

        timer = Timer()
        timeout = TimeoutChecker(
            timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        timeout.start_iterations()

        if sft_save_state is None:
            sft_save_state = {
                "epoch": 0,
                "step": 0,
                "total_steps": 0,
                "consumed_samples": 0,
                "total_valid_tokens": 0,
            }
            current_epoch = 0
            current_step = 0
            total_steps = 0
            total_valid_tokens = 0
        else:
            current_epoch = sft_save_state["epoch"]
            current_step = sft_save_state["step"]
            total_steps = sft_save_state["total_steps"]
            total_valid_tokens = sft_save_state.get("total_valid_tokens", 0)

        sft_config = master_config["sft"]
        val_period = sft_config["val_period"]
        val_at_start = sft_config["val_at_start"]
        val_at_end = sft_config["val_at_end"]
        max_num_epochs = sft_config["max_num_epochs"]

        # val_at_start — uses the existing patched validate (which knows about
        # ref_loss / origin_tag plumbing). We pass scoring_loss_fn so val
        # computes the same rho metrics as the score pass.
        if val_at_start and total_steps == 0:
            print("\n🔍 Running initial validation...")
            from nemo_rl.algorithms.sft import validate as validate_fn  # patched at import-time
            val_metrics, validation_timings = validate_fn(
                policy,
                val_dataloader,
                tokenizer,
                scoring_loss_fn,
                step=0,
                master_config=master_config,
                val_batches=sft_config["val_batches"],
                val_batch_size=sft_config["val_global_batch_size"],
                val_mbs=sft_config["val_micro_batch_size"],
            )
            logger.log_metrics(val_metrics, total_steps, prefix="validation")
            logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

        policy.prepare_for_training()

        while (
            current_epoch < max_num_epochs
            and total_steps < master_config["sft"]["max_num_steps"]
        ):
            print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

            for batch in train_dataloader:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/"
                    f"{min(len(train_dataloader), master_config['sft']['max_num_steps'])} "
                    f"{'=' * 25}"
                )
                maybe_gpu_profile_step(policy, total_steps + 1)
                val_metrics, validation_timings = None, None

                with timer.time("total_step_time"):
                    # ----------------------------------------------------------
                    # Step preparation — build the candidate BatchedDataDict
                    # ----------------------------------------------------------
                    print("▶ Preparing candidate batch...")
                    with timer.time("data_processing"):
                        add_loss_mask_to_message_log(
                            batch["message_log"],
                            roles_to_train_on=["assistant"],
                        )
                        cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                            batch["message_log"],
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["policy"][
                                "make_sequence_length_divisible_by"
                            ],
                        )

                        candidate_data: BatchedDataDict = BatchedDataDict({
                            "input_ids": cat_and_padded["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": cat_and_padded["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        })
                        candidate_data.update(
                            cat_and_padded.get_multimodal_dict(as_tensors=False)
                        )

                        # ref_loss / origin_tag / content_hash come from the
                        # rod_collate patch. They flow straight into
                        # the candidate_data for the score pass to consume.
                        # ref_loss is REQUIRED only for the RHO score/select path
                        # (enable_selection). Plain pre-training (enable_selection
                        # =false, e.g. the legal-specialist ref) trains without it.
                        if "ref_loss" in batch:
                            candidate_data["ref_loss"] = batch["ref_loss"]
                        if "origin_tag" in batch:
                            candidate_data["origin_tag"] = batch["origin_tag"]
                        else:
                            raise RuntimeError(
                                "rod_sft_train: batch missing 'origin_tag'. "
                                "Check the collate patch."
                            )
                        if "content_hash" in batch:
                            candidate_data["content_hash"] = batch["content_hash"]
                        # ``origin`` is the per-sample source string
                        # (e.g. ``casehold``, ``wk_web``).
                        # Plumbed by rod_collate. Carried verbatim on
                        # the driver so per-sub-origin selection metrics can
                        # use it after the score pass without re-extracting.
                        if "origin" in batch:
                            candidate_data["origin"] = batch["origin"]

                    if not enable_selection:
                        # ------------------------------------------------------
                        # Plain (no-RHO) pre-training: skip score + selection and
                        # train the FULL candidate batch with the LM loss. Builds
                        # reference models (legal-specialist adapt ref) + plain
                        # continual-pretraining baselines. No ref_loss required.
                        # ------------------------------------------------------
                        print("▶ Plain pre-training step on full candidate batch...")
                        selected_data = candidate_data
                        with timer.time("policy_training"):
                            train_results = policy.train(
                                selected_data,
                                training_loss_fn,
                                gbs=selected_data.size,
                                mbs=master_config["policy"]["train_micro_batch_size"],
                                timer=timer,
                            )
                        score_result = {}
                        rod_metrics = {}
                    else:
                        if "ref_loss" not in candidate_data:
                            raise RuntimeError(
                                "rod_sft_train: batch missing 'ref_loss' but "
                                "enable_selection=true. Set rod.ref_loss_cache_path "
                                "(or enable_selection=false for plain pre-training)."
                            )
                        # ------------------------------------------------------
                        # Score pass — full candidate pool, forward only, rho [GBS]
                        # ------------------------------------------------------
                        print("▶ Scoring candidates...")
                        with timer.time("score_pass"):
                            score_result = policy_score(policy=policy, candidate_batch=candidate_data)
                        rho = score_result["per_sample_rho"]                       # [GBS]
                        target_loss = score_result["per_sample_target_loss"]       # [GBS]
                        origin_tag = score_result["per_sample_origin_tag"]         # [GBS]
                        sample_mask_gbs = score_result["per_sample_sample_mask"]   # [GBS]

                        # ------------------------------------------------------
                        # Driver-side selection (joint top-k, or a fixed-share ablation)
                        # ------------------------------------------------------
                        selected_idx = _select_candidates(
                            rho=rho,
                            sample_mask=sample_mask_gbs,
                            k=top_k,
                            strategy=selection_strategy,
                            origin_tag=origin_tag,
                            fixed_replay_ratio=fixed_replay_ratio,
                        )
                        if selected_idx is None:
                            warnings.warn(
                                "Step has no valid samples (all sample_mask==0); skipping."
                            )
                            current_step += 1
                            continue

                        # ------------------------------------------------------
                        # Build the selected sub-batch and run the train pass
                        # ------------------------------------------------------
                        print("▶ Taking a training step on selected samples...")
                        selected_data = candidate_data.select_indices(selected_idx.tolist())
                        with timer.time("policy_training"):
                            train_results = policy.train(
                                selected_data,
                                training_loss_fn,
                                gbs=selected_data.size,
                                mbs=master_config["policy"]["train_micro_batch_size"],
                                timer=timer,
                            )

                        # ------------------------------------------------------
                        # Selection metrics (driver-side, full rho [GBS] visible)
                        # ------------------------------------------------------
                        with timer.time("selection_metrics"):
                            per_sample_origin = candidate_data.get("origin")
                            rod_metrics = _compute_selection_metrics(
                                rho=rho,
                                target_loss=target_loss,
                                origin_tag=origin_tag,
                                origin=per_sample_origin,
                                sample_mask=sample_mask_gbs,
                                selected_idx=selected_idx,
                                step=total_steps + 1,
                                rich_log_period=rich_log_period,
                            )

                        # ------------------------------------------------------
                        # Selection trace: record the SELECTED samples' content_hash
                        # (+ origin/rho) so replay picks can be categorized post-hoc
                        # (join content_hash -> corpus -> sub-domain / token_ids).
                        # ------------------------------------------------------
                        if sample_trace is not None and sample_trace.enabled:
                            per_sample_hash = candidate_data.get("content_hash")
                            if per_sample_hash is not None:
                                sel = selected_idx.tolist()
                                sample_trace.write_selected(
                                    step=total_steps + 1,
                                    content_hash=[per_sample_hash[i] for i in sel],
                                    origin=([per_sample_origin[i] for i in sel]
                                            if per_sample_origin is not None else None),
                                    origin_tag=origin_tag[selected_idx].tolist(),
                                    rho=rho[selected_idx].tolist(),
                                    target_loss=target_loss[selected_idx].tolist(),
                                )

                    is_last_step = total_steps + 1 >= master_config["sft"][
                        "max_num_steps"
                    ] or (
                        current_epoch + 1 == max_num_epochs
                        and current_step + 1 == len(train_dataloader)
                    )

                    # ----------------------------------------------------------
                    # Periodic validation
                    # ----------------------------------------------------------
                    if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                        val_at_end and is_last_step
                    ):
                        from nemo_rl.algorithms.sft import validate as validate_fn
                        val_metrics, validation_timings = validate_fn(
                            policy,
                            val_dataloader,
                            tokenizer,
                            scoring_loss_fn,
                            step=total_steps + 1,
                            master_config=master_config,
                            val_batches=sft_config["val_batches"],
                            val_batch_size=sft_config["val_global_batch_size"],
                            val_mbs=sft_config["val_micro_batch_size"],
                        )
                        logger.log_metrics(
                            validation_timings, total_steps + 1, prefix="timing/validation"
                        )
                        logger.log_metrics(
                            val_metrics, total_steps + 1, prefix="validation"
                        )

                    # ----------------------------------------------------------
                    # Metrics aggregation — mirror upstream sft_train's shape
                    # ----------------------------------------------------------
                    metrics = {
                        "loss": train_results["loss"].numpy(),
                        "grad_norm": train_results["grad_norm"].numpy(),
                    }
                    if "moe_metrics" in train_results:
                        metrics.update(
                            {f"moe/{k_}": v for k_, v in train_results["moe_metrics"].items()}
                        )
                    metrics.update(train_results["all_mb_metrics"])
                    for k_, v in metrics.items():
                        if k_ in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                            metrics[k_] = np.mean(v).item()
                        else:
                            metrics[k_] = np.sum(v).item()
                    total_valid_tokens += metrics.get("global_valid_toks", 0)

                    # Score-pass + selection metrics ride alongside the train
                    # metrics so the existing logger / dashboards pick them up.
                    score_metrics = score_result.get("aggregate_metrics", {})
                    metrics.update(score_metrics)
                    metrics.update(rod_metrics)

                    # ----------------------------------------------------------
                    # Checkpointing — mirror upstream sft_train verbatim
                    # ----------------------------------------------------------
                    sft_save_state["consumed_samples"] += master_config["policy"][
                        "train_global_batch_size"
                    ]
                    timeout.mark_iteration()
                    should_save_by_step = (
                        is_last_step
                        or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                        == 0
                    )
                    should_save_by_timeout = timeout.check_save()

                    if master_config["checkpointing"]["enabled"] and (
                        should_save_by_step or should_save_by_timeout
                    ):
                        sft_save_state["step"] = (current_step + 1) % len(train_dataloader)
                        sft_save_state["total_steps"] = total_steps + 1
                        sft_save_state["epoch"] = current_epoch
                        sft_save_state["total_valid_tokens"] = total_valid_tokens

                        full_metric_name = master_config["checkpointing"]["metric_name"]
                        if full_metric_name is not None:
                            assert full_metric_name.startswith(
                                "train:"
                            ) or full_metric_name.startswith("val:"), (
                                f"metric_name={full_metric_name} must start with "
                                "'val:' or 'train:'"
                            )
                            prefix, metric_name = full_metric_name.split(":", 1)
                            metrics_source = metrics if prefix == "train" else val_metrics
                            if not metrics_source:
                                warnings.warn(
                                    f"Asked to save based on {metric_name} but no "
                                    f"{prefix} metrics; skipping top-k accounting.",
                                    stacklevel=2,
                                )
                                if full_metric_name in sft_save_state:
                                    del sft_save_state[full_metric_name]
                            elif metric_name not in metrics_source:
                                raise ValueError(
                                    f"Metric {metric_name} not found in {prefix} metrics"
                                )
                            else:
                                sft_save_state[full_metric_name] = metrics_source[
                                    metric_name
                                ]

                        import os as _os
                        with timer.time("checkpointing"):
                            print(f"Saving checkpoint for step {total_steps + 1}...")
                            checkpoint_path = checkpointer.init_tmp_checkpoint(
                                total_steps + 1, sft_save_state, master_config
                            )
                            policy.save_checkpoint(
                                weights_path=_os.path.join(
                                    checkpoint_path, "policy", "weights"
                                ),
                                optimizer_path=_os.path.join(
                                    checkpoint_path, "policy", "optimizer"
                                )
                                if checkpointer.save_optimizer
                                else None,
                                tokenizer_path=_os.path.join(
                                    checkpoint_path, "policy", "tokenizer"
                                ),
                                checkpointing_cfg=master_config["checkpointing"],
                            )
                            import torch as _torch
                            _torch.save(
                                train_dataloader.state_dict(),
                                _os.path.join(checkpoint_path, "train_dataloader.pt"),
                            )
                            checkpointer.finalize_checkpoint(checkpoint_path)

                timing_metrics = timer.get_timing_metrics(reduction_op="sum")

                # ----------------------------------------------------------
                # Stdout summary
                # ----------------------------------------------------------
                print("\n📊 Training Results:")
                print(f"  • Loss: {float(metrics['loss']):.4f}")
                if "total_flops" in train_results:
                    total_tflops = (
                        train_results["total_flops"]
                        / timing_metrics["policy_training"]
                        / 1e12
                    )
                    num_ranks = train_results["num_ranks"]
                    print(
                        f"  • Training FLOPS: {total_tflops:.2f} TFLOPS "
                        f"({total_tflops / num_ranks:.2f} TFLOPS per rank)"
                    )
                    if "theoretical_tflops" in train_results:
                        theoretical_tflops = train_results["theoretical_tflops"]
                        print(
                            f"  • Training Model Floating Point Utilization: "
                            f"{100 * total_tflops / theoretical_tflops:.2f}%"
                        )
                        metrics["train_fp_utilization"] = (
                            total_tflops / theoretical_tflops
                        )
                print("\n⏱️  Timing:")
                total_time = timing_metrics.get("total_step_time", 0)
                print(f"  • Total step time: {total_time:.2f}s")
                for k_, v in sorted(
                    timing_metrics.items(), key=lambda item: item[1], reverse=True
                ):
                    if k_ != "total_step_time":
                        percent = (v / total_time * 100) if total_time > 0 else 0
                        print(f"  • {k_}: {v:.2f}s ({percent:.1f}%)")

                total_num_gpus = (
                    master_config["cluster"]["num_nodes"]
                    * master_config["cluster"]["gpus_per_node"]
                )
                if total_time > 0:
                    timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                        metrics.get("global_valid_toks", 0) / total_time / total_num_gpus
                    )
                else:
                    timing_metrics["valid_tokens_per_sec_per_gpu"] = 0.0
                logger.log_metrics(metrics, total_steps + 1, prefix="train")
                logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

                timer.reset()
                current_step += 1
                total_steps += 1

                if should_save_by_timeout:
                    print("Timeout reached, stopping training early", flush=True)
                    return
                if total_steps >= master_config["sft"]["max_num_steps"]:
                    print(
                        "Max number of steps reached, stopping training early",
                        flush=True,
                    )
                    return

            current_epoch += 1
            current_step = 0

    return patched_sft_train


# ---------------------------------------------------------------------------
# Driver-side selection — joint top-k (RoD) and the fixed-ratio control
# ---------------------------------------------------------------------------


def _select_candidates(
    *,
    rho,                              # torch.Tensor [GBS]
    sample_mask,                      # torch.Tensor [GBS]
    k: int,
    strategy: str,
    origin_tag=None,                  # torch.Tensor [GBS], 0=adapt 1=replay
    fixed_replay_ratio: float = 0.25, # only used by the fixed-share strategies
):
    """Pick ``k`` candidate indices per the chosen strategy.

    Args:
        rho: per-sample rho score, shape ``[GBS]``.
        sample_mask: 1 for valid samples, 0 for padding rows. Padding rows are
            mapped to ``-inf`` and can never be selected.
        k: number of samples to select (the trained batch size). Capped at the
            number of valid samples.
        strategy:
            - ``"top_k"`` (RoD): the ``k`` highest-rho candidates over the pooled
              adaptation and replay candidates, with no per-corpus quota.
              Deterministic.
            - Fixed-share ablations: ``round(fixed_replay_ratio · k)`` replay slots and
              the rest adaptation slots (if one corpus runs short, the other backfills).
              How the slots of each corpus are filled:

              =====================  ======================  =======================
              strategy               adaptation slots        replay slots
              =====================  ======================  =======================
              ``fixed_ratio``        uniform random          uniform random
              ``replay_only_rho``    uniform random          top-rho within replay
              ``fixed_ratio_rho``    top-rho within adapt    top-rho within replay
              =====================  ======================  =======================

              Random picks honor ``torch.manual_seed``.
        origin_tag: required for the fixed-share strategies.
        fixed_replay_ratio: replay fraction of the ``k`` slots for the fixed-share strategies.

    Returns:
        ``torch.Tensor`` of selected indices into ``[0, GBS)``, or ``None`` if there
        are no valid samples (the caller then skips the step).
    """
    import torch

    rho_for_select = rho.clone()
    # Mask padding rows so they're never selected.
    rho_for_select[sample_mask == 0] = float("-inf")

    n_valid = int(torch.isfinite(rho_for_select).sum().item())
    k_eff = min(k, n_valid)
    if k_eff <= 0:
        return None

    if strategy == "top_k":
        return torch.topk(rho_for_select, k=k_eff, largest=True).indices

    # (adaptation slots use rho?, replay slots use rho?) for the fixed-share strategies
    fixed_share = {"fixed_ratio": (False, False), "replay_only_rho": (False, True),
                   "fixed_ratio_rho": (True, True)}
    if strategy in fixed_share:
        if origin_tag is None:
            raise ValueError(f"{strategy} strategy requires origin_tag")
        rho_adapt, rho_replay = fixed_share[strategy]
        finite = torch.isfinite(rho_for_select)
        adapt = torch.nonzero(finite & (origin_tag == 0), as_tuple=False).flatten()
        replay = torch.nonzero(finite & (origin_tag == 1), as_tuple=False).flatten()
        k_rep = min(int(round(fixed_replay_ratio * k_eff)), replay.numel())
        k_ad = min(k_eff - k_rep, adapt.numel())
        k_rep = min(k_eff - k_ad, replay.numel())   # backfill replay if adapt was short

        def pick(idx, n, by_rho):
            if by_rho:
                return idx[torch.topk(rho_for_select[idx], k=n, largest=True).indices]
            return idx[torch.randperm(idx.numel(), device=idx.device)[:n]]

        return torch.cat([pick(adapt, k_ad, rho_adapt), pick(replay, k_rep, rho_replay)])

    raise ValueError(
        f"rod_sft_train._select_candidates: unsupported selection_strategy={strategy!r}; "
        "expected 'top_k', 'fixed_ratio', 'replay_only_rho' or 'fixed_ratio_rho'."
    )


# ---------------------------------------------------------------------------
# Driver-side selection metrics
# ---------------------------------------------------------------------------


def _compute_selection_metrics(
    *,
    rho,
    target_loss,
    origin_tag,
    sample_mask,
    selected_idx,
    step: int,
    rich_log_period: int,
    origin: list[str] | None = None,
) -> dict[str, float]:
    """Per-step selection metrics, computed on the full candidate-pool ``[GBS]`` rho.

    - Per-corpus (adapt / replay) means of target loss, reference loss and rho,
      all computed from the same per-sample tensors so ``target − ref = rho`` holds
      exactly.
    - Selection counts per corpus; the replay share of the trained batch is
      ``selection/n_selected_replay / selection/k_effective``.
    - Selected-vs-dropped means (``selection_sanity/*``): their gap confirms that
      selection prefers high-rho candidates.
    - Every ``rich_log_period`` steps: rho percentiles per corpus.
    - When ``origin`` (a length-``GBS`` list of sub-source strings) is provided, a
      per-origin breakdown under ``selection_by_origin/`` and ``rho_by_origin/``
      (which replay sources the selector picks from).
    """
    import torch

    metrics: dict[str, float] = {}

    valid = sample_mask > 0
    n_valid = float(valid.sum().item())
    if n_valid == 0:
        return metrics

    # Per-sample ref_loss is recoverable from target_loss and rho, since the
    # scoring pass computes ``rho = target_loss - ref_loss`` per sample.
    ref_loss = target_loss - rho

    # Selected mask
    selected_mask = torch.zeros_like(rho, dtype=torch.bool)
    selected_mask[selected_idx] = True
    dropped_mask = (~selected_mask) & valid

    # Per-corpus masks
    is_adapt = (origin_tag == 0) & valid
    is_replay = (origin_tag == 1) & valid

    def _safe_mean(values, mask):
        if mask.sum() == 0:
            return float("nan")
        return float(values[mask].mean().item())

    def _safe_count(mask):
        return float(mask.sum().item())

    # Per-corpus means + selection counts.
    metrics["loss/target_loss_adapt"] = _safe_mean(target_loss, is_adapt)
    metrics["loss/target_loss_replay"] = _safe_mean(target_loss, is_replay)
    metrics["loss/ref_loss_adapt"] = _safe_mean(ref_loss, is_adapt)
    metrics["loss/ref_loss_replay"] = _safe_mean(ref_loss, is_replay)
    metrics["rho/rho_mean_adapt"] = _safe_mean(rho, is_adapt)
    metrics["rho/rho_mean_replay"] = _safe_mean(rho, is_replay)
    metrics["rho/frac_rho_replay_above_zero"] = (
        float(((rho > 0) & is_replay).sum().item())
        / max(_safe_count(is_replay), 1.0)
    )

    n_sel_adapt = _safe_count(selected_mask & is_adapt)
    n_sel_replay = _safe_count(selected_mask & is_replay)
    metrics["selection/n_selected_adapt"] = n_sel_adapt
    metrics["selection/n_selected_replay"] = n_sel_replay
    metrics["selection/frac_selected_adapt"] = n_sel_adapt / max(_safe_count(is_adapt), 1.0)
    metrics["selection/frac_selected_replay"] = n_sel_replay / max(_safe_count(is_replay), 1.0)
    metrics["selection/frac_adapt_in_candidate"] = _safe_count(is_adapt) / n_valid
    metrics["selection/k_effective"] = float(selected_mask.sum().item())

    # Selected-vs-dropped means: together they confirm that selection picks
    # samples where the target has the most "room to learn relative to the reference".
    metrics["selection_sanity/rho_mean_selected"] = _safe_mean(rho, selected_mask & valid)
    metrics["selection_sanity/rho_mean_dropped"] = _safe_mean(rho, dropped_mask)
    metrics["selection_sanity/target_loss_mean_selected"] = _safe_mean(
        target_loss, selected_mask & valid
    )
    metrics["selection_sanity/target_loss_mean_dropped"] = _safe_mean(target_loss, dropped_mask)
    metrics["selection_sanity/ref_loss_mean_selected"] = _safe_mean(
        ref_loss, selected_mask & valid
    )
    metrics["selection_sanity/ref_loss_mean_dropped"] = _safe_mean(ref_loss, dropped_mask)

    # Periodic: percentiles of the per-corpus rho distributions.
    if step % rich_log_period == 0:
        for tag_mask, name in ((is_adapt, "adapt"), (is_replay, "replay")):
            rho_sub = rho[tag_mask]
            if rho_sub.numel() > 0:
                qs = torch.quantile(
                    rho_sub.float(),
                    torch.tensor([0.10, 0.50, 0.90, 0.99], device=rho_sub.device),
                )
                metrics[f"rho_dist/{name}/p10"] = float(qs[0].item())
                metrics[f"rho_dist/{name}/p50"] = float(qs[1].item())
                metrics[f"rho_dist/{name}/p90"] = float(qs[2].item())
                metrics[f"rho_dist/{name}/p99"] = float(qs[3].item())

    # ------------------------------------------------------------------
    # Per-sub-origin breakdown ("which capability in the replay is being
    # selected from?")
    # ------------------------------------------------------------------
    # Only runs when the collate patch plumbed ``origin`` through.
    if origin is not None:
        metrics.update(
            _compute_per_origin_metrics(
                rho=rho,
                origin=origin,
                origin_tag=origin_tag,
                sample_mask=sample_mask,
                selected_mask=selected_mask,
                step=step,
                rich_log_period=rich_log_period,
            )
        )

    return metrics


def _sanitize_origin_for_metric_key(origin_name: str) -> str:
    """Make ``origin_name`` safe to embed in a metric path.

    ``/`` separates metric groups and whitespace breaks metric names, so both are
    replaced with ``_``; an empty name becomes ``unknown``.
    """
    safe = (origin_name or "unknown").strip()
    if not safe:
        safe = "unknown"
    for bad in (" ", "\t", "\n", "/"):
        safe = safe.replace(bad, "_")
    return safe


def _compute_per_origin_metrics(
    *,
    rho,
    origin: list[str],
    origin_tag,
    sample_mask,
    selected_mask,
    step: int,
    rich_log_period: int,
) -> dict[str, float]:
    """Per-sub-origin (capability-level) selection + rho metrics.

    Emits, for each distinct ``origin`` string present in the candidate batch::

        selection_by_origin/<corpus>/<origin>/n_candidate
        selection_by_origin/<corpus>/<origin>/n_selected
        selection_by_origin/<corpus>/<origin>/frac_selected   (n_selected / n_candidate)
        rho_by_origin/<corpus>/<origin>/mean
        rho_by_origin/<corpus>/<origin>/frac_above_zero

    ``<corpus>`` is ``adapt`` for ``origin_tag == 0`` and ``replay`` for
    ``origin_tag == 1`` — derived per-sample from ``origin_tag`` so the
    namespace is unambiguous even if an origin name appears in both corpora.

    Padding samples (``sample_mask == 0``) are excluded from both counts and
    means.  Sub-origins with zero valid candidates in the current step are
    omitted entirely — ``nan``-filled entries clutter the dashboard for
    rare origins that don't appear in every batch.

    The rho summaries (``mean``, ``frac_above_zero``) are gated by
    ``rich_log_period`` like the percentiles; the counts are cheap and emit
    every step.
    """
    import torch

    out: dict[str, float] = {}
    if len(origin) != int(rho.shape[0]):
        log.warning(
            "_compute_per_origin_metrics: origin length (%d) != rho length (%d); "
            "skipping per-sub-origin breakdown for this step.",
            len(origin), int(rho.shape[0]),
        )
        return out

    # Bucket sample indices by (corpus, sub-origin). Single pass over the
    # candidate batch on CPU — len(origin) == GBS which is small (hundreds
    # of rows, not millions), so the Python loop is cheap relative to the
    # forward we just ran.
    sample_mask_cpu = sample_mask.detach().cpu()
    origin_tag_cpu = origin_tag.detach().cpu()
    buckets: dict[tuple[str, str], list[int]] = {}
    for i, sub_origin in enumerate(origin):
        if float(sample_mask_cpu[i].item()) <= 0:
            continue
        tag = int(origin_tag_cpu[i].item())
        corpus = "adapt" if tag == 0 else "replay" if tag == 1 else f"tag{tag}"
        key = (corpus, _sanitize_origin_for_metric_key(sub_origin))
        buckets.setdefault(key, []).append(i)

    if not buckets:
        return out

    selected_mask_cpu = selected_mask.detach().cpu()
    detailed = (step % rich_log_period == 0)

    for (corpus, sub_origin), indices in sorted(buckets.items()):
        idx_t = torch.tensor(indices, dtype=torch.long)
        n_candidate = float(len(indices))
        n_selected = float(selected_mask_cpu[idx_t].sum().item())
        prefix = f"selection_by_origin/{corpus}/{sub_origin}"
        out[f"{prefix}/n_candidate"] = n_candidate
        out[f"{prefix}/n_selected"] = n_selected
        out[f"{prefix}/frac_selected"] = n_selected / max(n_candidate, 1.0)

        if detailed:
            rho_sub = rho[idx_t.to(rho.device)]
            out[f"rho_by_origin/{corpus}/{sub_origin}/mean"] = float(
                rho_sub.float().mean().item()
            )
            out[f"rho_by_origin/{corpus}/{sub_origin}/frac_above_zero"] = float(
                (rho_sub > 0).float().mean().item()
            )

    return out


# ---------------------------------------------------------------------------
# Runtime slots for the validation loss fn + RoD cfg
# ---------------------------------------------------------------------------

_RUNTIME: dict[str, Any] = {
    "scoring_loss_fn": None,
    "training_loss_fn": None,
    "rod_cfg": None,
    "sample_trace": None,
}


def set_runtime(
    *,
    scoring_loss_fn: Any,
    rod_cfg: dict,
    training_loss_fn: Any | None = None,
    sample_trace: Any | None = None,
) -> None:
    """Set the runtime slots :func:`patched_sft_train` reads at iter time.

    Called once by ``run_rod.py`` before invoking ``sft_train``: slots in the
    validation loss fn, the ``rod`` config, and the optional ``sample_trace``
    writer (records selected content_hash/origin/rho per step).
    """
    _RUNTIME["scoring_loss_fn"] = scoring_loss_fn
    _RUNTIME["rod_cfg"] = rod_cfg
    _RUNTIME["training_loss_fn"] = training_loss_fn
    _RUNTIME["sample_trace"] = sample_trace


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


def apply() -> None:
    """Replace ``nemo_rl.algorithms.sft.sft_train`` with the RoD training loop.

    Idempotent. Must be called BEFORE ``run_rod.py`` does
    ``from nemo_rl.algorithms.sft import sft_train`` so the patched
    symbol is what the deferred import binds to.
    """
    global _APPLIED
    if _APPLIED:
        return

    from nemo_rl.algorithms import sft as _sft_mod

    _sft_mod.sft_train = _patched_sft_train_builder()

    _APPLIED = True
    log.info(
        "Patched nemo_rl.algorithms.sft.sft_train with rod_sft_train "
        "(driver-side two-pass score → select → train)"
    )
