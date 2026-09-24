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

"""Multi-rank Megatron -> HuggingFace conversion (local save).

Launch with:

    torchrun --standalone --nproc_per_node=8 convert_mcore_to_hf.py \
        --megatron-dir /local/path/to/checkpoint \
        --hf-model /path/to/qwen35_ref_dir \
        --ep 8

`--hf-model` is a local directory containing the HF model's config.json
and tokenizer files. Weights are NOT needed — we build the bridge config-
only via AutoConfig.from_pretrained -> AutoBridge.from_hf_config. If the
target architecture is native to the installed `transformers` (Qwen3.5 in
recent transformers is), no modeling_*.py is needed; if not, add the
modeling/configuration .py files to the same dir.

Uses gloo backend + CPU tensors throughout. `AutoBridge.load_megatron_model`
auto-detects gloo and sets `use_cpu_init=True` (see
`Megatron-Bridge/.../auto_bridge.py:1007`), avoiding the TE GroupedLinear
GPU-OOM at MoE init time without any internal-API trickery.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
import time
from pathlib import Path

# Print BEFORE the slow imports so we know the script actually got picked up
# by python3 — separate from anything Megatron's import-time warnings do.
_RANK = os.environ.get("RANK", "?")
print(f"[rank={_RANK}] convert_mcore_to_hf.py: module load start", file=sys.stderr, flush=True)

import torch.distributed as dist
from megatron.bridge import AutoBridge
from transformers import AutoConfig

print(f"[rank={_RANK}] convert_mcore_to_hf.py: imports done", file=sys.stderr, flush=True)


def _log(msg: str) -> None:
    """Force-flushed stderr print — bypasses Python's logging buffering and
    the WARNING-level root logger Megatron installs during its imports."""
    print(f"[rank={os.environ.get('RANK', '?')}] {msg}", file=sys.stderr, flush=True)


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def main() -> None:
    _log(
        f"main() start  RANK={os.environ.get('RANK')}  "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}  "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}  "
        f"is_rank_zero={_is_rank_zero()}"
    )

    p = argparse.ArgumentParser()
    p.add_argument("--megatron-dir", required=True,
                   help="Local dir containing the Megatron checkpoint "
                        "(iter_*/, latest_checkpointed_iteration.txt).")
    p.add_argument("--hf-model", required=True,
                   help="Local path to an HF reference dir containing "
                        "config.json and tokenizer files. Model weights "
                        "are NOT needed (config-only bridge construction). "
                        "Add modeling_*.py only if the architecture isn't "
                        "native to the installed `transformers`.")
    p.add_argument("--scratch-dir", default="mcore_to_hf_out",
                   help="Local working directory for the HF output")
    p.add_argument("--ep", type=int, default=1,
                   help="Expert parallel degree at load (MoE)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    args = p.parse_args()

    megatron_dir = Path(args.megatron_dir)
    scratch = Path(args.scratch_dir)
    hf_dir = scratch / "hf"

    # init_process_group before any rank-0-only work, so all ranks meet here as soon
    # as their imports finish; the long timeout covers slow cold imports.
    _log("init_process_group(backend=gloo, timeout=2h) start")
    t = time.monotonic()
    dist.init_process_group(
        backend="gloo",
        timeout=datetime.timedelta(hours=2),
    )
    _log(f"init_process_group done  ({time.monotonic()-t:.1f}s)")

    if _is_rank_zero():
        # Copy the entire source HF ref dir (config + tokenizer files +
        # any modeling_*.py) into the output. This is how the TOKENIZER
        # ends up in the final artifact — save_hf_pretrained's config-only
        # branch only writes config.json + safetensors, so without this
        # the output would have no tokenizer and vLLM/HF couldn't load it.
        # The subsequent save only touches config.json + *.safetensors,
        # leaving every other file we copied here intact.
        t = time.monotonic()
        if hf_dir.exists():
            shutil.rmtree(hf_dir)
        shutil.copytree(args.hf_model, hf_dir)
        _log(f"staged HF reference {args.hf_model} -> {hf_dir}  ({time.monotonic()-t:.1f}s)")

    _log("first dist.barrier() start (wait for rank 0 to finish staging)")
    t = time.monotonic()
    dist.barrier()
    _log(f"first dist.barrier() done  ({time.monotonic()-t:.1f}s)")

    # Config-only bridge: read the HF architecture from config.json (no weight
    # loading), so the same bridge NeMo-RL used for training maps the weights back.
    # For the multimodal Qwen3.5 classes this includes the (untrained) vision tower,
    # so the output is a complete checkpoint of the original model class.
    _log(f"AutoConfig.from_pretrained({args.hf_model}, trust_remote_code=True) start")
    t = time.monotonic()
    hf_config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)
    _log(f"AutoConfig.from_pretrained done  ({time.monotonic()-t:.1f}s)")

    _log("AutoBridge.from_hf_config start")
    t = time.monotonic()
    bridge = AutoBridge.from_hf_config(hf_config)
    _log(f"AutoBridge.from_hf_config done  ({time.monotonic()-t:.1f}s)")

    mp_overrides = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "expert_model_parallel_size": args.ep,
    }

    _log(
        f"bridge.load_megatron_model(from={megatron_dir}, "
        f"tp={args.tp}, pp={args.pp}, ep={args.ep}) start"
    )
    t = time.monotonic()
    model = bridge.load_megatron_model(
        str(megatron_dir),
        mp_overrides=mp_overrides,
        wrap_with_ddp=False,
    )
    _log(f"bridge.load_megatron_model done  ({time.monotonic()-t:.1f}s)")

    _log(f"bridge.save_hf_pretrained(to={hf_dir}, distributed_save=True) start")
    t = time.monotonic()
    bridge.save_hf_pretrained(
        model,
        str(hf_dir),
        show_progress=_is_rank_zero(),
        strict=False,
        distributed_save=True,
    )
    _log(f"bridge.save_hf_pretrained done  ({time.monotonic()-t:.1f}s)")

    _log("second dist.barrier() start (wait for all ranks to finish save)")
    t = time.monotonic()
    dist.barrier()
    _log(f"second dist.barrier() done  ({time.monotonic()-t:.1f}s)")

    if _is_rank_zero():
        # LOCAL SAVE ONLY: the HF model is fully materialized at hf_dir by
        # save_hf_pretrained above. There is no remote push; to share the model,
        # upload hf_dir to a Hugging Face repo yourself.
        _log(f"HF model saved locally at: {hf_dir}")
        _log(f"  reload with: transformers' from_pretrained('{hf_dir}', trust_remote_code=True)")

    _log("dist.destroy_process_group()")
    dist.destroy_process_group()
    _log("main() done")


if __name__ == "__main__":
    main()
