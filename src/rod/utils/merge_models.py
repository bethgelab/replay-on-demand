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

"""Linear weight-space merge of two HF checkpoints:  out = (1-w)*base + w*specialist.

Operates DIRECTLY on the safetensors weight files and never instantiates the HF model
class — so it needs no architecture-specific runtime deps (notably ``mamba_ssm`` for the
Nemotron-H hybrid, which the driver environment does not have). The output mirrors the base dir's
structure (same shard filenames + config + tokenizer + remote modeling code), so the
training driver loads the merged dir via Megatron-Bridge exactly like the base.

``w`` is the weight on the specialist:  w=0.0 -> base (low adapt/low forget),
w=1.0 -> specialist (high adapt/high forget). Interpolation is done in fp32 for precision,
then cast back to each tensor's original dtype. Runs single-process on CPU (I/O-bound).
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def _load_shard(path):
    tensors, meta = {}, {}
    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def _referenced_shards(model_dir):
    """The .safetensors shards a model ACTUALLY uses, per its *.index.json
    weight_map — so stale/unreferenced shard files left in the dir are ignored.

    This matters for HF bundles produced by convert_mcore_to_hf.py when the base
    template's shards are named differently from the converted output (e.g. the
    qwen base ships ``model.safetensors-00001-of-00004.safetensors`` while the
    convert writes ``model-00001-of-00004.safetensors``): the base copies then
    survive in the dir, and a blind ``glob('*.safetensors')`` load would sort
    them AFTER the real shards and silently overwrite the trained weights with
    base. The megatron/HF loader avoids this by loading via the index; so do we.

    Falls back to a sorted glob only when there is no index (single-file model).
    """
    import json

    idxs = sorted(glob.glob(os.path.join(model_dir, "*.index.json")))
    if idxs:
        with open(idxs[0]) as fh:
            wm = json.load(fh).get("weight_map", {})
        names = sorted(set(wm.values()))
        shards = [os.path.join(model_dir, n) for n in names]
        missing = [s for s in shards if not os.path.isfile(s)]
        if missing:
            raise FileNotFoundError(f"{model_dir} index {idxs[0]} references missing shards: {missing}")
        return shards
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors and no *.index.json in {model_dir}")
    return shards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="HF dir of the base model")
    ap.add_argument("--specialist", required=True, help="HF dir of the adapt-only specialist")
    ap.add_argument("--weight", type=float, required=True, help="w on specialist in [0,1]")
    ap.add_argument("--out", required=True, help="output HF dir for the merged model")
    a = ap.parse_args()
    if not 0.0 <= a.weight <= 1.0:
        raise ValueError(f"--weight must be in [0,1], got {a.weight}")
    w = float(a.weight)
    os.makedirs(a.out, exist_ok=True)

    base_shards = _referenced_shards(a.base)  # index-driven: ignores stale/unreferenced shard files

    # Specialist weights into one dict (base shard boundaries may differ from spec's).
    spec = {}
    spec_shards = _referenced_shards(a.specialist)  # index-driven: skips leftover base-named shards
    for sf in spec_shards:
        t, _ = _load_shard(sf)
        spec.update(t)
    print(f"[merge] loaded {len(spec)} specialist tensors from {len(spec_shards)} shards", flush=True)

    merged = skipped = 0
    for sf in base_shards:
        bt, meta = _load_shard(sf)
        for k in list(bt.keys()):
            st = spec.get(k)
            if st is not None and st.shape == bt[k].shape and bt[k].is_floating_point():
                dt = bt[k].dtype
                bt[k] = (bt[k].float() * (1.0 - w) + st.float() * w).to(dt)  # fp32 lerp -> orig dtype
                merged += 1
            else:
                skipped += 1  # non-float buffer / missing / shape mismatch -> keep base value
        meta.setdefault("format", "pt")
        save_file(bt, os.path.join(a.out, os.path.basename(sf)), metadata=meta)
        print(f"[merge] wrote shard {os.path.basename(sf)} ({len(bt)} tensors)", flush=True)
    print(f"[merge] w={w}: interpolated {merged} float tensors, kept {skipped} as base", flush=True)

    # Copy everything that isn't weights: config.json, *.index.json, tokenizer*, generation_config,
    # and the remote modeling code (*.py) — so the merged dir is a complete, loadable HF checkpoint.
    for fn in os.listdir(a.base):
        if fn.endswith(".safetensors"):
            continue
        src = os.path.join(a.base, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(a.out, fn))
    print(f"[merge] wrote merged HF (weights + config/tokenizer/modeling) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
