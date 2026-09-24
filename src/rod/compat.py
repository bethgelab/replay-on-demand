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

"""Compatibility shim for Qwen3.5 on the pinned NeMo-RL v0.6.0 stack.

Megatron-Bridge builds the Qwen3.5 vision tower from the Hugging Face vision config and reads
``hf_config.deepstack_visual_indexes`` (``qwen_vl/modelling_qwen3_vl/transformer_config.py``),
but the Qwen3.5 vision configs in transformers 5.3.0 do not define it, so building a Qwen3.5
model fails unless the checkpoint's ``config.json`` happens to carry the key. Qwen3.5 has no
deepstack mergers, so the missing value is an empty list.

The patch has to run in every process that builds the model (the Ray policy workers and the
checkpoint converter), not just the driver, so it is installed at interpreter startup by
``src/sitecustomize.py`` as a lazy post-import hook: it touches nothing until transformers
imports the config module, and leaves the class alone if the attribute already exists.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys

# module -> (class, attribute, default)
QWEN35_PATCHES = {
    "transformers.models.qwen3_5.configuration_qwen3_5": ("Qwen3_5VisionConfig", "deepstack_visual_indexes", []),
    "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe": ("Qwen3_5MoeVisionConfig", "deepstack_visual_indexes", []),
}


def _apply(module, cls_name: str, attr: str, default) -> None:
    cls = getattr(module, cls_name, None)
    if cls is not None and not hasattr(cls, attr):
        setattr(cls, attr, default)


class _PatchingLoader(importlib.abc.Loader):
    """Delegates to the real loader, then sets the missing class attribute."""

    def __init__(self, loader, patch):
        self._loader, self._patch = loader, patch

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _apply(module, *self._patch)

    def __getattr__(self, name):          # get_source, get_resource_reader, ...
        return getattr(self._loader, name)


class PostImportPatcher(importlib.abc.MetaPathFinder):
    def __init__(self, patches: dict):
        self._patches = patches

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._patches:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchingLoader(spec.loader, self._patches[fullname])
        return spec


def install(patches: dict = QWEN35_PATCHES) -> None:
    """Register the hook (idempotent); modules imported already are patched immediately."""
    for name, patch in patches.items():
        if name in sys.modules:
            _apply(sys.modules[name], *patch)
    if not any(isinstance(f, PostImportPatcher) and f._patches is patches for f in sys.meta_path):
        sys.meta_path.insert(0, PostImportPatcher(patches))
