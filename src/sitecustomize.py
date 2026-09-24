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

"""Interpreter-startup hook, imported automatically by ``site`` because ``src/`` is on PYTHONPATH.

Installs the Qwen3.5 compatibility shim (``rod.compat``) in every Python process of a run,
including the Ray policy workers and the checkpoint converter. Any ``sitecustomize`` that this
file shadows further down ``sys.path`` is still executed.
"""

import importlib.machinery
import importlib.util
import os
import sys

try:
    from rod.compat import install

    install()
except Exception as exc:  # never break interpreter startup
    print(f"[rod] compat hook not installed: {exc!r}", file=sys.stderr)

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.abspath(p or os.curdir) != _here])
if _spec is not None and _spec.loader is not None:
    _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
