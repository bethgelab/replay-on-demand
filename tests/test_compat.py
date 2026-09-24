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

"""Unit tests for the Qwen3.5 compatibility hook (no transformers needed).

Run: pytest tests/test_compat.py
"""
import os
import subprocess
import sys

from rod import compat

SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src")


def _fake_package(tmp_path, monkeypatch):
    pkg = tmp_path / "fakecfg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "plain.py").write_text("class VisionConfig:\n    pass\n")
    (pkg / "has_attr.py").write_text("class VisionConfig:\n    deepstack_visual_indexes = [8, 16]\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in ("fakecfg", "fakecfg.plain", "fakecfg.has_attr"):
        monkeypatch.delitem(sys.modules, m, raising=False)


def test_missing_attribute_is_set_on_import_and_existing_one_kept(tmp_path, monkeypatch):
    _fake_package(tmp_path, monkeypatch)
    patches = {"fakecfg.plain": ("VisionConfig", "deepstack_visual_indexes", []),
               "fakecfg.has_attr": ("VisionConfig", "deepstack_visual_indexes", [])}
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    compat.install(patches)
    compat.install(patches)                                   # idempotent
    assert sum(isinstance(f, compat.PostImportPatcher) and f._patches is patches for f in sys.meta_path) == 1

    import fakecfg.has_attr
    import fakecfg.plain
    assert fakecfg.plain.VisionConfig().deepstack_visual_indexes == []
    assert fakecfg.has_attr.VisionConfig.deepstack_visual_indexes == [8, 16]


def test_already_imported_module_is_patched_immediately(tmp_path, monkeypatch):
    _fake_package(tmp_path, monkeypatch)
    import fakecfg.plain
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    compat.install({"fakecfg.plain": ("VisionConfig", "deepstack_visual_indexes", [])})
    assert fakecfg.plain.VisionConfig.deepstack_visual_indexes == []


def test_sitecustomize_installs_the_hook_at_startup():
    env = dict(os.environ, PYTHONPATH=os.path.abspath(SRC))
    code = ("import sys; from rod.compat import PostImportPatcher; "
            "print(any(isinstance(f, PostImportPatcher) for f in sys.meta_path))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True", out.stderr
