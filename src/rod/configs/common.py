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

import logging
import warnings
from typing import Any, NotRequired, get_origin, get_type_hints

import nemo_rl
import nemo_rl.data
import nemo_rl.distributed.virtual_cluster
import nemo_rl.models.policy
import nemo_rl.utils.checkpoint
import nemo_rl.utils.logger
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from omegaconf import OmegaConf
from pydantic import BaseModel as _BaseModel

log = logging.getLogger(__name__)

# Our _validate below dispatches sub-config validation with strict=, but
# pydantic v2's deprecated BaseModel.validate doesn't accept that kwarg.
# Patch it: drop strict (and any other unknown kwargs), forward to the
# original method for correctness, then model_dump() so downstream dict
# indexing into nested pydantic-typed config blocks keeps working.
_orig_bm_validate = _BaseModel.validate.__func__


def _bm_validate_compat(cls, *args, **kwargs):
    kwargs.pop("strict", None)
    return _orig_bm_validate(cls, *args, **kwargs).model_dump()


_BaseModel.validate = classmethod(_bm_validate_compat)


def _validate(klass: type, data: dict[Any, Any], strict: bool = False) -> Any:
    """Validate a config dict against this TypedDict schema.

    nemo_rl configs are TypedDicts, which carry no runtime validation. This method
    fills that gap by checking required fields and flagging unexpected keys. Validation
    recurses into any nested field whose type is itself a verifiable config,
    so a single call on MasterConfig covers the full config tree.

    Args:
        data: Dict to validate against this class's TypedDict schema.
        strict: If True, raise on unexpected keys instead of warning.

    Returns:
        A new instance of the calling config class constructed from the validated data.

    Raises:
        ValueError: If any required field is missing, listing all missing fields.
        ValueError: If strict=True and unexpected keys are present.
    """
    try:
        hints = get_type_hints(klass)
    except Exception:
        hints = {}
        for base in reversed(klass.__mro__):
            hints.update(vars(base).get("__annotations__", {}))

    # REMARK: about this try/except
    # get_type_hints() resolves all annotations — including string forward references — by evaluating them against the
    #  module's __globals__. One of the nemo_rl nested TypedDicts has a field annotated with something like os.PathLike but
    # its defining module never imports os, so evaluation raises NameError. The fallback collects __annotations__
    # directly from each class in the MRO without evaluating them — it just reads the raw annotation values as stored.
    # For fields where the annotation is already a real type object (most of them), the recursion check
    # isinstance(hint, type) and hasattr(hint, "validate") works normally. For the problematic field, the annotation
    # stays as an unevaluated string like "os.PathLike" — isinstance(str, type) is False, so that field is skipped instead
    # of crashing. The tradeoff: we lose recursive validation into that one field, but that's acceptable since nemo_rl owns it and we can't fix its imports.

    known = set(hints)
    # Required keys, minus any that are actually optional. `from __future__ import
    # annotations` (PEP 563) stringizes a class's annotations, which can make a
    # TypedDict miscount NotRequired[...] keys as required in __required_keys__ — so
    # an ABSENT NotRequired key (e.g. `data_mixer` on MasterConfig) gets
    # falsely flagged as missing. Re-derive optionality from the resolved hints so
    # that never happens (harmless for classes whose keys are already correct).
    required = set(getattr(klass, "__required_keys__", frozenset()))
    optional = set(getattr(klass, "__optional_keys__", frozenset()))
    try:
        for _k, _h in get_type_hints(klass, include_extras=True).items():
            if get_origin(_h) is NotRequired:
                optional.add(_k)
    except Exception:
        pass
    required -= optional

    missing = sorted(required - data.keys())
    extra = sorted(data.keys() - known)

    if missing:
        fields = "\n".join(f"  - {f}" for f in missing)
        raise ValueError(f"{klass.__name__}: missing required fields:\n{fields}")
    if extra:
        msg = f"{klass.__name__}: unexpected fields: {extra}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)

    result = dict(data)
    for key, hint in hints.items():
        if key in result and isinstance(result[key], dict) and isinstance(hint, type) and hasattr(hint, "validate"):
            result[key] = hint.validate(result[key], strict=strict)

    return klass(**result)


def verifiable(cls: type) -> type:
    cls.validate = classmethod(_validate)  # type: ignore[attr-defined]
    return cls


@verifiable
class PolicyConfig(nemo_rl.models.policy.PolicyConfig):
    pass


@verifiable
class DataConfig(nemo_rl.data.DataConfig):
    pass


@verifiable
class ClusterConfig(nemo_rl.distributed.virtual_cluster.ClusterConfig):
    pass


@verifiable
class CheckpointingConfig(nemo_rl.utils.checkpoint.CheckpointingConfig):
    pass


@verifiable
class LoggerConfig(nemo_rl.utils.logger.LoggerConfig):
    pass


def build_config(cls: type, yaml_path: str, cli_overrides: list[str]) -> dict:
    """Load a YAML, apply CLI overrides, validate against ``cls`` schema.

    ``cls`` must be a MasterConfig class with a ``from_overrides(overrides_dict)`` classmethod that
    merges onto defaults (loaded internally via DEFAULT_PARAMS_FILE) and
    validates the merged result.
    """
    overrides = load_config(yaml_path)
    if cli_overrides:
        # Apply CLI overrides to the experiment merged onto the defaults, so any key of the
        # full config can be overridden, not only those the experiment YAML sets.
        log.info("CLI overrides: %s", cli_overrides)
        overrides = OmegaConf.merge(cls.get_default_params_dict(), overrides)
        overrides = parse_hydra_overrides(overrides, cli_overrides)
    return cls.from_overrides(
        OmegaConf.to_container(overrides, resolve=True),
    )
