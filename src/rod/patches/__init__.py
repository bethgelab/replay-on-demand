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

"""RoD runtime patches for public NeMo-RL, plus the loss / scoring modules they use.

- ``rod_collate`` / ``rod_validate`` / ``rod_sft_train``: monkey-patches applied by
  the entry points via the ``apply_*`` functions below.
- ``rod_loss`` / ``rod_score_pass`` / ``rod_scoring_loss``: plain modules (the shared
  loss formula, the per-step scoring pass, the validation loss fn).
"""

from rod.patches.rod_collate import apply as apply_rod_collate
from rod.patches.rod_scoring_loss import RHOScoringLossFn
from rod.patches.rod_sft_train import (
    apply as apply_rod_sft_train,
    set_runtime as set_rod_sft_train_runtime,
)
from rod.patches.rod_validate import apply as apply_rod_validate

__all__ = [
    "apply_rod_collate",
    "apply_rod_sft_train",
    "apply_rod_validate",
    "RHOScoringLossFn",
    "set_rod_sft_train_runtime",
]
