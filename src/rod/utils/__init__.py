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

import os


def get_job_name() -> str:
    """Run name from ``$MY_JOB_NAME`` (set by the launch scripts), else ``rod-run``.
    Used as the wandb run name and as the log / checkpoint directory under the log root."""
    return os.environ.get("MY_JOB_NAME", "rod-run")
