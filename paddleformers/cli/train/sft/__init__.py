# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from .workflow import run_sft

__all__ = ["run_sft", "run_sft_v2", "run_vl_sft_v2"]


def __getattr__(name):
    if name == "run_sft_v2":
        from .workflow2 import run_sft_v2

        return run_sft_v2
    if name == "run_vl_sft_v2":
        from .workflow_vl_v2 import run_vl_sft_v2

        return run_vl_sft_v2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
