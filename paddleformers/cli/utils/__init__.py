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

import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "process": [
        "terminate_process_tree",
        "is_env_enabled",
        "is_valid_model_dir",
        "detect_device",
        "set_ascend_environment",
        "remove_paddle_shm_files",
        "set_cuda_environment",
    ],
}

if TYPE_CHECKING:
    from .process import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
