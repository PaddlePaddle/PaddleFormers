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
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

import_structure = {
    "hparams": [
        "DataArguments",
        "ModelArguments",
        "GeneratingArguments",
        "FinetuningArguments",
        "ExportArguments",
        "ServerArguments",
        "get_train_args",
        "get_eval_args",
        "get_server_args",
        "get_export_args",
        "read_args",
    ],
    "train": [],
    "export": [],
    "utils": [
        "terminate_process_tree",
        "is_env_enabled",
        "is_valid_model_dir",
        "detect_device",
        "set_ascend_environment",
        "remove_paddle_shm_files",
        "set_cuda_environment",
    ],
    "cli": [],
    "launcher": [],
}

if TYPE_CHECKING:
    from .cli import *
    from .export import *
    from .hparams import *
    from .launcher import *
    from .train import *
    from .utils import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
