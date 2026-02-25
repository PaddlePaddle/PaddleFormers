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
        "set_env_if_empty",
    ],
    "llm_utils": [
        "compute_metrics",
        "get_prefix_tuning_params",
        "get_lora_target_modules",
        "pad_batch_data",
        "dybatch_preprocess",
        "load_real_time_tokens",
        "init_chat_template",
        "get_model_max_position_embeddings",
        "read_res",
        "read_res_dynamic_insert",
        "speculate_read_res",
        "get_rotary_position_embedding",
        "init_dist_env",
        "get_eos_token_id",
        "set_triton_cache",
    ],
    "mllm_utils": [
        "freeze_model_parameters",
        "get_multimodel_lora_target_modules",
    ],
}

if TYPE_CHECKING:
    from .llm_utils import *
    from .mllm_utils import *
    from .process import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
