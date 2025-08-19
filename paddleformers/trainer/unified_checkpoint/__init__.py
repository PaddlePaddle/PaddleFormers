# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ...utils.lazy_import import _LazyModule

import_structure = {
    "unified_checkpoint": [
        "load_state_dict",
        "unwrap_model",
        "dtype_byte_size",
        "empty_device_cache",
        "nested_copy",
        "strtobool",
        "distributed_file",
        "distributed_isfile",
        "check_unified_checkpoint",
        "check_unified_optimizer",
        "load_unified_checkpoint_dynamically",
        "load_unified_optimizer_dynamically",
        "load_unified_checkpoint_locally",
        "load_unified_optimizer_locally",
        "load_single_card_checkpoint",
        "load_single_card_optimizer",
        "save_single_card_checkpoint",
        "save_single_card_optimizer",
        "gather_splited_param_for_optimizer",
        "load_non_merge_optimizer_with_split_param",
        "filter_params",
        "filter_sync_parameters",
        "gather_sharded_object",
        "generate_base_static_name",
        "get_expected_state_dict",
        "get_sharded_file_name",
        "get_sharded_index",
        "is_need_master_weight",
        "is_sharding_split_param_mode",
        "merge_tensor_parallel_for_optimizer",
        "merge_tensor_parallel_with_shard",
        "reduce_master_weights_status",
        "rename_shard_file",
        "save_model_config",
        "unified_checkpoint_into_shards",
        "unified_optimizer_into_shards",
        "UnifiedCheckpointHandler",
    ],
    "lora_model": ["LoRAModel"],
    "prefix_model": ["PrefixModelForCausalLM"],
    "model_utils": ["PretrainedModel"],
    "async_handler": ["AsyncCheckpointHandler"],
    "shared_memory_utils": [],
    "utils": [],
    "check_completion": [],
    "load_dynamic": [],
    "sharding_split_param_utils": [],
    "load_local": [],
    "load_save_single_card": [],
}

if TYPE_CHECKING:
    from .unified_checkpoint import UnifiedCheckpointHandler
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
