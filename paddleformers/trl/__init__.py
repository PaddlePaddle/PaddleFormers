# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you smay not use this file except in compliance with the License.
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
    "dpo_auto_trainer": ["DPOAutoTrainer", "disable_dropout_in_model", "prepare_pipeline_dpo_inputs_func"],
    "dpo_trainer": ["disable_dropout_in_model", "DPOTrainer", "prepare_pipeline_dpo_inputs_func"],
    "embedding_trainer": ["EmbeddingTrainer"],
    "kto_trainer": ["disable_dropout_in_model", "KTOTrainer", "prepare_pipeline_dpo_inputs_func"],
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
    "model_config": ["ModelConfig"],
    "quant_config": ["QuantConfig"],
    "sft_auto_trainer": ["SFTAutoTrainer"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "sftdata_config": ["DataConfig"],
    "trl_data": [
        "check_preference_data",
        "preprocess_preference_data",
        "preference_collate_fn",
        "preference_collate_fn_auto_parallel",
    ],
    "trl_utils": ["calculate_effective_tokens"],
    "utils": ["ScriptArguments"],
}

from ..transformers.dpo_criterion import AutoDPOCriterion, DPOCriterion
from ..transformers.kto_criterion import KTOCriterion

if TYPE_CHECKING:
    from .dpo_auto_trainer import DPOAutoTrainer
    from .dpo_trainer import DPOTrainer
    from .embedding_trainer import EmbeddingTrainer
    from .kto_trainer import KTOTrainer
    from .model_config import *
    from .quant_config import *
    from .sft_auto_trainer import *
    from .sft_config import *
    from .sft_trainer import *
    from .sftdata_config import *
    from .trl_data import *
    from .trl_utils import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
        extra_objects={
            "AutoDPOCriterion": AutoDPOCriterion,
            "DPOCriterion": DPOCriterion,
            "KTOCriterion": KTOCriterion,
        },
    )
