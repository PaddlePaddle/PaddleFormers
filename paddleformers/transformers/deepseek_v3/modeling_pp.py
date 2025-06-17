# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from ..deepseek_v2.modeling_pp import DeepseekV2ForCausalLMPipe
from .configuration import DeepseekV3Config
from .modeling import DeepseekV3PretrainedModel

__all__ = [
    "DeepseekV3ForCausalLMPipe",
]


class DeepseekV3ForCausalLMPipe(DeepseekV2ForCausalLMPipe):
    """DeepseekV2ForPretraining adapted for pipeline parallelism.

    The largest change is flattening the DeepseekV2Model class so we can express it as a
    sequence of layers including embedding, transformer layers, and output.
    """

    config_class = DeepseekV3Config
    _base_model = DeepseekV3PretrainedModel
    _get_tensor_parallel_mappings = DeepseekV3PretrainedModel._get_tensor_parallel_mappings
    _init_weights = DeepseekV3PretrainedModel._init_weights
    _keys_to_ignore_on_load_unexpected = DeepseekV3PretrainedModel._keys_to_ignore_on_load_unexpected
    _get_model_flops = DeepseekV3PretrainedModel._get_model_flops
    _get_hardware_flops = DeepseekV3PretrainedModel._get_hardware_flops
    _tied_weights_keys = ["lm_head.weight"]
    base_model_prefix = DeepseekV3PretrainedModel.base_model_prefix
