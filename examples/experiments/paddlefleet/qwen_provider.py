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

# Refer to NVIDIA Megatron-Bridge https://github.com/NVIDIA-NeMo/Megatron-Bridge
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import paddle
import paddle.nn.functional as F
from gpt_provider import GPTModelProvider
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

if TYPE_CHECKING:
    from paddlefleet.spec_utils import LayerSpec


logger = logging.getLogger(__name__)


@dataclass
class Qwen3MoEModelProvider(GPTModelProvider):
    """Base provider for Qwen 3 MoE Models."""

    transformer_layer_spec: Union[
        "LayerSpec", Callable[["GPTModelProvider"], "LayerSpec"]
    ] = get_gpt_decoder_block_spec

    normalization: str = "RMSNorm"
    hidden_act: Callable = F.silu
    gated_linear_unit: bool = True
    use_bias: bool = False
    attention_bias: bool = False
    use_qk_norm: bool = True
    seq_length: int = 40960
    max_position_embeddings: int = 40960
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-6
    autocast_dtype: paddle.dtype = paddle.bfloat16
    params_dtype: paddle.dtype = paddle.bfloat16
    bf16: bool = True

    # Attention
    num_key_value_heads: int = 8
    attention_dropout: float = 0.0
    head_dim: int = 128

    # Rope
    position_embedding_type: str = "rope"
    rotary_base: float = 1000000.0

    # MoE specific parameters
    n_routed_experts: int = 128
    moe_router_load_balancing_type: str = "aux_loss"
    router_aux_loss_coef: float = 1e-3
    num_experts_per_tok: int = 8
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"

    # optimization
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True


@dataclass
class Qwen3MoEModelProvider30B_A3B(Qwen3MoEModelProvider):
    """
    Provider for Qwen 3 30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
    """

    num_hidden_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    intermediate_size: int = 6144
    moe_intermediate_size: int = 768


@dataclass
class Qwen3MoEModelSingleCardProvider(Qwen3MoEModelProvider):
    """
    Provider for short Qwen 3 30B-A3B to debug
    """

    num_hidden_layers: int = 10
    num_attention_heads: int = 32
    hidden_size: int = 128
    intermediate_size: int = 128
    num_key_value_heads: int = 4
    num_nextn_predict_layers: Optional[int] = 0
    use_bias: bool = False
    vocab_size: int = 37888

    n_shared_experts: int = 1
    moe_intermediate_size: int = 1408
