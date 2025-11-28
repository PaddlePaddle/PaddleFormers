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
class GLMMoEModelProvider(GPTModelProvider):
    """Base provider for GLM MoE Models."""

    transformer_layer_spec: Union[
        "LayerSpec", Callable[["GPTModelProvider"], "LayerSpec"]
    ] = get_gpt_decoder_block_spec

    normalization: str = "RMSNorm"
    act_fn: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    seq_length: int = 131072
    init_method_std: int = 0.02
    hidden_dropout_prob: float = 0.0
    vocab_size: int = 151552
    share_embeddings_and_output_weights: Optional[bool] = False
    rms_norm_eps: float = 1e-5
    autocast_dtype: paddle.dtype = paddle.bfloat16
    params_dtype: paddle.dtype = paddle.bfloat16
    bf16: bool = True

    # Attention
    num_key_value_heads: int = 8
    num_attention_heads: int = 96
    attention_dropout: float = 0.0
    head_dim: int = 128

    # RoPE
    position_embedding_type: str = "rope"
    rotary_base: float = 1000000.0
    rotary_percent: float = 0.5

    # MoE specific parameters
    moe_router_topk: int = 8
    moe_shared_expert_overlap: bool = True
    moe_token_dispatcher_type: str = "deepep"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_router_score_function: str = "sigmoid"
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0

    # optimization
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True

    # MTP
    num_nextn_predict_layers: Optional[int] = 1
    mtp_loss_scaling_factor: Optional[
        float
    ] = 0.3  # https://arxiv.org/pdf/2508.06471 0.3 for the first 15T tokens, 0.1 for the remaining tokens.


@dataclass
class GLM45ModelProvider355B(GLMMoEModelProvider):
    """
    Provider for GLM 4.5 355B-A32B: https://huggingface.co/zai-org/GLM-4.5
    """

    num_hidden_layers: int = 92
    moe_num_experts: int = 160
    hidden_size: int = 5120
    intermediate_size: int = 12288
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] * 3 + [1] * 89
    )  # first three layers are dense
    moe_ffn_hidden_size: int = 1536
    moe_shared_expert_intermediate_size: int = 1536
    use_qk_norm: bool = True
    moe_router_topk_scaling_factor: float = 2.5


@dataclass
class GLM45AirModelProvider106B(GLMMoEModelProvider):
    """
    Provider for GLM 4.5 Air 106B-A12B: https://huggingface.co/zai-org/GLM-4.5-Air
    """

    num_hidden_layers: int = 46
    moe_num_experts: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 10944
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 45
    )  # first one layer is dense
    moe_ffn_hidden_size: int = 1408
    moe_shared_expert_intermediate_size: int = 1408
    use_qk_norm: bool = False
    moe_router_topk_scaling_factor: float = 1.0


@dataclass
class GLM45AirModelDebugProvider(GLM45AirModelProvider106B):
    """
    Provider for GLM 4.5 Air 106B-A12B: https://huggingface.co/zai-org/GLM-4.5-Air
    """

    num_hidden_layers: int = 10
    moe_num_shared_experts: int = 1
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 9
    )  # first one layer is dense
    hidden_size: int = 128
    intermediate_size: int = 128
    moe_intermediate_size: int = 1408
    num_nextn_predict_layers: Optional[int] = 0
    use_bias: bool = False
    vocab_size: int = 37888


@dataclass
class GLM45AirModelSingleCardDebugProvider(GLMMoEModelProvider):
    """
    Provider for GLM 4.5 Air 106B-A12B: https://huggingface.co/zai-org/GLM-4.5-Air
    """

    use_bias: bool = False
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    moe_aux_loss_coeff: float = 1e-4
    num_key_value_heads: int = 8
    seq_length: int = 8192
    num_experts_per_tok: int = 4
    hidden_size: int = 512
    act_fn: Callable = F.silu
    intermediate_size: int = 1024
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 1
    )  # first one layer is dense
    moe_num_experts: int = 8
    topk_method: str = "noaux_tc"

    moe_intermediate_size: int = 1408
    moe_shared_expert_intermediate_size: int = 1408
    use_qk_norm: bool = False
    moe_router_topk_scaling_factor: float = 1.0
    num_nextn_predict_layers: Optional[int] = 0

    transpose_gate_weight: bool = True
