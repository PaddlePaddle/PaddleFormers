# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from abc import Callable

from dataclasses import dataclass
import paddle.nn.functional as F
from paddlefleet.spec_utils import LayerSpec

from paddlefleet.transformer.transformer_config import TransformerConfig

from .vision import Qwen3VisionModel

@dataclass
class Qwen3VLVisionProvider(TransformerConfig):
    """Qwen3VL Vidion Model Configuration."""
    patch_dim: int = 16,
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304,
    embed_dim: int = 1152,
    hidden_size: int = 1152,
    out_hidden_size: int = 4096,
    in_channels: int = 3,
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = True
    activation_func: Callable = F.gelu
    num_key_value_heads: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = None
    model_version: str = "qwen3_vl"
    
    def provide(self) -> "Qwen3VisionModel":
        pass
