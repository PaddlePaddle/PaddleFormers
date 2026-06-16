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
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec
from paddle.nn import functional as F

from ...transformer import TransformerConfig
from .qwen3_vl_builders import qwen3_vl_vision_builder
from .qwen3_vl_model import Qwen3VLVisionModel, Qwen3VLVisionTransformerLayer


@dataclass
class Qwen3VLVisionProvider(TransformerConfig):
    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304
    embed_dim: int = (1152,)
    hidden_size: int = 1152
    out_hidden_size: int = 4096
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = Qwen3VLVisionTransformerLayer
    model_version: str = "qwen3_vl"
    img_h: int = 336
    img_w: int = 336
    add_class_token: bool = False
    class_token_len: int = 1
    high_precision_rope: bool = True
    rotary_percent: float = 1.0
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
    }

    def provide(self) -> "Qwen3VLVisionModel":
        pp_size = self.pipeline_model_parallel_size

        is_pipeline_asymmetric = getattr(
            self, "account_for_embedding_in_pipeline_split", False
        ) or getattr(self, "account_for_loss_in_pipeline_split", False)
        is_pipeline_asymmetric |= (
            getattr(self, "num_empty_layers_add_in_head", None)
            or getattr(self, "num_empty_layers_add_in_tail", None)
        ) is not None

        # Initialize model as meta data instead of allocating data on a device
        model_init_device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            model_init_device_context = partial(paddle.device, device="meta")

        with model_init_device_context():
            res_model = qwen3_vl_vision_builder(
                self,
                seg_method="layer:TransformerLayer|EmptyLayer",
                num_stages=pp_size,
            )
        return res_model
