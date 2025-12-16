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

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from paddlefleet.transformer.attention import SelfAttention, SelfAttentionSublayersSpec
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.spec_utils import LayerSpec
from paddlefleet.transformer.transformer_layer import TransformerLayer, TransformerLayerSublayersSpec

HAVE_TE = False

from paddlefleet.fusions.fused_layer_norm import FusedLayerNorm
from paddlefleet.transformer.paddle_norm import FusedRMSNorm

def get_layer_spec(is_vit, normalization) -> LayerSpec:
    """Transformer Layer Spec"""
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal
    if normalization == "LayerNorm":
        norm = FusedLayerNorm
    elif normalization == "RMSNorm":
        norm = FusedRMSNorm
    else:
        raise RuntimeError("unknown normalization", normalization)

    mlp = get_mlp_module_spec(use_te=False)  # doesn't include norm.

    return LayerSpec(
        layer=TransformerLayer,
        sublayers_spec=TransformerLayerSublayersSpec(
            input_layernorm=norm,
            self_attn=LayerSpec(
                layer=SelfAttention,
                extra_kwargs={"attn_mask_type": attn_mask_type},
                sublayers_spec=SelfAttentionSublayersSpec(
                    qkv_proj=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    o_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_attention_layernorm=norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_mlp_module_spec(use_te: bool = True) -> LayerSpec:
    """MLP Submodule Spec"""
    # Dense MLP w/ or w/o TE modules.
    return LayerSpec(
        layer=MLP,
        sublayers_spec=MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            down_proj=RowParallelLinear,
        ),
    )
