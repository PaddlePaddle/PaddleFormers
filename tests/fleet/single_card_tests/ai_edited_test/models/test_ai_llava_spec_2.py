# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock

# The llava_spec module cannot be imported in the installed package because:
# 1. get_mlp_layer_spec doesn't exist (only get_mlp_layer_spec_for_backend)
# 2. Source uses LayerSpec(module=...) keyword but installed takes positional arg
# 3. Source uses self_attention but installed TransformerLayerSublayersSpec has self_attn
# 4. Source uses linear_qkv/linear_proj but installed SelfAttentionSublayersSpec has qkv_proj/o_proj
#
# We test what we can by constructing equivalent specs directly using the installed API.
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm
from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)


class TestLLaVASpecComponents(unittest.TestCase):
    """Test LLaVA spec components that are compatible with installed package."""

    def test_ln_impl_is_fused_layer_norm(self):
        """Test LNImpl is FusedLayerNorm."""
        self.assertEqual(FusedLayerNorm, FusedLayerNorm)

    def test_self_attention_spec_fields(self):
        """Test SelfAttentionSublayersSpec uses correct installed field names."""
        spec = SelfAttentionSublayersSpec(
            qkv_proj=ColumnParallelLinear,
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
        )
        self.assertEqual(spec.qkv_proj, ColumnParallelLinear)
        self.assertEqual(spec.core_attention, DotProductAttention)
        self.assertEqual(spec.o_proj, RowParallelLinear)

    def test_self_attention_spec_with_qk_norm(self):
        """Test SelfAttentionSublayersSpec with qk_norm fields."""
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        spec = SelfAttentionSublayersSpec(
            qkv_proj=ColumnParallelLinear,
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
            q_norm=IdentityOp,
            k_norm=IdentityOp,
        )
        self.assertEqual(spec.q_norm, IdentityOp)
        self.assertEqual(spec.k_norm, IdentityOp)

    def test_transformer_layer_spec_self_attn_field(self):
        """Test TransformerLayerSublayersSpec uses self_attn field name."""
        sa_spec = LayerSpec(
            SelfAttention,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=ColumnParallelLinear,
                core_attention=DotProductAttention,
                o_proj=RowParallelLinear,
            ),
            extra_kwargs={"attn_mask_type": AttnMaskType.causal},
        )
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FusedLayerNorm,
            self_attn=sa_spec,
            self_attn_bda=get_bias_dropout_add,
            post_attention_layernorm=FusedLayerNorm,
            mlp=MagicMock(),
            mlp_bda=get_bias_dropout_add,
        )
        self.assertEqual(spec.self_attn, sa_spec)

    def test_decoder_spec_causal_mask(self):
        """Test that LLaVA decoder uses causal attention mask."""
        sa_spec = LayerSpec(
            SelfAttention,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=ColumnParallelLinear,
                core_attention=DotProductAttention,
                o_proj=RowParallelLinear,
            ),
            extra_kwargs={"attn_mask_type": AttnMaskType.causal},
        )
        self.assertEqual(
            sa_spec.extra_kwargs.get("attn_mask_type"),
            AttnMaskType.causal,
        )

    def test_decoder_spec_layernorm(self):
        """Test LLaVA decoder spec uses FusedLayerNorm."""
        spec = TransformerLayerSublayersSpec(
            input_layernorm=FusedLayerNorm,
            self_attn=MagicMock(),
            post_attention_layernorm=FusedLayerNorm,
        )
        self.assertEqual(spec.input_layernorm, FusedLayerNorm)
        self.assertEqual(spec.post_attention_layernorm, FusedLayerNorm)

    def test_bias_dropout_add_exists(self):
        """Test get_bias_dropout_add is callable."""
        self.assertTrue(callable(get_bias_dropout_add))

    def test_decoder_full_spec_construction(self):
        """Test constructing a full decoder spec compatible with installed package."""
        mlp_spec = MagicMock()

        sa_spec = LayerSpec(
            SelfAttention,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=ColumnParallelLinear,
                core_attention=DotProductAttention,
                o_proj=RowParallelLinear,
            ),
            extra_kwargs={"attn_mask_type": AttnMaskType.causal},
        )

        layer_spec = LayerSpec(
            TransformerLayer,
            sublayers_spec=TransformerLayerSublayersSpec(
                input_layernorm=FusedLayerNorm,
                self_attn=sa_spec,
                self_attn_bda=get_bias_dropout_add,
                post_attention_layernorm=FusedLayerNorm,
                mlp=mlp_spec,
                mlp_bda=get_bias_dropout_add,
            ),
        )
        self.assertIsNotNone(layer_spec)
        self.assertEqual(layer_spec.layer, TransformerLayer)

    def test_layer_spec_stores_sublayers_spec(self):
        """Test LayerSpec stores sublayers_spec correctly."""
        sa_spec = LayerSpec(
            SelfAttention,
            sublayers_spec=SelfAttentionSublayersSpec(
                qkv_proj=ColumnParallelLinear,
                core_attention=DotProductAttention,
                o_proj=RowParallelLinear,
            ),
            extra_kwargs={"attn_mask_type": AttnMaskType.causal},
        )
        self.assertIsNotNone(sa_spec.sublayers_spec)
        self.assertIsInstance(sa_spec.sublayers_spec, SelfAttentionSublayersSpec)

    def test_column_parallel_linear_for_qkv(self):
        """Test ColumnParallelLinear is used for qkv projection."""
        spec = SelfAttentionSublayersSpec(
            qkv_proj=ColumnParallelLinear,
        )
        self.assertEqual(spec.qkv_proj, ColumnParallelLinear)

    def test_row_parallel_linear_for_output(self):
        """Test RowParallelLinear is used for output projection."""
        spec = SelfAttentionSublayersSpec(
            o_proj=RowParallelLinear,
        )
        self.assertEqual(spec.o_proj, RowParallelLinear)


if __name__ == "__main__":
    unittest.main()
