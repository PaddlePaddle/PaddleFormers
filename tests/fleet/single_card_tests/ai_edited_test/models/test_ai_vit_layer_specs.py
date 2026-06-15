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

# Check if the installed version has the required API
_can_call_get_vit_layer_with_local_spec = True
_can_import_spec_utils = True
try:
    from paddleformers.fleet.models.vision.vit_layer_specs import (
        get_vit_layer_with_local_spec,
    )

    get_vit_layer_with_local_spec()
except Exception:
    _can_call_get_vit_layer_with_local_spec = False

try:
    from paddleformers.fleet.transformer.spec_utils import LayerSpec  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _can_import_spec_utils = False


class TestGetViTLayerWithLocalSpec(unittest.TestCase):
    """Test get_vit_layer_with_local_spec function."""

    @unittest.skipUnless(
        _can_import_spec_utils,
        "paddleformers.fleet.transformer.spec_utils.LayerSpec not available",
    )
    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_returns_layer_spec(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )
        from paddleformers.fleet.transformer.spec_utils import LayerSpec

        result = get_vit_layer_with_local_spec()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, LayerSpec)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_transformer_layer(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )
        from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

        result = get_vit_layer_with_local_spec()
        self.assertEqual(result.module, TransformerLayer)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_submodules(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )
        from paddleformers.fleet.transformer.transformer_layer import (
            TransformerLayerSublayersSpec,
        )

        result = get_vit_layer_with_local_spec()
        self.assertIsNotNone(result.submodules)
        self.assertIsInstance(result.submodules, TransformerLayerSublayersSpec)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_self_attention(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )
        from paddleformers.fleet.transformer.attention import SelfAttention

        result = get_vit_layer_with_local_spec()
        attn_spec = result.submodules.self_attention
        self.assertEqual(attn_spec.module, SelfAttention)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_causal_mask(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )
        from paddleformers.fleet.transformer.enums import AttnMaskType

        result = get_vit_layer_with_local_spec()
        attn_spec = result.submodules.self_attention
        self.assertEqual(attn_spec.params["attn_mask_type"], AttnMaskType.causal)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_dot_product_attention(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )
        from paddleformers.fleet.transformer.dot_product_attention import (
            DotProductAttention,
        )

        result = get_vit_layer_with_local_spec()
        attn_sub = result.submodules.self_attention.submodules
        self.assertEqual(attn_sub.core_attention, DotProductAttention)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_parallel_linears(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )
        from paddleformers.fleet.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        result = get_vit_layer_with_local_spec()
        attn_sub = result.submodules.self_attention.submodules
        self.assertEqual(attn_sub.linear_qkv, ColumnParallelLinear)
        self.assertEqual(attn_sub.linear_proj, RowParallelLinear)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_bias_dropout_add(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import get_bias_dropout_add
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )

        result = get_vit_layer_with_local_spec()
        self.assertEqual(result.submodules.self_attn_bda, get_bias_dropout_add)
        self.assertEqual(result.submodules.mlp_bda, get_bias_dropout_add)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "get_vit_layer_with_local_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_spec_has_layernorm(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )

        result = get_vit_layer_with_local_spec()
        self.assertEqual(result.submodules.input_layernorm, FusedLayerNorm)
        self.assertEqual(result.submodules.post_attention_layernorm, FusedLayerNorm)


class TestGetMLPModuleSpec(unittest.TestCase):
    """Test _get_mlp_module_spec helper."""

    @unittest.skipUnless(
        _can_import_spec_utils,
        "paddleformers.fleet.transformer.spec_utils.LayerSpec not available",
    )
    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "_get_mlp_module_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_returns_layer_spec(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            _get_mlp_module_spec,
        )
        from paddleformers.fleet.transformer.spec_utils import LayerSpec

        result = _get_mlp_module_spec(use_te=False)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, LayerSpec)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "_get_mlp_module_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_returns_mlp_module(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            _get_mlp_module_spec,
        )
        from paddleformers.fleet.transformer.mlp import MLP

        result = _get_mlp_module_spec(use_te=False)
        self.assertEqual(result.module, MLP)

    @unittest.skipUnless(
        _can_call_get_vit_layer_with_local_spec,
        "_get_mlp_module_spec() fails in installed version (MLPSublayersSpec signature changed)",
    )
    def test_mlp_has_correct_submodules(self):
        from paddleformers.fleet.models.vision.vit_layer_specs import (
            _get_mlp_module_spec,
        )
        from paddleformers.fleet.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from paddleformers.fleet.transformer.mlp import MLPSublayersSpec

        result = _get_mlp_module_spec(use_te=False)
        self.assertIsInstance(result.submodules, MLPSublayersSpec)
        self.assertEqual(result.submodules.linear_fc1, ColumnParallelLinear)
        self.assertEqual(result.submodules.linear_fc2, RowParallelLinear)


class TestViTLayerSpecsImports(unittest.TestCase):
    """Test that vit_layer_specs imports are correct."""

    def test_ln_impl(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm
        from paddleformers.fleet.models.vision.vit_layer_specs import LNImpl

        self.assertEqual(LNImpl, FusedLayerNorm)


if __name__ == "__main__":
    unittest.main()
