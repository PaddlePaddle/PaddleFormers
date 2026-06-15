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


class TestBackendSpecProviderProtocol(unittest.TestCase):
    """Test BackendSpecProvider as a Protocol."""

    def test_protocol_has_required_methods(self):
        from paddleformers.fleet.models.backends import BackendSpecProvider

        # Protocol defines abstract methods - verify they exist
        self.assertTrue(hasattr(BackendSpecProvider, "column_parallel_linear"))
        self.assertTrue(hasattr(BackendSpecProvider, "row_parallel_linear"))
        self.assertTrue(hasattr(BackendSpecProvider, "fuse_layernorm_and_linear"))
        self.assertTrue(hasattr(BackendSpecProvider, "column_parallel_layer_norm_linear"))
        self.assertTrue(hasattr(BackendSpecProvider, "layer_norm"))
        self.assertTrue(hasattr(BackendSpecProvider, "core_attention"))
        self.assertTrue(hasattr(BackendSpecProvider, "grouped_mlp_layers"))
        self.assertTrue(hasattr(BackendSpecProvider, "hidden_act"))


class TestLocalSpecProviderLinear(unittest.TestCase):
    """Test LocalSpecProvider.linear method."""

    def test_linear_returns_linear(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider
        from paddleformers.fleet.tensor_parallel.layers import Linear

        provider = LocalSpecProvider()
        self.assertEqual(provider.linear(), Linear)


class TestLocalSpecProviderColumnParallelLinear(unittest.TestCase):
    """Test LocalSpecProvider.column_parallel_linear method."""

    def test_returns_column_parallel_linear(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider
        from paddleformers.fleet.tensor_parallel.layers import ColumnParallelLinear

        provider = LocalSpecProvider()
        self.assertEqual(provider.column_parallel_linear(), ColumnParallelLinear)


class TestLocalSpecProviderRowParallelLinear(unittest.TestCase):
    """Test LocalSpecProvider.row_parallel_linear method."""

    def test_returns_row_parallel_linear(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider
        from paddleformers.fleet.tensor_parallel.layers import RowParallelLinear

        provider = LocalSpecProvider()
        self.assertEqual(provider.row_parallel_linear(), RowParallelLinear)


class TestLocalSpecProviderFuseLayernormAndLinear(unittest.TestCase):
    """Test LocalSpecProvider.fuse_layernorm_and_linear method."""

    def test_returns_false(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider

        provider = LocalSpecProvider()
        self.assertFalse(provider.fuse_layernorm_and_linear())


class TestLocalSpecProviderColumnParallelLayerNormLinear(unittest.TestCase):
    """Test LocalSpecProvider.column_parallel_layer_norm_linear method."""

    def test_returns_none(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider

        provider = LocalSpecProvider()
        self.assertIsNone(provider.column_parallel_layer_norm_linear())


class TestLocalSpecProviderLayerNorm(unittest.TestCase):
    """Test LocalSpecProvider.layer_norm method."""

    def test_non_rms_norm_returns_ln_impl(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider
        from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm

        provider = LocalSpecProvider()
        result = provider.layer_norm(rms_norm=False)
        self.assertEqual(result, WrappedPaddleNorm)

    def test_rms_norm_returns_wrapped_paddle_norm(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider
        from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm

        provider = LocalSpecProvider()
        result = provider.layer_norm(rms_norm=True)
        self.assertEqual(result, WrappedPaddleNorm)


class TestLocalSpecProviderCoreAttention(unittest.TestCase):
    """Test LocalSpecProvider.core_attention method."""

    def test_returns_dot_product_attention(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider
        from paddleformers.fleet.transformer.dot_product_attention import (
            DotProductAttention,
        )

        provider = LocalSpecProvider()
        result = provider.core_attention()
        self.assertEqual(result, DotProductAttention)


class TestLocalSpecProviderGroupedMLPLayers(unittest.TestCase):
    """Test LocalSpecProvider.grouped_mlp_layers method."""

    def test_grouped_gemm_returns_grouped_mlp(self):
        from paddleformers.fleet.models.backends import GroupedMLP, LocalSpecProvider

        provider = LocalSpecProvider()
        layer, spec = provider.grouped_mlp_layers(moe_use_grouped_gemm=True, moe_use_legacy_grouped_gemm=False)
        self.assertEqual(layer, GroupedMLP)
        self.assertIsNone(spec)

    def test_non_grouped_gemm_returns_sequential_mlp(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider, SequentialMLP
        from paddleformers.fleet.transformer.mlp import MLPSublayersSpec

        provider = LocalSpecProvider()
        layer, spec = provider.grouped_mlp_layers(moe_use_grouped_gemm=False, moe_use_legacy_grouped_gemm=False)
        self.assertEqual(layer, SequentialMLP)
        self.assertIsNotNone(spec)
        self.assertIsInstance(spec, MLPSublayersSpec)


class TestLocalSpecProviderHiddenAct(unittest.TestCase):
    """Test LocalSpecProvider.hidden_act method."""

    def test_returns_none(self):
        from paddleformers.fleet.models.backends import LocalSpecProvider

        provider = LocalSpecProvider()
        self.assertIsNone(provider.hidden_act())


class TestGroupedMLPAndSequentialMLPClasses(unittest.TestCase):
    """Test placeholder GroupedMLP and SequentialMLP classes."""

    def test_grouped_mlp_exists(self):
        from paddleformers.fleet.models.backends import GroupedMLP

        # These are placeholder classes in backends.py
        self.assertTrue(callable(GroupedMLP))

    def test_sequential_mlp_exists(self):
        from paddleformers.fleet.models.backends import SequentialMLP

        self.assertTrue(callable(SequentialMLP))


if __name__ == "__main__":
    unittest.main()
