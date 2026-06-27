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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.transformer.paddle_norm import (
    FusedRMSNorm,
    L2Norm,
    LayerNorm,
    RMSNorm,
    RMSNormTriton,
    WrappedPaddleNorm,
    WrappedPaddleNormPipe,
    WrappedRMSNormTriton,
    get_norm_extra_args,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestRMSNormForwardDetailed(unittest.TestCase):
    """Detailed tests for RMSNorm forward."""

    def test_forward_returns_correct_dtype(self):
        """Test RMSNorm forward returns correct dtype."""
        config = _make_config(params_dtype="float32")
        norm = RMSNorm(config=config)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.dtype, paddle.float32)

    def test_forward_with_custom_normalized_shape(self):
        """Test RMSNorm with custom normalized_shape."""
        config = _make_config()
        norm = RMSNorm(config=config, normalized_shape=64)
        x = paddle.randn([2, 4, 64])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])

    def test_forward_with_custom_eps(self):
        """Test RMSNorm with custom eps."""
        config = _make_config()
        norm = RMSNorm(config=config, norm_eps=1e-6)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_enable_sequence_parallel(self):
        """Test RMSNorm enable_sequence_parallel."""
        config = _make_config()
        norm = RMSNorm(config=config, input_is_parallel=True)
        # Should not raise
        self.assertIsNotNone(norm.weight)


class TestLayerNormForwardDetailed(unittest.TestCase):
    """Detailed tests for LayerNorm forward."""

    def test_forward_basic(self):
        """Test LayerNorm forward basic."""
        config = _make_config(normalization="LayerNorm")
        norm = LayerNorm(config=config)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_with_custom_normalized_shape(self):
        """Test LayerNorm with custom normalized_shape."""
        config = _make_config(normalization="LayerNorm")
        norm = LayerNorm(config=config, normalized_shape=64)
        x = paddle.randn([2, 4, 64])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])

    def test_has_bias(self):
        """Test LayerNorm has bias parameter."""
        config = _make_config(normalization="LayerNorm")
        norm = LayerNorm(config=config)
        self.assertIsNotNone(norm.bias)


class TestFusedRMSNormDetailed(unittest.TestCase):
    """Detailed tests for FusedRMSNorm."""

    def test_forward_basic(self):
        """Test FusedRMSNorm forward basic."""
        config = _make_config()
        norm = FusedRMSNorm(config=config)
        x = paddle.randn([2, 4, 128])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 128])


class TestL2NormDetailed(unittest.TestCase):
    """Detailed tests for L2Norm."""

    def test_forward_basic(self):
        """Test L2Norm forward basic."""
        norm = L2Norm(hidden_size=64)
        x = paddle.randn([2, 4, 64])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])

    def test_forward_normalizes(self):
        """Test L2Norm normalizes the input (mean of squared values ~ 1)."""
        norm = L2Norm(hidden_size=64)
        x = paddle.randn([2, 4, 64])
        out = norm(x)
        # L2Norm normalizes so that mean of squared values along last dim is ~1
        mean_sq = out.float().pow(2).mean(-1)
        self.assertTrue(
            paddle.allclose(mean_sq, paddle.ones_like(mean_sq), atol=0.1).item()
        )

    def test_custom_eps(self):
        """Test L2Norm with custom eps."""
        norm = L2Norm(hidden_size=64, eps=1e-8)
        x = paddle.randn([2, 4, 64])
        out = norm(x)
        self.assertEqual(out.shape, [2, 4, 64])


class TestWrappedPaddleNormPipe(unittest.TestCase):
    """Tests for WrappedPaddleNormPipe."""

    def test_forward_basic(self):
        """Test WrappedPaddleNormPipe forward basic."""
        config = _make_config()
        norm_pipe = WrappedPaddleNormPipe(config=config, hidden_size=128)
        dict_args = {"hidden_states": paddle.randn([2, 4, 128])}
        result = norm_pipe(dict_args)
        self.assertIn("hidden_states", result)
        self.assertEqual(result["hidden_states"].shape, [2, 4, 128])

    def test_forward_with_mtp_enabled(self):
        """Test WrappedPaddleNormPipe with MTP layers enabled."""
        config = _make_config(
            num_nextn_predict_layers=2, mtp_load_weight_only=False
        )
        norm_pipe = WrappedPaddleNormPipe(config=config, hidden_size=128)
        # hidden_states is concatenated: [main, mtp_0, mtp_1]
        dict_args = {
            "hidden_states": paddle.randn([3, 4, 128]),
        }
        result = norm_pipe(dict_args)
        self.assertIn("hidden_states", result)
        self.assertEqual(result["hidden_states"].shape, [3, 4, 128])

    def test_build_schedule_node(self):
        """Test build_schedule_node returns ScheduleNode."""
        config = _make_config()
        norm_pipe = WrappedPaddleNormPipe(config=config, hidden_size=128)
        node = norm_pipe.build_schedule_node()
        self.assertIsNotNone(node)


class TestGetNormExtraArgs(unittest.TestCase):
    """Tests for get_norm_extra_args."""

    def test_with_wrapped_paddle_norm(self):
        """Test with WrappedPaddleNorm."""
        config = _make_config()
        extra_args = get_norm_extra_args(
            WrappedPaddleNorm, config, 128, 1e-5, False
        )
        self.assertIn("hidden_size", extra_args)
        self.assertIn("eps", extra_args)
        self.assertEqual(extra_args["hidden_size"], 128)

    def test_with_layer_spec(self):
        """Test with LayerSpec."""
        config = _make_config()
        layer_spec = LayerSpec(WrappedPaddleNorm)
        extra_args = get_norm_extra_args(layer_spec, config, 128, 1e-5, False)
        self.assertIn("hidden_size", extra_args)

    def test_with_other_norm(self):
        """Test with other norm class."""
        config = _make_config()
        extra_args = get_norm_extra_args(RMSNorm, config, 128, 1e-5, False)
        self.assertIn("normalized_shape", extra_args)
        self.assertIn("norm_eps", extra_args)
        self.assertEqual(extra_args["normalized_shape"], 128)


class TestWrappedRMSNormTriton(unittest.TestCase):
    """Tests for WrappedRMSNormTriton."""

    def test_construction(self):
        """Test WrappedRMSNormTriton construction."""
        config = _make_config()
        norm = WrappedRMSNormTriton(config=config, hidden_size=64, eps=1e-6)
        self.assertIsInstance(norm, RMSNormTriton)


if __name__ == "__main__":
    unittest.main()
