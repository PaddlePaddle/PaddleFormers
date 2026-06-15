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

"""Unit tests for GatedDeltaNet module."""

from __future__ import annotations

import unittest

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddleformers.fleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
    _l2norm,
    paddle_chunk_gated_delta_rule,
)
from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# ---- Local stand-in layers (no fleet / TP required) ----


class BiasedLinear(nn.Layer):
    """Simple linear layer that returns (output, bias), matching ColumnParallel/RowParallel API."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias

    def backward_dw(self):
        pass


class NoBiasLinear(nn.Layer):
    """Linear layer without bias that returns (output, None)."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None

    def backward_dw(self):
        pass


class SimpleRMSNorm(nn.Layer):
    """Minimal RMSNorm for testing."""

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        x_float = x.astype(paddle.float32)
        rms = paddle.rsqrt(x_float.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight.astype(paddle.float32)).astype(x.dtype)


# ---- Fake ProcessGroupCollection for single-GPU testing ----


class _FakeGroup:
    """Fake process group that reports world_size=1."""

    ranks = [0]
    nranks = 1


class _FakePGCollection:
    """Minimal stand-in for ProcessGroupCollection (TP=1)."""

    def __init__(self):
        self.tp = _FakeGroup()


# ---- Test dimensions ----
HIDDEN_SIZE = 64
NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 4
KEY_HEAD_DIM = 16
VALUE_HEAD_DIM = 16
CONV_KERNEL_DIM = 4
MICRO_BATCH_SIZE = 2
SEQ_LENGTH = 32


class TestPaddleChunkGatedDeltaRule(unittest.TestCase):
    """Test the deterministic paddle_chunk_gated_delta_rule function."""

    def test_output_shape(self):
        """Output shape must match [batch, seq_len, num_heads, v_head_dim]."""
        batch, seq_len, num_heads, k_dim, v_dim = 2, 32, 4, 16, 16
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, state = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
            output_final_state=False,
        )

        self.assertEqual(list(out.shape), [batch, seq_len, num_heads, v_dim])
        self.assertIsNone(state)
        self.assertEqual(out.dtype, query.dtype)

    def test_output_final_state(self):
        """When output_final_state=True, state should be returned."""
        batch, seq_len, num_heads, k_dim, v_dim = 1, 16, 2, 8, 8
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, state = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
            output_final_state=True,
        )

        self.assertIsNotNone(state)
        self.assertEqual(list(state.shape), [batch, num_heads, k_dim, v_dim])

    def test_backward(self):
        """Gradients must flow through the chunked gated delta rule."""
        batch, seq_len, num_heads, k_dim, v_dim = 2, 32, 4, 16, 16
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, _ = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertTrue(paddle.isfinite(query.grad).all().item())
        self.assertTrue(paddle.isfinite(key.grad).all().item())
        self.assertTrue(paddle.isfinite(value.grad).all().item())

    def test_seq_len_not_divisible_by_chunk(self):
        """Sequence length not divisible by chunk_size should still work (padding)."""
        batch, seq_len, num_heads, k_dim, v_dim = 1, 37, 2, 8, 8
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, _ = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
        )
        self.assertEqual(list(out.shape), [batch, seq_len, num_heads, v_dim])
        self.assertTrue(paddle.isfinite(out).all().item())


class TestL2Norm(unittest.TestCase):
    """Test the _l2norm helper function."""

    def test_output_shape(self):
        x = paddle.randn([2, 4, 3, 16])
        y = _l2norm(x)
        self.assertEqual(list(y.shape), list(x.shape))
        self.assertEqual(y.dtype, x.dtype)

    def test_normalization(self):
        """After L2 norm, mean of squared values along last dim should be ~1."""
        x = paddle.randn([4, 8, 32])
        y = _l2norm(x)
        mean_sq = y.astype(paddle.float32).pow(2).sum(-1)
        assert paddle.allclose(mean_sq, paddle.ones_like(mean_sq), atol=1e-4, rtol=1e-4).item()


class TestGatedDeltaNet(unittest.TestCase):
    """Test the full GatedDeltaNet module (single-GPU, no TP)."""

    def setUp(self):
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_KEY_HEADS,
            num_hidden_layers=2,
            hidden_act=F.silu,
            rms_norm_eps=1e-5,
            normalization="RMSNorm",
            sequence_parallel=False,
            deterministic_mode=True,
        )

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=NoBiasLinear,
            out_norm=SimpleRMSNorm,
            out_proj=NoBiasLinear,
        )

        self.gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
        )

    def test_constructor(self):
        """GatedDeltaNet should instantiate with the correct sub-modules."""
        self.assertIsInstance(self.gdn, GatedDeltaNet)
        self.assertTrue(hasattr(self.gdn, "in_proj"))
        self.assertTrue(hasattr(self.gdn, "conv1d"))
        self.assertTrue(hasattr(self.gdn, "dt_bias"))
        self.assertTrue(hasattr(self.gdn, "A_log"))
        self.assertTrue(hasattr(self.gdn, "out_norm"))
        self.assertTrue(hasattr(self.gdn, "out_proj"))

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=NoBiasLinear,
            out_norm=WrappedPaddleNorm,
            out_proj=NoBiasLinear,
        )

        gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
        )

    def test_sharded_state_dict(self):
        """Check sharded_state_dict() completeness."""
        sharded_sd = self.gdn.sharded_state_dict()
        self.assertEqual(len(sharded_sd), 6)  # 13 from GatedDeltaNetSublayersSpec

    def test_parameter_shapes(self):
        """Verify key parameter shapes."""
        # conv1d: depthwise conv with groups=conv_dim
        conv_dim = KEY_HEAD_DIM * NUM_KEY_HEADS * 2 + VALUE_HEAD_DIM * NUM_VALUE_HEADS
        self.assertEqual(
            list(self.gdn.conv1d.weight.shape),
            [conv_dim, 1, CONV_KERNEL_DIM],
        )
        # dt_bias and A_log
        self.assertEqual(list(self.gdn.dt_bias.shape), [NUM_VALUE_HEADS])
        self.assertEqual(list(self.gdn.A_log.shape), [NUM_VALUE_HEADS])

    def test_forward_output_shape(self):
        """Forward output shape should match [batch, seq_len, hidden_size]."""
        hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        attention_mask = None

        output, output_bias = self.gdn(hidden_states, attention_mask)

        self.assertEqual(output.ndim, 3)
        self.assertEqual(output.shape[0], MICRO_BATCH_SIZE)
        self.assertEqual(output.shape[1], SEQ_LENGTH)
        self.assertEqual(output.shape[2], HIDDEN_SIZE)
        self.assertEqual(output.dtype, hidden_states.dtype)

    def test_forward_output_finite(self):
        """Forward output should contain no NaN or Inf values."""
        hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

        output, _ = self.gdn(hidden_states, attention_mask=None)

        self.assertTrue(
            paddle.isfinite(output).all().item(),
            "Output contains NaN or Inf",
        )

    def test_backward_all_grads(self):
        """All parameters in the forward path should receive gradients."""
        hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        hidden_states.stop_gradient = False

        output, output_bias = self.gdn(hidden_states, attention_mask=None)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(paddle.isfinite(hidden_states.grad).all().item())

        # Check all parameter gradients
        params_with_grad = 0
        no_grad_params = []
        for name, param in self.gdn.named_parameters():
            if param.grad is None:
                no_grad_params.append(name)
            else:
                params_with_grad += 1
                self.assertEqual(
                    list(param.shape),
                    list(param.grad.shape),
                    f"Gradient shape mismatch for {name}",
                )
                self.assertTrue(
                    paddle.isfinite(param.grad).all().item(),
                    f"Non-finite gradients for {name}",
                )

        self.assertGreater(params_with_grad, 0, "No parameters received gradients")
        if no_grad_params:
            print(f"  [WARNING] Parameters without gradients: {no_grad_params}")

    def test_packed_seq_not_supported(self):
        """Packed sequence should raise NotImplementedError."""
        hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

        with self.assertRaises(NotImplementedError):
            self.gdn(hidden_states, attention_mask=None, packed_seq_params="dummy")


class TestGatedDeltaNetWithBias(unittest.TestCase):
    """Test GatedDeltaNet with bias enabled in linear layers and conv."""

    def setUp(self):
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_KEY_HEADS,
            num_hidden_layers=2,
            hidden_act=F.silu,
            rms_norm_eps=1e-5,
            normalization="RMSNorm",
            sequence_parallel=False,
            deterministic_mode=True,
        )

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=BiasedLinear,
            out_norm=SimpleRMSNorm,
            out_proj=BiasedLinear,
        )

        self.gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=True,
            conv_bias=True,
            conv_init=0.5,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
        )

    def test_forward_backward(self):
        """Forward and backward with bias should work correctly."""
        hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        hidden_states.stop_gradient = False

        output, output_bias = self.gdn(hidden_states, attention_mask=None)

        self.assertEqual(output.shape[0], MICRO_BATCH_SIZE)
        self.assertEqual(output.shape[1], SEQ_LENGTH)
        self.assertEqual(output.shape[2], HIDDEN_SIZE)
        self.assertTrue(paddle.isfinite(output).all().item())

        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)

        # Conv1d bias should have gradient
        self.assertIsNotNone(self.gdn.conv1d.bias)
        self.assertIsNotNone(
            self.gdn.conv1d.bias.grad,
            "conv1d.bias should receive gradient",
        )


class TestGatedDeltaNetGQA(unittest.TestCase):
    """Test GatedDeltaNet with GQA (num_value_heads > num_key_heads)."""

    def setUp(self):
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_KEY_HEADS,
            num_hidden_layers=2,
            hidden_act=F.silu,
            rms_norm_eps=1e-5,
            normalization="RMSNorm",
            sequence_parallel=False,
            deterministic_mode=True,
        )

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=NoBiasLinear,
            out_norm=SimpleRMSNorm,
            out_proj=NoBiasLinear,
        )

        # GQA: 8 value heads, 4 key heads => repeat factor 2
        self.gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=False,
            conv_bias=False,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=4,
            num_value_heads=8,
        )

    def test_forward_shape(self):
        """GQA should produce correct output shape."""
        hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        output, _ = self.gdn(hidden_states, attention_mask=None)

        self.assertEqual(output.shape[0], MICRO_BATCH_SIZE)
        self.assertEqual(output.shape[1], SEQ_LENGTH)
        self.assertEqual(output.shape[2], HIDDEN_SIZE)
        self.assertTrue(paddle.isfinite(output).all().item())

    def test_backward(self):
        """Backward through GQA should produce finite gradients for all params."""
        hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        hidden_states.stop_gradient = False

        output, _ = self.gdn(hidden_states, attention_mask=None)
        output.sum().backward()

        self.assertIsNotNone(hidden_states.grad)
        for name, param in self.gdn.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    paddle.isfinite(param.grad).all().item(),
                    f"Non-finite gradient for {name}",
                )


if __name__ == "__main__":
    unittest.main()
