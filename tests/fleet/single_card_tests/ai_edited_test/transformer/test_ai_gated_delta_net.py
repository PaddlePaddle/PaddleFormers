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
from unittest.mock import MagicMock, patch

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
    _l2norm,
    paddle_chunk_gated_delta_rule,
)
from paddleformers.fleet.transformer.paddle_norm import RMSNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class FakeInProj(paddle.nn.Layer):
    """Fake in_proj that returns (output, bias) like ColumnParallelLinear."""

    def __init__(self, in_f, out_f, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_f, out_f)
        self.skip_bias_add = kwargs.get("skip_bias_add", False)

    def forward(self, x, *args, **kwargs):
        out = self.linear(x)
        bias = self.linear.bias if not self.skip_bias_add else None
        return out, bias


class FakeOutProj(paddle.nn.Layer):
    """Fake out_proj that returns (output, bias) like RowParallelLinear."""

    def __init__(self, in_f, out_f, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_f, out_f)
        self.skip_bias_add = kwargs.get("skip_bias_add", False)

    def forward(self, x, *args, **kwargs):
        out = self.linear(x)
        bias = self.linear.bias if not self.skip_bias_add else None
        return out, bias


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "perform_initialization": True,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestL2Norm(unittest.TestCase):
    """Test _l2norm helper function."""

    def test_output_shape(self):
        x = paddle.randn([2, 4, 8])
        out = _l2norm(x)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_output_dtype(self):
        x = paddle.randn([2, 4, 8], dtype="float32")
        out = _l2norm(x)
        self.assertEqual(out.dtype, paddle.float32)

    def test_normalization_magnitude(self):
        x = paddle.ones([1, 1, 4]) * 2.0
        out = _l2norm(x)
        # Each element should be close to 1/sqrt(sum(4*4)=16) * 2 = 2/4 = 0.5
        expected = 2.0 / (16.0**0.5)
        for val in out.flatten().tolist():
            self.assertAlmostEqual(val, expected, places=4)

    def test_zero_input(self):
        x = paddle.zeros([1, 1, 4])
        out = _l2norm(x)
        # With eps, zero input should not produce NaN
        self.assertFalse(paddle.isnan(out).any())


class TestPaddleChunkGatedDeltaRule(unittest.TestCase):
    """Test paddle_chunk_gated_delta_rule function."""

    def test_output_shapes(self):
        batch, heads, seq, k_dim, v_dim = 1, 2, 8, 4, 8
        query = paddle.randn([batch, seq, heads, k_dim])
        key = paddle.randn([batch, seq, heads, k_dim])
        value = paddle.randn([batch, seq, heads, v_dim])
        g = paddle.randn([batch, seq, heads])
        beta = paddle.randn([batch, seq, heads])

        out, state = paddle_chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        self.assertEqual(out.shape, [batch, seq, heads, v_dim])
        self.assertIsNone(state)  # output_final_state=False

    def test_output_final_state(self):
        batch, heads, seq, k_dim, v_dim = 1, 2, 8, 4, 8
        query = paddle.randn([batch, seq, heads, k_dim])
        key = paddle.randn([batch, seq, heads, k_dim])
        value = paddle.randn([batch, seq, heads, v_dim])
        g = paddle.randn([batch, seq, heads])
        beta = paddle.randn([batch, seq, heads])

        out, state = paddle_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, output_final_state=True)
        self.assertIsNotNone(state)
        self.assertEqual(state.shape, [batch, heads, k_dim, v_dim])

    def test_with_initial_state(self):
        batch, heads, seq, k_dim, v_dim = 1, 2, 8, 4, 8
        query = paddle.randn([batch, seq, heads, k_dim])
        key = paddle.randn([batch, seq, heads, k_dim])
        value = paddle.randn([batch, seq, heads, v_dim])
        g = paddle.randn([batch, seq, heads])
        beta = paddle.randn([batch, seq, heads])
        initial = paddle.randn([batch, heads, k_dim, v_dim])

        out, state = paddle_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=initial)
        self.assertEqual(out.shape, [batch, seq, heads, v_dim])

    def test_custom_chunk_size(self):
        batch, heads, seq, k_dim, v_dim = 1, 2, 4, 4, 8
        query = paddle.randn([batch, seq, heads, k_dim])
        key = paddle.randn([batch, seq, heads, k_dim])
        value = paddle.randn([batch, seq, heads, v_dim])
        g = paddle.randn([batch, seq, heads])
        beta = paddle.randn([batch, seq, heads])

        out, state = paddle_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, chunk_size=2)
        self.assertEqual(out.shape, [batch, seq, heads, v_dim])

    def test_preserves_dtype(self):
        batch, heads, seq, k_dim, v_dim = 1, 1, 4, 4, 8
        query = paddle.randn([batch, seq, heads, k_dim], dtype="float32")
        key = paddle.randn([batch, seq, heads, k_dim], dtype="float32")
        value = paddle.randn([batch, seq, heads, v_dim], dtype="float32")
        g = paddle.randn([batch, seq, heads], dtype="float32")
        beta = paddle.randn([batch, seq, heads], dtype="float32")

        out, _ = paddle_chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        self.assertEqual(out.dtype, paddle.float32)

    def test_use_qk_l2norm_in_kernel(self):
        batch, heads, seq, k_dim, v_dim = 1, 1, 4, 4, 8
        query = paddle.randn([batch, seq, heads, k_dim])
        key = paddle.randn([batch, seq, heads, k_dim])
        value = paddle.randn([batch, seq, heads, v_dim])
        g = paddle.randn([batch, seq, heads])
        beta = paddle.randn([batch, seq, heads])

        out, _ = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertEqual(out.shape, [batch, seq, heads, v_dim])


class TestGatedDeltaNetSublayersSpec(unittest.TestCase):
    """Test GatedDeltaNetSublayersSpec dataclass."""

    def test_defaults(self):
        spec = GatedDeltaNetSublayersSpec()
        self.assertIsNotNone(spec.in_proj)
        self.assertIsNotNone(spec.out_norm)
        self.assertIsNotNone(spec.out_proj)

    def test_custom_values(self):
        spec = GatedDeltaNetSublayersSpec(
            in_proj=MagicMock(),
            out_norm=MagicMock(),
            out_proj=MagicMock(),
        )
        self.assertIsNotNone(spec.in_proj)


class TestGatedDeltaNetConstruction(unittest.TestCase):
    """Test GatedDeltaNet construction."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_basic_construction(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(hidden_act=F.silu, hidden_size=64)
        spec = GatedDeltaNetSublayersSpec()

        gdn = GatedDeltaNet(
            config,
            spec,
            layer_number=1,
            num_key_heads=2,
            num_value_heads=4,
            key_head_dim=16,
            value_head_dim=16,
        )
        self.assertIsNotNone(gdn.in_proj)
        self.assertIsNotNone(gdn.conv1d)
        self.assertIsNotNone(gdn.out_norm)
        self.assertIsNotNone(gdn.out_proj)
        self.assertEqual(gdn.num_key_heads, 2)
        self.assertEqual(gdn.num_value_heads, 4)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_invalid_a_init_range_raises(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(hidden_act=F.silu)
        spec = GatedDeltaNetSublayersSpec()
        with self.assertRaises(AssertionError):
            GatedDeltaNet(
                config,
                spec,
                A_init_range=(-1, 2),  # negative start
            )


class TestGatedDeltaNetForward(unittest.TestCase):
    """Test GatedDeltaNet forward pass."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_forward_shape(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        # in_proj_dim = qk_dim*2 + v_dim*2 + num_value_heads*2
        # With key_head_dim=8, num_key_heads=2, value_head_dim=8, num_value_heads=2:
        # qk_dim=16, v_dim=16, in_proj_dim = 32 + 32 + 4 = 68
        config = _make_config(
            hidden_act=F.silu,
            hidden_size=68,
            perform_initialization=False,
        )
        spec = GatedDeltaNetSublayersSpec(
            in_proj=FakeInProj,
            out_norm=RMSNorm,
            out_proj=FakeOutProj,
        )

        gdn = GatedDeltaNet(
            config,
            spec,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=8,
            value_head_dim=8,
        )
        x = paddle.randn([2, 8, 68], dtype="float32")
        # GatedDeltaNet expects 2D mask [batch, seq_len] (1=valid, 0=pad)
        mask = paddle.ones([2, 8], dtype="float32")
        out, bias = gdn(x, mask)
        self.assertEqual(out.shape, [2, 8, 68])

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_packed_seq_raises(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(
            hidden_act=F.silu,
            hidden_size=64,
            perform_initialization=False,
        )
        spec = GatedDeltaNetSublayersSpec()

        gdn = GatedDeltaNet(
            config,
            spec,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=16,
            value_head_dim=16,
        )
        x = paddle.randn([2, 8, 64], dtype="float32")
        with self.assertRaises(NotImplementedError):
            gdn(x, None, packed_seq_params="fake")


class TestGatedDeltaNetResetParameters(unittest.TestCase):
    """Test reset_parameters method."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_reset_does_not_raise(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(
            hidden_act=F.silu,
            perform_initialization=True,
        )
        spec = GatedDeltaNetSublayersSpec()
        gdn = GatedDeltaNet(
            config,
            spec,
            conv_init=0.01,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=16,
            value_head_dim=16,
        )
        gdn.reset_parameters()
        # Should not raise


class TestGatedDeltaNetShardedStateDict(unittest.TestCase):
    """Test sharded_state_dict method."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_returns_empty_when_import_fails(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(
            hidden_act=F.silu,
            perform_initialization=False,
        )
        spec = GatedDeltaNetSublayersSpec()
        gdn = GatedDeltaNet(
            config,
            spec,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=16,
            value_head_dim=16,
        )

        # Patch to make import fail
        with patch.dict(
            "sys.modules",
            {"paddle.distributed.flex_checkpoint.dcp.sharded_weight": None},
        ):
            result = gdn.sharded_state_dict()
            self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
