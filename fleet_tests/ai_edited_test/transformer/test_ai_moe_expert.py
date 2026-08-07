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
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.transformer.moe.moe_expert import (
    BMMFunction,
    GroupedMLPExpert,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "moe_intermediate_size": 128,
        "moe_deep_gemm": False,
        "use_bias": False,
        "gated_linear_unit": True,
        "hidden_act": paddle.nn.functional.silu,
        "rms_norm_eps": 1e-5,
        "fp8": False,
        "recompute_granularity": None,
        "recompute_modules": None,
        "using_sonic_moe": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestBMMFunctionDetailed(unittest.TestCase):
    """Detailed tests for BMMFunction."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_multiple_batch_sizes(self):
        """Test BMMFunction with 2D lhs as required by batched_gemm."""
        # batched_gemm expects lhs to be 2D [total_seq_len, input_hidden_size]
        x = paddle.randn([8, 64], dtype="float32")
        y = paddle.randn([8, 64, 32], dtype="float32")
        batch_sizes = [1, 1, 1, 1, 1, 1, 1, 1]

        out = BMMFunction.apply(x, y, batch_sizes, trans_y=False)
        self.assertEqual(list(out.shape), [8, 32])

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_preserves_dtype(self):
        """Test BMMFunction preserves float32 dtype."""
        x = paddle.randn([4, 64], dtype="float32")
        y = paddle.randn([4, 64, 32], dtype="float32")
        batch_sizes = [1, 1, 1, 1]

        out = BMMFunction.apply(x, y, batch_sizes, trans_y=False)
        self.assertEqual(out.dtype, paddle.float32)


class TestGroupedMLPExpertActivationRecompute(unittest.TestCase):
    """Tests for GroupedMLPExpert with activation recompute."""

    def test_activation_recompute_with_fp8_raises(self):
        """Test that activation_recompute with fp8 raises ValueError."""
        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["moe_act"],
            fp8=True,
        )
        with self.assertRaises(ValueError):
            GroupedMLPExpert(
                num_local_experts=2,
                config=config,
                moe_deep_gemm=False,
            )

    def test_no_glu_with_silu(self):
        """Test construction without GLU but with silu activation."""
        config = _make_config(
            gated_linear_unit=False,
            hidden_act=paddle.nn.functional.silu,
        )
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        # weight1 shape should be [2, 64, 128] (no doubling)
        self.assertEqual(expert.weight1.shape[0], 2)
        self.assertEqual(expert.weight1.shape[1], 64)
        self.assertEqual(expert.weight1.shape[2], 128)


class TestGroupedMLPExpertShardedStateDict(unittest.TestCase):
    """Tests for GroupedMLPExpert sharded_state_dict."""

    def test_sharded_state_dict_without_ep_group(self):
        """Test sharded_state_dict without ep_group."""
        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        expert.ep_group = None
        sd = expert.sharded_state_dict()
        self.assertIn("weight1", sd)
        self.assertIn("weight2", sd)


class TestGroupedMLPExpertWithPGCollection(unittest.TestCase):
    """Tests for GroupedMLPExpert with process group collection."""

    def test_construction_with_ep_group(self):
        """Test construction with ep_group set."""
        config = _make_config()
        mock_pg = MagicMock()
        mock_pg.ep = MagicMock()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg,
        )
        self.assertIsNotNone(expert.ep_group)


if __name__ == "__main__":
    unittest.main()
