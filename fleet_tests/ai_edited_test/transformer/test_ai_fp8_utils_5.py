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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.moe.fp8_utils import (
    FP8_ALIGN,
    ExpertsGroupGemmContiguousNode,
    _get_fp8_weight_and_scale,
    fused_stack_quant,
    has_config,
)


class TestFP8Align(unittest.TestCase):
    """Test FP8_ALIGN constant."""

    def test_fp8_align_value(self):
        self.assertEqual(FP8_ALIGN, 128)


class TestHasConfig(unittest.TestCase):
    """Test has_config helper function."""

    def test_none_config(self):
        self.assertFalse(has_config(None, "key"))

    def test_empty_config(self):
        self.assertFalse(has_config({}, "key"))

    def test_key_missing(self):
        self.assertFalse(has_config({"other": True}, "key"))

    def test_key_present_truthy(self):
        self.assertTrue(has_config({"key": True}, "key"))

    def test_key_present_falsy(self):
        self.assertFalse(has_config({"key": False}, "key"))

    def test_key_present_zero(self):
        self.assertFalse(has_config({"key": 0}, "key"))

    def test_key_present_none(self):
        self.assertFalse(has_config({"key": None}, "key"))


class TestGetFP8WeightAndScale(unittest.TestCase):
    """Test _get_fp8_weight_and_scale function."""

    def test_transpose_true(self):
        weight = MagicMock()
        weight.fp8_weight_stacked_transpose = paddle.randn([4, 4])
        weight.fp8_scale_stacked_transpose = paddle.randn([4])
        w, s = _get_fp8_weight_and_scale(weight, transpose=True)
        self.assertEqual(w.shape, [4, 4])
        self.assertEqual(s.shape, [4])

    def test_transpose_false(self):
        weight = MagicMock()
        weight.fp8_weight_stacked = paddle.randn([4, 4])
        weight.fp8_scale_stacked = paddle.randn([4])
        w, s = _get_fp8_weight_and_scale(weight, transpose=False)
        self.assertEqual(w.shape, [4, 4])
        self.assertEqual(s.shape, [4])


class TestFusedStackQuant(unittest.TestCase):
    """Test fused_stack_quant with cached weights."""

    @patch("paddleformers.fleet.transformer.moe.fp8_utils._get_fp8_weight_and_scale")
    def test_uses_cached_non_transpose(self, mock_get):
        mock_get.return_value = (paddle.randn([4, 4]), paddle.randn([4]))
        weight = MagicMock()
        weight.fp8_weight_stacked = True
        result = fused_stack_quant([weight], transpose=False)
        mock_get.assert_called_once_with(
            weight, transpose=False, num_expert=None, use_ue8m0=False
        )
        self.assertEqual(len(result), 2)

    @patch("paddleformers.fleet.transformer.moe.fp8_utils._get_fp8_weight_and_scale")
    def test_uses_cached_transpose(self, mock_get):
        mock_get.return_value = (paddle.randn([4, 4]), paddle.randn([4]))
        weight = MagicMock()
        weight.fp8_weight_stacked_transpose = True
        result = fused_stack_quant([weight], transpose=True)
        mock_get.assert_called_once_with(
            weight, transpose=True, num_expert=None, use_ue8m0=False
        )
        self.assertEqual(len(result), 2)

    @patch("paddleformers.fleet.transformer.moe.fp8_utils._get_fp8_weight_and_scale")
    @patch(
        "paddleformers.fleet.transformer.moe.fp8_utils.fused_stack_quant_without_cache"
    )
    def test_fallback_to_non_transpose_cache(
        self, mock_without_cache, mock_get
    ):
        """Only fp8_weight_stacked_transpose set (no fp8_weight_stacked):
        cache path is NOT entered, falls through to fused_stack_quant_without_cache."""
        mock_without_cache.return_value = (
            paddle.randn([4, 4]),
            paddle.randn([4]),
        )
        weight = MagicMock()
        del weight.fp8_weight_stacked
        weight.fp8_weight_stacked_transpose = True
        result = fused_stack_quant([weight], transpose=False)
        mock_get.assert_not_called()
        mock_without_cache.assert_called_once()

    @patch("paddleformers.fleet.transformer.moe.fp8_utils._get_fp8_weight_and_scale")
    def test_fallback_to_transpose_cache(self, mock_get):
        """fp8_weight_stacked set but no fp8_weight_stacked_transpose,
        transpose=True: enters cache path, _get_fp8_weight_and_scale handles
        on-the-fly transpose internally."""
        mock_get.return_value = (paddle.randn([4, 4]), paddle.randn([4]))
        weight = MagicMock(spec=["fp8_weight_stacked"])
        weight.fp8_weight_stacked = True
        result = fused_stack_quant([weight], transpose=True)
        mock_get.assert_called_once_with(
            weight, transpose=True, num_expert=None, use_ue8m0=False
        )


class TestExpertsGroupGemmContiguousNode(unittest.TestCase):
    """Test ExpertsGroupGemmContiguousNode."""

    def test_construction_default(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(custom_map)
        self.assertEqual(node.expert_id, None)
        self.assertFalse(node.recompute_moe_gate_up)
        self.assertFalse(node.dequant_input)
        self.assertTrue(node.use_fp8_mlp)
        self.assertFalse(node.moe_deep_gemm)

    def test_construction_with_expert_id(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(custom_map, expert_id=1)
        self.assertEqual(len(node.experts), 1)

    def test_construction_grouped_gemm(self):
        custom_map = MagicMock()
        custom_map.grouped_gemm_experts = MagicMock()
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            moe_expert_fusion=True,
            use_fp8_mlp=False,
        )
        self.assertTrue(hasattr(node, "grouped_gemm_experts"))

    def test_cached_tensors(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(custom_map)
        tensors = node.cached_tensors()
        self.assertEqual(len(tensors), 6)

    def test_set_and_clear_cached_tensors(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(custom_map)
        node.set_cached_tensors([None, None, None, None, None, None])
        self.assertIsNone(node.tokens_per_expert)
        self.assertIsNone(node.m_indices)
        node.clear_cached_tensors()
        self.assertIsNone(node.input)

    def test_reset_state(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(custom_map)
        node.tokens_per_expert = [1, 2, 3]
        node.input = paddle.randn([4, 64])
        node.reset_state()
        self.assertIsNone(node.tokens_per_expert)
        self.assertIsNone(node.input)

    def test_clear_activation_tensors(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(custom_map)
        node.input = paddle.randn([4, 64])
        node.input_fp8 = paddle.randn([4, 64])
        node.input_scale = paddle.randn([4, 1])
        node.o1 = paddle.randn([4, 64])
        node.clear_activation_tensors()
        self.assertIsNone(node.input)
        self.assertIsNone(node.input_fp8)
        self.assertIsNone(node.input_scale)
        self.assertIsNone(node.o1)

    def test_moe_subbatch_assertion(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                custom_map,
                moe_subbatch_token_num_after_dispatch=100,
            )

    def test_moe_subbatch_not_aligned(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                custom_map,
                moe_subbatch_token_num_after_dispatch=129,
            )

    def test_moe_subbatch_valid(self):
        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            moe_subbatch_token_num_after_dispatch=128,
        )
        self.assertEqual(node.moe_subbatch_token_num_after_dispatch, 128)


class TestExpertsGroupGemmContiguousNodeGenMIndices(unittest.TestCase):
    """Test gen_m_indices method."""

    @patch(
        "paddleformers.fleet.transformer.moe.fp8_utils.ExpertsGroupGemmContiguousNode.__init__",
        return_value=None,
    )
    def test_gen_m_indices(self, mock_init):
        node = ExpertsGroupGemmContiguousNode.__new__(
            ExpertsGroupGemmContiguousNode
        )
        tokens_per_expert = [2, 3, 1]
        indices = node.gen_m_indices(tokens_per_expert)
        self.assertEqual(indices.shape[0], 6)
        expected = [0, 0, 1, 1, 1, 2]
        self.assertEqual(indices.tolist(), expected)

    @patch(
        "paddleformers.fleet.transformer.moe.fp8_utils.ExpertsGroupGemmContiguousNode.__init__",
        return_value=None,
    )
    def test_gen_m_indices_empty(self, mock_init):
        node = ExpertsGroupGemmContiguousNode.__new__(
            ExpertsGroupGemmContiguousNode
        )
        indices = node.gen_m_indices([0, 0, 0])
        self.assertEqual(indices.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
