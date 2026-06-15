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

from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
    FP8_ALIGN,
    MlpNode,
    UnZipNode,
    ZipNode,
)


class TestFP8AlignImport(unittest.TestCase):
    """Test FP8_ALIGN is imported correctly."""

    def test_fp8_align_value(self):
        self.assertEqual(FP8_ALIGN, 128)


class TestUnZipNode(unittest.TestCase):
    """Test UnZipNode."""

    def test_construction(self):
        token_dispatcher = MagicMock()
        node = UnZipNode(token_dispatcher)
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)
        self.assertEqual(node.name, "unzip")

    def test_cached_tensors(self):
        token_dispatcher = MagicMock()
        node = UnZipNode(token_dispatcher)
        tensors = node.cached_tensors()
        self.assertEqual(len(tensors), 2)

    def test_set_cached_tensors(self):
        token_dispatcher = MagicMock()
        node = UnZipNode(token_dispatcher)
        node.set_cached_tensors([paddle.randn([4]), paddle.randn([4, 4])])
        self.assertIsNotNone(node.unzipped_probs)
        self.assertIsNotNone(node.zipped_expertwise_rowmap)

    def test_clear_cached_tensors(self):
        token_dispatcher = MagicMock()
        node = UnZipNode(token_dispatcher)
        node.unzipped_probs = paddle.randn([4])
        node.zipped_expertwise_rowmap = paddle.randn([4, 4])
        node.clear_cached_tensors()
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_reset_state(self):
        token_dispatcher = MagicMock()
        node = UnZipNode(token_dispatcher)
        node.unzipped_probs = paddle.randn([4])
        node.zipped_expertwise_rowmap = paddle.randn([4, 4])
        node.reset_state()
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)


class TestZipNode(unittest.TestCase):
    """Test ZipNode."""

    def test_construction(self):
        token_dispatcher = MagicMock()
        node = ZipNode(token_dispatcher)
        self.assertEqual(node.name, "zip")

    def test_cached_tensors_empty(self):
        token_dispatcher = MagicMock()
        node = ZipNode(token_dispatcher)
        tensors = node.cached_tensors()
        self.assertEqual(len(tensors), 0)

    def test_set_cached_tensors_asserts_empty(self):
        token_dispatcher = MagicMock()
        node = ZipNode(token_dispatcher)
        node.set_cached_tensors([])

    def test_set_cached_tensors_non_empty_raises(self):
        token_dispatcher = MagicMock()
        node = ZipNode(token_dispatcher)
        with self.assertRaises(AssertionError):
            node.set_cached_tensors([1])

    def test_clear_cached_tensors_noop(self):
        token_dispatcher = MagicMock()
        node = ZipNode(token_dispatcher)
        node.clear_cached_tensors()  # Should not raise


class TestMlpNodeConstruction(unittest.TestCase):
    """Test MlpNode construction."""

    @patch("paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode")
    def test_basic_construction(self, mock_gemm_node):
        mock_gemm_node.return_value = MagicMock()
        custom_map = MagicMock()
        custom_map.token_dispatcher = MagicMock()
        custom_map.token_dispatcher._comm_manager = MagicMock()
        custom_map.token_dispatcher._comm_manager.tokens_per_expert = [
            2,
            2,
            2,
            2,
        ]

        node = MlpNode(
            custom_map,
            num_experts_per_tok=2,
        )
        self.assertFalse(node.moe_expert_fusion)
        self.assertFalse(node.recompute_moe_premute)
        self.assertIsNotNone(node.experts_group_gemm_node)
        self.assertIsNotNone(node.unzip_node)
        self.assertIsNotNone(node.zip_node)

    @patch("paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode")
    def test_moe_subbatch_assertion(self, mock_gemm_node):
        mock_gemm_node.return_value = MagicMock()
        custom_map = MagicMock()
        custom_map.token_dispatcher = MagicMock()
        custom_map.token_dispatcher._comm_manager = MagicMock()
        custom_map.token_dispatcher._comm_manager.tokens_per_expert = [2, 2]

        with self.assertRaises(AssertionError):
            MlpNode(
                custom_map,
                num_experts_per_tok=2,
                moe_subbatch_token_num_after_dispatch=100,
            )

    @patch("paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode")
    def test_moe_expert_fusion_false_init(self, mock_gemm_node):
        mock_gemm_node.return_value = MagicMock()
        custom_map = MagicMock()
        custom_map.token_dispatcher = MagicMock()
        custom_map.token_dispatcher._comm_manager = MagicMock()
        custom_map.token_dispatcher._comm_manager.tokens_per_expert = [2, 2]

        node = MlpNode(custom_map, num_experts_per_tok=2, moe_expert_fusion=False)
        self.assertIsNotNone(node)
        self.assertFalse(node.moe_expert_fusion)


class TestMlpNodeCachedTensors(unittest.TestCase):
    """Test MlpNode cached tensors management."""

    @patch("paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode")
    def test_cached_tensors_not_empty(self, mock_gemm_node):
        mock_gem = MagicMock()
        mock_gem.cached_tensors.return_value = [None] * 6
        mock_gemm_node.return_value = mock_gem
        custom_map = MagicMock()
        custom_map.token_dispatcher = MagicMock()
        custom_map.token_dispatcher._comm_manager = MagicMock()
        custom_map.token_dispatcher._comm_manager.tokens_per_expert = [2, 2]

        node = MlpNode(custom_map, num_experts_per_tok=2)
        tensors = node.cached_tensors()
        self.assertGreater(len(tensors), 0)

    @patch("paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode")
    def test_clear_cached_tensors(self, mock_gemm_node):
        mock_gem = MagicMock()
        mock_gem.cached_tensors.return_value = [None] * 6
        mock_gemm_node.return_value = mock_gem
        custom_map = MagicMock()
        custom_map.token_dispatcher = MagicMock()
        custom_map.token_dispatcher._comm_manager = MagicMock()
        custom_map.token_dispatcher._comm_manager.tokens_per_expert = [2, 2]

        node = MlpNode(custom_map, num_experts_per_tok=2)
        node.clear_cached_tensors()
        self.assertIsNone(node.hs_2d_dispatched_fp8)

    @patch("paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode")
    def test_reset_state(self, mock_gemm_node):
        mock_gem = MagicMock()
        mock_gem.cached_tensors.return_value = [None] * 6
        mock_gemm_node.return_value = mock_gem
        custom_map = MagicMock()
        custom_map.token_dispatcher = MagicMock()
        custom_map.token_dispatcher._comm_manager = MagicMock()
        custom_map.token_dispatcher._comm_manager.tokens_per_expert = [2, 2]

        node = MlpNode(custom_map, num_experts_per_tok=2)
        node.dispatched_indices = paddle.randn([4])
        node.reset_state()
        self.assertIsNone(node.dispatched_indices)
        self.assertIsNone(node.dispatched_probs)


class TestMlpNodeTokenOffsets(unittest.TestCase):
    """Test MlpNode token offset computation."""

    @patch("paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode")
    def test_token_offsets_computed(self, mock_gemm_node):
        mock_gem = MagicMock()
        mock_gem.cached_tensors.return_value = [None] * 6
        mock_gemm_node.return_value = mock_gem
        custom_map = MagicMock()
        custom_map.token_dispatcher = MagicMock()
        custom_map.token_dispatcher._comm_manager = MagicMock()
        custom_map.token_dispatcher._comm_manager.tokens_per_expert = [
            2,
            4,
            1,
            3,
        ]

        node = MlpNode(custom_map, num_experts_per_tok=2)
        # padding should be aligned to FP8_ALIGN=128
        expected_padded = [
            128,
            128,
            128,
            128,
        ]
        self.assertEqual(node.padding_token_per_experts, expected_padded)
        self.assertEqual(node.token_offsets, [0, 128, 256, 384, 512])


class TestFusionMoePyLayer(unittest.TestCase):
    """Test FusionMoePyLayer exists with correct interface."""

    def test_has_forward_and_backward(self):
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
            FusionMoePyLayer,
        )

        self.assertTrue(hasattr(FusionMoePyLayer, "forward"))
        self.assertTrue(hasattr(FusionMoePyLayer, "backward"))

    def test_is_pylayer(self):
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
            FusionMoePyLayer,
        )

        self.assertTrue(issubclass(FusionMoePyLayer, paddle.autograd.PyLayer))


if __name__ == "__main__":
    unittest.main()
