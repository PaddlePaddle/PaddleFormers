# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle


def _make_mock_custom_map(num_experts=1):
    """Create a mock custom_map with proper token_dispatcher setup."""
    mock_custom_map = MagicMock()
    mock_custom_map.experts = [MagicMock() for _ in range(num_experts)]
    for e in mock_custom_map.experts:
        e.up_gate_proj = MagicMock()
        e.up_gate_proj.weight = paddle.randn([128, 64], dtype=paddle.bfloat16)
        e.down_proj = MagicMock()
        e.down_proj.weight = paddle.randn([64, 128], dtype=paddle.bfloat16)

    mock_comm_manager = MagicMock()
    mock_comm_manager.tokens_per_expert = [4] * num_experts
    mock_custom_map.token_dispatcher._comm_manager = mock_comm_manager

    return mock_custom_map


class TestFusionLayerUtils(unittest.TestCase):
    """Unit tests for fusion_layer_utils module."""

    def test_unzip_node_init(self):
        """Test UnZipNode initialization."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import UnZipNode

        mock_dispatcher = MagicMock()
        node = UnZipNode(mock_dispatcher)
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_unzip_node_reset_state(self):
        """Test UnZipNode.reset_state."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import UnZipNode

        mock_dispatcher = MagicMock()
        node = UnZipNode(mock_dispatcher)
        node.unzipped_probs = paddle.randn([4, 2])
        node.zipped_expertwise_rowmap = paddle.randn([4, 2])
        node.reset_state()
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_unzip_node_cached_tensors(self):
        """Test UnZipNode.cached_tensors."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import UnZipNode

        mock_dispatcher = MagicMock()
        node = UnZipNode(mock_dispatcher)
        cached = node.cached_tensors()
        self.assertEqual(len(cached), 2)
        self.assertIsNone(cached[0])
        self.assertIsNone(cached[1])

    def test_unzip_node_set_cached_tensors(self):
        """Test UnZipNode.set_cached_tensors."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import UnZipNode

        mock_dispatcher = MagicMock()
        node = UnZipNode(mock_dispatcher)
        t1 = paddle.randn([4, 2])
        t2 = paddle.randn([4, 2])
        node.set_cached_tensors([t1, t2])
        self.assertTrue(paddle.allclose(node.unzipped_probs, t1))
        self.assertTrue(paddle.allclose(node.zipped_expertwise_rowmap, t2))

    def test_unzip_node_clear_cached_tensors(self):
        """Test UnZipNode.clear_cached_tensors."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import UnZipNode

        mock_dispatcher = MagicMock()
        node = UnZipNode(mock_dispatcher)
        node.unzipped_probs = paddle.randn([4, 2])
        node.zipped_expertwise_rowmap = paddle.randn([4, 2])
        node.clear_cached_tensors()
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_zip_node_cached_tensors_empty(self):
        """Test ZipNode.cached_tensors returns empty list."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import ZipNode

        mock_dispatcher = MagicMock()
        node = ZipNode(mock_dispatcher)
        self.assertEqual(node.cached_tensors(), [])

    def test_zip_node_set_cached_tensors_empty(self):
        """Test ZipNode.set_cached_tensors accepts empty list."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import ZipNode

        mock_dispatcher = MagicMock()
        node = ZipNode(mock_dispatcher)
        node.set_cached_tensors([])
        # Should not raise

    def test_zip_node_clear_cached_tensors_noop(self):
        """Test ZipNode.clear_cached_tensors is no-op."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import ZipNode

        mock_dispatcher = MagicMock()
        node = ZipNode(mock_dispatcher)
        node.clear_cached_tensors()
        # Should not raise

    def test_fusion_moe_pylayer_forward_returns_output(self):
        """Test FusionMoePyLayer forward produces output."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import (
            FusionMoePyLayer,
        )

        mock_custom_map = _make_mock_custom_map()

        with patch(
            "paddleformers.fleet.transformer.moe.fusion_layer_utils.MlpNode"
        ) as MockMlpNode:
            mock_node = MagicMock()
            mock_node.forward.return_value = paddle.randn(
                [4, 64], dtype=paddle.bfloat16
            )
            # Return empty list of tensors for cached_tensors
            mock_node.cached_tensors.return_value = []
            mock_node.clear_cached_tensors.return_value = None
            MockMlpNode.return_value = mock_node

            hidden = paddle.randn([4, 64], dtype=paddle.bfloat16)
            probs = paddle.randn([4, 2], dtype=paddle.float32)
            indices = paddle.randint(0, 2, [4, 2])

            out = FusionMoePyLayer.apply(
                hidden,
                probs,
                indices,
                mock_custom_map,
                2,
                use_fp8_mlp=False,
                moe_deep_gemm=False,
                moe_expert_fusion=False,
                is_first_fwd=True,
            )
            self.assertIsNotNone(out)

    def test_mlp_node_release_mem(self):
        """Test MlpNode.release_mem."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import MlpNode

        mock_custom_map = _make_mock_custom_map()

        with patch(
            "paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode"
        ) as MockGemm:
            mock_gemm = MagicMock()
            MockGemm.return_value = mock_gemm

            node = MlpNode(
                mock_custom_map,
                2,
                recompute_moe_gate_up=False,
                dequant_input=False,
                moe_expert_fusion=True,
                use_fp8_mlp=False,
                moe_deep_gemm=False,
            )
            node.release_mem()
            self.assertIsNone(node.experts_group_gemm_node)

    def test_mlp_node_init_assertions(self):
        """Test MlpNode init assertions."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import MlpNode

        mock_custom_map = _make_mock_custom_map()

        # recompute_moe_premute requires moe_expert_fusion=False
        with self.assertRaises(AssertionError):
            MlpNode(
                mock_custom_map,
                2,
                recompute_moe_premute=True,
                recompute_moe_gate_up=True,
                dequant_input=True,
                moe_expert_fusion=True,
                use_fp8_mlp=False,
                moe_deep_gemm=False,
            )

    def test_mlp_node_non_fusion_init(self):
        """Test MlpNode with moe_expert_fusion=False initializes correctly."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import MlpNode

        mock_custom_map = _make_mock_custom_map()

        with patch(
            "paddleformers.fleet.transformer.moe.fusion_layer_utils.ExpertsGroupGemmContiguousNode"
        ) as MockGemm:
            MockGemm.return_value = MagicMock()

            node = MlpNode(
                mock_custom_map,
                2,
                moe_expert_fusion=False,
                use_fp8_mlp=False,
                moe_deep_gemm=False,
            )
            self.assertIsNotNone(node)

    def test_mlp_node_subbatch_assertions(self):
        """Test MlpNode init with subbatch asserts."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import MlpNode

        mock_custom_map = _make_mock_custom_map()

        with self.assertRaises(AssertionError):
            MlpNode(
                mock_custom_map,
                2,
                moe_subbatch_token_num_after_dispatch=-1,
                moe_expert_fusion=True,
                recompute_moe_gate_up=True,
                dequant_input=True,
                use_fp8_mlp=False,
                moe_deep_gemm=False,
            )

    def _bare_mlp_node(self, **attrs):
        """Build an MlpNode instance without running __init__, setting only the
        attributes the pure helper methods read."""
        from paddleformers.fleet.transformer.moe.fusion_layer_utils import MlpNode

        node = object.__new__(MlpNode)
        node.experts = None
        node.experts_group_gemm_node = None
        for k, v in attrs.items():
            setattr(node, k, v)
        return node

    def test_gate_up_out_dim_per_expert(self):
        """_gate_up_out_dim: per-expert (non-fused) layout reads up_gate_proj."""
        expert = SimpleNamespace(
            up_gate_proj=SimpleNamespace(
                weight=paddle.randn([128, 64], dtype=paddle.bfloat16)
            )
        )
        # first entry None exercises the `if expert is None: continue` branch
        node = self._bare_mlp_node(experts=[None, expert])
        self.assertEqual(node._gate_up_out_dim(128), 64)

    def test_gate_up_out_dim_grouped(self):
        """_gate_up_out_dim: grouped deep_gemm layout reads stacked weight1,
        with experts_group_gemm_node given as a list."""
        parent = SimpleNamespace(
            weight1=paddle.randn([2, 128, 64], dtype=paddle.bfloat16)
        )
        gemm = SimpleNamespace(grouped_gemm_experts=parent)
        node = self._bare_mlp_node(experts=None, experts_group_gemm_node=[gemm])
        self.assertEqual(node._gate_up_out_dim(128), 64)

    def test_gate_up_out_dim_fallback(self):
        """_gate_up_out_dim: falls back to 2*hidden when weight unresolved."""
        gemm = SimpleNamespace()  # no grouped_gemm_experts attribute
        node = self._bare_mlp_node(experts=None, experts_group_gemm_node=gemm)
        self.assertEqual(node._gate_up_out_dim(128), 256)

    def test_bwd_feature_sizes_bf16_inplace(self):
        """bf16 wgrad + no clamp -> inplace peak (do1 reuses o1): 5 buffers.
        Also exercises the list-form experts_group_gemm_node branch."""
        gemm = SimpleNamespace(use_bf16_gemm_weight_grad=True, clamp_value=None)
        node = self._bare_mlp_node(experts_group_gemm_node=[gemm])
        sizes = node._bwd_pre_permute_feature_sizes(128, 64, 32)
        self.assertEqual(len(sizes), 5)

    def test_bwd_feature_sizes_bf16_clamp_out_of_place(self):
        """bf16 wgrad + clamp_value>0 -> out-of-place peak (do1 is a separate
        buffer, matching fused_swiglu_weighted_clamp_bwd): 6 buffers."""
        gemm = SimpleNamespace(use_bf16_gemm_weight_grad=True, clamp_value=1.0)
        node = self._bare_mlp_node(experts_group_gemm_node=gemm)
        inplace = self._bare_mlp_node(
            experts_group_gemm_node=SimpleNamespace(
                use_bf16_gemm_weight_grad=True, clamp_value=None
            )
        )._bwd_pre_permute_feature_sizes(128, 64, 32)
        sizes = node._bwd_pre_permute_feature_sizes(128, 64, 32)
        self.assertEqual(len(sizes), 6)
        # clamp path must estimate a strictly larger footprint than inplace
        self.assertGreater(sum(sizes), sum(inplace))

    def test_bwd_feature_sizes_fp8_inplace(self):
        """fp8 wgrad + no clamp -> inplace peak: 5 buffers."""
        gemm = SimpleNamespace(
            use_bf16_gemm_weight_grad=False, clamp_value=None
        )
        node = self._bare_mlp_node(experts_group_gemm_node=gemm)
        sizes = node._bwd_pre_permute_feature_sizes(128, 64, 32)
        self.assertEqual(len(sizes), 5)

    def test_bwd_feature_sizes_fp8_clamp_out_of_place(self):
        """fp8 wgrad + clamp_value>0 -> out-of-place peak: 6 buffers."""
        gemm = SimpleNamespace(use_bf16_gemm_weight_grad=False, clamp_value=2.0)
        node = self._bare_mlp_node(experts_group_gemm_node=gemm)
        sizes = node._bwd_pre_permute_feature_sizes(128, 64, 32)
        self.assertEqual(len(sizes), 6)


if __name__ == "__main__":
    unittest.main()
