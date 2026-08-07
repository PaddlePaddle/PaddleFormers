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

import unittest

import numpy as np
import paddle

from paddleformers.fleet.transformer.moe.moe_router import _apply_routing_map_fusion


def reference_topk(
    gate_probs,
    probs_for_choice,
    moe_k,
    use_node_limit,
    n_group,
    topk_group,
    norm_gate_logits,
):
    """
    Reference implementation of MoE TopK selection using Paddle ops.
    """
    seq_len, n_experts = gate_probs.shape

    if use_node_limit and n_group > 1:
        # Node limit logic: select topk_group groups based on top-2 sum
        epg = n_experts // n_group
        group_scores = []
        for g in range(n_group):
            g_start = g * epg
            g_end = g_start + epg
            group_probs = probs_for_choice[:, g_start:g_end]
            # Get top-2 sum for each group
            top2_vals, _ = paddle.topk(group_probs, k=min(2, epg), axis=-1)
            group_score = top2_vals.sum(axis=-1)  # [seq_len]
            group_scores.append(group_score)

        group_scores = paddle.stack(group_scores, axis=-1)  # [seq_len, n_group]
        _, selected_groups = paddle.topk(
            group_scores, k=topk_group, axis=-1
        )  # [seq_len, topk_group]

        # Create mask for selected groups
        mask = paddle.zeros_like(probs_for_choice)
        for g in range(topk_group):
            group_idx = selected_groups[:, g : g + 1]  # [seq_len, 1]
            for e in range(epg):
                expert_idx = group_idx * epg + e
                mask = paddle.scatter(
                    mask,
                    expert_idx,
                    paddle.ones([seq_len, 1], dtype=mask.dtype),
                    axis=1,
                )

        probs_for_choice = probs_for_choice * mask + (1 - mask) * (-1e9)

    # TopK selection
    _, topk_indices = paddle.topk(probs_for_choice, k=moe_k, axis=-1)
    topk_probs = paddle.take_along_axis(gate_probs, topk_indices, axis=-1)

    # Normalization
    if norm_gate_logits:
        denom = topk_probs.sum(axis=-1, keepdim=True) + 1e-12
        topk_probs = topk_probs / denom

    return topk_probs, topk_indices


class TestMoETopkFusionTriton(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        self.seq_len = 1024
        self.n_experts = 32
        self.moe_k = 8
        self.n_group = 4
        self.topk_group = 2

    def test_forward_no_node_limit(self):
        """Test forward pass without node limit."""
        gate_probs = paddle.rand(
            [self.seq_len, self.n_experts], dtype="float32"
        )
        gate_probs = paddle.nn.functional.softmax(gate_probs, axis=-1)
        probs_for_choice = gate_probs.clone()

        # Reference
        ref_probs, ref_indices = reference_topk(
            gate_probs,
            probs_for_choice,
            self.moe_k,
            use_node_limit=False,
            n_group=1,
            topk_group=1,
            norm_gate_logits=True,
        )

        # Triton
        paddle.enable_compat(scope={"triton"}, silent=True)
        from paddleformers.fleet.triton_ops import MoETopkFusion

        triton_probs, triton_indices = MoETopkFusion.apply(
            gate_probs, probs_for_choice, self.moe_k, False, 1, 1, True
        )
        paddle.disable_compat()

        # Check indices match (may differ in tie-breaking, so check probs instead)
        np.testing.assert_allclose(
            ref_probs.numpy(), triton_probs.numpy(), rtol=1e-4, atol=1e-4
        )

    def test_forward_with_node_limit(self):
        """Test forward pass with node limit (group selection)."""
        gate_probs = paddle.rand(
            [self.seq_len, self.n_experts], dtype="float32"
        )
        gate_probs = paddle.nn.functional.softmax(gate_probs, axis=-1)

        # Add correction bias for choice
        correction_bias = paddle.randn([self.n_experts], dtype="float32") * 0.1
        probs_for_choice = gate_probs + correction_bias.unsqueeze(0)

        # Triton
        paddle.enable_compat(scope={"triton"}, silent=True)
        from paddleformers.fleet.triton_ops import MoETopkFusion

        triton_probs, triton_indices = MoETopkFusion.apply(
            gate_probs,
            probs_for_choice,
            self.moe_k,
            True,
            self.n_group,
            self.topk_group,
            True,
        )
        paddle.disable_compat()

        # Verify output shapes
        self.assertEqual(triton_probs.shape, [self.seq_len, self.moe_k])
        self.assertEqual(triton_indices.shape, [self.seq_len, self.moe_k])

        # Verify normalization (probs should sum to 1)
        prob_sums = triton_probs.sum(axis=-1)
        np.testing.assert_allclose(
            prob_sums.numpy(), np.ones(self.seq_len), rtol=1e-4, atol=1e-4
        )

    def test_backward(self):
        """Test backward pass gradient computation."""
        gate_probs = paddle.rand(
            [self.seq_len, self.n_experts], dtype="float32"
        )
        gate_probs = paddle.nn.functional.softmax(gate_probs, axis=-1)
        gate_probs.stop_gradient = False
        probs_for_choice = gate_probs.clone().detach()

        paddle.enable_compat(scope={"triton"}, silent=True)
        from paddleformers.fleet.triton_ops import MoETopkFusion

        triton_probs, triton_indices = MoETopkFusion.apply(
            gate_probs, probs_for_choice, self.moe_k, False, 1, 1, True
        )

        # Backward
        dy = paddle.randn_like(triton_probs) * 0.01
        triton_probs.backward(dy)
        grad = gate_probs.grad
        paddle.disable_compat()

        # Verify gradient shape and non-zero
        self.assertEqual(grad.shape, gate_probs.shape)
        self.assertGreater(paddle.abs(grad).sum().item(), 0)

    def test_routing_map_forward(self):
        """Test routing map generation."""
        gate_probs = paddle.rand(
            [self.seq_len, self.n_experts], dtype="float32"
        )
        topk_indices = paddle.randint(
            0, self.n_experts, [self.seq_len, self.moe_k], dtype="int64"
        )

        paddle.enable_compat(scope={"triton"}, silent=True)
        from paddleformers.fleet.triton_ops import routing_map_fusion_forward

        routing_map, topk_out, dispatch_mask = routing_map_fusion_forward(
            gate_probs, topk_indices, input_ids=None, is_pure_text_line=None
        )
        paddle.disable_compat()

        # Verify shapes
        self.assertEqual(routing_map.shape, [self.seq_len, self.n_experts])
        self.assertEqual(topk_out.shape, [self.seq_len, self.moe_k])
        self.assertEqual(dispatch_mask.shape, [self.n_experts])

        # Verify routing_map is binary
        unique_vals = paddle.unique(routing_map)
        self.assertTrue(len(unique_vals) <= 2)  # Only 0 and 1

    def test_different_dtypes(self):
        """Test with bfloat16 dtype."""
        gate_probs = paddle.rand([512, self.n_experts], dtype="bfloat16")
        probs_for_choice = gate_probs.clone()

        paddle.enable_compat(scope={"triton"}, silent=True)
        from paddleformers.fleet.triton_ops import MoETopkFusion

        triton_probs, triton_indices = MoETopkFusion.apply(
            gate_probs, probs_for_choice, self.moe_k, False, 1, 1, True
        )
        paddle.disable_compat()

        self.assertEqual(triton_probs.dtype, paddle.bfloat16)


def _router_branch_reference(gates, top_idx, input_ids_none_zero_mask):
    mask = paddle.zeros_like(gates).put_along_axis(
        top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1
    )
    if input_ids_none_zero_mask is not None:
        valid_mask = input_ids_none_zero_mask
        mask = mask * valid_mask.cast(mask.dtype)
        top_idx = top_idx.masked_fill(~valid_mask.cast(paddle.bool), -1)
    return mask, top_idx


class TestRoutingMapFusionRouterBranch(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        self.seq_len = 1024
        self.n_experts = 64
        self.moe_k = 8

    def _make_inputs(self, dtype="float32"):
        gates = paddle.rand([self.seq_len, self.n_experts], dtype=dtype)
        top_idx = paddle.randint(
            0, self.n_experts, [self.seq_len, self.moe_k], dtype="int64"
        )
        return gates, top_idx

    def test_equivalence_without_padding(self):
        gates, top_idx = self._make_inputs("float32")

        paddle.enable_compat(scope={"triton"}, silent=True)
        fused_mask, fused_top_idx, dispatch_mask = _apply_routing_map_fusion(
            gates, top_idx.clone(), None
        )
        paddle.disable_compat()
        ref_mask, ref_top_idx = _router_branch_reference(
            gates, top_idx.clone(), None
        )

        self.assertEqual(fused_mask.shape, [self.seq_len, self.n_experts])
        self.assertEqual(dispatch_mask.shape, [self.n_experts])
        np.testing.assert_array_equal(fused_mask.numpy(), ref_mask.numpy())
        np.testing.assert_array_equal(
            fused_top_idx.numpy(), ref_top_idx.numpy()
        )
        np.testing.assert_array_equal(
            dispatch_mask.numpy(),
            ref_mask.cast("int64").sum(axis=0).numpy(),
        )

    def test_equivalence_with_padding_mask(self):
        gates, top_idx = self._make_inputs("float32")
        padded = paddle.rand([self.seq_len], dtype="float32") <= 0.2
        input_ids = paddle.where(
            padded,
            paddle.zeros([self.seq_len], dtype="int64"),
            paddle.ones([self.seq_len], dtype="int64"),
        )
        valid_mask = (input_ids != 0).reshape([-1, 1]).cast("float32")

        paddle.enable_compat(scope={"triton"}, silent=True)
        fused_mask, fused_top_idx, fused_dispatch_mask = (
            _apply_routing_map_fusion(
                gates, top_idx.clone(), valid_mask, input_ids=input_ids
            )
        )
        paddle.disable_compat()
        ref_mask, ref_top_idx = _router_branch_reference(
            gates, top_idx.clone(), valid_mask
        )

        np.testing.assert_array_equal(fused_mask.numpy(), ref_mask.numpy())
        np.testing.assert_array_equal(
            fused_top_idx.numpy(), ref_top_idx.numpy()
        )
        padded_rows = (valid_mask.squeeze(-1) == 0).numpy()
        self.assertTrue((fused_top_idx.numpy()[padded_rows] == -1).all())
        self.assertTrue((fused_mask.numpy()[padded_rows] == 0).all())
        np.testing.assert_array_equal(
            fused_dispatch_mask.numpy(),
            ref_mask.cast("int64").sum(axis=0).numpy(),
        )

    def test_bfloat16_dtype(self):
        gates, top_idx = self._make_inputs("bfloat16")

        paddle.enable_compat(scope={"triton"}, silent=True)
        fused_mask, fused_top_idx, fused_dispatch_mask = (
            _apply_routing_map_fusion(gates, top_idx.clone(), None)
        )
        paddle.disable_compat()
        ref_mask, ref_top_idx = _router_branch_reference(
            gates, top_idx.clone(), None
        )

        self.assertEqual(fused_mask.dtype, paddle.bfloat16)
        np.testing.assert_array_equal(
            fused_mask.cast("float32").numpy(),
            ref_mask.cast("float32").numpy(),
        )
        np.testing.assert_array_equal(
            fused_top_idx.numpy(), ref_top_idx.numpy()
        )
        np.testing.assert_array_equal(
            fused_dispatch_mask.numpy(),
            ref_mask.cast("int64").sum(axis=0).numpy(),
        )


if __name__ == "__main__":
    unittest.main()
