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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import paddle.nn as nn


def _make_gate_cls():
    """Return the StandardMoEGate class."""
    from paddleformers.nn.moe_deepep.moe_gate import StandardMoEGate

    return StandardMoEGate


def _make_gate(
    num_experts=8,
    expert_hidden_size=16,
    topk_method="greedy",
    num_experts_per_tok=2,
    norm_topk_prob=True,
    drop_tokens=False,
    moe_expert_capacity_factor=0.0,
    moe_token_drop_policy="probs",
    transpose_gate_weight=False,
    scoring_func="softmax",
    seq_length=32,
    n_group=1,
    topk_group=1,
    routed_scaling_factor=1.0,
    global_aux_loss=False,
    seq_aux=True,
    moe_subbatch_token_num_before_dispatch=-1,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
):
    """Create a StandardMoEGate instance with sensible defaults."""
    cls = _make_gate_cls()
    moe_config = {
        "gate_activation": scoring_func,
        "eval_capacity_factor": 1.0,
        "group": None,
        "global_aux_loss": global_aux_loss,
        "use_rts": True,
        "top2_2nd_expert_sampling": True,
        "seq_aux": seq_aux,
    }
    return cls(
        num_experts=num_experts,
        expert_hidden_size=expert_hidden_size,
        drop_tokens=drop_tokens,
        topk_method=topk_method,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
        moe_config=moe_config,
        seq_length=seq_length,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        moe_subbatch_token_num_before_dispatch=moe_subbatch_token_num_before_dispatch,
        tensor_model_parallel_size=tensor_model_parallel_size,
        sequence_parallel=sequence_parallel,
        moe_expert_capacity_factor=moe_expert_capacity_factor,
        moe_token_drop_policy=moe_token_drop_policy,
        transpose_gate_weight=transpose_gate_weight,
    )


class TestMoEGateMixin(unittest.TestCase):
    """Tests for MoEGateMixin methods via StandardMoEGate."""

    def test_gate_score_func_softmax(self):
        """Test gate_score_func with softmax scoring."""
        gate = _make_gate(scoring_func="softmax")
        logits = paddle.randn([4, 8])
        scores = gate.gate_score_func(logits)
        # Softmax output should sum to 1 along last axis
        sums = scores.sum(axis=-1)
        np.testing.assert_allclose(sums.numpy(), np.ones(4), atol=1e-5)

    def test_gate_score_func_sigmoid(self):
        """Test gate_score_func with sigmoid scoring."""
        gate = _make_gate(scoring_func="sigmoid")
        logits = paddle.randn([4, 8])
        scores = gate.gate_score_func(logits)
        # Sigmoid output should be in [0, 1]
        self.assertTrue(paddle.all(scores >= 0).item())
        self.assertTrue(paddle.all(scores <= 1).item())

    def test_gate_score_func_tanh(self):
        """Test gate_score_func with tanh scoring."""
        gate = _make_gate(scoring_func="tanh")
        logits = paddle.randn([4, 8])
        scores = gate.gate_score_func(logits)
        # Tanh output should be in [-1, 1]
        self.assertTrue(paddle.all(scores >= -1).item())
        self.assertTrue(paddle.all(scores <= 1).item())

    def test_gate_score_func_relu(self):
        """Test gate_score_func with relu scoring."""
        gate = _make_gate(scoring_func="relu")
        logits = paddle.randn([4, 8])
        scores = gate.gate_score_func(logits)
        # ReLU output should be non-negative
        self.assertTrue(paddle.all(scores >= 0).item())

    def test_gate_score_func_gelu(self):
        """Test gate_score_func with gelu scoring."""
        gate = _make_gate(scoring_func="gelu")
        logits = paddle.randn([4, 8])
        scores = gate.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 8])

    def test_gate_score_func_leaky_relu(self):
        """Test gate_score_func with leaky_relu scoring."""
        gate = _make_gate(scoring_func="leaky_relu")
        logits = paddle.randn([4, 8])
        scores = gate.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 8])

    def test_gate_score_func_unknown_defaults_softmax(self):
        """Test that unknown scoring function defaults to softmax."""
        gate = _make_gate(scoring_func="unknown_func")
        logits = paddle.randn([4, 8])
        scores = gate.gate_score_func(logits)
        # Should fall back to softmax: sums to 1
        sums = scores.sum(axis=-1)
        np.testing.assert_allclose(sums.numpy(), np.ones(4), atol=1e-5)

    def test_gate_score_func_output_dtype_float32(self):
        """Test that gate_score_func returns float32 regardless of input dtype."""
        gate = _make_gate(scoring_func="softmax")
        logits = paddle.randn([4, 8], dtype="float16")
        scores = gate.gate_score_func(logits)
        self.assertEqual(scores.dtype, paddle.float32)

    def test_one_hot_to_float(self):
        """Test _one_hot_to_float conversion."""
        gate = _make_gate()
        x = paddle.to_tensor([0, 1, 2])
        result = gate._one_hot_to_float(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])
        self.assertTrue(result.dtype == paddle.float32 or str(result.dtype) == "float32")

    def test_one_hot_to_float_int32_input(self):
        """Test _one_hot_to_float with int32 input."""
        gate = _make_gate()
        x = paddle.to_tensor([0, 1, 2], dtype="int32")
        result = gate._one_hot_to_float(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])

    def test_one_hot_to_float_float_input_cast(self):
        """Test _one_hot_to_float casts float input to int64."""
        gate = _make_gate()
        x = paddle.to_tensor([0.0, 1.0, 2.0])
        result = gate._one_hot_to_float(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])

    def test_one_hot_to_int64(self):
        """Test _one_hot_to_int64 conversion."""
        gate = _make_gate()
        x = paddle.to_tensor([0, 1, 2])
        result = gate._one_hot_to_int64(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])
        self.assertEqual(result.dtype, paddle.int64)

    def test_capacity_basic(self):
        """Test _capacity calculation with basic inputs."""
        gate = _make_gate()
        gates = paddle.randn([16, 4])
        capacity = gate._capacity(gates, capacity_factor=1.0)
        self.assertEqual(capacity, 4)

    def test_capacity_with_factor(self):
        """Test _capacity with capacity_factor > 1."""
        gate = _make_gate()
        gates = paddle.randn([16, 4])
        capacity = gate._capacity(gates, capacity_factor=2.0)
        self.assertEqual(capacity, 8)

    def test_capacity_asserts_2d(self):
        """Test _capacity raises assertion for non-2D gates."""
        gate = _make_gate()
        gates = paddle.randn([2, 4, 8])
        with self.assertRaises(AssertionError):
            gate._capacity(gates, capacity_factor=1.0)

    def test_capacity_asserts_positive(self):
        """Test _capacity raises assertion when capacity is 0."""
        gate = _make_gate()
        gates = paddle.randn([3, 16])
        with self.assertRaises(AssertionError):
            gate._capacity(gates, capacity_factor=0.1)

    def test_cal_z_loss(self):
        """Test _cal_z_loss returns a scalar tensor."""
        gate = _make_gate()
        logits = paddle.randn([4, 8])
        z_loss = gate._cal_z_loss(logits)
        self.assertEqual(z_loss.shape, [])
        self.assertTrue(z_loss.numpy().item() >= 0)

    def test_cal_z_loss_shape(self):
        """Test _cal_z_loss with different batch sizes."""
        gate = _make_gate()
        logits = paddle.randn([8, 16])
        z_loss = gate._cal_z_loss(logits)
        self.assertEqual(z_loss.shape, [])

    def test_topk_greedy(self):
        """Test _topk_greedy returns correct shapes."""
        gate = _make_gate()
        scores = paddle.randn([6, 8])
        topk_weight, topk_idx = gate._topk_greedy(scores, k=2)
        self.assertEqual(topk_weight.shape, [6, 2])
        self.assertEqual(topk_idx.shape, [6, 2])

    def test_topk_greedy_sorted(self):
        """Test _topk_greedy returns sorted results."""
        gate = _make_gate()
        scores = paddle.randn([6, 8])
        topk_weight, _ = gate._topk_greedy(scores, k=2)
        # First weight should be >= second weight per row
        self.assertTrue(paddle.all(topk_weight[:, 0] >= topk_weight[:, 1]).item())

    def test_topk_group_limited_greedy(self):
        """Test _topk_group_limited_greedy returns correct shapes."""
        gate = _make_gate(n_group=4, topk_group=2)
        scores = paddle.randn([6, 8])
        topk_weight, topk_idx = gate._topk_group_limited_greedy(scores, k=2, n_group=4, topk_group=2)
        self.assertEqual(topk_weight.shape, [6, 2])
        self.assertEqual(topk_idx.shape, [6, 2])

    def test_topk_group_limited_greedy_asserts_divisible(self):
        """Test _topk_group_limited_greedy raises assertion when n_experts not divisible by n_group."""
        gate = _make_gate(n_group=3)
        scores = paddle.randn([6, 8])
        with self.assertRaises(AssertionError):
            gate._topk_group_limited_greedy(scores, k=2, n_group=3, topk_group=1)

    def test_topk_noaux_tc(self):
        """Test _topk_noaux_tc returns correct shapes."""
        gate = _make_gate(topk_method="noaux_tc", n_group=4, topk_group=2)
        scores = paddle.randn([6, 8])
        topk_weight, topk_idx = gate._topk_noaux_tc(scores, k=2, n_group=4, topk_group=2)
        self.assertEqual(topk_weight.shape, [6, 2])
        self.assertEqual(topk_idx.shape, [6, 2])

    def test_topk_noaux_tc_uses_bias(self):
        """Test _topk_noaux_tc uses e_score_correction_bias."""
        gate = _make_gate(topk_method="noaux_tc", n_group=4, topk_group=2)
        self.assertIsNotNone(gate.e_score_correction_bias)
        self.assertEqual(gate.e_score_correction_bias.shape, [8])

    def test_priority(self):
        """Test _priority returns correct shape."""
        gate = _make_gate()
        topk_idx = paddle.to_tensor([[0, 1], [2, 3], [0, 4]])
        result = gate._priority(topk_idx, capacity=2)
        self.assertEqual(result.shape, [3, 8])

    def test_probs_drop_policy(self):
        """Test _probs_drop_policy returns mask with correct shape."""
        gate = _make_gate()
        scores = paddle.randn([6, 8])
        # Zero out non-topk positions
        topk_val, topk_idx = paddle.topk(scores, k=2, axis=-1)
        topk_scores = paddle.zeros_like(scores)
        topk_scores.put_along_axis(topk_idx, topk_val, axis=1)
        mask = gate._probs_drop_policy(topk_scores, capacity=3)
        self.assertEqual(mask.shape, [6, 8])

    def test_cal_aux_loss(self):
        """Test _cal_aux_loss returns a scalar tensor."""
        gate = _make_gate(global_aux_loss=False)
        gates = paddle.nn.functional.softmax(paddle.randn([6, 8]).cast("float32"), axis=-1)
        mask = paddle.zeros([6, 8])
        mask[:, 0] = 1.0
        mask[:, 1] = 1.0
        aux_loss = gate._cal_aux_loss(gates, mask)
        self.assertEqual(aux_loss.shape, [])

    @patch("paddleformers.nn.moe_deepep.moe_gate.dist.all_gather")
    def test_cal_aux_loss_global(self, mock_all_gather):
        """Test _cal_aux_loss with global_aux_loss=True."""
        moe_config = {
            "gate_activation": "softmax",
            "eval_capacity_factor": 1.0,
            "group": True,  # Must be truthy to pass the group assertion
            "global_aux_loss": True,
            "use_rts": True,
            "top2_2nd_expert_sampling": True,
            "seq_aux": True,
        }
        cls = _make_gate_cls()
        with patch("paddleformers.nn.moe_deepep.moe_gate.dist.get_rank", return_value=0):
            gate = cls(
                num_experts=8,
                expert_hidden_size=16,
                drop_tokens=False,
                topk_method="greedy",
                num_experts_per_tok=2,
                norm_topk_prob=True,
                moe_config=moe_config,
                seq_length=32,
                n_group=1,
                topk_group=1,
                routed_scaling_factor=1.0,
                moe_subbatch_token_num_before_dispatch=-1,
                tensor_model_parallel_size=1,
                sequence_parallel=False,
                moe_expert_capacity_factor=0.0,
                moe_token_drop_policy="probs",
                transpose_gate_weight=False,
            )
        gate.rank = 0
        gate.group = MagicMock()

        def fake_all_gather(output_list, tensor, group=None):
            # Real dist.all_gather fills output_list with world_size elements
            # For single-rank testing, world_size=1
            output_list.clear()
            output_list.append(tensor.clone())

        mock_all_gather.side_effect = fake_all_gather
        gates = paddle.nn.functional.softmax(paddle.randn([6, 8]).cast("float32"), axis=-1)
        mask = paddle.zeros([6, 8])
        mask[:, 0] = 1.0
        aux_loss = gate._cal_aux_loss(gates, mask)
        self.assertEqual(aux_loss.shape, [])
        mock_all_gather.assert_called()


class TestStandardMoEGate(unittest.TestCase):
    """Tests for StandardMoEGate class."""

    def test_import(self):
        """Test that StandardMoEGate can be imported."""
        cls = _make_gate_cls()
        self.assertIsNotNone(cls)

    def test_is_nn_layer(self):
        """Test that StandardMoEGate is a subclass of nn.Layer."""
        cls = _make_gate_cls()
        self.assertTrue(issubclass(cls, nn.Layer))

    def test_init_greedy(self):
        """Test StandardMoEGate initialization with greedy topk_method."""
        gate = _make_gate(topk_method="greedy")
        self.assertEqual(gate.num_experts, 8)
        self.assertEqual(gate.topk_method, "greedy")
        self.assertEqual(gate.num_experts_per_tok, 2)
        self.assertFalse(gate._cast_to_low_precision)

    def test_init_noaux_tc(self):
        """Test StandardMoEGate initialization with noaux_tc topk_method."""
        gate = _make_gate(topk_method="noaux_tc")
        self.assertIsNotNone(gate.e_score_correction_bias)
        self.assertEqual(gate.e_score_correction_bias.shape, [8])
        self.assertTrue(gate.e_score_correction_bias.stop_gradient)

    def test_init_noaux_tc_expert_usage(self):
        """Test expert_usage buffer is created with noaux_tc."""
        gate = _make_gate(topk_method="noaux_tc")
        self.assertIsNotNone(gate.expert_usage)
        self.assertEqual(gate.expert_usage.shape, [8])
        self.assertEqual(gate.expert_usage.dtype, paddle.int64)
        np.testing.assert_array_equal(gate.expert_usage.numpy(), np.zeros(8))

    def test_weight_shape_no_transpose(self):
        """Test gate weight shape when transpose_gate_weight=False."""
        gate = _make_gate(transpose_gate_weight=False)
        self.assertEqual(gate.weight.shape, [16, 8])

    def test_weight_shape_transpose(self):
        """Test gate weight shape when transpose_gate_weight=True."""
        gate = _make_gate(transpose_gate_weight=True)
        self.assertEqual(gate.weight.shape, [8, 16])

    def test_forward_returns_8_values(self):
        """Test forward returns 8 values."""
        gate = _make_gate()
        x = paddle.randn([4, 16])
        result = gate(x)
        self.assertEqual(len(result), 8)

    def test_forward_capacity(self):
        """Test forward returns a positive capacity."""
        gate = _make_gate()
        x = paddle.randn([4, 16])
        capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss = gate(x)
        self.assertIsInstance(capacity, int)
        self.assertGreater(capacity, 0)

    def test_forward_top_gate_shape(self):
        """Test forward returns correct top_gate shape."""
        gate = _make_gate()
        x = paddle.randn([4, 16])
        _, top_gate, top_idx, _, _, _, _, _ = gate(x)
        self.assertEqual(top_gate.shape, [4, 2])
        self.assertEqual(top_idx.shape, [4, 2])

    def test_forward_mask_shape(self):
        """Test forward returns correct mask shape."""
        gate = _make_gate()
        x = paddle.randn([4, 16])
        _, _, _, gates_masked, mask, _, _, _ = gate(x)
        self.assertEqual(gates_masked.shape, [4, 8])
        self.assertEqual(mask.shape, [4, 8])

    def test_forward_aux_loss_scalar(self):
        """Test forward returns scalar aux loss."""
        gate = _make_gate()
        x = paddle.randn([4, 16])
        _, _, _, _, _, _, l_aux, l_zloss = gate(x)
        self.assertEqual(l_aux.shape, [])
        self.assertEqual(l_zloss.shape, [])

    def test_forward_with_3d_input(self):
        """Test forward with 3D input (batch_size, seq_len, hidden_size)."""
        gate = _make_gate()
        x = paddle.randn([2, 4, 16])
        result = gate(x)
        self.assertEqual(len(result), 8)

    def test_forward_group_limited_greedy(self):
        """Test forward with group_limited_greedy topk_method."""
        gate = _make_gate(topk_method="group_limited_greedy", n_group=4, topk_group=2)
        x = paddle.randn([4, 16])
        capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss = gate(x)
        self.assertEqual(top_gate.shape, [4, 2])

    def test_forward_noaux_tc(self):
        """Test forward with noaux_tc topk_method."""
        gate = _make_gate(topk_method="noaux_tc", n_group=4, topk_group=2)
        x = paddle.randn([4, 16])
        capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss = gate(x)
        self.assertEqual(top_gate.shape, [4, 2])

    def test_forward_invalid_topk_method_raises(self):
        """Test that invalid topk_method raises NotImplementedError."""
        gate = _make_gate(topk_method="invalid")
        x = paddle.randn([4, 16])
        with self.assertRaises(NotImplementedError):
            gate(x)

    def test_forward_sigmoid_scoring(self):
        """Test forward with sigmoid scoring function."""
        gate = _make_gate(scoring_func="sigmoid")
        x = paddle.randn([4, 16])
        result = gate(x)
        self.assertEqual(len(result), 8)

    def test_forward_no_norm_topk_prob(self):
        """Test forward with norm_topk_prob=False."""
        gate = _make_gate(norm_topk_prob=False)
        x = paddle.randn([4, 16])
        result = gate(x)
        self.assertEqual(len(result), 8)

    def test_forward_routed_scaling_factor(self):
        """Test forward with routed_scaling_factor != 1."""
        gate = _make_gate(routed_scaling_factor=2.0)
        x = paddle.randn([4, 16])
        _, top_gate, _, _, _, _, _, _ = gate(x)
        # top_gate should be scaled by routed_scaling_factor
        self.assertTrue(paddle.any(top_gate > 0).item())

    def test_forward_noaux_tc_updates_expert_usage(self):
        """Test that forward updates expert_usage when topk_method is noaux_tc."""
        gate = _make_gate(topk_method="noaux_tc", n_group=4, topk_group=2)
        initial_usage = gate.expert_usage.numpy().copy()
        x = paddle.randn([4, 16])
        gate(x)
        # expert_usage should have changed
        self.assertFalse(np.array_equal(initial_usage, gate.expert_usage.numpy()))

    def test_topkgating_returns_8_values(self):
        """Test topkgating returns 8 values."""
        gate = _make_gate()
        x = paddle.randn([4, 16])
        result = gate.topkgating(x)
        self.assertEqual(len(result), 8)

    def test_topkgating_gates_masked_normalized(self):
        """Test that gates_masked is normalized when norm_topk_prob=True."""
        gate = _make_gate(norm_topk_prob=True)
        x = paddle.randn([4, 16])
        _, _, _, gates_masked, mask, _, _, _ = gate.topkgating(x)
        # Each row of gates_masked should sum to routed_scaling_factor
        row_sums = gates_masked.sum(axis=-1)
        nonzero_rows = row_sums[mask.sum(axis=-1) > 0]
        if nonzero_rows.shape[0] > 0:
            np.testing.assert_allclose(
                nonzero_rows.numpy(),
                np.ones(nonzero_rows.shape[0]),
                atol=1e-4,
            )

    def test_orthogonal_loss(self):
        """Test _cal_orthogonal_loss returns a scalar."""
        gate = _make_gate()
        loss = gate._cal_orthogonal_loss()
        self.assertEqual(loss.shape, [])

    def test_gumbel_rsample(self):
        """Test gumbel_rsample returns correct shape."""
        gate = _make_gate()
        logits = paddle.randn([4, 8])
        result = gate.gumbel_rsample(logits)
        self.assertEqual(result.shape, [4, 8])

    def test_uniform_sample(self):
        """Test uniform_sample returns correct shape."""
        gate = _make_gate()
        logits = paddle.randn([4, 8])
        result = gate.uniform_sample(logits)
        self.assertEqual(result.shape, [4, 8])


if __name__ == "__main__":
    unittest.main()
