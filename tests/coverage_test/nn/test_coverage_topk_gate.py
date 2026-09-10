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

import paddle
import paddle.nn.functional as F


def _randn(shape, dtype="float32"):
    return paddle.randn(shape, dtype=dtype)


def _make_fake_group():
    """Create a fake distributed group for testing."""
    group = MagicMock()
    group.nranks = 1
    group.rank = 0
    group.ranks = [0]
    return group


class FakeGateConfig:
    """Fake config with dict-like get() for TopKGate."""

    def __init__(self, **overrides):
        self.hidden_size = 64
        self.moe_num_experts = 8
        self.moe_capacity = [1.0, 1.0, 1.0]
        self.moe_k = 2
        self.fuse_gate_detach_matmul = False
        self.scoring_func = "softmax"
        self.global_aux_loss = False
        self.sinkhorn_2gate = False
        self.sinkhorn_temp = 1.0
        self.moe_use_aux_free = False
        self.router_aux_loss_coef = 0.01
        self.router_z_loss_coef = 0.0
        self.moe_orthogonal_loss_lambda = 0.0
        self.moe_norm_gate_logits = False
        self.moe_group_experts = False
        self.moe_use_token_type_bias = False
        self.moe_world_size = 1
        self.multimodel_experts = False
        self.moe_use_hard_gate = False
        self.moe_group_orthogonal_loss = False
        for k, v in overrides.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


class TestMaskedFill(unittest.TestCase):
    """Tests for masked_fill function."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import masked_fill

        self.masked_fill = masked_fill

    def test_basic_mask(self):
        """Test basic masked fill operation."""
        x = paddle.ones([4, 4])
        mask = paddle.zeros([4, 4], dtype="bool")
        mask[0, 0] = True
        mask[1, 1] = True
        result = self.masked_fill(x, mask, -1e9)
        self.assertAlmostEqual(result[0, 0].item(), -1e9, places=3)
        self.assertAlmostEqual(result[1, 1].item(), -1e9, places=3)
        self.assertAlmostEqual(result[2, 2].item(), 1.0, places=3)

    def test_all_false_mask(self):
        """Test with all-False mask (no elements filled)."""
        x = paddle.ones([2, 3])
        mask = paddle.zeros([2, 3], dtype="bool")
        result = self.masked_fill(x, mask, 0.0)
        self.assertTrue(paddle.allclose(result, x))

    def test_all_true_mask(self):
        """Test with all-True mask (all elements filled)."""
        x = paddle.ones([2, 3])
        mask = paddle.ones([2, 3], dtype="bool")
        result = self.masked_fill(x, mask, -100.0)
        self.assertTrue(paddle.allclose(result, paddle.full([2, 3], -100.0, x.dtype)))


class TestCastIfNeeded(unittest.TestCase):
    """Tests for cast_if_needed function."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import cast_if_needed

        self.cast_if_needed = cast_if_needed

    def test_same_dtype(self):
        """Test when tensor is already the target dtype."""
        x = _randn([4], "float32")
        result = self.cast_if_needed(x, paddle.float32)
        self.assertEqual(result.dtype, paddle.float32)

    def test_different_dtype(self):
        """Test when tensor needs casting."""
        x = _randn([4], "float16")
        result = self.cast_if_needed(x, paddle.float32)
        self.assertEqual(result.dtype, paddle.float32)


class TestGateDetachMatmul(unittest.TestCase):
    """Tests for gate_detach_matmul function."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import gate_detach_matmul

        self.gate_detach_matmul = gate_detach_matmul

    def test_without_fuse(self):
        """Test without fuse."""
        x = _randn([4, 8])
        weight = _randn([8, 4])
        result = self.gate_detach_matmul(x, weight, use_fuse=False)
        self.assertEqual(result.shape, [4, 4])

    def test_with_fuse(self):
        """Test with fuse (uses FusedGateDetachMatmul)."""
        x = _randn([4, 8])
        weight = _randn([8, 4])
        result = self.gate_detach_matmul(x, weight, use_fuse=True)
        self.assertEqual(result.shape, [4, 4])


class TestFusedGateDetachMatmul(unittest.TestCase):
    """Tests for FusedGateDetachMatmul class."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import FusedGateDetachMatmul

        self.FusedGateDetachMatmul = FusedGateDetachMatmul

    def test_forward(self):
        """Test forward pass."""
        x = _randn([4, 8])
        w = _randn([8, 4])
        result = self.FusedGateDetachMatmul.apply(x, w)
        self.assertEqual(result.shape, [4, 4])

    def test_backward_weight_not_stop_gradient(self):
        """Test backward pass when weight requires grad."""
        x = _randn([4, 8])
        x.stop_gradient = False
        w = _randn([8, 4])
        w.stop_gradient = False
        result = self.FusedGateDetachMatmul.apply(x, w)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_backward_weight_stop_gradient(self):
        """Test backward pass when weight has stop_gradient=True."""
        x = _randn([4, 8])
        x.stop_gradient = False
        w = _randn([8, 4])
        w.stop_gradient = True
        result = self.FusedGateDetachMatmul.apply(x, w)
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(w.grad)


class TestTopKGateInit(unittest.TestCase):
    """Tests for TopKGate __init__."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import TopKGate

        self.TopKGate = TopKGate

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_basic_init_softmax(self, mock_rank):
        """Test basic initialization with softmax scoring."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        self.assertEqual(gate.model_dim, 64)
        self.assertEqual(gate.num_experts, 8)
        self.assertEqual(gate.config.scoring_func, "softmax")
        self.assertFalse(gate.use_multimodel_experts)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_init_sigmoid(self, mock_rank):
        """Test initialization with sigmoid scoring."""
        config = FakeGateConfig(scoring_func="sigmoid")
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        self.assertEqual(gate.config.scoring_func, "sigmoid")

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_invalid_scoring_func(self, mock_rank):
        """Test that invalid scoring func raises ValueError."""
        config = FakeGateConfig(scoring_func="invalid")
        with self.assertRaises(ValueError):
            self.TopKGate(config, layer_idx=0, group=_make_fake_group())

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_init_with_gate_weight(self, mock_rank):
        """Test initialization with external gate_weight."""
        config = FakeGateConfig()
        gate_weight = _randn([64, 8])
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group(), gate_weight=gate_weight)
        self.assertIs(gate.weight, gate_weight)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_init_global_aux_loss(self, mock_rank):
        """Test initialization with global_aux_loss=True."""
        config = FakeGateConfig(global_aux_loss=True)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        self.assertTrue(gate.global_aux_loss)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_init_multimodel_experts(self, mock_rank):
        """Test initialization with multimodel_experts."""
        config = FakeGateConfig(moe_num_experts=[4, 4], multimodel_experts=True)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        self.assertTrue(gate.use_multimodel_experts)
        self.assertEqual(gate.num_experts, [4, 4])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_init_multimodel_experts_hard_gate(self, mock_rank):
        """Test initialization with multimodel_experts and hard gate."""
        config = FakeGateConfig(
            moe_num_experts=[4, 4],
            multimodel_experts=True,
            moe_use_hard_gate=True,
            moe_group_experts=True,
            moe_world_size=1,
        )
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        self.assertTrue(gate.use_multimodel_experts)
        self.assertIsNotNone(gate.experts_type_ids)


class TestTopKGateForward(unittest.TestCase):
    """Tests for TopKGate forward method."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import TopKGate

        self.TopKGate = TopKGate

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_basic_forward(self, mock_rank):
        """Test basic forward pass."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        x = _randn([8, 64])
        logits, capacity, router_loss = gate(x)
        self.assertEqual(logits.shape, [8, 8])
        self.assertIsInstance(capacity, int)
        self.assertEqual(router_loss.shape, [1])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_forward_transform_weight_false(self, mock_rank):
        """Test forward with transform_weight=False."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        x = _randn([8, 64])
        logits, capacity, router_loss = gate(x, transform_weight=False)
        self.assertEqual(logits.shape, [8, 8])


class TestTopKGateGetCapacity(unittest.TestCase):
    """Tests for TopKGate get_capacity method."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import TopKGate

        self.TopKGate = TopKGate

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_training_capacity(self, mock_rank):
        """Test capacity during training."""
        config = FakeGateConfig(moe_capacity=[1.0, 1.5, 2.0])
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate.training = True
        cap = gate.get_capacity(16)
        self.assertEqual(cap, 2)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_eval_small_tokens(self, mock_rank):
        """Test capacity during eval with small number of tokens."""
        config = FakeGateConfig(moe_capacity=[1.0, 1.5, 2.0])
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate.training = False
        cap = gate.get_capacity(4)
        self.assertEqual(cap, 1)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_eval_large_tokens(self, mock_rank):
        """Test capacity during eval with large number of tokens."""
        config = FakeGateConfig(moe_capacity=[1.0, 1.5, 2.0])
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate.training = False
        cap = gate.get_capacity(32)
        self.assertEqual(cap, 6)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_custom_cap_factor(self, mock_rank):
        """Test capacity with custom cap_factor."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        cap = gate.get_capacity(16, cap_factor=2.0)
        self.assertEqual(cap, 4)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_capacity_assertion(self, mock_rank):
        """Test that capacity assertion fails for very small cap."""
        config = FakeGateConfig(moe_capacity=[0.001, 0.001, 0.001])
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate.training = True
        with self.assertRaises(AssertionError):
            gate.get_capacity(8)


class TestTopKGateGetGateWeight(unittest.TestCase):
    """Tests for TopKGate get_gate_weight method."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import TopKGate

        self.TopKGate = TopKGate

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_non_multimodel(self, mock_rank):
        """Test get_gate_weight without multimodel_experts."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        w = gate.get_gate_weight(transform_weight=True)
        self.assertEqual(w.shape, [64, 8])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_non_multimodel_no_transform(self, mock_rank):
        """Test get_gate_weight without transform when not multimodel."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        w = gate.get_gate_weight(transform_weight=False)
        self.assertIs(w, gate.weight)

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_multimodel_transform(self, mock_rank):
        """Test get_gate_weight with multimodel_experts and transform."""
        config = FakeGateConfig(
            moe_num_experts=[4, 4],
            multimodel_experts=True,
            moe_group_experts=False,
            moe_world_size=1,
        )
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        w = gate.get_gate_weight(transform_weight=True)
        self.assertEqual(w.shape, [64, 8])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_multimodel_no_transform(self, mock_rank):
        """Test get_gate_weight with multimodel_experts and no transform."""
        config = FakeGateConfig(
            moe_num_experts=[4, 4],
            multimodel_experts=True,
            moe_group_experts=False,
            moe_world_size=1,
        )
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        w = gate.get_gate_weight(transform_weight=False)
        self.assertEqual(w.shape, [64, 8])


class TestTopKGateCalAuxLoss(unittest.TestCase):
    """Tests for TopKGate _cal_aux_loss method."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import TopKGate

        self.TopKGate = TopKGate

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_basic_aux_loss(self, mock_rank):
        """Test basic auxiliary loss computation."""
        config = FakeGateConfig(moe_k=2)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate_prob = F.softmax(_randn([8, 8]))
        dispatch_mask = paddle.randint(0, 8, [8])
        with patch("paddleformers.nn.moe.topk_gate.cal_aux_loss") as mock_cal:
            mock_cal.return_value = (paddle.zeros([]), paddle.to_tensor(8.0), paddle.zeros([8]))
            loss = gate._cal_aux_loss(gate_prob, dispatch_mask)
        self.assertEqual(loss.shape, [])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_aux_loss_sigmoid(self, mock_rank):
        """Test aux loss with sigmoid scoring (normalizes gate_prob)."""
        config = FakeGateConfig(scoring_func="sigmoid", moe_k=2)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate_prob = F.sigmoid(_randn([8, 8]))
        dispatch_mask = paddle.randint(0, 8, [8])
        with patch("paddleformers.nn.moe.topk_gate.cal_aux_loss") as mock_cal:
            mock_cal.return_value = (paddle.zeros([]), paddle.to_tensor(8.0), paddle.zeros([8]))
            loss = gate._cal_aux_loss(gate_prob, dispatch_mask)
        self.assertEqual(loss.shape, [])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_aux_loss_with_correction_bias_no_tokens_mask(self, mock_rank):
        """Test aux loss with correction_bias but no tokens_mask."""
        config = FakeGateConfig(moe_use_aux_free=True, moe_k=2)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate_prob = F.softmax(_randn([8, 8]))
        dispatch_mask = paddle.randint(0, 8, [8])
        with patch("paddleformers.nn.moe.topk_gate.int_bincount") as mock_bc:
            mock_bc.return_value = paddle.zeros([8], dtype="int64")
            with patch("paddleformers.nn.moe.topk_gate.dist.stream.all_reduce"):
                with patch("paddleformers.nn.moe.topk_gate.cal_aux_loss") as mock_cal:
                    mock_cal.return_value = (paddle.zeros([]), paddle.to_tensor(8.0), paddle.zeros([8]))
                    loss = gate._cal_aux_loss(gate_prob, dispatch_mask)
        self.assertEqual(loss.shape, [])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_aux_loss_fallback_path(self, mock_rank):
        """Test aux loss fallback path when shape condition not met."""
        config = FakeGateConfig(moe_k=2)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        # gate_prob [4, 8] where rows < cols triggers fallback
        gate_prob = F.softmax(_randn([4, 8]))
        # dispatch_mask should match vocab dimension (8), not batch (4)
        dispatch_mask = paddle.randint(0, 8, [8])
        loss = gate._cal_aux_loss(gate_prob, dispatch_mask)
        self.assertEqual(loss.shape, [])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_aux_loss_with_group(self, mock_rank):
        """Test aux loss with moe_group_experts."""
        config = FakeGateConfig(moe_k=2, moe_group_experts=True)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        gate_prob = F.softmax(_randn([8, 8]))
        dispatch_mask = paddle.randint(0, 8, [8])
        loss = gate._cal_aux_loss(gate_prob, dispatch_mask)
        self.assertEqual(loss.shape, [])


class TestTopKGateCalZLoss(unittest.TestCase):
    """Tests for TopKGate _cal_z_loss method."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import TopKGate

        self.TopKGate = TopKGate

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_z_loss_no_mask(self, mock_rank):
        """Test z-loss without loss_mask."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        logits = _randn([8, 8])
        loss = gate._cal_z_loss(logits)
        self.assertEqual(loss.shape, [])

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_z_loss_with_mask(self, mock_rank):
        """Test z-loss with loss_mask."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        logits = _randn([8, 8])
        loss_mask = paddle.ones([8])
        loss = gate._cal_z_loss(logits, loss_mask=loss_mask)
        self.assertEqual(loss.shape, [])


class TestTopKGateCalOrthogonalLoss(unittest.TestCase):
    """Tests for TopKGate _cal_orthogonal_loss methods."""

    def setUp(self):
        from paddleformers.nn.moe.topk_gate import TopKGate

        self.TopKGate = TopKGate

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_orthogonal_loss_no_group(self, mock_rank):
        """Test orthogonal loss without group."""
        config = FakeGateConfig(moe_group_experts=False, moe_group_orthogonal_loss=False)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        loss = gate._cal_orthogonal_loss()
        self.assertEqual(loss.shape, [1])  # _squared_l2_norm returns [1]

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_orthogonal_loss_with_group(self, mock_rank):
        """Test orthogonal loss with group."""
        config = FakeGateConfig(moe_k=2, moe_group_experts=True, moe_group_orthogonal_loss=True)
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        loss = gate._cal_orthogonal_loss()
        self.assertEqual(loss.shape, [1])  # _squared_l2_norm returns [1]

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_orthogonal_loss_with_weight_id(self, mock_rank):
        """Test orthogonal loss with specific weight_id."""
        config = FakeGateConfig()
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        loss = gate._cal_orthogonal_loss(weight_id=0)
        self.assertEqual(loss.shape, [1])  # _squared_l2_norm returns [1]

    @patch("paddleformers.nn.moe.topk_gate.dist.get_rank", return_value=0)
    def test_orthogonal_loss_multimodel(self, mock_rank):
        """Test orthogonal loss with multimodel_experts."""
        config = FakeGateConfig(
            moe_num_experts=[4, 4],
            multimodel_experts=True,
            moe_group_experts=False,
            moe_world_size=1,
        )
        gate = self.TopKGate(config, layer_idx=0, group=_make_fake_group())
        loss = gate._cal_orthogonal_loss()
        self.assertEqual(loss.shape, [1])  # _squared_l2_norm returns [1]


if __name__ == "__main__":
    unittest.main()
