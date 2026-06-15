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


class TestLanguageLossInit(unittest.TestCase):
    """Test LanguageLoss initialization."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_init_no_parallel(self, mock_dist, mock_tp):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = LanguageLoss(config=mock_config)
        self.assertFalse(loss.enable_parallel_cross_entropy)
        self.assertFalse(loss.use_subbatch)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=4,
    )
    @patch("paddle.distributed.is_initialized", return_value=True)
    @patch("paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy")
    def test_init_with_parallel(self, mock_pce, mock_dist, mock_tp):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = LanguageLoss(config=mock_config)
        self.assertTrue(loss.enable_parallel_cross_entropy)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_init_with_subbatch(self, mock_dist, mock_tp):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 512

        loss = LanguageLoss(config=mock_config)
        self.assertTrue(loss.use_subbatch)


class TestLanguageLossForwardImpl(unittest.TestCase):
    """Test LanguageLoss.forward_impl method."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_impl_basic(self, mock_dist, mock_tp, mock_cp):
        import paddle

        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 10, 100])
        labels = paddle.randint(0, 100, [2, 10])
        result = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(result)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_impl_all_ignored(self, mock_dist, mock_tp, mock_cp):
        import paddle

        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 10, 100])
        labels = paddle.full([2, 10], -100, dtype="int64")
        result = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(result)


class TestLanguageLossForwardWithMTP(unittest.TestCase):
    """Test LanguageLoss.forward with Multi-Token Prediction."""

    @patch("paddle.device.cuda.empty_cache")
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_list_logits(self, mock_dist, mock_tp, mock_cp, mock_cache):
        import paddle

        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.num_nextn_predict_layers = 2
        mock_config.mtp_load_weight_only = False
        mock_config.mtp_distillation_loss = False
        mock_config.train_mtp_only = False
        mock_config.add_mtp_loss = False
        mock_config.mtp_loss_scaling_factor = 1.0
        mock_config.recompute_modules = None
        mock_config.gpt_model_use_experimental_version = False

        loss_fn = LanguageLoss(config=mock_config)
        logits = [
            paddle.randn([2, 10, 100]),
            paddle.randn([2, 10, 100]),
            paddle.randn([2, 10, 100]),
        ]
        labels = paddle.randint(0, 100, [2, 12])
        result = loss_fn.forward(logits, labels)
        self.assertIsNotNone(result)


class TestLanguageLossBuildScheduleNode(unittest.TestCase):
    """Test LanguageLoss.build_schedule_node."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_returns_schedule_node(self, mock_dist, mock_tp):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss_fn = LanguageLoss(config=mock_config)
        node = loss_fn.build_schedule_node()
        self.assertIsNotNone(node)


class TestDistributedSoftmaxOp(unittest.TestCase):
    """Test DistributedSoftmaxOp static methods."""

    def test_forward_method_exists(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            DistributedSoftmaxOp,
        )

        self.assertTrue(hasattr(DistributedSoftmaxOp, "forward"))
        self.assertTrue(callable(DistributedSoftmaxOp.forward))

    def test_backward_method_exists(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            DistributedSoftmaxOp,
        )

        self.assertTrue(hasattr(DistributedSoftmaxOp, "backward"))
        self.assertTrue(callable(DistributedSoftmaxOp.backward))


class TestSubbatch(unittest.TestCase):
    """Test subbatch function."""

    def test_subbatch_small_input(self):
        import paddle

        from paddleformers.fleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        def simple_fn(x, y):
            return x + y

        sb_fn = subbatch(simple_fn, arg_idx=[0, 1], axis=[0, 0], bs=100, out_idx=0)
        x = paddle.randn([5, 10])
        y = paddle.randn([5, 10])
        result = sb_fn(x, y)
        # Input smaller than batch size, should call function directly
        self.assertIsNotNone(result)

    def test_subbatch_equal_batch_size(self):
        import paddle

        from paddleformers.fleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        def simple_fn(x, y):
            return x + y

        sb_fn = subbatch(simple_fn, arg_idx=[0, 1], axis=[0, 0], bs=5, out_idx=0)
        x = paddle.randn([5, 10])
        y = paddle.randn([5, 10])
        result = sb_fn(x, y)
        self.assertIsNotNone(result)

    def test_subbatch_assert_arg_axis_length(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            subbatch,
        )

        def simple_fn(x, y):
            return x + y

        # Mismatched arg_idx and axis lengths should raise
        sb_fn = subbatch(simple_fn, arg_idx=[0], axis=[0, 1], bs=10, out_idx=0)
        import paddle

        x = paddle.randn([10, 10])
        y = paddle.randn([10, 10])
        with self.assertRaises(AssertionError):
            sb_fn(x, y)


class TestLanguageLossMTPTracker(unittest.TestCase):
    """Test LanguageLoss.mtp_loss_tracker class attribute."""

    def test_tracker_is_dict(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        self.assertIsInstance(LanguageLoss.mtp_loss_tracker, dict)

    def test_tracker_initially_empty(self):
        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        LanguageLoss.mtp_loss_tracker.clear()
        self.assertEqual(len(LanguageLoss.mtp_loss_tracker), 0)


class TestLanguageLossForwardSingleLogits(unittest.TestCase):
    """Test LanguageLoss.forward with single (non-list) logits."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_single_logits(self, mock_dist, mock_tp, mock_cp):
        import paddle

        from paddleformers.fleet.models.common.language_loss.language_loss import (
            LanguageLoss,
        )

        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.recompute_modules = None

        loss_fn = LanguageLoss(config=mock_config)
        logits = paddle.randn([2, 10, 100])
        labels = paddle.randint(0, 100, [2, 10])
        result = loss_fn.forward(logits, labels)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
