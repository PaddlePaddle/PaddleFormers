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


# Tests for paddleformers/fleet/context_parallel_utils.py

import unittest
from unittest import mock


class TestContextParallelUtils(unittest.TestCase):
    """Tests for context_parallel_utils module."""

    def test_mark_context_parallel_parameter_disable_scale_grad_layer(self):
        """Test marking a layer to disable scale grad."""
        from paddle import nn

        from paddleformers.fleet.context_parallel_utils import (
            mark_context_parallel_parameter_disable_scale_grad,
        )

        layer = nn.Linear(10, 10)
        mark_context_parallel_parameter_disable_scale_grad(layer)
        self.assertTrue(layer.weight.context_parallel_disable_scale_grad)
        self.assertTrue(layer.bias.context_parallel_disable_scale_grad)

    def test_mark_context_parallel_parameter_disable_scale_grad_layer_no_bias(
        self,
    ):
        """Test marking a layer without bias."""
        from paddle import nn

        from paddleformers.fleet.context_parallel_utils import (
            mark_context_parallel_parameter_disable_scale_grad,
        )

        layer = nn.Linear(10, 10, bias_attr=False)
        mark_context_parallel_parameter_disable_scale_grad(layer)
        self.assertTrue(layer.weight.context_parallel_disable_scale_grad)

    def test_mark_context_parallel_parameter_disable_scale_grad_param(self):
        """Test marking a parameter directly."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            context_parallel_parameter_disable_scale_grad,
            mark_context_parallel_parameter_disable_scale_grad,
        )

        param = paddle.randn([10, 10])
        mark_context_parallel_parameter_disable_scale_grad(param)
        self.assertTrue(context_parallel_parameter_disable_scale_grad(param))

    def test_mark_context_parallel_parameter_disable_scale_grad_invalid_type(
        self,
    ):
        """Test marking an invalid type raises TypeError."""
        from paddleformers.fleet.context_parallel_utils import (
            mark_context_parallel_parameter_disable_scale_grad,
        )

        with self.assertRaises(TypeError):
            mark_context_parallel_parameter_disable_scale_grad("not_a_param")

    def test_context_parallel_parameter_disable_scale_grad_default(self):
        """Test default value of context_parallel_parameter_disable_scale_grad."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            context_parallel_parameter_disable_scale_grad,
        )

        param = paddle.randn([10, 10])
        self.assertFalse(context_parallel_parameter_disable_scale_grad(param))

    def test_scatter_balance_single_rank(self):
        """Test scatter_balance with single rank returns clone."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import scatter_balance

        group = mock.MagicMock()
        group.nranks = 1
        x = paddle.randn([4, 8])
        result = scatter_balance(x, group=group)
        self.assertEqual(result.shape, [4, 8])

    def test_scatter_balance_multi_rank(self):
        """Test scatter_balance with multi-rank."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import scatter_balance

        group = mock.MagicMock()
        group.nranks = 2
        group.rank = 0
        x = paddle.randn([8, 4], dtype="float32")
        # seq_len=8, parallelism=2, 8 % (2*2)=0 OK
        result = scatter_balance(x, group=group, axis=0)
        self.assertEqual(result.shape[0], 4)

    def test_scatter_balance_invalid_seq_len(self):
        """Test scatter_balance with invalid seq_len raises."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import scatter_balance

        group = mock.MagicMock()
        group.nranks = 2
        group.rank = 0
        x = paddle.randn([5, 4], dtype="float32")
        # 5 % (2*2) = 1, not 0
        with self.assertRaises(AssertionError):
            scatter_balance(x, group=group, axis=0)

    def test_scatter_balance_default_group(self):
        """Test scatter_balance with default group from fleet."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import scatter_balance

        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group.nranks = 1
        mock_hcg.get_model_parallel_group.return_value = mock_group
        x = paddle.randn([4, 8])
        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = scatter_balance(x)
        self.assertEqual(result.shape, [4, 8])

    def test_all_gather_balance_single_rank(self):
        """Test all_gather_balance with single rank."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import all_gather_balance

        group = mock.MagicMock()
        group.nranks = 1
        x = paddle.randn([4, 8])
        result = all_gather_balance(x, group=group)
        self.assertEqual(result.shape, [4, 8])

    def test_all_gather_balance_axis0(self):
        """Test all_gather_balance axis=0 path."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import all_gather_balance

        group = mock.MagicMock()
        group.nranks = 2
        group.rank = 0
        x = paddle.randn([4, 8], dtype="float32")

        with (
            mock.patch("paddle.distributed.stream.all_gather"),
            mock.patch(
                "paddleformers.fleet.triton_ops.balanced_reorder.balanced_gather_reorder_kernel"
            ),
        ):
            result = all_gather_balance(x, group=group, axis=0)
            # Should call all_gather twice (start and end chunks)

    def test_all_gather_balance_axis1(self):
        """Test all_gather_balance axis=1 path."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import all_gather_balance

        group = mock.MagicMock()
        group.nranks = 2
        group.rank = 0
        x = paddle.randn([4, 4], dtype="float32")

        with (
            mock.patch("paddle.distributed.stream.all_gather"),
            mock.patch(
                "paddleformers.fleet.triton_ops.balanced_reorder.balanced_gather_reorder_kernel"
            ),
        ):
            result = all_gather_balance(x, group=group, axis=1)

    def test_reduce_scatter_any_axis_single_rank(self):
        """Test reduce_scatter_any_axis with single rank."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import reduce_scatter_any_axis

        group = mock.MagicMock()
        group.nranks = 1
        x = paddle.randn([4, 8])
        result = reduce_scatter_any_axis(x, axis=0, group=group)
        self.assertEqual(result.shape, [4, 8])

    def test_reduce_scatter_any_axis_axis0(self):
        """Test reduce_scatter_any_axis axis=0 path."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import reduce_scatter_any_axis

        group = mock.MagicMock()
        group.nranks = 2
        x = paddle.randn([8, 4], dtype="float32")

        with (
            mock.patch("paddle.distributed.stream.reduce_scatter"),
            mock.patch("paddle.distributed.ReduceOp") as mock_reduce_op,
        ):
            mock_reduce_op.SUM = "sum"
            result = reduce_scatter_any_axis(x, axis=0, group=group)

    def test_reduce_scatter_any_axis_invalid(self):
        """Test reduce_scatter_any_axis with invalid size raises."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import reduce_scatter_any_axis

        group = mock.MagicMock()
        group.nranks = 2
        x = paddle.randn([5, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            reduce_scatter_any_axis(x, axis=0, group=group)

    def test_reduce_scatter_any_axis_balance_single_rank(self):
        """Test reduce_scatter_any_axis_balance with single rank."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_any_axis_balance,
        )

        group = mock.MagicMock()
        group.nranks = 1
        x = paddle.randn([4, 8])
        result = reduce_scatter_any_axis_balance(x, axis=0, group=group)
        self.assertEqual(result.shape, [4, 8])

    def test_reduce_scatter_any_axis_balance_multi_rank(self):
        """Test reduce_scatter_any_axis_balance with multi-rank."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_any_axis_balance,
        )

        group = mock.MagicMock()
        group.nranks = 2
        x = paddle.randn([8, 4], dtype="float32")

        with (
            mock.patch("paddle.distributed.stream.alltoall_single"),
            mock.patch(
                "paddleformers.fleet.triton_ops.balanced_reorder.balanced_scatter_reorder_kernel"
            ),
        ):
            result = reduce_scatter_any_axis_balance(x, axis=0, group=group)

    def test_reduce_scatter_any_axis_balance_invalid(self):
        """Test reduce_scatter_any_axis_balance with invalid size raises."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            reduce_scatter_any_axis_balance,
        )

        group = mock.MagicMock()
        group.nranks = 2
        x = paddle.randn([5, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            reduce_scatter_any_axis_balance(x, axis=0, group=group)

    def test_preprocess_index(self):
        """Test preprocess_index function."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import preprocess_index

        indices = paddle.to_tensor([[0, 10], [5, 15]], dtype="int64")
        result = preprocess_index(
            indices, chunk_id=1, seq_blocksize=16, max_seqlen_q=8
        )
        self.assertEqual(result.shape, [2, 2])

    def test_preprocess_index_dual_chunks(self):
        """Test preprocess_index_dual_chunks function."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            preprocess_index_dual_chunks,
        )

        indices = paddle.to_tensor([[0, 10], [5, 15]], dtype="int64")
        result = preprocess_index_dual_chunks(
            indices,
            chunk_id_first=0,
            chunk_id_second=1,
            seq_blocksize=16,
            max_seqlen_q=8,
        )
        self.assertEqual(result.shape, [2, 2])

    def test_scatter_with_padding_normal_rank(self):
        """Test scatter_with_padding for normal rank."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import scatter_with_padding

        group = mock.MagicMock()
        group.nranks = 2
        group.rank = 0
        x = paddle.randn([8, 4], dtype="float32")
        result = scatter_with_padding(x, num_pad=0, axis=0, group=group)
        self.assertEqual(result.shape[0], 4)

    def test_scatter_with_padding_beyond_rank_idx(self):
        """Test scatter_with_padding for rank beyond split."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import scatter_with_padding

        group = mock.MagicMock()
        group.nranks = 2
        group.rank = 5
        x = paddle.randn([8, 4], dtype="float32")
        result = scatter_with_padding(x, num_pad=0, axis=0, group=group)
        self.assertEqual(result.shape[0], 4)

    def test_all_gather_without_padding_no_pad(self):
        """Test all_gather_without_padding without padding."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            all_gather_without_padding,
        )

        group = mock.MagicMock()
        group.nranks = 2
        x = paddle.randn([4, 4], dtype="float32")
        with mock.patch("paddle.distributed.stream.all_gather"):
            result = all_gather_without_padding(
                x, num_pad=0, axis=0, group=group
            )

    def test_all_gather_without_padding_with_pad(self):
        """Test all_gather_without_padding with padding."""
        import paddle

        from paddleformers.fleet.context_parallel_utils import (
            all_gather_without_padding,
        )

        group = mock.MagicMock()
        group.nranks = 2
        x = paddle.randn([4, 4], dtype="float32")
        with mock.patch("paddle.distributed.stream.all_gather"):
            result = all_gather_without_padding(
                x, num_pad=2, axis=0, group=group
            )

    def test_FlashMaskContextParallel_dropout_raises(self):
        """Test FlashMaskContextParallel raises on dropout."""
        from paddleformers.fleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        mock_q = mock.MagicMock()
        mock_q.shape = [2, 8, 4, 16]
        with self.assertRaises(NotImplementedError):
            FlashMaskContextParallel.forward(
                mock_ctx,
                mock_q,
                mock_q,
                mock_q,
                mock_q,
                dropout=0.5,
            )

    def test_FlashMaskContextParallel_causal_raises(self):
        """Test FlashMaskContextParallel raises on causal."""
        from paddleformers.fleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        mock_q = mock.MagicMock()
        mock_q.shape = [2, 8, 4, 16]
        with self.assertRaises(NotImplementedError):
            FlashMaskContextParallel.forward(
                mock_ctx,
                mock_q,
                mock_q,
                mock_q,
                mock_q,
                causal=True,
            )

    def test_FlashMaskContextParallel_fixed_seed_raises(self):
        """Test FlashMaskContextParallel raises on fixed seed offset."""
        from paddleformers.fleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        mock_q = mock.MagicMock()
        mock_q.shape = [2, 8, 4, 16]
        with self.assertRaises(NotImplementedError):
            FlashMaskContextParallel.forward(
                mock_ctx,
                mock_q,
                mock_q,
                mock_q,
                mock_q,
                fixed_seed_offset=mock.MagicMock(),
            )

    def test_FlashMaskContextParallel_odd_seq_len_raises(self):
        """Test FlashMaskContextParallel raises on odd sequence length."""
        from paddleformers.fleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        mock_q = mock.MagicMock()
        mock_q.shape = [2, 7, 4, 16]  # 7 is odd

        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group.rank = 0
        mock_group.world_size = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=mock_hcg,
            ),
            self.assertRaises(AssertionError),
        ):
            FlashMaskContextParallel.forward(
                mock_ctx,
                mock_q,
                mock_q,
                mock_q,
                mock_q,
            )
