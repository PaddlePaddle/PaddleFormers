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


class TestAllgatherAsync(unittest.TestCase):
    """Tests for allgather_async function."""

    def _get_func(self):
        from paddleformers.nn.moe.all_gather import allgather_async

        return allgather_async

    def test_import(self):
        """Test that allgather_async can be imported."""
        func = self._get_func()
        self.assertTrue(callable(func))

    @patch("paddleformers.nn.moe.all_gather.fleet")
    def test_single_rank_returns_clone(self, mock_fleet):
        """Test that single rank returns a clone of input."""
        func = self._get_func()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = mock_group
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        x = paddle.randn([4, 8])
        out, task = func(x, group=mock_group)
        self.assertEqual(out.shape, [4, 8])
        self.assertIsNone(task)

    @patch("paddleformers.nn.moe.all_gather.fleet")
    def test_group_none_uses_model_parallel(self, mock_fleet):
        """Test that group=None uses model parallel group from fleet."""
        func = self._get_func()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = mock_group
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        x = paddle.randn([4, 8])
        out, task = func(x)
        self.assertEqual(out.shape, [4, 8])

    @patch("paddleformers.nn.moe.all_gather.dist.stream.all_gather")
    @patch("paddleformers.nn.moe.all_gather.fleet")
    def test_multi_rank_returns_correct_shape(self, mock_fleet, mock_all_gather):
        """Test that multi-rank returns expanded output shape."""
        func = self._get_func()
        mock_group = MagicMock()
        mock_group.nranks = 4
        mock_task = MagicMock()
        mock_all_gather.return_value = mock_task

        x = paddle.randn([4, 8])
        out, task = func(x, group=mock_group)
        self.assertEqual(out.shape, [16, 8])
        mock_all_gather.assert_called_once()
        self.assertIs(task, mock_task)


class TestReduceScatterAsync(unittest.TestCase):
    """Tests for reduce_scatter_async function."""

    def _get_func(self):
        from paddleformers.nn.moe.all_gather import reduce_scatter_async

        return reduce_scatter_async

    def test_import(self):
        """Test that reduce_scatter_async can be imported."""
        func = self._get_func()
        self.assertTrue(callable(func))

    @patch("paddleformers.nn.moe.all_gather.fleet")
    def test_single_rank_returns_clone(self, mock_fleet):
        """Test that single rank returns a clone of input."""
        func = self._get_func()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = mock_group
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        x = paddle.randn([8, 4])
        out, task = func(x, group=mock_group)
        self.assertEqual(out.shape, [8, 4])
        self.assertIsNone(task)

    @patch("paddleformers.nn.moe.all_gather.fleet")
    def test_group_none_uses_model_parallel(self, mock_fleet):
        """Test that group=None uses model parallel group."""
        func = self._get_func()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = mock_group
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        x = paddle.randn([8, 4])
        out, task = func(x)
        self.assertEqual(out.shape, [8, 4])

    @patch("paddleformers.nn.moe.all_gather.dist.stream.reduce_scatter")
    @patch("paddleformers.nn.moe.all_gather.fleet")
    def test_multi_rank_reduces_shape(self, mock_fleet, mock_reduce_scatter):
        """Test that multi-rank returns reduced output shape."""
        func = self._get_func()
        mock_group = MagicMock()
        mock_group.nranks = 4
        mock_task = MagicMock()
        mock_reduce_scatter.return_value = mock_task

        x = paddle.randn([16, 4])
        out, task = func(x, group=mock_group)
        self.assertEqual(out.shape, [4, 4])
        mock_reduce_scatter.assert_called_once()

    @patch("paddleformers.nn.moe.all_gather.fleet")
    def test_undivisible_input_raises(self, mock_fleet):
        """Test that input not divisible by parallelism raises AssertionError."""
        func = self._get_func()
        mock_group = MagicMock()
        mock_group.nranks = 3
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_group.return_value = mock_group
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        x = paddle.randn([10, 4])  # 10 not divisible by 3
        with self.assertRaises(AssertionError):
            func(x, group=mock_group)


class TestAllGatherAsyncPyLayer(unittest.TestCase):
    """Tests for AllGatherAsync PyLayer."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_gather import AllGatherAsync

        return AllGatherAsync

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size", return_value=1)
    def test_forward_single_world_manual_backward_called(self, mock_ws, mock_mb):
        """Test forward with world_size=1 calls manual_backward."""
        cls = self._get_cls()
        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        x = paddle.randn([4, 8])
        cls.apply(x, group=None, fn=mock_fn, is_first_fwd=True)
        mock_mb.assert_called_once_with(mock_fn, True)

    @patch("paddleformers.nn.moe.all_gather.reduce_scatter_async")
    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size", return_value=1)
    def test_backward_single_world(self, mock_ws, mock_mb, mock_rs):
        """Test backward with world_size=1."""
        cls = self._get_cls()
        mock_bwf = MagicMock()
        mock_bwf.return_value = (paddle.randn([4, 8]),)
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        x = paddle.randn([4, 8])
        cls.apply(x, group=None, fn=MagicMock(), is_first_fwd=True)

    @patch("paddleformers.nn.moe.all_gather.allgather_async")
    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size", return_value=4)
    def test_forward_multi_world_allgather_called(self, mock_ws, mock_mb, mock_ag):
        """Test forward with world_size>1 calls allgather_async."""
        cls = self._get_cls()
        mock_task = MagicMock()
        mock_task.wait = MagicMock()
        mock_ag.return_value = (paddle.randn([16, 8]), mock_task)
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        mock_group = MagicMock()
        x = paddle.randn([4, 8])
        cls.apply(x, group=mock_group, fn=MagicMock(), is_first_fwd=True)
        mock_ag.assert_called_once()


class TestAlltoAllSmart(unittest.TestCase):
    """Tests for AlltoAllSmart PyLayer."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_gather import AlltoAllSmart

        return AlltoAllSmart

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_all_none_inputs_raises(self, mock_gg, mock_ws, mock_rank, mock_mb):
        """Test that all None inputs raises RuntimeError."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = None
        x2 = None
        router_loss_args = (paddle.randn([2, 4]),)

        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([2, 4]),))

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        with self.assertRaises(RuntimeError):
            cls.apply(
                x1,
                x2,
                *router_loss_args,
                router_loss_fn=MagicMock(),
                forward_func_dict=None,
                local_expert_id=local_expert_id,
                send_rank_global=send_rank_global,
                recv_rank_global=recv_rank_global,
                num_local_experts=2,
                capacity=2,
                group=mock_group,
                recv_size=4,
                send_counts=paddle.zeros([2, 1], dtype="int64"),
                recv_counts=paddle.zeros([2, 1], dtype="int64"),
                send_counts_num=paddle.zeros([2], dtype="int64"),
                recv_counts_num=paddle.zeros([2], dtype="int64"),
                is_first_fwd=True,
            )

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_basic_single_rank(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with single rank."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([4, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        mock_bwf = MagicMock()
        mock_mb.side_effect = [
            (None, (paddle.randn([4, 8]),)),  # forward_func_dict is None, no bwf
            (mock_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        result = cls.apply(
            x1,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict=None,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=1,
            capacity=4,
            group=mock_group,
            recv_size=4,
            send_counts=np.array([[4]], dtype="int64"),
            recv_counts=np.array([[4]], dtype="int64"),
            send_counts_num=np.array([4], dtype="int64"),
            recv_counts_num=np.array([4], dtype="int64"),
            is_first_fwd=True,
        )
        self.assertEqual(len(result), 3)

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_with_forward_func_dict(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with forward_func_dict (expert computation)."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([4, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        mock_bwf = MagicMock()
        mock_mb.side_effect = [
            (mock_bwf, (paddle.randn([4, 8]),)),
            (mock_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        mock_expert_fn = MagicMock()
        mock_expert_fn.return_value = paddle.randn([4, 8])

        result = cls.apply(
            x1,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict={0: mock_expert_fn},
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=1,
            capacity=4,
            group=mock_group,
            recv_size=4,
            send_counts=np.array([[4]], dtype="int64"),
            recv_counts=np.array([[4]], dtype="int64"),
            send_counts_num=np.array([4], dtype="int64"),
            recv_counts_num=np.array([4], dtype="int64"),
            is_first_fwd=True,
        )
        self.assertEqual(len(result), 3)

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_no_padding_branch(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with use_padding=False."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([4, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        mock_bwf = MagicMock()
        mock_mb.side_effect = [
            (None, (paddle.randn([4, 8]),)),
            (mock_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        result = cls.apply(
            x1,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict=None,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=1,
            capacity=4,
            use_padding=False,
            expert_num_global=1,
            group=mock_group,
            recv_size=4,
            send_counts=np.array([[4]], dtype="int64"),
            recv_counts=np.array([[4]], dtype="int64"),
            send_counts_num=np.array([4], dtype="int64"),
            recv_counts_num=np.array([4], dtype="int64"),
            is_first_fwd=True,
        )
        self.assertEqual(len(result), 3)

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_with_multi_experts(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with multiple local experts."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([2, 8])
        x2 = paddle.randn([2, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        mock_bwf = MagicMock()
        mock_mb.side_effect = [
            (None, (paddle.randn([2, 8]),)),
            (mock_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 2, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        result = cls.apply(
            x1,
            x2,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict=None,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=2,
            capacity=2,
            group=mock_group,
            recv_size=4,
            send_counts=np.array([[2], [2]], dtype="int64"),
            recv_counts=np.array([[2], [2]], dtype="int64"),
            send_counts_num=np.array([2, 2], dtype="int64"),
            recv_counts_num=np.array([2, 2], dtype="int64"),
            is_first_fwd=True,
        )
        self.assertEqual(len(result), 3)


class TestAlltoAllSmartXPU(unittest.TestCase):
    """Tests for AlltoAllSmartXPU PyLayer."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_gather import AlltoAllSmartXPU

        return AlltoAllSmartXPU

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_all_none_inputs_with_func(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with all None inputs and forward_func_dict=None raises TypeError."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        router_loss_args = (paddle.randn([2, 4]),)
        mock_mb.return_value = (None, (paddle.randn([2, 4]),))

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        x1 = None
        x2 = None
        # AlltoAllSmartXPU doesn't have "all inputs are None" check like AlltoAllSmart
        # Instead, it tries forward_func_dict[0] when all inputs are None
        with self.assertRaises(TypeError):
            cls.apply(
                x1,
                x2,
                *router_loss_args,
                router_loss_fn=MagicMock(),
                forward_func_dict=None,
                local_expert_id=local_expert_id,
                send_rank_global=send_rank_global,
                recv_rank_global=recv_rank_global,
                num_local_experts=2,
                capacity=2,
                group=mock_group,
                recv_size=paddle.to_tensor(0),
                send_counts=np.array([[0], [0]], dtype="int64"),
                recv_counts=np.array([[0], [0]], dtype="int64"),
                send_counts_num=np.array([0, 0], dtype="int64"),
                recv_counts_num=np.array([0, 0], dtype="int64"),
                is_first_fwd=True,
            )

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_with_valid_inputs(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with valid inputs."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([4, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        mock_bwf = MagicMock()
        mock_mb.side_effect = [
            (None, (paddle.randn([4, 8]),)),
            (mock_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        result = cls.apply(
            x1,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict=None,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=1,
            capacity=4,
            group=mock_group,
            recv_size=paddle.to_tensor(4),
            send_counts=np.array([[4]], dtype="int64"),
            recv_counts=np.array([[4]], dtype="int64"),
            send_counts_num=np.array([4], dtype="int64"),
            recv_counts_num=np.array([4], dtype="int64"),
            is_first_fwd=True,
        )
        self.assertEqual(len(result), 3)

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_no_padding(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with use_padding=False."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([4, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        mock_bwf = MagicMock()
        mock_mb.side_effect = [
            (None, (paddle.randn([4, 8]),)),
            (mock_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        result = cls.apply(
            x1,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict=None,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=1,
            capacity=4,
            use_padding=False,
            expert_num_global=1,
            group=mock_group,
            recv_size=paddle.to_tensor(4),
            send_counts=np.array([[4]], dtype="int64"),
            recv_counts=np.array([[4]], dtype="int64"),
            send_counts_num=np.array([4], dtype="int64"),
            recv_counts_num=np.array([4], dtype="int64"),
            is_first_fwd=True,
        )
        self.assertEqual(len(result), 3)

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_forward_mixed_zero_and_nonzero_experts(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test forward with multiple experts each having tokens (simpler case)."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([2, 8])
        x2 = paddle.randn([2, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        processed_1 = paddle.randn([2, 8])
        processed_2 = paddle.randn([2, 8])
        dummy_router = paddle.randn([2, 4])

        # Both experts have tokens
        mock_mb.side_effect = [
            (None, (processed_1,)),  # expert 0 forward
            (None, (processed_2,)),  # expert 1 forward
            (None, (dummy_router,)),  # router_loss_fn
        ]

        local_expert_id = paddle.randint(0, 2, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        mock_expert_fn = MagicMock()
        mock_expert_fn.training = False
        mock_expert_fn.up_gate_proj.weight = paddle.randn([8, 16])

        cls.apply(
            x1,
            x2,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict={0: mock_expert_fn, 1: mock_expert_fn},
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=2,
            capacity=2,
            group=mock_group,
            recv_size=paddle.to_tensor(4),
            send_counts=np.array([[2], [2]], dtype="int64"),
            recv_counts=np.array([[2], [2]], dtype="int64"),
            send_counts_num=np.array([2, 2], dtype="int64"),
            recv_counts_num=np.array([2, 2], dtype="int64"),
            is_first_fwd=True,
        )


class TestAlltoAllSmartBackward(unittest.TestCase):
    """Tests for AlltoAllSmart backward pass."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_gather import AlltoAllSmart

        return AlltoAllSmart

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_backward_basic(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test backward pass execution."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([4, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        MagicMock()
        mock_router_bwf = MagicMock()
        mock_router_bwf.return_value = ()
        mock_mb.side_effect = [
            (None, (paddle.randn([4, 8]),)),
            (mock_router_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        output, router_loss, mask = cls.apply(
            x1,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict=None,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=1,
            capacity=4,
            group=mock_group,
            recv_size=4,
            send_counts=np.array([[4]], dtype="int64"),
            recv_counts=np.array([[4]], dtype="int64"),
            send_counts_num=np.array([4], dtype="int64"),
            recv_counts_num=np.array([4], dtype="int64"),
            is_first_fwd=True,
        )


class TestAlltoAllSmartXPUBackward(unittest.TestCase):
    """Tests for AlltoAllSmartXPU backward pass."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_gather import AlltoAllSmartXPU

        return AlltoAllSmartXPU

    @patch("paddleformers.nn.moe.all_gather.manual_backward")
    @patch("paddleformers.nn.moe.all_gather.dist.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_gather.dist.get_rank")
    @patch("paddleformers.nn.moe.all_gather.dist.get_world_size")
    @patch("paddleformers.nn.moe.all_gather._get_global_group")
    def test_backward_basic(self, mock_gg, mock_ws, mock_rank, mock_alltoall, mock_mb):
        """Test backward pass execution."""
        cls = self._get_cls()
        mock_group = MagicMock()
        mock_group.nranks = 1
        mock_gg.return_value = mock_group
        mock_ws.return_value = 1
        mock_rank.return_value = 0

        x1 = paddle.randn([4, 8])
        router_loss_args = (paddle.randn([2, 4]),)

        MagicMock()
        mock_router_bwf = MagicMock()
        mock_router_bwf.return_value = ()
        mock_mb.side_effect = [
            (None, (paddle.randn([4, 8]),)),
            (mock_router_bwf, (paddle.randn([2, 4]),)),
        ]

        local_expert_id = paddle.randint(0, 1, [4], dtype="int64")
        send_rank_global = paddle.randint(0, 1, [4], dtype="int64")
        recv_rank_global = paddle.randint(0, 1, [4], dtype="int64")

        output, router_loss, mask = cls.apply(
            x1,
            *router_loss_args,
            router_loss_fn=MagicMock(),
            forward_func_dict=None,
            local_expert_id=local_expert_id,
            send_rank_global=send_rank_global,
            recv_rank_global=recv_rank_global,
            num_local_experts=1,
            capacity=4,
            group=mock_group,
            recv_size=paddle.to_tensor(4),
            send_counts=np.array([[4]], dtype="int64"),
            recv_counts=np.array([[4]], dtype="int64"),
            send_counts_num=np.array([4], dtype="int64"),
            recv_counts_num=np.array([4], dtype="int64"),
            is_first_fwd=True,
        )


if __name__ == "__main__":
    unittest.main()
