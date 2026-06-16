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


import unittest
from unittest.mock import MagicMock, patch


class TestP2PAsyncHandle(unittest.TestCase):
    """Tests for P2PAsyncHandle dataclass."""

    def test_forward_handle_wait(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        mock_fn = MagicMock()
        handle = P2PAsyncHandle(
            forward_handle_wait_fn=mock_fn,
            forward_async_comm_fn=MagicMock(),
            backward_handle_wait_fn=MagicMock(),
            backward_async_comm_fn=MagicMock(),
        )
        handle.forward_handle_wait()
        mock_fn.assert_called_once()

    def test_backward_handle_wait(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        mock_fn = MagicMock()
        handle = P2PAsyncHandle(
            forward_handle_wait_fn=MagicMock(),
            forward_async_comm_fn=MagicMock(),
            backward_handle_wait_fn=mock_fn,
            backward_async_comm_fn=MagicMock(),
        )
        handle.backward_handle_wait()
        mock_fn.assert_called_once()

    def test_forward_async_comm(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        mock_fn = MagicMock(return_value=(1, "input", None))
        handle = P2PAsyncHandle(
            forward_handle_wait_fn=MagicMock(),
            forward_async_comm_fn=mock_fn,
            backward_handle_wait_fn=MagicMock(),
            backward_async_comm_fn=MagicMock(),
        )
        handle.forward_async_comm("output_tensor")
        mock_fn.assert_called_once_with(output_tensor="output_tensor")
        self.assertEqual(handle.next_forward_virtual_pp_rank, 1)
        self.assertEqual(handle.input_tensor, "input")

    def test_backward_async_comm(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        mock_fn = MagicMock(return_value=(2, "grad", True, None))
        handle = P2PAsyncHandle(
            forward_handle_wait_fn=MagicMock(),
            forward_async_comm_fn=MagicMock(),
            backward_handle_wait_fn=MagicMock(),
            backward_async_comm_fn=mock_fn,
        )
        handle.backward_async_comm("input_tensor_grad")
        mock_fn.assert_called_once_with(input_tensor_grad="input_tensor_grad")
        self.assertEqual(handle.next_backward_virtual_pp_rank, 2)
        self.assertEqual(handle.output_tensor_grad, "grad")
        self.assertTrue(handle.recv_next)


class TestPipelineParallelWithInterleaveVirtualRank(unittest.TestCase):
    """Tests for _get_virtual_pp_rank logic."""

    def test_get_virtual_pp_rank_first_chunk(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp.accumulate_steps = 4
        pp.num_stages = 4
        pp.num_model_chunks = 2
        pp._best_unbalanced_scheduler = False

        # micro_step 0 -> virtual_pp_stage = 0
        rank = pp._get_virtual_pp_rank(0, forward=True)
        self.assertEqual(rank, 0)

    def test_get_virtual_pp_rank_backward(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp.accumulate_steps = 8
        pp.num_stages = 4
        pp.num_model_chunks = 2
        pp._best_unbalanced_scheduler = False

        rank = pp._get_virtual_pp_rank(0, forward=False)
        self.assertEqual(rank, pp.num_model_chunks - 1)


class TestPipelineParallelWithInterleaveCheckSanity(unittest.TestCase):
    """Tests for _check_sanity."""

    def test_check_sanity_pass(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp.num_stages = 4
        pp.accumulate_steps = 8
        with patch("paddle.framework.in_dynamic_mode", return_value=True):
            pp._check_sanity()

    def test_check_sanity_fail_not_dynamic(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp.num_stages = 4
        pp.accumulate_steps = 8
        with (
            patch("paddle.framework.in_dynamic_mode", return_value=False),
            self.assertRaises(AssertionError),
        ):
            pp._check_sanity()

    def test_check_sanity_fail_stages(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp.num_stages = 2
        pp.accumulate_steps = 8
        with (
            patch("paddle.framework.in_dynamic_mode", return_value=True),
            self.assertRaises(AssertionError),
        ):
            pp._check_sanity()

    def test_check_sanity_fail_acc_steps(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp.num_stages = 4
        pp.accumulate_steps = 4
        with (
            patch("paddle.framework.in_dynamic_mode", return_value=True),
            self.assertRaises(AssertionError),
        ):
            pp._check_sanity()


class TestGetSchedulerName(unittest.TestCase):
    """Tests for _get_scheduler_name."""

    def test_scheduler_name(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp.overlap_schedule_mode = False
        pp._overlap_p2p_comm = False
        name = pp._get_scheduler_name()
        self.assertIn("PipelineParallelWithInterleave", name)
        self.assertIn("False", name)


class TestOverlapCommGrads(unittest.TestCase):
    """Tests for _overlap_comm_grads."""

    def test_no_comm_overlap(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp._comm_overlap = False
        # Should return early
        pp._overlap_comm_grads()

    def test_stage_zero(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp._comm_overlap = True
        pp.stage_id = 0
        pp._backward_step_count = 1
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        pp.num_stages = 4
        pp.num_model_chunks = 2
        # Should return early after stage_id check
        pp._overlap_comm_grads()


class TestSyncOverlapGrads(unittest.TestCase):
    """Tests for _sync_overlap_grads."""

    def test_no_comm_overlap(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp._comm_overlap = False
        pp._sync_overlap_grads()

    def test_with_comm_overlap(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp._comm_overlap = True
        pp._backward_step_count = 8
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        pp.num_stages = 4
        pp.num_model_chunks = 2
        pp._chunk_2_comm_buffers = {
            0: [MagicMock()],
            1: [MagicMock()],
        }
        pp._sync_overlap_grads()
        for buffers in pp._chunk_2_comm_buffers.values():
            for buf in buffers:
                buf.scale_grads.assert_called_once()

    def test_with_comm_overlap_mismatch_count(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleave,
        )

        pp = PipelineParallelWithInterleave.__new__(
            PipelineParallelWithInterleave
        )
        pp._comm_overlap = True
        pp._backward_step_count = 5
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        pp.num_stages = 4
        pp.num_model_chunks = 2
        with self.assertRaises(AssertionError):
            pp._sync_overlap_grads()


if __name__ == "__main__":
    unittest.main()
