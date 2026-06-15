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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock, patch


class TestFthenBInit(unittest.TestCase):
    """Tests for PipelineParallelWithInterleaveFthenB initialization."""

    def test_overlap_schedule_mode_default(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp.overlap_schedule_mode = False
        self.assertFalse(pp.overlap_schedule_mode)

    def test_get_scheduler_name(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        name = pp._get_scheduler_name()
        self.assertEqual(name, "PipelineParallelWithInterleaveFthenB")

    def test_init_user_bubble_hooks(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._init_user_bubble_hooks()
        self.assertIsNone(pp.bubble_hooks)


class TestFthenBCheckSanity(unittest.TestCase):
    """Tests for _check_sanity."""

    def test_check_sanity_pass(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp.num_stages = 4
        with patch("paddle.framework.in_dynamic_mode", return_value=True):
            pp._check_sanity()

    def test_check_sanity_fail_not_dynamic(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp.num_stages = 4
        with (
            patch("paddle.framework.in_dynamic_mode", return_value=False),
            self.assertRaises(AssertionError),
        ):
            pp._check_sanity()

    def test_check_sanity_fail_stages(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp.num_stages = 2
        with (
            patch("paddle.framework.in_dynamic_mode", return_value=True),
            self.assertRaises(AssertionError),
        ):
            pp._check_sanity()


class TestFthenBGetVirtualPPRank(unittest.TestCase):
    """Tests for _get_virtual_pp_rank."""

    def test_forward_first_step(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp.accumulate_steps = 4
        pp.num_stages = 4
        pp.num_model_chunks = 2

        rank = pp._get_virtual_pp_rank(0, forward=True)
        self.assertEqual(rank, 0)

    def test_forward_second_step(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp.accumulate_steps = 4
        pp.num_stages = 4
        pp.num_model_chunks = 2

        # micro_step 4 -> 4 % (4*2) = 4, 4 // 4 = 1
        rank = pp._get_virtual_pp_rank(4, forward=True)
        self.assertEqual(rank, 1)

    def test_backward_first_step(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp.accumulate_steps = 4
        pp.num_stages = 4
        pp.num_model_chunks = 2

        rank = pp._get_virtual_pp_rank(0, forward=False)
        self.assertEqual(rank, pp.num_model_chunks - 1)


class TestFthenBOverlapCommGrads(unittest.TestCase):
    """Tests for _overlap_comm_grads."""

    def test_no_comm_overlap(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._comm_overlap = False
        pp._overlap_comm_grads()

    def test_with_comm_overlap(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._comm_overlap = True
        pp.stage_id = 1
        pp._backward_step_count = 1
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        pp._chunk_2_comm_buffers = {
            0: [MagicMock()],
            1: [MagicMock()],
        }
        pp._overlap_comm_grads()

    def test_stage_zero_early_return(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._comm_overlap = True
        pp.stage_id = 0
        pp._backward_step_count = 1
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        pp._chunk_2_comm_buffers = {}
        pp._overlap_comm_grads()

    def test_final_sync_step(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._comm_overlap = True
        pp.stage_id = 1
        pp._backward_step_count = 8
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        mock_buf = MagicMock()
        pp._chunk_2_comm_buffers = {0: [mock_buf]}
        pp._overlap_comm_grads()
        mock_buf.comm_grads.assert_called_once()


class TestFthenBSyncOverlapGrads(unittest.TestCase):
    """Tests for _sync_overlap_grads."""

    def test_no_comm_overlap(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._comm_overlap = False
        pp._sync_overlap_grads()

    def test_with_comm_overlap(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._comm_overlap = True
        pp._backward_step_count = 8
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        pp._chunk_2_comm_buffers = {
            0: [MagicMock()],
            1: [MagicMock()],
        }
        pp._sync_overlap_grads()
        for buffers in pp._chunk_2_comm_buffers.values():
            for buf in buffers:
                buf.scale_grads.assert_called_once()

    def test_count_mismatch(self):
        from paddle.distributed.fleet.meta_parallel import (
            PipelineParallelWithInterleaveFthenB,
        )

        pp = PipelineParallelWithInterleaveFthenB.__new__(PipelineParallelWithInterleaveFthenB)
        pp._comm_overlap = True
        pp._backward_step_count = 5
        pp.accumulate_steps = 4
        pp._virtual_pp_world_size = 2
        with self.assertRaises(AssertionError):
            pp._sync_overlap_grads()


if __name__ == "__main__":
    unittest.main()
