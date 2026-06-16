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
import unittest

# Enable sync_send mode before importing the module
os.environ["PADDLE_USE_FOUR_DIRECTIONS_P2P"] = "True"
os.environ["PADDLE_P2P_SYNC_SEND"] = "1"

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddleformers.fleet.pipeline_parallel.pp_utils.four_directions_p2p_communication import (
    SendRecvMeta,
    _p2p_helper,
    initialize_p2p_groups,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 2


def _init_pp():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)


def setUpModule():
    _init_pp()
    hcg = fleet.get_hybrid_communicate_group()
    initialize_p2p_groups(hcg, enable_partial_send_recv=True)
    np.random.seed(42)
    paddle.seed(42)


class TestSyncSendMode(unittest.TestCase):
    """Test _p2p_helper with PADDLE_P2P_SYNC_SEND=1."""

    def test_p2p_helper_sync_send_recv_prev(self):
        """Test _p2p_helper recv_prev with sync_send mode."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 4]
        meta.send_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_shape_message = [2, 4]
        meta.recv_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_stop_gradient = False

        if pp_rank == 0:
            tensor = paddle.randn([2, 4], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_prev)

        dist.barrier()

    def test_p2p_helper_sync_send_recv_next(self):
        """Test _p2p_helper recv_next with sync_send mode."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 4]
        meta.send_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_shape_message = [2, 4]
        meta.recv_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_stop_gradient = False

        if pp_rank == 1:
            tensor = paddle.randn([2, 4], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=tensor,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_next)

        dist.barrier()

    def test_p2p_helper_sync_send_tuple_recv_prev(self):
        """Test _p2p_helper recv_prev with tuple in sync_send mode."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = ([2, 4], [3, 5])
        meta.send_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float32),
        )
        meta.recv_shape_message = ([2, 4], [3, 5])
        meta.recv_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float32),
        )
        meta.recv_stop_gradient = (False, False)

        if pp_rank == 0:
            t1 = paddle.randn([2, 4], dtype="float32")
            t2 = paddle.randn([3, 5], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=(t1, t2),
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_prev)
            self.assertIsInstance(recv_prev, tuple)

        dist.barrier()

    def test_p2p_helper_sync_send_tuple_recv_next(self):
        """Test _p2p_helper recv_next with tuple in sync_send mode."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = ([2, 4], [3, 5])
        meta.send_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float32),
        )
        meta.recv_shape_message = ([2, 4], [3, 5])
        meta.recv_dtype_message = (
            paddle_2_number(paddle.float32),
            paddle_2_number(paddle.float32),
        )
        meta.recv_stop_gradient = (False, False)

        if pp_rank == 1:
            t1 = paddle.randn([2, 4], dtype="float32")
            t2 = paddle.randn([3, 5], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=(t1, t2),
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNotNone(recv_next)
            self.assertIsInstance(recv_next, tuple)

        dist.barrier()

    def test_p2p_helper_sync_send_non_blocking(self):
        """Test _p2p_helper with sync_recv=False in sync_send mode."""
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 4]
        meta.send_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_shape_message = [2, 4]
        meta.recv_dtype_message = paddle_2_number(paddle.float32)
        meta.recv_stop_gradient = False

        if pp_rank == 0:
            tensor = paddle.randn([2, 4], dtype="float32")
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=False,
                send_recv_meta=meta,
            )
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=False,
                send_recv_meta=meta,
            )

        dist.barrier()


if __name__ == "__main__":
    unittest.main()
