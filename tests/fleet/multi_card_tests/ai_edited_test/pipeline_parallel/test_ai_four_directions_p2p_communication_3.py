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

os.environ["PADDLE_USE_FOUR_DIRECTIONS_P2P"] = "True"

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
    _is_valid_send_recv_partial,
    initialize_p2p_groups,
    recv_partial,
    send_partial,
)
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 2
MP_DEGREE = 2


def _init_pp_mp():
    """Initialize with PP=2, MP=2 to test partial send/recv."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": MP_DEGREE,
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
    _init_pp_mp()
    hcg = fleet.get_hybrid_communicate_group()
    initialize_p2p_groups(
        hcg, enable_partial_send_recv=True, enable_timer=False
    )
    np.random.seed(42)
    paddle.seed(42)


class TestPartialSendRecvValidity(unittest.TestCase):
    """Test _is_valid_send_recv_partial with MP=2 configuration."""

    def test_is_valid_with_mp2_divisible(self):
        """Should return True when tensor numel is divisible by mp_degree > 1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertTrue(_is_valid_send_recv_partial(tensor, MP_DEGREE))

    def test_is_valid_with_mp2_not_divisible(self):
        """Should return False when tensor numel is not divisible by mp_degree."""
        tensor = paddle.randn([3, 7], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, MP_DEGREE))

    def test_is_valid_with_mp_degree_one(self):
        """Should return False when mp_degree is 1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 1))


class TestPartialSendRecvDistributed(unittest.TestCase):
    """Test partial send/recv operations with PP=2, MP=2."""

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() and dist.is_initialized()),
        "Requires CUDA and distributed environment",
    )
    def test_send_recv_partial_asymmetric(self):
        """Test send_partial and recv_partial with asymmetric pattern.

        Rank 0 (PP stage 0) sends, Rank 2 (PP stage 1) receives.
        This avoids deadlock in distributed communication.
        """
        hcg = fleet.get_hybrid_communicate_group()
        pp_rank = hcg.get_stage_id()
        mp_rank = hcg.get_model_parallel_rank()

        tensor = paddle.randn([4, 8], dtype="float32")

        # Verify _is_valid_send_recv_partial returns True for this tensor
        is_valid = _is_valid_send_recv_partial(tensor, MP_DEGREE)
        self.assertTrue(
            is_valid,
            f"Tensor should be valid for partial send/recv: numel={tensor.numel()}, mp_degree={MP_DEGREE}",
        )

        if pp_rank == 0:
            # Send to next PP stage
            send_partial(tensor, dst=1, nranks=MP_DEGREE, rank_id=mp_rank)
            dist.barrier()
        elif pp_rank == 1:
            # Receive from prev PP stage
            recv_tensor = paddle.empty([4, 8], dtype="float32")
            result = recv_partial(
                recv_tensor, src=0, nranks=MP_DEGREE, rank_id=mp_rank
            )
            self.assertIsNotNone(result)
            dist.barrier()


if __name__ == "__main__":
    unittest.main()
