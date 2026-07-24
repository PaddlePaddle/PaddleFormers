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

import paddle.distributed as dist
from paddle.distributed import fleet

from paddleformers.fleet import parallel_state as ps
from paddleformers.fleet.training.initialize import initialize_fleet


class TestParallelState(unittest.TestCase):
    """
    Unit tests for ParallelState.

    Tests include:
    - [tp, pp, dp, ep, cp, expt_dp, cp_dp] group correctness
    """

    def test_comm_group(self):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 2,
            "sharding_degree": 2,
            "sep_degree": 1,
            "cp_degree": 2,
            "ep_degree": 4,
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
        initialize_fleet(strategy=strategy)

        tp_group = ps.get_tensor_model_parallel_group()
        pp_group = ps.get_pipeline_model_parallel_group()
        dp_group = ps.get_data_parallel_group()
        cp_group = ps.get_context_parallel_group()
        ep_group = ps.get_expert_model_parallel_group()
        expt_dp_group = ps.get_expert_data_parallel_group()
        cp_dp_group = ps.get_data_parallel_group(with_context_parallel=True)

        rank = dist.get_rank()
        # check tp_group ranks
        expected_tp_group_ranks = (
            [rank, rank + 1] if rank % 2 == 0 else [rank - 1, rank]
        )
        assert tp_group.ranks == expected_tp_group_ranks, (
            f"Expected tp group rank: {expected_tp_group_ranks}, got {tp_group.ranks}"
        )
        expected_pp_group = [rank, rank + 4] if rank < 4 else [rank - 4, rank]
        assert pp_group.ranks == expected_pp_group, (
            f"Expected pp group rank: {expected_pp_group}, got {pp_group.ranks}"
        )

        expected_dp_group_map = {
            0: [0, 2],
            1: [1, 3],
            2: [0, 2],
            3: [1, 3],
            4: [4, 6],
            5: [5, 7],
            6: [4, 6],
            7: [5, 7],
        }
        assert dp_group.ranks == expected_dp_group_map[rank], (
            f"Expected dp group rank: {expected_dp_group_map[rank]}, got {dp_group.ranks}"
        )
        assert cp_group.ranks == expected_dp_group_map[rank], (
            f"Expected cp group rank: {expected_dp_group_map[rank]}, got {cp_group.ranks}"
        )

        expected_ep_group_map = {
            0: [0, 1, 2, 3],
            1: [0, 1, 2, 3],
            2: [0, 1, 2, 3],
            3: [0, 1, 2, 3],
            4: [4, 5, 6, 7],
            5: [4, 5, 6, 7],
            6: [4, 5, 6, 7],
            7: [4, 5, 6, 7],
        }
        assert ep_group.ranks == expected_ep_group_map[rank], (
            f"Expected ep group rank: {expected_ep_group_map[rank]}, got {ep_group.ranks}"
        )

        assert expt_dp_group.ranks == [rank], (
            f"Expected expt_dp group rank: {[rank]}, got {expt_dp_group.ranks}"
        )
        assert cp_dp_group.ranks == [rank], (
            f"Expected cp_dp group rank: {[rank]}, got {cp_dp_group.ranks}"
        )


if __name__ == "__main__":
    unittest.main()
