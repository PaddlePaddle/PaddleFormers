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


import paddle.distributed as dist
from paddle.distributed import fleet

from paddleformers.fleet.training.initialize import initialize_fleet


class Utils:
    inited = False
    store = None

    @staticmethod
    def initialize_model_parallel(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        sharding_parallel_size=1,
        expert_parallel_size=1,
        context_parallel_size=1,
    ):
        strategy = fleet.DistributedStrategy()
        world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size * sharding_parallel_size
        moe_sharding_parallel_size = 1
        if expert_parallel_size > 1:
            moe_sharding_parallel_size = world_size // (pipeline_parallel_size * expert_parallel_size)
        strategy.hybrid_configs = {
            "dp_degree": data_parallel_size,
            "mp_degree": tensor_parallel_size,
            "pp_degree": pipeline_parallel_size,
            "sharding_degree": sharding_parallel_size,
            "sep_degree": 1,
            "cp_degree": context_parallel_size,
            "ep_degree": expert_parallel_size,
            "moe_sharding_degree": moe_sharding_parallel_size,
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
        Utils.inited = True
        Utils.world_size = world_size
        Utils.rank = dist.get_rank()
