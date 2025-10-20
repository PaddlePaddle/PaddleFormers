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

"""
Set random seed for reproducibility in hybrid parallel training.
"""
import random

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker


def set_seed(seed):
    """set random seed for reproducibility in hybrid parallel training."""
    # NOTE(shenliang03): For parameter init seed:
    # seed: dp/mp_undistributed_paramter/sharding is same; others is different
    # For compute seed(dropout):
    # global seed: only mp group is same.
    # local seed: all groups are different

    if hasattr(fleet, "_hcg"):  # 混合并行下，才分开设置local-seed和global-seed
        # obtain rank message of hybrid parallel
        hcg = fleet.get_hybrid_communicate_group()

        mp_rank = hcg.get_model_parallel_rank()
        mp_size = hcg.get_model_parallel_world_size()

        pp_rank = hcg.get_stage_id()
        pp_size = hcg.get_pipe_parallel_world_size()

        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()

        sharding_rank = hcg.get_sharding_parallel_rank()
        sharding_size = hcg.get_sharding_parallel_world_size()
    else:
        mp_rank, mp_size = 0, 1
        pp_rank, pp_size = 0, 1
        dp_rank, dp_size = dist.get_rank(), dist.get_world_size()
        sharding_rank, sharding_size = 0, 1

    # NOTE: the commented seeds are set only for precision validation
    # 与框架中的实现对齐,
    # 无论是否启用混合并行，都设置 model_parallel_rng 用于同步初始化参数
    model_parallel_rng = seed + 1 + mp_rank * pp_size + pp_rank

    seed += (
        1 * dp_rank
    )  # EB4框架中数据流并不需要全局seed，。此处操作对数据没什么影响，对组网也没什么影响。只是为了兼容 fleet 传统而设置。
    random.seed(seed)
    np.random.seed(seed)

    # seed = mp_rank +
    #        pp_rank * (mp_size) +
    #        dp_rank * (mp_size * pp_size) +
    #        sharding_rank * (mp_size * pp_size * dp_size)
    # seed offset is order to avoid conflicts with the parameter initialization seed

    seed_offset = seed + 1024 + paddle.distributed.get_world_size()
    global_seed = (
        seed_offset
        + pp_rank * (mp_size)
        + dp_rank * (mp_size * pp_size)
        + sharding_rank * (mp_size * pp_size * dp_size)
    )

    seed_offset += paddle.distributed.get_world_size()
    local_seed = (
        seed_offset
        + mp_rank
        + pp_rank * (mp_size)
        + dp_rank * (mp_size * pp_size)
        + sharding_rank * (mp_size * pp_size * dp_size)
    )

    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)
    if "model_parallel_rng" not in tracker.states_:
        tracker.add("model_parallel_rng", model_parallel_rng)
    paddle.seed(global_seed)

    print(
        f"""
        The global seed is set to {global_seed} and local seed is set to {local_seed}.
        mp_init_seed={model_parallel_rng}
        """
    )
