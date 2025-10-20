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

"""MoE utils"""

from paddle.distributed import fleet


def distributed_optimizer_for_moe(
    optimizer,
    use_moe=False,
):
    """
    Create a distributed optimizer with MoE (Mixture of Experts) support.

    Args:
        optimizer: Base optimizer to decorate.
        use_moe (bool): Whether to enable MoE expert parallel.

    Returns:
        HybridParallelOptimizer: Configured optimizer for distributed training.
    """

    if not use_moe:
        return fleet.distributed_optimizer(optimizer)

    from ernie.moe.distributed.hybrid_parallel_optimizer import (
        HybridParallelOptimizer as MoEHybridParallelOptimizer,
    )

    fleet_env = fleet.fleet
    fleet_env.user_defined_optimizer = optimizer
    hp_optim = MoEHybridParallelOptimizer(optimizer, fleet_env._hcg, fleet_env._user_defined_strategy)

    if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].dp_comm_overlap:
        hp_optim._dp_enable = False

    if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].sharding_comm_overlap:
        hp_optim._sharding_enable = False
    return hp_optim
