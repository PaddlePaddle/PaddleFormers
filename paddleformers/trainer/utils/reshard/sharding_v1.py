# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
)

from ....transformers.model_utils import unwrap_optimizer


def shard(node_model_state, model, optimizer):
    cur_rank = max(node_model_state.group.rank, 0)
    optimizer = unwrap_optimizer(optimizer, DygraphShardingOptimizer)
    assert optimizer is not None
    param2rank = optimizer._param2rank

    def filter_func(key):
        names = key
        param_name = names[1]
        assert param_name in param2rank
        dst_rank = param2rank[param_name]
        return dst_rank == cur_rank

    node_model_state.reshard(filter_func)
    return node_model_state


def restore(node_model_state, model, optimizer):
    node_model_state.drop_rank()
    return node_model_state
