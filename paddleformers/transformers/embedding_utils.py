# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed import fleet


def dist_gather_tensor_with_gradient(tensor):
    if tensor is None:
        return None

    if paddle.distributed.get_world_size() <= 1:
        return tensor

    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    data_group = hcg.get_data_parallel_group()
    data_rank = data_group.rank

    if sharding_group.nranks == 1 and data_group.nranks == 1:
        return tensor

    if sharding_group.nranks > 1:
        all_tensors = []
        paddle.distributed.all_gather(all_tensors, tensor.contiguous(), group=sharding_group)
        all_tensors[sharding_rank] = tensor
        all_tensors = paddle.concat(all_tensors, axis=0)
    else:
        all_tensors = tensor

    if data_group.nranks > 1:
        final_tensors = []
        paddle.distributed.all_gather(final_tensors, all_tensors.contiguous(), group=data_group)
        final_tensors[data_rank] = all_tensors
        final_tensors = paddle.concat(final_tensors, axis=0)
    else:
        final_tensors = all_tensors

    return final_tensors
