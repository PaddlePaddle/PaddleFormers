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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import paddle

from ..utils import get_tensor_model_parallel_group_if_none

_MAX_DATA_DIM = 5


def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert data[key].dtype == target_dtype, (
            f"{key} has data type {data[key].dtype} which " f"is different than {target_dtype}"
        )


def _build_key_size_numel_dictionaries(keys, data, tp_group=None):
    """Build the size on rank 0 and broadcast."""
    tp_group = get_tensor_model_parallel_group_if_none(tp_group)
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    # Pack the sizes on rank zero.
    if tp_group.rank == 0:
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, "you should increase MAX_DATA_DIM"
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = paddle.tensor(sizes, dtype=paddle.int32)
    global_ranks_in_group = tp_group.ranks
    paddle.distributed.broadcast(sizes_cuda, global_ranks_in_group[0], group=tp_group)

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype, tp_group=None):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Args:
        keys: list of keys in the data dictionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: paddle data type of all tensors in data associated
                  with keys.
        tp_group: the tensor model parallel group to broadcast to.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)
    tp_group = get_tensor_model_parallel_group_if_none(tp_group)
    # Pack on rank zero.
    if tp_group.rank == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = paddle.concat([data[key].cuda().contiguous().view(-1) for key in keys], dim=0)
    else:
        flatten_data = paddle.empty([total_numel], dtype=datatype)

    # Broadcast
    global_ranks_in_group = tp_group.ranks
    paddle.distributed.broadcast(flatten_data, global_ranks_in_group[0], group=tp_group)

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = paddle.narrow(flatten_data, 0, offset, numel).view(size)
        offset += numel

    return output
