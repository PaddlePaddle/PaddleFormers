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
from paddle.distributed.fleet import fleet


def split_inputs_sequence_dim_load_balance(inputs, rank=None, degree=None):
    if degree is None and rank is None:
        _hcg = fleet.get_hybrid_communicate_group()
        degree = _hcg.get_sep_parallel_world_size()
        rank = _hcg.get_sep_parallel_rank()
    assert isinstance(degree, int) and isinstance(
        rank, int
    ), f"degree:{type(degree)} and rank:{type(rank)} must be int"
    if degree <= 1:
        return inputs

    def do_split_sequence_dim_load_balance(data, rank, degree):
        if data is None:
            return None
        assert isinstance(data, paddle.Tensor), f"data should be paddle.Tensor, but is type:{type(data)}"
        assert len(data.shape) == 2, f"data dims should be 2, but shaped: {data.shape}"
        sliced_datas = paddle.split(data, num_or_sections=degree * 2, axis=-1)
        sliced_data0, sliced_data1 = sliced_datas[rank], sliced_datas[degree * 2 - 1 - rank]
        return paddle.concat([sliced_data0, sliced_data1], axis=-1)

    if isinstance(inputs, paddle.Tensor):
        return do_split_sequence_dim_load_balance(inputs, rank, degree)
    elif isinstance(inputs, dict):
        res = {}
        for k, tensor in inputs.items():
            res[k] = do_split_sequence_dim_load_balance(tensor, rank, degree)
    elif isinstance(inputs, list):
        res = []
        for tensor in inputs:
            res.append(do_split_sequence_dim_load_balance(tensor, rank, degree))
    else:
        raise ValueError(f"the inputs should be a list or a dict, but is type: {type(inputs)}")
    return res
