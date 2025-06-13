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
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet import fleet


def split_inputs_sequence_dim(inputs, sep_rank=None, sep_degree=None):
    if sep_degree is None and sep_rank is None:
        _hcg = fleet.get_hybrid_communicate_group()
        sep_degree = _hcg.get_sep_parallel_world_size()
        sep_rank = _hcg.get_sep_parallel_rank()
    assert isinstance(sep_degree, int) and isinstance(
        sep_rank, int
    ), f"sep_degree:{type(sep_degree)} and sep_rank:{type(sep_rank)} must be int"
    if sep_degree <= 1:
        return inputs

    def do_split_sequence_dim(data, sep_rank, sep_degree):
        if data is None:
            return None
        assert isinstance(data, paddle.Tensor), f"data should be paddle.Tensor, but is type:{type(data)}"
        assert len(data.shape) == 2, f"data dims should be 2, but shaped: {data.shape}"
        sliced_data = paddle.split(data, num_or_sections=sep_degree, axis=-1)[sep_rank]
        return sliced_data

    if isinstance(inputs, paddle.Tensor):
        return do_split_sequence_dim(inputs, sep_rank, sep_degree)
    elif isinstance(inputs, dict):
        res = {}
        for k, tensor in inputs.items():
            res[k] = do_split_sequence_dim(tensor, sep_rank, sep_degree)
    elif isinstance(inputs, list):
        res = []
        for tensor in inputs:
            res.append(do_split_sequence_dim(tensor, sep_rank, sep_degree))
        raise ValueError(f"the inputs should be a list or a dict, but is type: {type(inputs)}")
    return res


@paddle.no_grad()
def _reshard_qkv(x, group, split_axis=2, concat_axis=0):
    # [s/sep, b, h] -> [s, b, h/sep]
    # [s, b, h/sep] -> [s/sep, b, h]
    group = _get_global_group() if group is None else group
    nranks = dist.get_world_size(group=group)
    shape = x.shape

    assert len(shape) == 3, "Only support 3D tensor, but got {}".format(len(shape))
    assert shape[split_axis] % nranks == 0, "Only support evenly split, but got {} % {} != 0".format(shape[2], nranks)

    comm_tensor_list = paddle.split(x, nranks, axis=split_axis)
    output_list = [paddle.empty_like(comm_tensor_list[0]) for _ in comm_tensor_list]
    dist.alltoall(output_list, comm_tensor_list, group=group)
    reshard_tensor = paddle.concat(output_list, axis=concat_axis)

    return reshard_tensor


class ReshardQKV(PyLayer):
    @staticmethod
    def forward(ctx, x, group=None, split_axis=2, concat_axis=0):
        ctx.group = _get_global_group() if group is None else group
        ctx.split_axis = split_axis
        ctx.concat_axis = concat_axis
        res = _reshard_qkv(x, group, split_axis=ctx.split_axis, concat_axis=ctx.concat_axis)
        return res

    @staticmethod
    def backward(ctx, dy):
        res = _reshard_qkv(dy, ctx.group, split_axis=ctx.concat_axis, concat_axis=ctx.split_axis)
        return res


class ReshardLayer(paddle.nn.Layer):
    def __init__(self, sep_group=None) -> None:
        if sep_group is None:
            _hcg = fleet.get_hybrid_communicate_group()
            sep_group = _hcg.get_sep_parallel_group() if sep_group is None else sep_group
        self.sep_group = sep_group
        self.sep_degree = dist.get_world_size(group=self.sep_group)
        super(ReshardLayer, self).__init__()

    def forward(
        self,
        x,
        split_axis=1,
        concat_axis=2,
    ):
        # if x dims==3, its shape can be [s/sep, b, h] or [b, s/sep, h], the output shape can be [s, b, h/sep] or [b, s, h/sep]
        # if x dims==4, its shape can be [s, b, num_head/sep, head_dim] or [b, s, num_head/sep, head_dim], the output shape can be [s/sep, b, num_head, head_dim] or [b, s/sep, num_head, head_dim]
        shape = x.shape
        assert len(shape) == 3 or len(shape) == 4, "Only support 3D or 4D tensor"
        if len(shape) == 4:
            assert shape[split_axis] % self.sep_degree == 0
            shape[split_axis] = shape[split_axis] // self.sep_degree
            shape[concat_axis] = shape[concat_axis] * self.sep_degree

        input_data = x
        if len(shape) == 3:
            reshard_tensor = ReshardQKV.apply(
                input_data, self.sep_group, split_axis=split_axis, concat_axis=concat_axis
            )
        else:
            input_data = input_data.reshape([0, 0, -1])
            reshard_tensor = ReshardQKV.apply(
                input_data, self.sep_group, split_axis=split_axis, concat_axis=concat_axis
            )
            reshard_tensor.reshape_(shape)
        return reshard_tensor
