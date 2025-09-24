# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.communication.group import _get_global_group

from paddleformers.transformers.segment_parallel_utils import (
    ReshardLayer,
    split_inputs_sequence_dim,
)


def prepare_data(batch_major=True, dim_size=4, batch_size=2, seq_len=2, num_head=2, h=4):
    assert dim_size == 3 or dim_size == 4, f"dim_size should be 3 or 4, but {dim_size}"
    batch_size = batch_size
    seq_len = seq_len
    h = h
    num_head = num_head
    sep = dist.get_world_size()
    # assert sep == 2, f"sep should be 2, but {sep}"
    num_elem = batch_size * seq_len // sep * num_head * h // num_head
    local_rank = dist.get_rank()
    input_data_list = []
    split_tensor_list = []
    if dim_size == 4:
        shape = (
            [batch_size, seq_len, num_head // sep, h // num_head]
            if batch_major
            else [seq_len, batch_size, num_head // sep, h // num_head]
        )
        split_axis = 1 if batch_major else 0
        concat_axis = 2
    else:
        shape = [batch_size, seq_len // sep, h] if batch_major else [seq_len // sep, batch_size, h]
        split_axis = 2
        concat_axis = 1 if batch_major else 0
    for rank in range(sep):
        t = paddle.to_tensor(
            np.reshape(np.arange(rank * num_elem, (rank + 1) * num_elem) + 1, shape), dtype=paddle.float32
        )
        input_data_list.append(t)
        split_tensor_list.append(paddle.split(t, sep, axis=split_axis))
    input_data = input_data_list[dist.get_rank()]
    expected_output_data = [t[local_rank] for t in split_tensor_list]
    expected_output_data = paddle.cat(expected_output_data, axis=concat_axis)
    return input_data, expected_output_data


def run_forward_backward(x, y_grad, split_axis=0, concat_axis=2):
    x = x.detach()
    x.stop_gradient = False
    reshard_layer = ReshardLayer(sep_group=_get_global_group())
    y = reshard_layer(
        x,
        split_axis=split_axis,
        concat_axis=concat_axis,
    )
    paddle.autograd.backward([y], [y_grad], True)
    return y, x.grad


def should_test(sep_degree):
    if sep_degree <= 1:
        print(f"sep degree should greater than 1, but is {sep_degree}, skip this test")
    return sep_degree > 1


class TestReshardLayer(unittest.TestCase):
    def setUp(self):
        dist.init_parallel_env()

    def test_split_inputs(self):
        batch_size = 8
        seq_len = 4096
        sep = dist.get_world_size()
        sep_rank = dist.get_rank()
        if not should_test(sep):
            return

        inputs_ids = paddle.randint(low=0, high=65535, shape=(batch_size, seq_len))
        labels = paddle.randint(low=0, high=2, shape=(batch_size, seq_len))
        inputs = {"inputs_ids": inputs_ids, "labels": labels}

        splited_inputs = split_inputs_sequence_dim(inputs, sep_rank, sep)
        expected_local_inputs = {}
        for k, v in inputs.items():
            expected_local_inputs[k] = paddle.split(v, sep, axis=1)[sep_rank]
            assert k in splited_inputs
            np.testing.assert_equal(expected_local_inputs[k].numpy(), splited_inputs[k].numpy())

    def test_reshard(self):
        # [s / sep, b, h] -> [s, b, h / sep]
        seq_len = 16
        sep = dist.get_world_size()
        if not should_test(sep):
            return
        assert seq_len % sep == 0, f"seq_len should be divisible by sep, seq_len:{seq_len}, sep:{sep}"

        def check_equal(input_data, expected_output_data, split_axis=0, concat_axis=2):
            bin_bout_output_grad = expected_output_data
            bin_bout_output, bin_bout_input_grad = run_forward_backward(
                input_data,
                bin_bout_output_grad,
                split_axis=split_axis,
                concat_axis=concat_axis,
            )
            np.testing.assert_equal(bin_bout_output.numpy(), expected_output_data.numpy())
            np.testing.assert_equal(bin_bout_input_grad.numpy(), input_data.numpy())

        dim_size = 3
        for batch_major in [True, False]:
            for dim_size in [3, 4]:
                for batch_size in [1, 2]:
                    for seq_len in [4096, 4096 * 2, 4096 * 4]:
                        for num_head in [32, 64]:
                            for hidden_size in [num_head * 16, num_head * 32]:
                                if dim_size == 3:
                                    # check reshard before flash attn, shaped: [b, s, h]
                                    input_data, expected_output_data = prepare_data(
                                        batch_major=batch_major,
                                        dim_size=dim_size,
                                        batch_size=batch_size,
                                        seq_len=seq_len,
                                        num_head=num_head,
                                        h=hidden_size,
                                    )
                                    split_axis_for_num_head = 2
                                    concat_axis_for_seq = 1 if batch_major else 0
                                    check_equal(
                                        input_data,
                                        expected_output_data,
                                        split_axis=split_axis_for_num_head,
                                        concat_axis=concat_axis_for_seq,
                                    )
                                elif dim_size == 4:
                                    # check reshard after flash attn, shaped: [b, s, num_head, head_dim]
                                    input_data, expected_output_data = prepare_data(
                                        batch_major=batch_major,
                                        dim_size=dim_size,
                                        batch_size=batch_size,
                                        seq_len=seq_len,
                                        num_head=num_head,
                                        h=hidden_size,
                                    )
                                    split_axis_for_seq = 1 if batch_major else 0
                                    concat_axis_for_num_head = 2
                                    check_equal(
                                        input_data,
                                        expected_output_data,
                                        split_axis=split_axis_for_seq,
                                        concat_axis=concat_axis_for_num_head,
                                    )


if __name__ == "__main__":
    unittest.main()
