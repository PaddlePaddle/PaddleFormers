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

# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import paddle

from paddleformers.fleet.tensor_parallel import mappings
from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_CopyToModelParallelRegion():
    input_data = paddle.ones(1).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

    class Ctx:
        group = tp_group

    output_data = mappings._CopyToModelParallelRegion.backward(
        Ctx(), input_data
    )
    result = paddle.ones([1]).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert paddle.equal_all(output_data, result)
    assert paddle.equal_all(
        input_data, mappings.copy_to_tensor_model_parallel_region(input_data)
    )
    assert paddle.equal_all(
        input_data,
        mappings._CopyToModelParallelRegion.symbolic(
            None, input_data, tp_group
        ),
    )


def test_ReduceFromModelParallelRegion():
    input_data = paddle.ones(1).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings._ReduceFromModelParallelRegion.symbolic(
        None, input_data, tp_group
    )

    result = paddle.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert paddle.equal_all(output_data, result)

    input_data = paddle.ones(1).cuda() * Utils.rank
    assert paddle.equal_all(
        mappings.reduce_from_tensor_model_parallel_region(input_data), result
    )

    class Ctx:
        group = tp_group

    output_data = mappings._ReduceFromModelParallelRegion.backward(
        Ctx(), input_data
    )
    assert paddle.equal_all(input_data, output_data)


def test_ScatterToModelParallelRegion():
    input_data = paddle.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.scatter_to_tensor_model_parallel_region(input_data)

    req_dim = int(Utils.rank)
    assert paddle.equal_all(output_data, input_data[:, req_dim].reshape((8, 1)))
    output_data = mappings._ScatterToModelParallelRegion.symbolic(
        None, input_data, tp_group
    )
    assert paddle.equal_all(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = paddle.ones([8]).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    actual_output_data = mappings._ScatterToModelParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.cat(
        (
            paddle.ones([8]) * 0,
            paddle.ones([8]) * 1,
            paddle.ones([8]) * 2,
            paddle.ones([8]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(actual_output_data, expected_output)


def test_GatherFromModelParallelRegion():
    input_data = paddle.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = Utils.rank

    class Ctx:
        group = tp_group

    output_data = mappings._GatherFromModelParallelRegion.backward(
        Ctx(), input_data
    )
    assert paddle.equal_all(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = paddle.ones([8]).cuda() * Utils.rank
    actual_output_data = mappings.gather_from_tensor_model_parallel_region(
        input_data
    )
    expected_output = paddle.cat(
        (
            paddle.ones([8]) * 0,
            paddle.ones([8]) * 1,
            paddle.ones([8]) * 2,
            paddle.ones([8]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(actual_output_data, expected_output)
    assert paddle.equal_all(
        mappings._GatherFromModelParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected_output,
    )


def test_ScatterToSequenceParallelRegion():
    input_data = paddle.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = Utils.rank * 2
    output_data = mappings._ScatterToSequenceParallelRegion.symbolic(
        None, input_data, tp_group
    )
    assert paddle.equal_all(output_data, input_data[req_dim : req_dim + 2, :])
    output_data = mappings.scatter_to_sequence_parallel_region(input_data)
    assert paddle.equal_all(output_data, input_data[req_dim : req_dim + 2, :])

    input_data = paddle.ones([4]).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    output_data = mappings._ScatterToModelParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.concat(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(output_data, expected_output)


def test_GatherFromSequenceParallelRegion():
    input_data = paddle.ones([4]).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.gather_from_sequence_parallel_region(input_data)
    expected_output = paddle.concat(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(output_data, expected_output)
    assert paddle.equal_all(
        mappings._GatherFromSequenceParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected_output,
    )
    input_data = paddle.vstack(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()

    class Ctx:
        tensor_parallel_output_grad = True
        output_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._GatherFromSequenceParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.ones((1, 4)).cuda() * 4 * int(Utils.rank % 4)
    assert paddle.equal_all(output_data, expected_output)


def test_ReduceScatterToSequenceParallelRegion():
    input_data = paddle.vstack(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.reduce_scatter_to_sequence_parallel_region(
        input_data
    )
    expected_output = paddle.ones([1, 4]).cuda() * 4 * int(Utils.rank % 4)
    assert paddle.equal_all(output_data, expected_output)
    assert paddle.equal_all(
        mappings._ReduceScatterToSequenceParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected_output.reshape((1, 4)),
    )
    input_data = paddle.ones([4]).cuda() * Utils.rank

    class Ctx:
        input_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._ReduceScatterToSequenceParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.concat(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(output_data, expected_output)


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    test_CopyToModelParallelRegion()
    test_ReduceFromModelParallelRegion()
    test_ScatterToModelParallelRegion()
    test_GatherFromModelParallelRegion()
    test_ReduceScatterToSequenceParallelRegion()
    test_GatherFromSequenceParallelRegion()
