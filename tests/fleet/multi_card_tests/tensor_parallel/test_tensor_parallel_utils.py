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

import paddleformers.fleet.tensor_parallel.utils as util
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_split_tensor_along_last_dim():
    input_tensor = paddle.rand((3, 4))
    paddle.equal_all(
        input_tensor[0:2, 0:2],
        util.split_tensor_along_last_dim(input_tensor, 2)[0],
    )
    paddle.equal_all(
        input_tensor[2:, 2:],
        util.split_tensor_along_last_dim(input_tensor, 2)[1],
    )


def test_split_tensor_into_1d_equal_chunks():
    input_tensor = paddle.rand((3, 4))
    output_tensor = util.split_tensor_into_1d_equal_chunks(input_tensor)
    if Utils.rank % 2 == 0:
        start = 0
        end = int(input_tensor.numel() / 2)
    else:
        start = int(input_tensor.numel() / 2)
        end = input_tensor.numel()

    assert paddle.equal_all(output_tensor, input_tensor.flatten()[start:end])


def test_gather_split_1d_tensor():
    input_tensor = paddle.ones((2, 4)).cuda() * Utils.rank
    actual_output_tensor = util.gather_split_1d_tensor(input_tensor.flatten())
    if Utils.rank % 2 == 0:
        expected_output_tensor = paddle.concat((input_tensor.flatten(), input_tensor.flatten() + 1))
    else:
        expected_output_tensor = paddle.concat((input_tensor.flatten() - 1, input_tensor.flatten()))
    assert paddle.equal_all(actual_output_tensor, expected_output_tensor)


def test_vocab():
    global_vocab_size = 1600
    per_partition_vocab_size = 1600 / Utils.world_size
    assert (Utils.rank * per_partition_vocab_size, (Utils.rank + 1) * per_partition_vocab_size,) == (
        util.VocabUtility.vocab_range_from_per_partition_vocab_size(
            global_vocab_size // Utils.world_size, Utils.rank, Utils.world_size
        )
    )
    assert (
        Utils.rank * per_partition_vocab_size,
        (Utils.rank + 1) * per_partition_vocab_size,
    ) == (util.VocabUtility.vocab_range_from_global_vocab_size(global_vocab_size, Utils.rank, Utils.world_size))


if __name__ == "__main__":
    Utils.initialize_model_parallel(tensor_parallel_size=2, pipeline_parallel_size=2)
    test_split_tensor_along_last_dim()
    test_split_tensor_into_1d_equal_chunks()
    test_gather_split_1d_tensor()
    test_vocab()
