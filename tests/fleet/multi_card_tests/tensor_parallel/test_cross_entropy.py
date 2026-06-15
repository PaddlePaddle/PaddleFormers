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

import numpy as np
import paddle

from paddleformers.fleet.tensor_parallel.cross_entropy import (
    vocab_parallel_cross_entropy,
)
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4, 1)
    np_vocab_parallel_logits = (np.arange(32)).reshape((1, 32)) % 8
    np_vocab_parallel_logits = np.repeat(np_vocab_parallel_logits, 16, 0)
    vocab_parallel_logits = paddle.tensor(np_vocab_parallel_logits)
    target = paddle.arange(0, 32, 2).cuda()
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = paddle.tensor(
        [
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
            10.2309,
            8.2309,
            6.2309,
            4.2309,
        ]
    ).cuda()
    output.backward()
    assert paddle.equal_all(paddle.round(expected_output), paddle.round(output))


if __name__ == "__main__":
    test_vocab_parallel_cross_entropy()
