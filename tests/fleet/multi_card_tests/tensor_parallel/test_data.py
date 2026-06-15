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

import paddle

from paddleformers.fleet.tensor_parallel.data import broadcast_data
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_broadcast_data():
    Utils.initialize_model_parallel(2, 2)
    input_data = {
        0: paddle.ones((8, 8)).cuda() * 0.0,
        1: paddle.ones((8, 8)).cuda() * 1.0,
        2: paddle.ones((8, 8)).cuda() * 2.0,
        3: paddle.ones((8, 8)).cuda() * 3.0,
        4: paddle.ones((8, 8)).cuda() * 4.0,
        5: paddle.ones((8, 8)).cuda() * 5.0,
        6: paddle.ones((8, 8)).cuda() * 6.0,
        7: paddle.ones((8, 8)).cuda() * 7.0,
    }
    dtype = paddle.float32
    actual_output = broadcast_data([0, 1], input_data, dtype)
    assert paddle.equal_all(actual_output[0], input_data[0])
    assert paddle.equal_all(actual_output[1], input_data[1])


if __name__ == "__main__":
    test_broadcast_data()
