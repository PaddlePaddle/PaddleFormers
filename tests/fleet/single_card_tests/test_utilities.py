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


import unittest
from contextlib import contextmanager

import paddle

from paddleformers.fleet.utils import GlobalMemoryBuffer


class TestModel(paddle.nn.Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        bias: bool,
        shared_embedding: bool = False,
    ):
        super().__init__()
        self.layers = paddle.nn.LayerList(
            [
                paddle.nn.Linear(input_dim, output_dim, bias)
                for _ in range(num_layers)
            ]
        )
        if shared_embedding:
            self.layers[-1].weight.shared_embedding = True


class TestGlobalMemoryBuffer(unittest.TestCase):
    def test_get_tensor(self):
        gmb = GlobalMemoryBuffer()

        # Test 1: Initial allocation
        shape1 = [10, 10]
        dtype = paddle.float32
        name = "buffer1"
        tensor1 = gmb.get_tensor(shape1, dtype, name)

        self.assertEqual(tensor1.shape, shape1)
        self.assertEqual(tensor1.dtype, dtype)
        self.assertIn((name, dtype), gmb.buffer)
        self.assertEqual(
            gmb.buffer[(name, dtype)].shape, [100]
        )  # Flattened size

        # Test 2: Reuse buffer (smaller size)
        shape2 = [5, 5]
        tensor2 = gmb.get_tensor(shape2, dtype, name)
        self.assertEqual(tensor2.shape, shape2)
        # Buffer should still be size 100
        self.assertEqual(gmb.buffer[(name, dtype)].shape, [100])

        # Test 3: Reallocation (larger size)
        shape3 = [20, 10]  # 200 elements
        tensor3 = gmb.get_tensor(shape3, dtype, name)
        self.assertEqual(tensor3.shape, shape3)
        # Buffer should now be size 200
        self.assertEqual(gmb.buffer[(name, dtype)].shape, [200])

        # Test 4: Different name
        name2 = "buffer2"
        tensor4 = gmb.get_tensor(shape1, dtype, name2)
        self.assertEqual(tensor4.shape, shape1)
        self.assertIn((name2, dtype), gmb.buffer)

        # Test 5: Different dtype
        dtype2 = paddle.int32
        tensor5 = gmb.get_tensor(shape1, dtype2, name)
        self.assertEqual(tensor5.dtype, dtype2)
        self.assertIn((name, dtype2), gmb.buffer)

        # Test 6: Context manager
        entered = False

        @contextmanager
        def my_context():
            nonlocal entered
            entered = True
            yield

        gmb.get_tensor([300], dtype, name, mem_alloc_context=my_context)
        self.assertTrue(entered)


if __name__ == "__main__":
    unittest.main()
