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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest


class TestJit(unittest.TestCase):
    """Tests for paddleformers.fleet.jit module."""

    def test_jit_fuser_returns_original_function(self):
        from paddleformers.fleet.jit import jit_fuser

        def original_fn(x):
            return x * 2

        result = jit_fuser(original_fn)
        self.assertEqual(result, original_fn)
        # Verify the function still works
        self.assertEqual(result(5), 10)

    def test_jit_fuser_with_lambda(self):
        from paddleformers.fleet.jit import jit_fuser

        fn = lambda x: x + 1
        result = jit_fuser(fn)
        self.assertEqual(result, fn)
        self.assertEqual(result(10), 11)

    def test_jit_fuser_with_class_method(self):
        from paddleformers.fleet.jit import jit_fuser

        class MyClass:
            def method(self, x):
                return x**2

        obj = MyClass()
        result = jit_fuser(obj.method)
        self.assertEqual(result, obj.method)
        self.assertEqual(result(3), 9)

    def test_jit_fuser_is_callable(self):
        from paddleformers.fleet.jit import jit_fuser

        def my_func():
            return "hello"

        wrapped = jit_fuser(my_func)
        self.assertTrue(callable(wrapped))
        self.assertEqual(wrapped(), "hello")
