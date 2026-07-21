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

import sys
import types
import unittest

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.train.sft.make_data_utils import DataGenerator


class TestDataGenerator(unittest.TestCase):
    """Tests for DataGenerator class"""

    def setUp(self):
        self.simple_list = [1, 2, 3]
        self.generator = DataGenerator(self.simple_list)

    def test_init(self):
        self.assertIs(self.generator.data_source, self.simple_list)
        self.assertIsNotNone(self.generator.data_source_iter)

    def test_iter_returns_self(self):
        result = iter(self.generator)
        self.assertIs(result, self.generator)

    def test_next_returns_items_in_order(self):
        self.assertEqual(next(self.generator), 1)
        self.assertEqual(next(self.generator), 2)
        self.assertEqual(next(self.generator), 3)

    def test_next_wraps_around_after_stop_iteration(self):
        # Consume all items
        self.assertEqual(next(self.generator), 1)
        self.assertEqual(next(self.generator), 2)
        self.assertEqual(next(self.generator), 3)
        # Should restart from the beginning
        self.assertEqual(next(self.generator), 1)
        self.assertEqual(next(self.generator), 2)
        self.assertEqual(next(self.generator), 3)
        self.assertEqual(next(self.generator), 1)

    def test_infinite_iteration(self):
        results = []
        for i, val in enumerate(self.generator):
            results.append(val)
            if i >= 9:
                break
        # Should repeat the list [1,2,3] multiple times
        expected = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
        self.assertEqual(results, expected)

    def test_single_element_source(self):
        gen = DataGenerator([42])
        self.assertEqual(next(gen), 42)
        self.assertEqual(next(gen), 42)
        self.assertEqual(next(gen), 42)

    def test_string_data_source(self):
        gen = DataGenerator(["a", "b", "c"])
        self.assertEqual(next(gen), "a")
        self.assertEqual(next(gen), "b")
        self.assertEqual(next(gen), "c")
        self.assertEqual(next(gen), "a")

    def test_dict_data_source(self):
        data = [{"x": 1}, {"x": 2}]
        gen = DataGenerator(data)
        self.assertEqual(next(gen), {"x": 1})
        self.assertEqual(next(gen), {"x": 2})
        self.assertEqual(next(gen), {"x": 1})


if __name__ == "__main__":
    unittest.main()
