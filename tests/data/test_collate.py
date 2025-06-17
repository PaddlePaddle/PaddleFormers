# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.data import Dict, Pad, Stack, Tuple
from tests import testing_utils
from tests.common_test import CpuCommonTest


class TestStack(CpuCommonTest):
    def setUp(self):
        self.input = [[1, 2, 3, 4], [4, 5, 6, 8], [8, 9, 1, 2]]
        self.expected_result = np.array(self.input)

    def test_stack(self):
        result = Stack()(self.input)
        self.check_output_equal(self.expected_result, result)


class TestPad(CpuCommonTest):
    def setUp(self):
        self.input = [[1, 2, 3, 4], [4, 5, 6], [8, 9]]
        self.expected_result = np.array([[1, 2, 3, 4], [4, 5, 6, 0], [8, 9, 0, 0]])

    def test_pad(self):
        result = Pad()(self.input)
        self.check_output_equal(self.expected_result, result)


class TestPadLeft(CpuCommonTest):
    def setUp(self):
        self.input = [[1, 2, 3, 4], [4, 5, 6], [8, 9]]
        self.expected_result = np.array([[1, 2, 3, 4], [0, 4, 5, 6], [0, 0, 8, 9]])

    def test_pad(self):
        result = Pad(pad_right=False)(self.input)
        self.check_output_equal(self.expected_result, result)


class TestPadRetLength(CpuCommonTest):
    def setUp(self):
        self.input = [[1, 2, 3, 4], [4, 5, 6], [8, 9]]
        self.expected_result = np.array([[1, 2, 3, 4], [4, 5, 6, 0], [8, 9, 0, 0]])

    def test_pad(self):
        result, length = Pad(ret_length=True)(self.input)
        self.check_output_equal(self.expected_result, result)
        self.check_output_equal(length, np.array([4, 3, 2]))


class TestTuple(CpuCommonTest):
    def setUp(self):
        self.input = [[[1, 2, 3, 4], [1, 2, 3, 4]], [[4, 5, 6, 8], [4, 5, 6]], [[8, 9, 1, 2], [8, 9]]]
        self.expected_result = (
            np.array([[1, 2, 3, 4], [4, 5, 6, 8], [8, 9, 1, 2]]),
            np.array([[1, 2, 3, 4], [4, 5, 6, 0], [8, 9, 0, 0]]),
        )

    def _test_impl(self, list_fn=True):
        if list_fn:
            batchify_fn = Tuple([Stack(), Pad(axis=0, pad_val=0)])
        else:
            batchify_fn = Tuple(Stack(), Pad(axis=0, pad_val=0))
        result = batchify_fn(self.input)
        self.check_output_equal(result[0], self.expected_result[0])
        self.check_output_equal(result[1], self.expected_result[1])

    def test_tuple(self):
        self._test_impl()

    def test_tuple_list(self):
        self._test_impl(False)

    @testing_utils.assert_raises
    def test_empty_fn(self):
        Tuple([Stack()], Pad(axis=0, pad_val=0))


class TestDict(CpuCommonTest):
    def setUp(self):
        self.input = [
            {"text": [1, 2, 3, 4], "label": [1]},
            {"text": [4, 5, 6], "label": [0]},
            {"text": [7, 8], "label": [1]},
        ]
        self.expected_result = (np.array([[1, 2, 3, 4], [4, 5, 6, 0], [7, 8, 0, 0]]), np.array([[1], [0], [1]]))

    def test_dict(self):
        batchify_fn = Dict({"text": Pad(axis=0, pad_val=0), "label": Stack()})
        result = batchify_fn(self.input)
        self.check_output_equal(result[0], self.expected_result[0])
        self.check_output_equal(result[1], self.expected_result[1])


if __name__ == "__main__":
    unittest.main()
