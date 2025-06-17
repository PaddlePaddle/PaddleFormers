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

import random
import time
import unittest

import paddle

from paddleformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return paddle.to_tensor(data=values).reshape(shape)


class StoppingCriteriaTestCase(unittest.TestCase):
    def _get_tensors(self, length):
        batch_size = 3
        vocab_size = 250

        input_ids = ids_tensor((batch_size, length), vocab_size)
        scores = paddle.ones((batch_size, length)) / length
        return input_ids, scores

    def test_list_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=10),
                MaxTimeCriteria(max_time=0.1),
            ]
        )
        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(9)
        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(10)
        self.assertTrue(criteria(input_ids, scores))

    def test_max_length_criteria(self):
        criteria = MaxLengthCriteria(max_length=10)

        input_ids, scores = self._get_tensors(5)
        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(9)
        self.assertFalse(criteria(input_ids, scores))

        input_ids, scores = self._get_tensors(10)
        self.assertTrue(criteria(input_ids, scores))

    def test_max_time_criteria(self):
        input_ids, scores = self._get_tensors(5)

        criteria = MaxTimeCriteria(max_time=0.1)
        self.assertFalse(criteria(input_ids, scores))

        criteria = MaxTimeCriteria(max_time=0.1, initial_timestamp=time.time() - 0.2)
        self.assertTrue(criteria(input_ids, scores))

    def test_validate_stopping_criteria(self):
        validate_stopping_criteria(StoppingCriteriaList([MaxLengthCriteria(10)]), 10)

        with self.assertWarns(UserWarning):
            validate_stopping_criteria(StoppingCriteriaList([MaxLengthCriteria(10)]), 11)

        stopping_criteria = validate_stopping_criteria(StoppingCriteriaList(), 11)

        self.assertEqual(len(stopping_criteria), 1)
