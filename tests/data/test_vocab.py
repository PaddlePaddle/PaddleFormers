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
from collections import Counter

from paddleformers.data import Vocab
from tests import testing_utils
from tests.common_test import CpuCommonTest


class TestVocab(CpuCommonTest):
    def create_counter(self):
        counter = Counter()
        counter["一万七千多"] = 2
        counter["一万七千余"] = 3
        counter["一万万"] = 1
        counter["一万七千多户"] = 3
        counter["一万七千"] = 4
        counter["一万七"] = 0
        self.counter = counter

    def setUp(self):
        self.create_counter()

    @testing_utils.assert_raises(ValueError)
    def test_invalid_specail_token(self):
        Vocab(wrong_kwarg="")

    @testing_utils.assert_raises(ValueError)
    def test_invalid_identifier(self):
        Vocab(counter=self.counter, _special_token="")

    @testing_utils.assert_raises(ValueError)
    def test_sort_index_value_error1(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "IP地址": 3}
        Vocab(counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx)

    @testing_utils.assert_raises(ValueError)
    def test_sort_index_value_error2(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万七千": 2}
        Vocab(counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx)

    @testing_utils.assert_raises(ValueError)
    def test_sort_index_value_error3(self):
        token_to_idx = {"一万七千多": -1, "一万七千余": 2, "一万七千": 3}
        Vocab(counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx)

    @testing_utils.assert_raises(ValueError)
    def test_to_token_excess_size(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万万": 3}
        vocab = Vocab(counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx)
        vocab.to_tokens(len(vocab))

    def test_counter(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万万": 3}
        vocab = Vocab(counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx)
        self.check_output_equal(vocab.to_tokens(1), "一万七千多")
        self.check_output_equal(vocab.to_tokens(2), "一万七千余")
        self.check_output_equal(vocab.to_tokens(3), "一万万")

    def test_json(self):
        token_to_idx = {"一万七千多": 1, "一万七千余": 2, "一万万": 3}
        vocab = Vocab(counter=self.counter, unk_token="[UNK]", token_to_idx=token_to_idx)
        json_str = vocab.to_json()
        copied_vocab = Vocab.from_json(json_str)
        for key, value in copied_vocab.token_to_idx.items():
            self.check_output_equal(value, vocab[key])


if __name__ == "__main__":
    unittest.main()
