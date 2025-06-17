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

from paddleformers.data import JiebaTokenizer, Vocab
from tests.common_test import CpuCommonTest
from tests.testing_utils import create_test_data


class TestJiebaTokenizer(CpuCommonTest):
    def setUp(self):
        test_data_file = create_test_data(__file__)
        self.vocab = Vocab.load_vocabulary(test_data_file, unk_token="[UNK]")
        self.tokenizer = JiebaTokenizer(self.vocab)

    def test_jieba(self):
        text = "一万一"
        token_arr = self.tokenizer.cut(text)
        idx_arr = self.tokenizer.encode(text)
        for i, token in enumerate(token_arr):
            self.check_output_equal(self.vocab(token), idx_arr[i])

        jieba_tokenizer = self.tokenizer.get_tokenizer()
        jieba_token_arr = jieba_tokenizer.lcut(text, False, True)
        self.check_output_equal(token_arr, jieba_token_arr)

    def test_unk(self):
        text = "中国"
        idx_arr = self.tokenizer.encode(text)
        self.check_output_equal(self.vocab[self.vocab.unk_token] in idx_arr, True)


if __name__ == "__main__":
    unittest.main()
