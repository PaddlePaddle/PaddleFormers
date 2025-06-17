# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Team. All rights reserved.
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

from paddleformers.transformers import QWenTokenizer


class Qwen2TokenizationTest(unittest.TestCase):
    from_pretrained_id = "qwen/qwen-7b"
    tokenizer_class = QWenTokenizer
    test_slow_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def setUp(self):
        super().setUp()

    def get_tokenizer(self, **kwargs):
        return QWenTokenizer.from_pretrained(self.from_pretrained_id, **kwargs)

    def test_add_special_tokens(self):
        tokenizer = self.get_tokenizer()
        origin_tokens_len = len(tokenizer)

        add_tokens_num = tokenizer.add_special_tokens({"additional_special_tokens": ["<img>"]})
        assert add_tokens_num == 1
        assert len(tokenizer) == origin_tokens_len + 1

        add_tokens_num = tokenizer.add_special_tokens({"unk_token": "<unk>"})
        assert add_tokens_num == 1
        assert len(tokenizer) == origin_tokens_len + 2
