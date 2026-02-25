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
import tempfile
import unittest

from paddleformers.transformers import Ernie4_5_VLTokenizer
from paddleformers.utils.download import DownloadSource
from tests.testing_utils import set_proxy

HUB_FLAG = "aistudio"


class Ernie4_5_VL_TokenizationTest(unittest.TestCase):
    from_pretrained_id = "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Base-PT"
    tokenizer_class = Ernie4_5_VLTokenizer
    test_slow_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    @set_proxy(DownloadSource.AISTUDIO)
    def test_slow_tokenizer_from_pretrained(self):
        tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
            self.from_pretrained_id, download_hub=HUB_FLAG, trust_remote_code=True
        )
        self.assertTrue(tokenizer is not None)

    @set_proxy(DownloadSource.AISTUDIO)
    def test_slow_tokenizer_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
                self.from_pretrained_id, download_hub=HUB_FLAG, trust_remote_code=True
            )
            tokenizer.model_max_length = 512
            tokenizer.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")))

    @set_proxy(DownloadSource.AISTUDIO)
    def test_tokenize(self):
        tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
            self.from_pretrained_id, download_hub=HUB_FLAG, trust_remote_code=True
        )
        text = "hello world, this is a tokenizer test"
        output_dict = tokenizer(text)
        decode_text = tokenizer.decode(output_dict["input_ids"], skip_special_tokens=True)
        self.assertEqual(text, decode_text)

    @set_proxy(DownloadSource.AISTUDIO)
    def test_special_token(self):
        tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
            self.from_pretrained_id, download_hub=HUB_FLAG, trust_remote_code=True
        )

        self.assertEqual(tokenizer.space_token, "<mask:1>")
        self.assertEqual(tokenizer.space_token_id, 100274)
        self.assertEqual(tokenizer.gend_token, "<mask:7>")
        self.assertEqual(tokenizer.gend_token_id, 100280)
        self.assertEqual(tokenizer.im_start_id, 101304)
        self.assertEqual(tokenizer.im_end_id, 101305)

        vocab = tokenizer.get_vocab()
        self.assertTrue("<|IMAGE_PLACEHOLDER|>" in vocab)


Ernie4_5_VL_TokenizationTest().test_slow_tokenizer_from_pretrained()
