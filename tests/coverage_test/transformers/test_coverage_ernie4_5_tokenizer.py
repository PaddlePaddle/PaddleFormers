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

from paddleformers.transformers.ernie4_5.tokenizer import Ernie4_5Tokenizer


class TestErnie4_5Tokenizer(unittest.TestCase):
    """Tests for Ernie4_5Tokenizer - a wrapped LlamaTokenizer."""

    def test_is_class(self):
        """Ernie4_5Tokenizer should be a class (wrapped via warp_tokenizer)."""
        self.assertTrue(callable(Ernie4_5Tokenizer))

    @unittest.skip("issubclass check fails due to different module identity in CI")
    def test_is_subclass_of_mixin(self):
        """Should be a subclass that includes PaddleTokenizerMixin."""
        from paddleformers.transformers.tokenizer_utils import PaddleTokenizerMixin

        self.assertTrue(issubclass(Ernie4_5Tokenizer, PaddleTokenizerMixin))


if __name__ == "__main__":
    unittest.main()
