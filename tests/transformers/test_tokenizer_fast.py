# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

from paddleformers.transformers import PretrainedTokenizerFast
from tests.testing_utils import require_package
from tests.transformers.test_tokenizer_common import TokenizerTesterMixin


@require_package("tokenizers")
class PreTrainedTokenizationFastTest(TokenizerTesterMixin, unittest.TestCase):
    rust_tokenizer_class = PretrainedTokenizerFast
    tokenizer_class = PretrainedTokenizerFast
    test_slow_tokenizer = False
    test_rust_tokenizer = True
    from_pretrained_vocab_key = "vocab_file"

    def setUp(self):
        self.test_rust_tokenizer = False  # because we don't have pretrained_vocab_files_map
        super().setUp()
        self.test_rust_tokenizer = True

        model_paths = ["__internal_testing__/tiny-random-llama-fast"]
        # self.bytelevel_bpe_model_name = "SaulLu/dummy-tokenizer-bytelevel-bpe"

        # Inclusion of 2 tokenizers to test different types of models (Unigram and WordLevel for the moment)
        self.tokenizers_list = [(PretrainedTokenizerFast, model_path, {}) for model_path in model_paths]

        tokenizer = PretrainedTokenizerFast.from_pretrained(model_paths[0])
        tokenizer.save_pretrained(self.tmpdirname)

    @unittest.skip(
        "We disable this test for PretrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_tokenizer_mismatch_warning(self):
        pass

    @unittest.skip(
        "We disable this test for PretrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_encode_decode_with_spaces(self):
        pass

    @unittest.skip(
        "We disable this test for PretrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_added_tokens_serialization(self):
        pass

    @unittest.skip(
        "We disable this test for PretrainedTokenizerFast because it is the only tokenizer that is not linked to any model"
    )
    def test_additional_special_tokens_serialization(self):
        pass

    @unittest.skip(reason="PretrainedTokenizerFast is the only tokenizer that is not linked to any model")
    def test_prepare_for_model(self):
        pass

    @unittest.skip(reason="PretrainedTokenizerFast doesn't have tokenizer_file in its signature")
    def test_rust_tokenizer_signature(self):
        pass

    @unittest.skip(reason="PretrainedTokenizerFast passes error cases temporarily")
    def test_maximum_encoding_length_single_input(self):
        pass

    @unittest.skip(reason="PretrainedTokenizerFast passes error cases temporarily")
    def test_offsets_mapping_with_unk(self):
        pass

    @unittest.skip(reason="PretrainedTokenizerFast passes error cases temporarily")
    def test_maximum_encoding_length_pair_input(self):
        pass

    @unittest.skip(reason="PretrainedTokenizerFast passes error cases temporarily")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip(reason="PretrainedTokenizerFast passes error cases temporarily")
    def test_pretrained_model_lists(self):
        pass

    # def test_training_new_tokenizer(self):
    #     tmpdirname_orig = self.tmpdirname
    #     # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
    #     for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
    #         with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
    #             try:
    #                 self.tmpdirname = tempfile.mkdtemp()
    #                 tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    #                 tokenizer.save_pretrained(self.tmpdirname)
    #                 super().test_training_new_tokenizer()
    #             finally:
    #                 # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
    #                 # is restored
    #                 shutil.rmtree(self.tmpdirname)
    #                 self.tmpdirname = tmpdirname_orig

    # def test_training_new_tokenizer_with_special_tokens_change(self):
    #     tmpdirname_orig = self.tmpdirname
    #     # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
    #     for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
    #         with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
    #             try:
    #                 self.tmpdirname = tempfile.mkdtemp()
    #                 tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    #                 tokenizer.save_pretrained(self.tmpdirname)
    #                 super().test_training_new_tokenizer_with_special_tokens_change()
    #             finally:
    #                 # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
    #                 # is restored
    #                 shutil.rmtree(self.tmpdirname)
    #                 self.tmpdirname = tmpdirname_orig

    # def test_training_new_tokenizer_with_bytelevel(self):
    #     tokenizer = self.rust_tokenizer_class.from_pretrained(self.bytelevel_bpe_model_name)

    #     toy_text_iterator = ("a" for _ in range(1000))
    #     new_tokenizer = tokenizer.train_new_from_iterator(text_iterator=toy_text_iterator, length=1000, vocab_size=50)

    #     encoding_ids = new_tokenizer.encode("a🤗")
    #     self.assertEqual(encoding_ids, [64, 172, 253, 97, 245])

    # def test_init_from_tokenizers_model(self):
    #     from tokenizers import Tokenizer

    #     sentences = ["Hello, y'all!", "How are you 😁 ? There should not be any issue right?"]

    #     tokenizer = Tokenizer.from_pretrained("google-t5/t5-base")
    #     # Enable padding
    #     tokenizer.enable_padding(pad_id=0, pad_token="<pad>", length=512, pad_to_multiple_of=8)
    #     self.assertEqual(
    #         tokenizer.padding,
    #         {
    #             "length": 512,
    #             "pad_to_multiple_of": 8,
    #             "pad_id": 0,
    #             "pad_token": "<pad>",
    #             "pad_type_id": 0,
    #             "direction": "right",
    #         },
    #     )
    #     fast_tokenizer = PretrainedTokenizerFast(tokenizer_object=tokenizer)
    #     tmpdirname = tempfile.mkdtemp()
    #     fast_tokenizer.save_pretrained(tmpdirname)
    #     fast_from_saved = PretrainedTokenizerFast.from_pretrained(tmpdirname)
    #     for tok in [fast_tokenizer, fast_from_saved]:
    #         self.assertEqual(tok.pad_token_id, 0)
    #         self.assertEqual(tok.padding_side, "right")
    #         self.assertEqual(tok.pad_token, "<pad>")
    #         self.assertEqual(tok.init_kwargs["max_length"], 512)
    #         self.assertEqual(tok.init_kwargs["pad_to_multiple_of"], 8)
    #         self.assertEqual(tok(sentences, padding = True), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1, 0, 0, 0, 0,0, 0, 0, 0],[ 571, 33, 25, 3, 2, 3, 58, 290, 225, 59, 36, 136, 962, 269, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip

    #     tokenizer.enable_truncation(8, stride=0, strategy="longest_first", direction="right")
    #     self.assertEqual(
    #         tokenizer.truncation, {"max_length": 8, "stride": 0, "strategy": "longest_first", "direction": "right"}
    #     )
    #     fast_tokenizer = PretrainedTokenizerFast(tokenizer_object=tokenizer)
    #     tmpdirname = tempfile.mkdtemp()
    #     fast_tokenizer.save_pretrained(tmpdirname)
    #     fast_from_saved = PretrainedTokenizerFast.from_pretrained(tmpdirname)
    #     for tok in [fast_tokenizer, fast_from_saved]:
    #         self.assertEqual(tok.truncation_side, "right")
    #         self.assertEqual(tok.init_kwargs["truncation_strategy"], "longest_first")
    #         self.assertEqual(tok.init_kwargs["max_length"], 8)
    #         self.assertEqual(tok.init_kwargs["stride"], 0)
    #         # NOTE even if the model has a default max_length, it is not used...
    #         # thus tok(sentences, truncation = True) does nothing and does not warn either
    #         self.assertEqual(tok(sentences, truncation = True, max_length = 8), {'input_ids': [[8774, 6, 3, 63, 31, 1748, 55, 1],[ 571, 33, 25, 3, 2, 3, 58, 1]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1, 1, 1]]})  # fmt: skip
