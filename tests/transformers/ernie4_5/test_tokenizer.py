# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

from parameterized import parameterized_class

from paddleformers.transformers.auto.tokenizer import AutoTokenizer
from paddleformers.transformers.llama.tokenizer import LlamaTokenizer
from paddleformers.transformers.tokenizer_utils import PretrainedTokenizer
from paddleformers.transformers.tokenizer_utils_fast import PretrainedTokenizerFast

from ..test_tokenizer_common import TokenizerTesterMixin

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
}


class LlamaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = LlamaTokenizer
    test_decode_token = True
    # from_pretrained_kwargs = {"add_prefix_space": True}
    # test_seq2seq = False

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        tokenizer = LlamaTokenizer.from_pretrained("__internal_testing__/tiny-random-llama", **kwargs)
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = "lower newer"
        bpe_tokens = ["▁lower", "▁newer"]
        tokens = tokenizer.tokenize(text, add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [5224, 20687, 0]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_pretokenized_inputs(self, *args, **kwargs):
        pass

    def test_tokenizers_common_ids_setters(self, *args, **kwargs):
        pass

    def test_mask_output(self):
        pass

    def test_offsets_mapping(self):
        pass

    def test_offsets_mapping_with_unk(self):
        pass

    def test_special_tokens_mask(self):
        pass

    def test_special_tokens_mask_input_pairs(self):
        pass

    def test_padding_side_in_kwargs(self):
        tokenizer = self.get_tokenizer(padding_side="left")
        self.assertEqual(tokenizer.padding_side, "left")

        tokenizer = self.get_tokenizer(padding_side="right")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_truncation_side_in_kwargs(self):
        tokenizer = self.get_tokenizer(truncation_side="left")
        self.assertEqual(tokenizer.truncation_side, "left")

        tokenizer = self.get_tokenizer(truncation_side="right")
        self.assertEqual(tokenizer.truncation_side, "right")

    def test_add_tokens(self):
        tokenizer = self.get_tokenizer()

        vocab_size = len(tokenizer)
        self.assertEqual(tokenizer.add_tokens(""), 0)
        self.assertEqual(tokenizer.add_tokens("testoken"), 1)
        self.assertEqual(tokenizer.add_tokens(["testoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer), vocab_size + 3)

        self.assertEqual(tokenizer.add_special_tokens({}), 0)
        self.assertRaises(AssertionError, tokenizer.add_special_tokens, {"additional_special_tokens": "<testtoken1>"})
        self.assertEqual(tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
        self.assertEqual(
            tokenizer.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
        )
        self.assertIn("<testtoken3>", tokenizer.special_tokens_map["additional_special_tokens"])
        self.assertIsInstance(tokenizer.special_tokens_map["additional_special_tokens"], list)
        self.assertGreaterEqual(len(tokenizer.special_tokens_map["additional_special_tokens"]), 2)

        self.assertEqual(len(tokenizer), vocab_size + 6)

    def test_add_tokens_tokenizer(self):
        tokenizer = self.get_tokenizer()

        vocab_size = tokenizer.vocab_size
        all_size = len(tokenizer)

        self.assertNotEqual(vocab_size, 0)

        new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
        added_toks = tokenizer.add_tokens(new_toks)
        vocab_size_2 = tokenizer.vocab_size
        all_size_2 = len(tokenizer)

        self.assertNotEqual(vocab_size_2, 0)
        self.assertEqual(vocab_size, vocab_size_2)
        self.assertEqual(added_toks, len(new_toks))
        self.assertEqual(all_size_2, all_size + len(new_toks))

        tokens = tokenizer.encode(
            "aaaaa bbbbbb low cccccccccdddddddd l", return_token_type_ids=None, add_special_tokens=False
        )["input_ids"]
        self.assertGreaterEqual(len(tokens), 4)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

    def test_consecutive_unk_string(self):
        tokenizer = self.get_tokenizer(add_bos_token=False)

        tokens = [tokenizer.unk_token for _ in range(2)]
        string = tokenizer.convert_tokens_to_string(tokens)
        encoding = tokenizer(
            text=string,
            runcation=True,
            return_offsets_mapping=True,
        )
        self.assertEqual(len(encoding["input_ids"]), 2)
        self.assertEqual(len(encoding["offset_mapping"]), 2)

    def test_padding_if_pad_token_set_slow(self):
        tokenizer = self.get_tokenizer()

        # Simple input
        s = "This is a simple input"
        s2 = ["This is a simple input looooooooong", "This is a simple input"]
        p = ("This is a simple input", "This is a pair")

        pad_token_id = tokenizer.pad_token_id

        out_s = tokenizer(s, padding="max_length", max_length=30, return_tensors="np", return_attention_mask=True)
        out_s2 = tokenizer(s2, padding=True, truncate=True, return_tensors="np", return_attention_mask=True)
        out_p = tokenizer(*p, padding="max_length", max_length=60, return_tensors="np", return_attention_mask=True)

        # s
        # test single string max_length padding
        self.assertEqual(out_s["input_ids"].shape[-1], 30)
        self.assertTrue(pad_token_id in out_s["input_ids"])
        self.assertTrue(0 in out_s["attention_mask"])

        # s2
        # test automatic padding
        self.assertEqual(out_s2["input_ids"].shape[-1], 12)
        # long slice doesn't have padding
        self.assertFalse(pad_token_id in out_s2["input_ids"][0])
        self.assertFalse(0 in out_s2["attention_mask"][0])
        # short slice does have padding
        self.assertTrue(pad_token_id in out_s2["input_ids"][1])
        self.assertTrue(0 in out_s2["attention_mask"][1])

        # p
        # test single pair max_length padding
        self.assertEqual(out_p["input_ids"].shape[-1], 60)
        self.assertTrue(pad_token_id in out_p["input_ids"])
        self.assertTrue(0 in out_p["attention_mask"])

    def test_add_bos_token_slow(self):
        bos_token = "<s>"
        tokenizer = self.get_tokenizer()
        s = "This is a simple input"
        s2 = ["This is a simple input 1", "This is a simple input 2"]

        bos_token_id = tokenizer.bos_token_id

        out_s = tokenizer(s)
        out_s2 = tokenizer(s2)

        self.assertEqual(out_s.input_ids[0], bos_token_id)
        self.assertTrue(all(o[0] == bos_token_id for o in out_s2["input_ids"]))

        decode_s = tokenizer.decode(out_s["input_ids"])
        decode_s2 = tokenizer.batch_decode(out_s2["input_ids"])

        self.assertEqual(decode_s.split()[0][:3], bos_token)
        self.assertTrue(all(d.split()[0][:3] == bos_token for d in decode_s2))

    def test_pretrained_model_lists(self):
        # No max_model_input_sizes
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_resource_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_resource_files_map.values())[0]), 1)


@parameterized_class(
    ["model_name_or_path"],
    [
        ["facebook/llama-7b"],
        ["meta-llama/Meta-Llama-3.1-8B"],
        ["meta-llama/Llama-3.2-1B"],
        ["meta-llama/Llama-3.3-70B-Instruct"],
    ],
)
class LlamaTokenizationLoadTest(unittest.TestCase):
    model_name_or_path: str = None

    def get_tokenizer(self, **kwargs) -> PretrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    def test_load_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = "lower newer"
        tokenizer.tokenize(text, add_prefix_space=True)


class TikTokenIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    def test_tiktoken_llama(self):
        model_path = "hf-internal-testing/llama-3-8b-internal"
        subfolder = ""
        test_text = "This is a test sentence."
        test_tokens = [128000, 2028, 374, 264, 1296, 11914, 13, 128001]
        num_reserved_special_tokens = 256
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
            "<|python_tag|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, num_reserved_special_tokens - 5)]

        tiktoken_tokenizer = PretrainedTokenizerFast.from_pretrained(
            model_path,
            subfolder=subfolder,
            additional_special_tokens=special_tokens,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
        )
        tokens = tiktoken_tokenizer.tokenize("<|begin_of_text|> " + test_text)
        self.assertEqual(tokens[0], "<|begin_of_text|>")

        tiktoken_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder=subfolder,
            additional_special_tokens=special_tokens,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            add_bos_token=True,
            add_eos_token=True,
            use_fast=True,
        )
        self.assertTrue(isinstance(tiktoken_tokenizer, PretrainedTokenizerFast))
        tokens = tiktoken_tokenizer.encode(test_text, add_special_tokens=True)["input_ids"]
        self.assertEqual(tokens, test_tokens)
        tmpdirname = tempfile.mkdtemp()
        tiktoken_tokenizer.save_pretrained(tmpdirname)
        tokenizer_reload = AutoTokenizer.from_pretrained(tmpdirname, use_fast=True)
        self.assertTrue(isinstance(tokenizer_reload, PretrainedTokenizerFast))
        tokens = tokenizer_reload.encode(test_text, add_special_tokens=True)["input_ids"]
        self.assertEqual(tokens, test_tokens)
        shutil.rmtree(tmpdirname)
        tiktoken_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder=subfolder,
            additional_special_tokens=special_tokens,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            from_slow=True,
            add_bos_token=True,
            add_eos_token=True,
            use_fast=True,
        )
        tokens = tiktoken_tokenizer.encode(test_text, add_special_tokens=True)["input_ids"]
        self.assertEqual(tokens, test_tokens)
