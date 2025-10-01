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

import unittest
from queue import Empty
from threading import Thread

import paddle

from paddleformers.generation import TextIteratorStreamer, TextStreamer
from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer
from paddleformers.transformers.utils import CaptureStd
from tests.testing_utils import slow
from tests.transformers.test_modeling_common import ids_tensor


class StreamerTester(unittest.TestCase):
    def get_inputs(self, model):
        input_ids = ids_tensor([1, 5], vocab_size=model.config.vocab_size, dtype="int64")
        attention_mask = paddle.ones_like(input_ids, dtype="bool")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decode_strategy": "greedy_search",
            "max_length": 10,
        }

    def test_text_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("Paddleformers/tiny-random-llama")
        model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-llama")
        model.config.eos_token_id = -1

        input_kwargs = self.get_inputs(model)
        greedy_ids = model.generate(**input_kwargs)
        greedy_text = tokenizer.decode(greedy_ids[0][0])

        with CaptureStd(out=True, err=False, replay=True) as cs:
            streamer = TextStreamer(tokenizer)
            model.generate(**input_kwargs, streamer=streamer)
        # The greedy text should be printed to stdout, except for the final "\n" in the streamer
        streamer_text = cs.out[:-1]

        self.assertEqual(streamer_text, greedy_text)

    def test_iterator_streamer_matches_non_streaming(self):
        tokenizer = AutoTokenizer.from_pretrained("Paddleformers/tiny-random-llama")
        model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-llama")
        model.config.eos_token_id = -1

        input_kwargs = self.get_inputs(model)
        greedy_ids = model.generate(**input_kwargs)
        greedy_text = tokenizer.decode(greedy_ids[0][0])

        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = {**input_kwargs, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        streamer_text = ""
        for new_text in streamer:
            streamer_text += new_text

        self.assertEqual(streamer_text, greedy_text)

    @slow
    def test_text_streamer_decode_kwargs(self):
        # Tests that we can pass `decode_kwargs` to the streamer to control how the tokens are decoded. Must be tested
        # with actual models -- the dummy models' tokenizers are not aligned with their models, and
        # `skip_special_tokens=True` has no effect on them
        tokenizer = AutoTokenizer.from_pretrained("Paddleformers/tiny-random-qwen3")
        model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-qwen3", convert_from_hf=True)
        model.config.eos_token_id = -1
        model.config.bos_token_id = 1

        input_ids = paddle.ones([1, 5], dtype="int64") * model.config.bos_token_id
        attention_mask = paddle.ones_like(input_ids, dtype="bool")

        with CaptureStd(out=True, err=False, replay=True) as cs:
            streamer = TextStreamer(tokenizer, skip_special_tokens=True)
            model.generate(input_ids, attention_mask=attention_mask, max_length=1, do_sample=False, streamer=streamer)

        # The prompt contains a special token, so the streamer should not print it. As such, the output text, when
        # re-tokenized, must only contain one token
        streamer_text = cs.out[:-1]  # Remove the final "\n"
        streamer_text_tokenized = tokenizer(streamer_text, return_tensors="pd")
        self.assertEqual(streamer_text_tokenized.input_ids.shape, [1, 1])

    def test_iterator_streamer_timeout(self):
        tokenizer = AutoTokenizer.from_pretrained("Paddleformers/tiny-random-llama")
        model = AutoModelForCausalLM.from_pretrained("Paddleformers/tiny-random-llama")
        model.config.eos_token_id = -1

        input_kwargs = self.get_inputs(model)
        streamer = TextIteratorStreamer(tokenizer, timeout=0.001)
        generation_kwargs = {**input_kwargs, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # The streamer will timeout after 0.001 seconds, so an exception will be raised
        with self.assertRaises(Empty):
            streamer_text = ""
            for new_text in streamer:
                streamer_text += new_text
