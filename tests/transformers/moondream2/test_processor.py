# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from unittest import mock

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from paddleformers.transformers import Moondream2Processor
from paddleformers.transformers.image_processing_utils import PaddleImageProcessingMixin


class _ImageProcessor(PaddleImageProcessingMixin):
    model_input_names = ["pixel_values"]

    def __init__(self):
        pass

    def __call__(self, images, **kwargs):
        return {"pixel_values": images}


def _make_tokenizer():
    tokenizer = mock.MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.init_kwargs = {}
    tokenizer.model_input_names = ["input_ids", "attention_mask"]

    def _tokenize(text, add_special_tokens=False, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[ord(char) for char in item] for item in text]}
        return {"input_ids": [ord(char) for char in text]}

    def _pad(encoded_inputs, padding=False, **kwargs):
        input_ids = encoded_inputs["input_ids"]
        max_length = max(len(ids) for ids in input_ids) if padding else None
        padded_ids = [ids + [0] * (max_length - len(ids)) if padding else ids for ids in input_ids]
        return {
            "input_ids": padded_ids,
            "attention_mask": [
                [1] * len(ids) + [0] * (len(padded) - len(ids)) for ids, padded in zip(input_ids, padded_ids)
            ],
        }

    tokenizer.side_effect = _tokenize
    tokenizer.pad.side_effect = _pad
    return tokenizer


class Moondream2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = Moondream2Processor(image_processor=_ImageProcessor(), tokenizer=_make_tokenizer())

    def test_task_prompt_batch(self):
        output = self.processor(images=["image 1", "image 2"], text=["q1", "q2"], task="query")

        self.assertEqual(output["input_ids"], [[1, 15381, 2, 113, 49, 3], [1, 15381, 2, 113, 50, 3]])
        self.assertEqual(output["attention_mask"], [[1] * 6, [1] * 6])
        self.assertEqual(output["pixel_values"], ["image 1", "image 2"])

    def test_task_prompt_single_text(self):
        output = self.processor(images=["image"], text="question", task="query")

        self.assertEqual(output["input_ids"], [[1, 15381, 2, 113, 117, 101, 115, 116, 105, 111, 110, 3]])
        self.assertEqual(output["attention_mask"], [[1] * 12])

    def test_task_prompt_batch_padding(self):
        output = self.processor(images=["image 1", "image 2"], text=["q", "question"], task="query", padding=True)

        self.assertEqual(
            output["input_ids"],
            [
                [1, 15381, 2, 113, 3, 0, 0, 0, 0, 0, 0, 0],
                [1, 15381, 2, 113, 117, 101, 115, 116, 105, 111, 110, 3],
            ],
        )
        self.assertEqual(output["attention_mask"], [[1] * 5 + [0] * 7, [1] * 12])

    def test_caption_prompt(self):
        output = self.processor(images=["image 1", "image 2"], task="caption")

        self.assertEqual(
            output["input_ids"],
            [[1, 32708, 2, 6382, 3], [1, 32708, 2, 6382, 3]],
        )

    def test_task_prompt_batch_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same batch size"):
            self.processor(images=["image 1", "image 2"], text=["q1"], task="query")

    def test_task_prompt_invalid_text(self):
        with self.assertRaisesRegex(ValueError, "non-empty list of strings"):
            self.processor(images=["image"], text=[], task="query")


if __name__ == "__main__":
    unittest.main()
