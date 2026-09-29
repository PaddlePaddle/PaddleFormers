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

import re
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from paddleformers.transformers import Ovis2ImageProcessor, Ovis2Processor


class DummyTokenizer:
    init_kwargs = {}
    model_input_names = ["input_ids", "attention_mask"]

    token_to_id = {
        "<image>": 1,
        "<IMG_START>": 2,
        "<IMG_ATOM>": 3,
        "<IMG_GRID>": 4,
        "<IMG_COL>": 5,
        "<IMG_ROW>": 6,
        "<IMG_END>": 7,
    }
    special_token_pattern = re.compile(
        "(" + "|".join(re.escape(token) for token in token_to_id if token != "<image>") + ")"
    )

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, 8)

    def __call__(self, text, **kwargs):
        sequences = []
        for prompt in text:
            tokens = [token for token in self.special_token_pattern.findall(prompt)]
            sequences.append([self.token_to_id[token] for token in tokens] or [8])

        max_length = max(len(sequence) for sequence in sequences)
        input_ids = np.zeros((len(sequences), max_length), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = sequence
            attention_mask[index, : len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class Ovis2ProcessorTest(unittest.TestCase):
    def setUp(self):
        image_processor = Ovis2ImageProcessor(
            size={"height": 16, "width": 16},
            crop_to_patches=True,
            min_patches=1,
            max_patches=6,
        )
        with patch.object(Ovis2Processor, "check_argument_for_proper_class"):
            self.processor = Ovis2Processor(
                image_processor=image_processor,
                tokenizer=DummyTokenizer(),
                image_seq_length=4,
            )
        self.image = Image.fromarray(np.zeros((16, 32, 3), dtype=np.uint8))

    def test_processor_expands_image_placeholder_for_grid(self):
        inputs = self.processor(
            images=self.image,
            text="<image> describe the image",
            return_tensors="np",
        )

        self.assertEqual(set(inputs.keys()), {"input_ids", "attention_mask", "pixel_values"})
        self.assertEqual(inputs["pixel_values"].shape, (3, 3, 16, 16))
        self.assertEqual(int((inputs["input_ids"] == 3).sum()), 12)
        self.assertEqual(int((inputs["input_ids"] == 2).sum()), 1)
        self.assertEqual(int((inputs["input_ids"] == 4).sum()), 1)
        self.assertEqual(int((inputs["input_ids"] == 6).sum()), 1)
        self.assertEqual(int((inputs["input_ids"] == 5).sum()), 0)
        self.assertEqual(int((inputs["input_ids"] == 7).sum()), 1)

    def test_processor_builds_prompt_when_text_is_omitted(self):
        square_image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

        inputs = self.processor(images=square_image, return_tensors="np")

        self.assertEqual(inputs["pixel_values"].shape, (1, 3, 16, 16))
        self.assertEqual(int((inputs["input_ids"] == 3).sum()), 4)

    def test_text_only_input(self):
        inputs = self.processor(text="plain text", return_tensors="np")

        self.assertEqual(set(inputs.keys()), {"input_ids", "attention_mask"})
        self.assertEqual(inputs["input_ids"].shape, (1, 1))

    def test_processor_rejects_image_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "image tokens but received"):
            self.processor(images=self.image, text="no placeholder", return_tensors="np")

    def test_processor_rejects_mismatched_batches(self):
        with self.assertRaisesRegex(ValueError, "image batches but 2 text prompts"):
            self.processor(images=self.image, text=["<image>", "<image>"], return_tensors="np")

    def test_processor_validates_required_inputs_and_text_type(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.processor()
        with self.assertRaisesRegex(TypeError, "must be a string"):
            self.processor(text=[1, 2])

    def test_model_input_names_exclude_processor_only_grid(self):
        self.assertEqual(self.processor.model_input_names, ["input_ids", "attention_mask", "pixel_values"])


if __name__ == "__main__":
    unittest.main()

