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
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from paddleformers.transformers import InternVL3Processor, PreTrainedTokenizerFast
from paddleformers.transformers.auto.processing import processor_class_from_name
from paddleformers.transformers.image_processing_utils import warp_base_image_processor


class InternVL3ProcessorTest(unittest.TestCase):
    def setUp(self):
        from transformers import CLIPImageProcessor

        vocab = {"[UNK]": 0, "[PAD]": 1, "<img>": 2, "</img>": 3, "<IMG_CONTEXT>": 4}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
            unk_token="[UNK]",
            pad_token="[PAD]",
            additional_special_tokens=["<img>", "</img>", "<IMG_CONTEXT>"],
        )
        image_processor = warp_base_image_processor(CLIPImageProcessor)(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
        )
        self.processor = InternVL3Processor(
            image_processor=image_processor, tokenizer=tokenizer, image_seq_length=4, max_dynamic_patch=2
        )
        self.square = Image.new("RGB", (16, 16), color="red")
        self.wide = Image.new("RGB", (32, 16), color="blue")

    def test_dynamic_patches_and_grid(self):
        output = self.processor.image_processor([self.square, self.wide], max_dynamic_patch=2, return_tensors="np")
        self.assertEqual(output["num_patches"], [1, 3])
        self.assertEqual(list(output["pixel_values"].shape), [4, 3, 16, 16])
        np.testing.assert_array_equal(output["image_grid_thw"], [[1, 1, 4], [3, 1, 4]])
        no_thumbnail = self.processor.image_processor(
            [self.wide], max_dynamic_patch=2, use_thumbnail=False, return_tensors="np"
        )
        self.assertEqual(no_thumbnail["num_patches"], [2])

    def test_placeholder_expansion_and_paddle_output(self):
        for return_tensors in ("np", "pd"):
            output = self.processor(
                images=[self.square, self.wide],
                text=["<image>", "<image>"],
                padding=True,
                max_dynamic_patch=2,
                return_tensors=return_tensors,
                return_mm_token_type_ids=True,
            )
            ids = np.asarray(output["input_ids"])
            self.assertEqual((ids[0] == self.processor.image_token_id).sum(), 4)
            self.assertEqual((ids[1] == self.processor.image_token_id).sum(), 12)
            self.assertEqual(list(output["pixel_values"].shape), [4, 3, 16, 16])
            np.testing.assert_array_equal(np.asarray(output["image_grid_thw"]), [[1, 1, 4], [3, 1, 4]])
            np.testing.assert_array_equal(
                np.asarray(output["mm_token_type_ids"]), np.isin(ids, self.processor.image_ids).astype("int64")
            )

    def test_placeholder_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.processor(images=[self.square], text="<image><image>")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.processor(images=[self.square, self.square], text="<image>")

    def test_auto_processor_resolution(self):
        self.assertIs(processor_class_from_name("InternVL3Processor"), InternVL3Processor)


if __name__ == "__main__":
    unittest.main()
