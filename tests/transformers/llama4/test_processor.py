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

import numpy as np
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from paddleformers.datasets.template.mm_plugin import Llama4Plugin
from paddleformers.transformers import Llama4ImageProcessor, Llama4Processor
from paddleformers.transformers.llama4.image_processor import (
    find_supported_resolutions,
    get_best_fit,
)

SPECIAL_TOKENS = [
    "<unk>",
    "<|image|>",
    "<|patch|>",
    "<|image_start|>",
    "<|image_end|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
]


def get_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({token: index for index, token in enumerate(SPECIAL_TOKENS)}, "<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        additional_special_tokens=SPECIAL_TOKENS[1:],
    )


class Llama4ImageProcessorTest(unittest.TestCase):
    def setUp(self):
        self.image_processor = Llama4ImageProcessor()

    def test_best_fit_matches_reference_layout(self):
        supported = find_supported_resolutions(16, 336, 336)
        self.assertEqual(get_best_fit((400, 800), supported), (672, 1008))
        self.assertEqual(get_best_fit((800, 400), supported), (1008, 672))

    def test_square_image_uses_one_tile(self):
        image = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        inputs = self.image_processor(image, return_tensors="pd")
        self.assertEqual(inputs["pixel_values"].shape, [1, 3, 336, 336])
        self.assertEqual(inputs["aspect_ratios"].numpy().tolist(), [[1, 1]])

    def test_tiled_image_appends_global_tile(self):
        image = Image.fromarray(np.zeros((400, 800, 3), dtype=np.uint8))
        inputs = self.image_processor(image, return_tensors="pd")
        self.assertEqual(inputs["pixel_values"].shape, [7, 3, 336, 336])
        self.assertEqual(inputs["aspect_ratios"].numpy().tolist(), [[2, 3]])


class Llama4ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = get_tokenizer()
        self.processor = Llama4Processor(Llama4ImageProcessor(), self.tokenizer)

    def test_processor_matches_visual_feature_count(self):
        image = Image.fromarray(np.zeros((400, 800, 3), dtype=np.uint8))
        inputs = self.processor(text="<|image|>", images=image, return_tensors="pd")
        patch_token_id = self.tokenizer.convert_tokens_to_ids("<|patch|>")
        num_patch_tokens = int((inputs["input_ids"] == patch_token_id).astype("int64").sum())
        self.assertEqual(num_patch_tokens, inputs["pixel_values"].shape[0] * 144)
        self.assertNotIn("aspect_ratios", inputs)

    def test_processor_rejects_placeholder_mismatch(self):
        image = Image.fromarray(np.zeros((336, 336, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "1 placeholders.*2 flattened images"):
            self.processor(text="<|image|>", images=[image, image])

    def test_multimodal_token_count_uses_image_aspect_ratio(self):
        multimodal_data = self.processor._get_num_multimodal_tokens(image_sizes=[(800, 400), (336, 336)])
        self.assertEqual(multimodal_data.num_image_patches, [7, 1])
        self.assertEqual(multimodal_data.num_image_tokens, [1008, 144])

    def test_plugin_expands_dataset_placeholder(self):
        plugin = Llama4Plugin(image_token="<|image|>", video_token=None, audio_token=None)
        messages = [{"role": "user", "content": "<image>What is shown?"}]
        processed = plugin.process_messages(
            messages,
            images=[object()],
            videos=[],
            audios=[],
            mm_inputs={"aspect_ratios": np.array([[2, 3]])},
            processor=self.processor,
        )
        self.assertEqual(processed[0]["content"].count("<|patch|>"), 7 * 144)
        self.assertNotIn("<image>", processed[0]["content"])
        self.assertEqual(messages[0]["content"], "<image>What is shown?")


if __name__ == "__main__":
    unittest.main()
