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

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor

from paddleformers.datasets.template.mm_plugin import FastVLMPlugin
from paddleformers.datasets.template.template import TEMPLATES, FastVLMTemplate
from paddleformers.transformers.fastvlm.image_processor import FastVLMImageProcessor


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(text)] if text else []


class FastVLMTemplateTest(unittest.TestCase):
    def test_non_square_images_are_padded_before_clip_in_processor_and_plugin(self):
        processor = FastVLMImageProcessor(image_size=12, image_mean=[0.2, 0.3, 0.4])
        reference = CLIPImageProcessor(**processor.to_dict())
        plugin = FastVLMPlugin(image_token="<image>", video_token=None, audio_token=None)
        for width, height in ((10, 4), (4, 10), (10, 10)):
            with self.subTest(size=(width, height)), tempfile.TemporaryDirectory() as directory:
                image = Image.fromarray(np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3))
                size = max(width, height)
                square = Image.new("RGB", (size, size), tuple(int(x * 255) for x in processor.image_mean))
                square.paste(image, ((size - width) // 2, (size - height) // 2))
                expected = reference(square, return_tensors="np").pixel_values
                actual = processor(image, return_tensors="np").pixel_values
                np.testing.assert_array_equal(actual, expected)
                array = np.asarray(image).transpose(2, 0, 1)
                np.testing.assert_array_equal(
                    processor(array, input_data_format="channels_first", return_tensors="np").pixel_values,
                    expected,
                )
                path = os.path.join(directory, "image.png")
                image.save(path)
                output = plugin.get_mm_inputs(
                    [path], [], [], SimpleNamespace(image_processor=processor, tokenizer=object())
                )
                np.testing.assert_array_equal(output["pixel_values"].numpy(), expected)
                processor.save_pretrained(directory)
                restored = FastVLMImageProcessor.from_pretrained(directory)
                self.assertEqual(restored.image_aspect_ratio, "pad")
                np.testing.assert_array_equal(restored(image, return_tensors="np").pixel_values, expected)

    def test_registered_template_injects_negative_image_token(self):
        template = TEMPLATES["fastvlm"]
        self.assertIsInstance(template, FastVLMTemplate)
        pairs = template.encode_multiturn(
            DummyTokenizer(),
            [
                {"role": "user", "content": "<image>Describe it."},
                {"role": "assistant", "content": "An image."},
            ],
        )
        prompt_ids, _ = pairs[0]
        self.assertEqual(prompt_ids.count(-200), 1)

    def test_plugin_validates_single_image_and_masks_placeholder(self):
        plugin = FastVLMPlugin(image_token="<image>", video_token=None, audio_token=None)
        processor = SimpleNamespace(image_processor=object(), tokenizer=object())
        messages = [{"role": "user", "content": "<image>Describe it."}]
        self.assertIs(plugin.process_messages(messages, [object()], [], [], {}, processor), messages)
        self.assertEqual(plugin.process_tokens([1, -200, 2], processor), [1, -100, 2])

        with self.assertRaisesRegex(ValueError, "exactly one image"):
            plugin.process_messages(
                [{"role": "user", "content": "<image><image>"}],
                [object(), object()],
                [],
                [],
                {},
                processor,
            )


if __name__ == "__main__":
    unittest.main()
