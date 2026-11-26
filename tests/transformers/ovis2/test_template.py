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
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from paddleformers.datasets.template.mm_plugin import Ovis2Plugin
from paddleformers.datasets.template.template import TEMPLATES
from paddleformers.transformers import Ovis2ImageProcessor, Ovis2Processor
from tests.transformers.ovis2.test_processor import DummyTokenizer


class Ovis2TemplateTest(unittest.TestCase):
    def test_registered_template_expands_images_like_processor(self):
        self._check_template_processor_agreement("<image>", prompt_image_token="<image>")

    def test_template_honors_custom_prompt_marker(self):
        self._check_template_processor_agreement("<photo>", prompt_image_token="<photo>")

    def test_template_honors_environment_prompt_marker(self):
        with patch("paddleformers.datasets.template.mm_plugin.IMAGE_PLACEHOLDER", "<env_image>"):
            self._check_template_processor_agreement("<image>", template_marker="<env_image>")

    def test_template_marker_does_not_mutate_custom_processor(self):
        with patch("paddleformers.datasets.template.mm_plugin.IMAGE_PLACEHOLDER", "<env_image>"):
            self._check_template_processor_agreement(
                "<photo>", template_marker="<env_image>", prompt_image_token="<photo>"
            )

    def test_late_environment_changes_do_not_change_template_or_processor(self):
        with patch.dict(os.environ, {"IMAGE_PLACEHOLDER": "<late_change>"}):
            self._check_template_processor_agreement("<image>")

    def _check_template_processor_agreement(self, marker, template_marker="<image>", **kwargs):
        plugin = TEMPLATES["ovis2"].mm_plugin
        self.assertIsInstance(plugin, Ovis2Plugin)
        self.assertEqual(plugin.image_token, "<IMG_ATOM>")
        with patch.object(Ovis2Processor, "check_argument_for_proper_class"):
            processor = Ovis2Processor(
                image_processor=Ovis2ImageProcessor(
                    size={"height": 16, "width": 16}, min_patches=1, max_patches=6, crop_to_patches=True
                ),
                tokenizer=DummyTokenizer(),
                image_seq_length=4,
                **kwargs,
            )
        self.assertEqual(processor.prompt_image_token, marker)
        processor.image_min_pixels = 1
        processor.image_max_pixels = 1024
        for sizes in [[(16, 16)], [(32, 16)], [(16, 16), (32, 16)]]:
            with self.subTest(sizes=sizes), tempfile.TemporaryDirectory() as tmpdir:
                images = [Image.new("RGB", size) for size in sizes]
                paths = []
                for index, image in enumerate(images):
                    path = Path(tmpdir) / f"image-{index}.png"
                    image.save(path)
                    paths.append(str(path))
                prompts = [marker + " describe the image" for _ in images]
                template_prompts = [template_marker + " describe the image" for _ in images]
                messages = [{"role": "system", "content": "Describe images."}] + [
                    {"role": "user", "content": prompt} for prompt in template_prompts
                ]
                mm_inputs = plugin._get_mm_inputs(paths, [], [], processor)
                expanded = plugin.process_messages(messages, paths, [], [], mm_inputs, processor)
                self.assertEqual([message["content"] for message in messages[1:]], template_prompts)
                actual_text = [message["content"] for message in expanded[1:]]
                self.assertTrue(all(template_marker not in text for text in actual_text))
                self.assertEqual(expanded[0], messages[0])
                grids = mm_inputs["grids"].numpy().tolist()
                expected_atoms = sum(4 * (1 + r * c if r * c > 1 else 1) for r, c in grids)
                self.assertEqual(sum(text.count("<IMG_ATOM>") for text in actual_text), expected_atoms)
                self.assertEqual(processor.prompt_image_token, marker)
                self.assertEqual(actual_text, processor.expand_image_placeholders(prompts, grids))
                self.assertEqual(
                    actual_text,
                    processor.expand_image_placeholders(template_prompts, grids, prompt_image_token=template_marker),
                )
                expected = processor(images=[[image] for image in images], text=prompts, return_tensors="np")
                actual = processor.tokenizer(actual_text)
                np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])
                with self.assertRaisesRegex(ValueError, "number of images"):
                    plugin.process_messages(messages, paths[:-1], [], [], mm_inputs, processor)


if __name__ == "__main__":
    unittest.main()
