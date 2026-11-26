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
            )
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
                prompt = " ".join("<image>" for _ in images) + " describe the images"
                messages = [{"role": "user", "content": prompt}]
                mm_inputs = plugin._get_mm_inputs(paths, [], [], processor)
                expanded = plugin.process_messages(messages, paths, [], [], mm_inputs, processor)
                self.assertEqual(messages[0]["content"], prompt)
                self.assertNotIn("<image>", expanded[0]["content"])
                grids = mm_inputs["grids"].numpy().tolist()
                expected_atoms = sum(4 * (1 + r * c if r * c > 1 else 1) for r, c in grids)
                self.assertEqual(expanded[0]["content"].count("<IMG_ATOM>"), expected_atoms)
                expected = processor(images=images, text=prompt, return_tensors="np")
                actual = processor.tokenizer([expanded[0]["content"]])
                np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"])


if __name__ == "__main__":
    unittest.main()
