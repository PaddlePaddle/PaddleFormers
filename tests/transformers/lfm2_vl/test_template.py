# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import os
import tempfile
import unittest

from PIL import Image

from paddleformers.datasets.template import get_mm_plugin
from paddleformers.datasets.template.mm_plugin import Lfm2VlPlugin
from paddleformers.datasets.template.template import TEMPLATES
from paddleformers.transformers.lfm2_vl.image_processor import Lfm2VlImageProcessor
from paddleformers.transformers.lfm2_vl.processor import Lfm2VlProcessor


class DummyTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.token_ids = {
            "<image>": 396,
            "<|image_start|>": 397,
            "<|image_end|>": 398,
            "<|img_thumbnail|>": 399,
            "<|img_row_1_col_1|>": 400,
            "<|img_row_2_col_1|>": 401,
        }

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.token_ids.get(tokens, self.unk_token_id)
        return [self.token_ids.get(token, self.unk_token_id) for token in tokens]


def get_processor():
    processor = object.__new__(Lfm2VlProcessor)
    processor.image_processor = Lfm2VlImageProcessor(
        encoder_patch_size=2,
        downsample_factor=2,
        tile_size=8,
        max_tiles=2,
        use_thumbnail=True,
    )
    processor.tokenizer = DummyTokenizer()
    processor.image_token = "<image>"
    processor.image_start_token = "<|image_start|>"
    processor.image_end_token = "<|image_end|>"
    processor.image_thumbnail_token = "<|img_thumbnail|>"
    return processor


class Lfm2VlTemplateTest(unittest.TestCase):
    def test_registered_template_uses_lfm2_vl_plugin(self):
        self.assertIsInstance(TEMPLATES["lfm2_vl"].mm_plugin, Lfm2VlPlugin)

    def test_plugin_expands_and_masks_visual_tokens(self):
        processor = get_processor()
        plugin = get_mm_plugin(name="lfm2_vl", image_token="<image>")
        messages = [{"role": "user", "content": "Describe <image>."}]
        mm_inputs = {"image_rows": [2], "image_cols": [1], "image_sizes": [[8, 4]]}

        processed = plugin.process_messages(messages, ["image.jpg"], [], [], mm_inputs, processor)

        self.assertEqual(messages[0]["content"], "Describe <image>.")
        self.assertEqual(processed[0]["content"].count("<image>"), 10)
        self.assertIn("<|img_row_1_col_1|>", processed[0]["content"])
        self.assertIn("<|img_thumbnail|>", processed[0]["content"])
        self.assertEqual(plugin.process_tokens([11, 396, 397, 400, 0], processor), [11, -100, -100, -100, 0])

    def test_plugin_provides_common_collator_aliases(self):
        processor = get_processor()
        plugin = get_mm_plugin(name="lfm2_vl", image_token="<image>")
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "image.png")
            Image.new("RGB", (8, 8), (20, 30, 40)).save(image_path)
            mm_inputs = plugin.get_mm_inputs([image_path], [], [], processor)

        self.assertIs(mm_inputs["image_grid_thw"], mm_inputs["spatial_shapes"])
        self.assertIs(mm_inputs["feature_attention_mask"], mm_inputs["pixel_attention_mask"])


if __name__ == "__main__":
    unittest.main()
