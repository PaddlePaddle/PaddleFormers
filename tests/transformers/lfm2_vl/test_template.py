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
from unittest.mock import Mock

import numpy as np
import paddle
from PIL import Image

from paddleformers.datasets.collate import mm_collate_fn
from paddleformers.datasets.SFTDataset import Sequence
from paddleformers.datasets.template import get_mm_plugin
from paddleformers.datasets.template.mm_plugin import Lfm2VlPlugin
from paddleformers.datasets.template.template import TEMPLATES
from paddleformers.transformers.lfm2_vl.image_processor import Lfm2VlImageProcessor
from paddleformers.transformers.lfm2_vl.processor import Lfm2VlProcessor

from .test_modeling import Lfm2VlForConditionalGeneration, get_config


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
        labels = [-100, 396, 397, 400, 0]
        self.assertEqual(
            plugin.process_tokens([11, 396, 397, 400, 0], processor, labels=labels),
            [-100, -100, -100, -100, 0],
        )
        self.assertEqual(labels, [-100, 396, 397, 400, 0])
        self.assertFalse(any(key in mm_inputs for key in ("image_rows", "image_cols", "image_sizes")))

    def test_plugin_provides_native_model_keys(self):
        processor = get_processor()
        plugin = get_mm_plugin(name="lfm2_vl", image_token="<image>")
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "image.png")
            Image.new("RGB", (8, 8), (20, 30, 40)).save(image_path)
            mm_inputs = plugin.get_mm_inputs([image_path], [], [], processor)

        self.assertIn("spatial_shapes", mm_inputs)
        self.assertIn("pixel_attention_mask", mm_inputs)
        self.assertNotIn("image_grid_thw", mm_inputs)
        self.assertNotIn("feature_attention_mask", mm_inputs)

    def test_processor_special_tokens_match_hf_default_and_override(self):
        processor = get_processor()
        for override in (None, False, True):
            processor.tokenizer = Mock(return_value={"input_ids": [[1, 2]]})
            kwargs = {} if override is None else {"add_special_tokens": override}
            processor(text="<|startoftext|>hello", **kwargs)
            self.assertEqual(processor.tokenizer.call_args.kwargs["add_special_tokens"], bool(override))
            self.assertEqual(processor.tokenizer.call_args.args[0], ["<|startoftext|>hello"])

    def test_plugin_collator_model_training_path(self):
        processor = get_processor()
        plugin = get_mm_plugin(name="lfm2_vl", image_token="<image>")
        samples = []
        for size in ((8, 8), (8, 4), (4, 8)):
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "image.png")
                Image.new("RGB", size).save(path)
                mm_inputs = plugin.get_mm_inputs([path], [], [], processor)
            plugin.process_messages(
                [{"role": "user", "content": "Describe <image>"}], ["image"], [], [], mm_inputs, processor
            )
            count = sum(int(h) // 2 * (int(w) // 2) for h, w in mm_inputs["spatial_shapes"].tolist())
            ids = [5] + [7] * count + [6]
            samples.append(
                Sequence(
                    token_ids=ids, labels=ids, position_ids=list(range(len(ids))), num_examples=1, mm_inputs=mm_inputs
                )
            )
        training = SimpleNamespace(
            num_nextn_predict_layers=0,
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            fp8=False,
        )
        model_args = SimpleNamespace(use_attn_mask_startend_row_indices=False, use_global_causal_attn=False)
        expected_shapes = paddle.concat([s.mm_inputs["spatial_shapes"] for s in samples]).numpy()
        model = Lfm2VlForConditionalGeneration(get_config()).eval()
        for packed in (False, True):
            with self.subTest(packed=packed):
                batch = [samples[:2], samples[2:]] if packed else [[s] for s in samples]
                inputs = mm_collate_fn(
                    batch,
                    template=None,
                    processor=None,
                    tokenizer=SimpleNamespace(pad_token_id=0),
                    training_args=training,
                    model_args=model_args,
                    max_seq_len=256,
                    padding_free=packed,
                    model=None,
                )
                np.testing.assert_array_equal(inputs["spatial_shapes"].numpy(), expected_shapes)
                self.assertNotIn("image_sizes", inputs)
                self.assertNotIn("feature_attention_mask", inputs)
                self.assertNotIn("image_grid_thw", inputs)
                # This check covers image-field plumbing, not text packing isolation.
                visual_inputs = {
                    key: inputs[key] for key in ("input_ids", "pixel_values", "pixel_attention_mask", "spatial_shapes")
                }
                self.assertEqual(model(**visual_inputs).logits.shape[0], inputs["input_ids"].shape[0])


if __name__ == "__main__":
    unittest.main()
