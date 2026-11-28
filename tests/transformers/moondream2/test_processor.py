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
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from paddleformers.datasets.SFTDataset import Sequence
from paddleformers.datasets.collate import mm_collate_fn
from paddleformers.datasets.template.mm_plugin import Moondream2Plugin
from paddleformers.transformers import Moondream2Processor
from paddleformers.transformers.image_processing_utils import PaddleImageProcessingMixin
from paddleformers.transformers.moondream2.image_processor import Moondream2ImageProcessor


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


class Moondream2CropCollateTest(unittest.TestCase):
    """Crop padding and collation for samples that need different crop counts."""

    def setUp(self):
        # 8x8 crops with 1-patch overlap: a 1x1 image needs 2 crops in total
        # (global + one tile), while a wide image needs 3 (global + 2 tiles).
        self.image_processor = Moondream2ImageProcessor(
            crop_size=8, patch_size=2, max_crops=2, overlap_margin=1
        )
        self.processor = Moondream2Processor(
            image_processor=self.image_processor, tokenizer=_make_tokenizer()
        )
        self.plugin = Moondream2Plugin(image_token="<image>", video_token=None, audio_token=None)

    def test_mm_collate_fn_preserves_per_sample_crop_counts(self):
        small = self.plugin.get_mm_inputs([Image.new("RGB", (8, 8))], [], [], self.processor)
        wide = self.plugin.get_mm_inputs([Image.new("RGB", (40, 16))], [], [], self.processor)

        # ``max_crops`` bounds the local tiles, and the global crop is always
        # prepended, so every sample is padded to ``max_crops + 1``.
        expected_crops = self.image_processor.max_crops + 1
        self.assertEqual(small["pixel_values"].shape[1], expected_crops)
        self.assertEqual(wide["pixel_values"].shape[1], expected_crops)
        self.assertEqual(small["crop_counts"].tolist(), [2])
        self.assertEqual(wide["crop_counts"].tolist(), [3])

        common = {"position_ids": [0, 1], "labels": [-100, 1], "num_examples": 1}
        inputs = mm_collate_fn(
            batch=[
                [Sequence(token_ids=[1, 2], mm_inputs=dict(small), **common)],
                [Sequence(token_ids=[3, 4], mm_inputs=dict(wide), **common)],
            ],
            template=None,
            processor=None,
            tokenizer=SimpleNamespace(pad_token_id=0),
            training_args=SimpleNamespace(
                num_nextn_predict_layers=0,
                context_parallel_size=1,
                tensor_model_parallel_size=1,
                sequence_parallel=False,
                fp8=False,
            ),
            model_args=SimpleNamespace(use_attn_mask_startend_row_indices=True, use_global_causal_attn=False),
            max_seq_len=2,
            padding_free=False,
            model=None,
        )

        self.assertEqual(inputs["pixel_values"].shape, [2, expected_crops, 3, 8, 8])
        self.assertEqual(inputs["image_sizes"].tolist(), [[1, 1], [1, 2]])
        # The real crop count must reach the model; otherwise the padded zero
        # crops are encoded as if they were image content.
        self.assertEqual(inputs["crop_counts"].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
