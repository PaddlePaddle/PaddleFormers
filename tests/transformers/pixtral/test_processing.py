# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import re
import unittest
from unittest.mock import patch

import numpy as np

from paddleformers.transformers.pixtral import PixtralProcessor


class FakeTokenizer:
    init_kwargs = {}
    model_input_names = ["input_ids", "attention_mask"]
    token_ids = {"[IMG]": 10, "[IMG_BREAK]": 11, "[IMG_END]": 12}

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]

    def __call__(self, texts, **kwargs):
        input_ids = []
        for text in texts:
            tokens = re.findall(r"\[IMG_BREAK\]|\[IMG_END\]|\[IMG\]", text)
            input_ids.append([self.token_ids[token] for token in tokens])
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }


class FakeImageProcessor:
    model_input_names = ["pixel_values", "image_sizes"]
    size = {"longest_edge": 1024}

    def __call__(self, images, patch_size=None, **kwargs):
        return {
            "pixel_values": np.zeros((1, 3, 32, 48), dtype=np.float32),
            "image_sizes": np.asarray([[32, 48]], dtype=np.int64),
        }


def make_processor():
    processor = object.__new__(PixtralProcessor)
    processor.image_processor = FakeImageProcessor()
    processor.tokenizer = FakeTokenizer()
    processor.patch_size = 16
    processor.spatial_merge_size = 1
    processor.image_token = "[IMG]"
    processor.image_break_token = "[IMG_BREAK]"
    processor.image_end_token = "[IMG_END]"
    processor.image_token_id = 10
    processor.image_break_token_id = 11
    processor.image_end_token_id = 12
    processor.image_ids = [10, 11, 12]
    return processor


class PixtralProcessorTest(unittest.TestCase):
    def test_expands_image_tokens(self):
        processor = make_processor()

        inputs = processor(
            images=[np.zeros((30, 47, 3), dtype=np.uint8)],
            text="Describe [IMG]",
            return_tensors=None,
            return_mm_token_type_ids=True,
        )

        self.assertEqual(inputs["input_ids"][0].count(processor.image_token_id), 6)
        self.assertEqual(inputs["input_ids"][0].count(processor.image_break_token_id), 1)
        self.assertEqual(inputs["input_ids"][0].count(processor.image_end_token_id), 1)
        self.assertEqual(inputs["mm_token_type_ids"][0], [1] * 8)

    def test_rejects_mismatched_image_count(self):
        processor = make_processor()

        with self.assertRaisesRegex(ValueError, "exceeds the number of provided images"):
            processor(
                images=[np.zeros((30, 47, 3), dtype=np.uint8)],
                text="[IMG] and [IMG]",
                return_tensors=None,
            )

    def test_get_num_multimodal_tokens(self):
        processor = make_processor()

        multimodal_data = processor._get_num_multimodal_tokens(image_sizes=[(30, 47)])

        self.assertEqual(multimodal_data.num_image_tokens, [8])
        self.assertEqual(multimodal_data.num_image_patches, [1])

    def test_from_pretrained_uses_processor_mixin_loading_path(self):
        processor = object()
        processor_dict = {"patch_size": 14, "spatial_merge_size": 2}
        arguments = [object(), object()]

        with (
            patch.object(PixtralProcessor, "get_processor_dict", return_value=(processor_dict, {})) as get_dict,
            patch.object(PixtralProcessor, "_get_arguments_from_pretrained", return_value=arguments) as get_args,
            patch.object(PixtralProcessor, "from_args_and_dict", return_value=processor) as from_dict,
        ):
            loaded = PixtralProcessor.from_pretrained(
                "organization/model",
                cache_dir="/tmp/cache",
                subfolder="vision",
                download_hub="modelscope",
                local_files_only=True,
            )

        self.assertIs(loaded, processor)
        get_dict.assert_called_once_with(
            "organization/model",
            cache_dir="/tmp/cache",
            subfolder="vision",
            download_hub="modelscope",
            local_files_only=True,
        )
        get_args.assert_called_once()
        from_dict.assert_called_once_with(arguments, processor_dict)


if __name__ == "__main__":
    unittest.main()
