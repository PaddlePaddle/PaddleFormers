# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import unittest

import numpy as np

from paddleformers.transformers import LlavaNextProcessor


class LlavaNextProcessorTest(unittest.TestCase):
    class DummyImageProcessor:
        model_input_names = ["pixel_values", "image_sizes"]
        image_grid_pinpoints = [[16, 16]]
        size = {"shortest_edge": 16}
        crop_size = {"height": 16, "width": 16}

        def __call__(self, images, **kwargs):
            self.kwargs = kwargs
            return {
                "pixel_values": [np.zeros((2, 3, 16, 16), dtype="float32")],
                "image_sizes": [[16, 16]],
            }

    class DummyTokenizer:
        init_kwargs = {}
        model_input_names = ["input_ids", "attention_mask"]
        image_token = "<image>"
        image_token_id = 99
        vocab = {"<image>": image_token_id, "look": 1}

        def convert_tokens_to_ids(self, token):
            return self.vocab[token]

        def __call__(self, text, **kwargs):
            self.kwargs = kwargs
            input_ids = []
            for sample in text:
                sample_ids = []
                index = 0
                special_tokens = sorted(self.vocab, key=len, reverse=True)
                while index < len(sample):
                    for token in special_tokens:
                        if sample.startswith(token, index):
                            sample_ids.append(self.vocab[token])
                            index += len(token)
                            break
                    else:
                        index += 1
                input_ids.append(sample_ids)
            return {"input_ids": input_ids, "attention_mask": [[1] * len(ids) for ids in input_ids]}

        def batch_decode(self, sequences, **kwargs):
            return [" ".join(str(token_id) for token_id in sequence) for sequence in sequences]

    def get_processor(self):
        processor = object.__new__(LlavaNextProcessor)
        processor.image_processor = self.DummyImageProcessor()
        processor.tokenizer = self.DummyTokenizer()
        processor.chat_template = None
        processor.patch_size = 4
        processor.num_additional_image_tokens = 0
        processor.vision_feature_select_strategy = "full"
        processor.image_token = "<image>"
        processor.image_token_id = processor.tokenizer.image_token_id
        return processor

    def test_processor_expands_image_placeholders_and_splits_kwargs(self):
        processor = self.get_processor()
        outputs = processor(
            text="look <image>",
            images=[np.zeros((16, 16, 3), dtype=np.uint8)],
            do_pad=False,
            padding=True,
            return_mm_token_type_ids=True,
        )

        self.assertEqual(outputs["input_ids"][0].count(processor.image_token_id), 36)
        self.assertTrue(processor.tokenizer.kwargs["padding"])
        self.assertNotIn("padding", processor.image_processor.kwargs)
        self.assertFalse(processor.image_processor.kwargs["do_pad"])
        self.assertIn("mm_token_type_ids", outputs)
        self.assertEqual(outputs["mm_token_type_ids"][0].count(1), 36)

    def test_get_num_multimodal_tokens(self):
        processor = self.get_processor()
        multimodal_data = processor._get_num_multimodal_tokens(image_sizes=[[16, 16]])

        self.assertEqual(multimodal_data["num_image_tokens"], [36])
        self.assertEqual(multimodal_data["num_image_patches"], [1])

    def test_post_process_image_text_to_text(self):
        processor = self.get_processor()
        decoded = processor.post_process_image_text_to_text([[1, 99, 2]], skip_special_tokens=True)

        self.assertIsInstance(decoded, list)


if __name__ == "__main__":
    unittest.main()
