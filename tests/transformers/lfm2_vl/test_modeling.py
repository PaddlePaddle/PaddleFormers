# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import unittest

import paddle
from PIL import Image

from paddleformers.transformers import AutoModelForConditionalGeneration
from paddleformers.transformers.lfm2_vl.configuration import (
    Lfm2Config,
    Lfm2VlConfig,
    Siglip2VisionConfig,
)
from paddleformers.transformers.lfm2_vl.image_processor import Lfm2VlImageProcessor
from paddleformers.transformers.lfm2_vl.modeling import Lfm2VlForConditionalGeneration


def get_config():
    text_config = Lfm2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv"],
        block_auto_adjust_ff_dim=False,
        pad_token_id=0,
    )
    vision_config = Siglip2VisionConfig(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_patches=4,
        patch_size=2,
    )
    return Lfm2VlConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=7,
        projector_hidden_size=32,
        downsample_factor=2,
    )


class Lfm2VlModelTest(unittest.TestCase):
    def get_inputs(self):
        input_ids = paddle.to_tensor([[1, 7, 2, 3]], dtype="int64")
        return {
            "input_ids": input_ids,
            "attention_mask": paddle.ones_like(input_ids),
            "pixel_values": paddle.randn([1, 4, 12]),
            "pixel_attention_mask": paddle.ones([1, 4], dtype="int64"),
            "spatial_shapes": paddle.to_tensor([[2, 2]], dtype="int64"),
            "labels": input_ids,
        }

    def test_multimodal_forward_and_backward(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        outputs = model(**self.get_inputs())
        self.assertEqual(list(outputs.logits.shape), [1, 4, 128])
        self.assertEqual(list(outputs.image_hidden_states.shape), [1, 32])
        self.assertIsNotNone(outputs.loss)
        outputs.loss.backward()

    def test_text_only_forward(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        inputs = self.get_inputs()
        for key in ["pixel_values", "pixel_attention_mask", "spatial_shapes"]:
            inputs.pop(key)
        outputs = model(**inputs)
        self.assertEqual(list(outputs.logits.shape), [1, 4, 128])

    def test_auto_model(self):
        model = AutoModelForConditionalGeneration.from_config(get_config())
        self.assertIsInstance(model, Lfm2VlForConditionalGeneration)

    def test_image_processor(self):
        processor = Lfm2VlImageProcessor(
            do_image_splitting=False,
            min_image_tokens=4,
            max_image_tokens=4,
            encoder_patch_size=2,
            downsample_factor=2,
            tile_size=8,
        )
        outputs = processor(Image.new("RGB", (8, 8), (20, 30, 40)), return_tensors="pd")
        self.assertEqual(list(outputs.pixel_values.shape), [1, 16, 12])
        self.assertEqual(list(outputs.spatial_shapes.shape), [1, 2])


if __name__ == "__main__":
    unittest.main()
