# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest

import paddle
from PIL import Image

from paddleformers.transformers import AutoModelForConditionalGeneration
from paddleformers.transformers.fastvlm import (
    FastVLMConfig,
    FastVLMForConditionalGeneration,
    FastVLMImageProcessor,
)


class FastVLMModelTest(unittest.TestCase):
    def setUp(self):
        self.config = FastVLMConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            mm_vision_tower="mobileclip_l_64",
            mm_hidden_size=128,
            vision_config={"layers": [1, 1, 1, 1], "embed_dims": [8, 16, 32, 64]},
            tie_word_embeddings=False,
            use_cache=False,
            architectures=["LlavaQwen2ForCausalLM"],
        )

    def get_inputs(self):
        input_ids = paddle.to_tensor([[1, -200, 2, 3]], dtype="int64")
        labels = paddle.to_tensor([[1, -100, 2, 3]], dtype="int64")
        pixel_values = paddle.randn([1, 3, 64, 64])
        return input_ids, labels, pixel_values

    def test_multimodal_forward_and_loss(self):
        model = FastVLMForConditionalGeneration(self.config)
        input_ids, labels, pixel_values = self.get_inputs()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            return_dict=True,
        )
        self.assertEqual(outputs.logits.shape, [1, 7, self.config.vocab_size])
        self.assertIsNotNone(outputs.loss)

    def test_backward(self):
        model = FastVLMForConditionalGeneration(self.config)
        input_ids, labels, pixel_values = self.get_inputs()
        loss = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            return_dict=True,
        ).loss
        loss.backward()
        self.assertIsNotNone(model.model.mm_projector[0].weight.grad)

    def test_auto_model_from_config(self):
        model = AutoModelForConditionalGeneration.from_config(self.config)
        self.assertIsInstance(model, FastVLMForConditionalGeneration)

    def test_image_processor(self):
        outputs = FastVLMImageProcessor(image_size=64)(Image.new("RGB", (80, 60), (20, 30, 40)), return_tensors="pd")
        self.assertEqual(list(outputs.pixel_values.shape), [1, 3, 64, 64])


if __name__ == "__main__":
    unittest.main()
