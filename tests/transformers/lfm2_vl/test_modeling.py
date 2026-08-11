# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import paddle
from PIL import Image
from safetensors.numpy import save_file

from paddleformers.transformers import AutoModelForConditionalGeneration
from paddleformers.transformers.lfm2_vl.configuration import (
    Lfm2Config,
    Lfm2VlConfig,
    Siglip2VisionConfig,
)
from paddleformers.transformers.lfm2_vl.image_processor import Lfm2VlImageProcessor
from paddleformers.transformers.lfm2_vl.modeling import Lfm2VlForConditionalGeneration
from paddleformers.transformers.lfm2_vl.processor import Lfm2VlProcessor


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

    def test_images_with_inputs_embeds_raise_clear_error(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        inputs = self.get_inputs()
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        with self.assertRaisesRegex(ValueError, "input_ids must be provided"):
            model.model(
                inputs_embeds=inputs_embeds,
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
            )

    def test_processor_rejects_image_token_without_image(self):
        processor = object.__new__(Lfm2VlProcessor)
        processor.image_token = "<image>"
        with self.assertRaisesRegex(ValueError, "image must be supplied"):
            processor(text="Describe <image>", images=None)

    def test_sharded_safetensors_iterator(self):
        from paddleformers.transformers.lfm2_vl.modeling import _iter_hf_tensors

        with tempfile.TemporaryDirectory() as model_dir:
            save_file({"first": np.ones([2], dtype="float32")}, os.path.join(model_dir, "part-1.safetensors"))
            save_file({"second": np.zeros([3], dtype="float32")}, os.path.join(model_dir, "part-2.safetensors"))
            with open(os.path.join(model_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "metadata": {"total_size": 20},
                        "weight_map": {"first": "part-1.safetensors", "second": "part-2.safetensors"},
                    },
                    file,
                )
            tensors = dict(_iter_hf_tensors(model_dir))
        np.testing.assert_array_equal(tensors["first"].numpy(), np.ones([2], dtype="float32"))
        np.testing.assert_array_equal(tensors["second"].numpy(), np.zeros([3], dtype="float32"))

    def test_official_model_id_uses_lfm2_vl_conversion(self):
        config = get_config()
        reference = Lfm2VlForConditionalGeneration(config)
        linear_weight_suffixes = (
            ".in_proj.weight",
            ".out_proj.weight",
            ".q_proj.weight",
            ".k_proj.weight",
            ".v_proj.weight",
            ".w1.weight",
            ".w2.weight",
            ".w3.weight",
            ".patch_embedding.weight",
            ".linear_1.weight",
            ".linear_2.weight",
            ".fc1.weight",
            ".fc2.weight",
        )
        hf_tensors = []
        for name, tensor in reference.state_dict().items():
            source = tensor.transpose([1, 0]) if name.endswith(linear_weight_suffixes) else tensor
            hf_tensors.append((name, source.clone()))
        with (
            mock.patch(
                "paddleformers.transformers.lfm2_vl.modeling._resolve_hf_checkpoint_dir",
                return_value="/tmp/lfm2-vl-hf-checkpoint",
            ),
            mock.patch(
                "paddleformers.transformers.lfm2_vl.modeling.Lfm2VlConfig.from_pretrained",
                return_value=(config, {}),
            ),
            mock.patch(
                "paddleformers.transformers.lfm2_vl.modeling._iter_hf_tensors",
                return_value=hf_tensors,
            ) as load_checkpoint,
        ):
            loaded = Lfm2VlForConditionalGeneration.from_pretrained("LiquidAI/LFM2.5-VL-450M")
        load_checkpoint.assert_called_once_with("/tmp/lfm2-vl-hf-checkpoint")
        for name, tensor in reference.state_dict().items():
            np.testing.assert_array_equal(tensor.numpy(), loaded.state_dict()[name].numpy())

    def test_default_save_pretrained_round_trip(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir)
            self.assertTrue(os.path.exists(os.path.join(save_dir, "model_state.pdparams")))
            loaded = Lfm2VlForConditionalGeneration.from_pretrained(save_dir)
        for name, tensor in model.state_dict().items():
            np.testing.assert_array_equal(tensor.numpy(), loaded.state_dict()[name].numpy())


if __name__ == "__main__":
    unittest.main()
