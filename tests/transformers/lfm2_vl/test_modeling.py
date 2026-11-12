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

from paddleformers.transformers import (
    AutoModelForCausalLM,
    AutoModelForConditionalGeneration,
)
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
    def test_text_auto_model_loads_hf_linear_weights(self):
        self._check_text_hf_round_trip(tied=True)

    def test_text_auto_model_loads_untied_head_and_conv_bias(self):
        self._check_text_hf_round_trip(tied=False)

    def _check_text_hf_round_trip(self, tied):
        from paddleformers.transformers.lfm2_vl.modeling_lfm2 import Lfm2ForCausalLM

        config = get_config().text_config
        config.tie_word_embeddings = tied
        config.conv_bias = not tied
        config.architectures = ["Lfm2ForCausalLM"]
        reference = Lfm2ForCausalLM(config).eval()
        hf_weights = {}
        for name, tensor in reference.state_dict().items():
            array = tensor.numpy()
            if name.endswith(".weight") and name.split(".")[-2] in reference.transpose_weight_keys:
                array = array.T.copy()
            hf_weights[name] = array
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            save_file(hf_weights, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
            actual = AutoModelForCausalLM.from_pretrained(directory, convert_from_hf=True).eval()
            exported = os.path.join(directory, "exported")
            actual.save_pretrained(exported, save_checkpoint_format="flex_checkpoint")
            reloaded = AutoModelForCausalLM.from_pretrained(exported).eval()
        self.assertIsInstance(actual, Lfm2ForCausalLM)
        for name, expected in reference.state_dict().items():
            np.testing.assert_array_equal(actual.state_dict()[name].numpy(), expected.numpy(), err_msg=name)
            np.testing.assert_array_equal(reloaded.state_dict()[name].numpy(), expected.numpy(), err_msg=name)
        ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        with paddle.no_grad():
            np.testing.assert_allclose(actual(ids).logits.numpy(), reference(ids).logits.numpy(), atol=1e-6, rtol=1e-6)

    def test_text_layer_types_must_match_hidden_layers(self):
        with self.assertRaisesRegex(ValueError, "num_hidden_layers"):
            Lfm2Config(num_hidden_layers=2, layer_types=["full_attention"])

        with self.assertRaisesRegex(ValueError, "layer_types"):
            Lfm2Config(num_hidden_layers=1, layer_types=["unsupported"])

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

    def test_multimodal_recompute_forward_and_backward(self):
        config = get_config()
        config.text_config.recompute_granularity = "full"
        config.text_config.recompute_use_reentrant = False
        config.vision_config.recompute_granularity = "full"
        config.vision_config.recompute_use_reentrant = False
        model = Lfm2VlForConditionalGeneration(config)
        model.train()

        outputs = model(**self.get_inputs())
        outputs.loss.backward()

        self.assertIsNotNone(model.model.language_model.layers[0].feed_forward.w1.weight.grad)
        self.assertIsNotNone(model.model.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj.weight.grad)

    def test_text_only_forward(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        inputs = self.get_inputs()
        for key in ["pixel_values", "pixel_attention_mask", "spatial_shapes"]:
            inputs.pop(key)
        outputs = model(**inputs)
        self.assertEqual(list(outputs.logits.shape), [1, 4, 128])

    def test_common_collator_multimodal_aliases(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        inputs = self.get_inputs()
        inputs["feature_attention_mask"] = inputs.pop("pixel_attention_mask")
        inputs["image_grid_thw"] = inputs.pop("spatial_shapes")
        outputs = model(**inputs)
        self.assertEqual(list(outputs.logits.shape), [1, 4, 128])
        self.assertEqual(list(outputs.image_hidden_states.shape), [1, 32])

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

    def test_image_processor_loads_standard_and_nested_configs(self):
        settings = dict(
            do_image_splitting=False,
            min_image_tokens=4,
            max_image_tokens=4,
            encoder_patch_size=2,
            downsample_factor=2,
            tile_size=8,
        )
        for filename, data in (
            ("preprocessor_config.json", settings),
            ("processor_config.json", {"image_processor": settings}),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, filename)
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(data, file)
                processor = Lfm2VlImageProcessor.from_pretrained(directory)
                self.assertEqual(processor.encoder_patch_size, 2)
                # Exercise the shared Hub resolution path without network access.
                with mock.patch(
                    "paddleformers.transformers.image_processing_utils.resolve_file_path", return_value=path
                ) as resolve:
                    remote = Lfm2VlImageProcessor.from_pretrained("test/lfm2-vl", local_files_only=True)
                self.assertEqual(remote.tile_size, 8)
                self.assertTrue(resolve.call_args.kwargs["local_files_only"])
                # Overrides must also update derived padding sizes.
                overridden = Lfm2VlImageProcessor.from_pretrained(directory, min_image_tokens=9, max_image_tokens=9)
                output = overridden(Image.new("RGB", (12, 12)), return_tensors="pd")
                self.assertEqual(list(output.pixel_values.shape), [1, 36, 12])
                saved = os.path.join(directory, "saved")
                overridden.save_pretrained(saved)
                reloaded = Lfm2VlImageProcessor.from_pretrained(saved)
                np.testing.assert_array_equal(
                    reloaded(Image.new("RGB", (12, 12)), return_tensors="pd").pixel_values.numpy(),
                    output.pixel_values.numpy(),
                )

    def test_image_processor_missing_and_invalid_config_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No image processor"):
                Lfm2VlImageProcessor.from_pretrained(directory)
            with open(os.path.join(directory, "preprocessor_config.json"), "w", encoding="utf-8") as file:
                file.write("{invalid")
            with self.assertRaisesRegex(OSError, "not a valid JSON"):
                Lfm2VlImageProcessor.from_pretrained(directory)

    def test_checkpoint_resolution_errors_are_not_swallowed(self):
        from paddleformers.transformers.model_utils import PretrainedModel

        for error in (OSError("network failure"), ValueError("bad index"), FileNotFoundError("missing shard")):
            with self.subTest(error=error), mock.patch(
                "paddleformers.transformers.lfm2_vl.modeling._resolve_hf_checkpoint_dir", side_effect=error
            ), mock.patch.object(PretrainedModel, "from_pretrained") as fallback:
                with self.assertRaises(type(error)) as caught:
                    Lfm2VlForConditionalGeneration.from_pretrained("test/lfm2-vl")
                self.assertIs(caught.exception, error)
                fallback.assert_not_called()

    def test_invalid_hf_index_and_missing_shard_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.safetensors.index.json")
            for contents, error in (
                ("{broken", json.JSONDecodeError),
                (json.dumps({"weight_map": {"weight": "missing.safetensors"}}), FileNotFoundError),
            ):
                with self.subTest(contents=contents):
                    with open(path, "w", encoding="utf-8") as file:
                        file.write(contents)
                    with self.assertRaises(error):
                        Lfm2VlForConditionalGeneration.from_pretrained(directory, config=get_config())

    def test_native_loader_rejects_unused_kwargs(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            for options in ({}, {"config": get_config()}):
                with self.subTest(options=options), self.assertRaisesRegex(TypeError, "unexpected_option"):
                    Lfm2VlForConditionalGeneration.from_pretrained(directory, unexpected_option=True, **options)

    def test_real_hf_safetensors_conversion_and_kwargs(self):
        reference = Lfm2VlForConditionalGeneration(get_config()).eval()
        linear_names = {
            name + ".weight" for name, layer in reference.named_sublayers() if isinstance(layer, paddle.nn.Linear)
        }
        hf_weights = {
            name: (tensor.numpy().T.copy() if name in linear_names else tensor.numpy())
            for name, tensor in reference.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            subdir = os.path.join(directory, "weights")
            os.makedirs(subdir)
            reference.config.save_pretrained(subdir)
            save_file(hf_weights, os.path.join(subdir, "model.safetensors"), metadata={"format": "pt"})
            loaded = Lfm2VlForConditionalGeneration.from_pretrained(
                directory, subfolder="weights", convert_from_hf=True, local_files_only=True
            ).eval()
            for options in ({}, {"config": get_config()}):
                with self.subTest(options=options), self.assertRaisesRegex(TypeError, "unexpected_option"):
                    Lfm2VlForConditionalGeneration.from_pretrained(subdir, unexpected_option=True, **options)
            with self.assertRaisesRegex(ValueError, "convert_from_hf"):
                Lfm2VlForConditionalGeneration.from_pretrained(subdir, convert_from_hf=False)
        for name, tensor in reference.state_dict().items():
            np.testing.assert_array_equal(loaded.state_dict()[name].numpy(), tensor.numpy(), err_msg=name)
        inputs = self.get_inputs()
        with paddle.no_grad():
            np.testing.assert_allclose(loaded(**inputs).logits.numpy(), reference(**inputs).logits.numpy(), atol=1e-6)

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
