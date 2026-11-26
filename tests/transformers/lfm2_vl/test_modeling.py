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

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import paddle
from PIL import Image
from safetensors.numpy import load_file, save_file

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
    def test_checkpoint_api_is_inherited_without_model_specific_loader(self):
        from paddleformers.transformers.model_utils import PretrainedModel

        self.assertIs(
            Lfm2VlForConditionalGeneration.from_pretrained.__func__, PretrainedModel.from_pretrained.__func__
        )
        self.assertIs(Lfm2VlForConditionalGeneration.save_pretrained, PretrainedModel.save_pretrained)

    def test_native_checkpoint_round_trip(self):
        model = Lfm2VlForConditionalGeneration(get_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, save_checkpoint_format="paddle", save_safetensors=False)
            loaded = AutoModelForConditionalGeneration.from_pretrained(
                directory, load_checkpoint_format="paddle", convert_from_hf=False
            ).eval()
            for name, tensor in model.state_dict().items():
                np.testing.assert_array_equal(loaded.state_dict()[name].numpy(), tensor.numpy(), err_msg=name)

    def test_multimodal_standard_aoa_checkpoint_round_trip(self):
        from pathlib import Path

        for tied in (False, True):
            for optional_parameters in (False, True):
                with self.subTest(
                    tied=tied, optional_parameters=optional_parameters
                ), tempfile.TemporaryDirectory() as directory:
                    config = get_config()
                    config.tie_word_embeddings = config.text_config.tie_word_embeddings = tied
                    config.projector_bias = config.projector_use_layernorm = optional_parameters
                    config.text_config.conv_bias = optional_parameters
                    reference = Lfm2VlForConditionalGeneration(config).eval()
                    linear_names = {
                        name + ".weight"
                        for name, layer in reference.named_sublayers()
                        if isinstance(layer, paddle.nn.Linear)
                    }
                    hf_weights = {
                        name: (tensor.numpy().T.copy() if name in linear_names else tensor.numpy())
                        for name, tensor in reference.state_dict().items()
                        if not (tied and name == "lm_head.weight")
                    }
                    config.architectures = ["Lfm2VlForConditionalGeneration"]
                    config.save_pretrained(directory)
                    # Exercise the real shared sharded loader, not a mocked converter.
                    names = list(hf_weights)
                    shards = (names[::2], names[1::2])
                    weight_map = {}
                    for index, keys in enumerate(shards):
                        filename = f"model-{index}.safetensors"
                        save_file(
                            {key: hf_weights[key] for key in keys},
                            os.path.join(directory, filename),
                            metadata={"format": "pt"},
                        )
                        weight_map.update({key: filename for key in keys})
                    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as file:
                        json.dump(
                            {
                                "metadata": {"total_size": sum(value.nbytes for value in hf_weights.values())},
                                "weight_map": weight_map,
                            },
                            file,
                        )
                    loaded = AutoModelForConditionalGeneration.from_pretrained(
                        directory, load_checkpoint_format="flex_checkpoint"
                    ).eval()
                    exported = os.path.join(directory, "exported")
                    loaded.save_pretrained(exported, save_checkpoint_format="flex_checkpoint")
                    exported_weights = {}
                    for shard in Path(exported).glob("*.safetensors"):
                        exported_weights.update(load_file(shard))
                    self.assertEqual(set(exported_weights), set(hf_weights))
                    for name, value in hf_weights.items():
                        np.testing.assert_array_equal(exported_weights[name], value, err_msg=name)
                    reloaded = AutoModelForConditionalGeneration.from_pretrained(
                        exported, load_checkpoint_format="flex_checkpoint"
                    ).eval()
                    inputs = self.get_inputs()
                    for actual in (loaded, reloaded):
                        self.assertEqual(set(actual.state_dict()), set(reference.state_dict()))
                        for name, tensor in reference.state_dict().items():
                            np.testing.assert_array_equal(
                                actual.state_dict()[name].numpy(), tensor.numpy(), err_msg=name
                            )
                        with paddle.no_grad():
                            np.testing.assert_allclose(
                                actual(**inputs).logits.numpy(), reference(**inputs).logits.numpy(), atol=1e-6
                            )

    def test_packed_attention_and_shortconv_match_independent_documents(self):
        from paddleformers.transformers.lfm2_vl.modeling_lfm2 import Lfm2ForCausalLM

        config = get_config().text_config
        config._attn_implementation = "eager"
        config.conv_bias = True
        model = Lfm2ForCausalLM(config).eval()
        first = paddle.to_tensor([[5, 6]])
        second = paddle.to_tensor([[8, 9, 10]])
        expected = paddle.concat([model(input_ids=first).logits, model(input_ids=second).logits], axis=1)
        expected.sum().backward()
        expected_grad = model.model.layers[0].conv.conv.weight.grad.clone()
        model.clear_gradients()
        ids = paddle.concat([first, second], axis=1)
        positions = paddle.to_tensor([[0, 1, 0, 1, 2]])
        indices = paddle.to_tensor([[[2, 2, 5, 5, 5]]], dtype="int32")
        allowed = np.zeros([1, 1, 5, 5], dtype=bool)
        allowed[:, :, :2, :2] = np.tril(np.ones([2, 2], dtype=bool))
        allowed[:, :, 2:, 2:] = np.tril(np.ones([3, 3], dtype=bool))
        for mask_args in ({"attn_mask_startend_row_indices": indices}, {"attention_mask": paddle.to_tensor(allowed)}):
            actual = model(input_ids=ids, position_ids=positions, **mask_args).logits
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
            actual.sum().backward()
            np.testing.assert_allclose(
                model.model.layers[0].conv.conv.weight.grad.numpy(), expected_grad.numpy(), atol=2e-5, rtol=2e-5
            )
            model.clear_gradients()
            changed = ids.clone()
            changed[:, :2] = paddle.to_tensor([[13, 14]])
            changed_logits = model(input_ids=changed, position_ids=positions, **mask_args).logits
            np.testing.assert_allclose(changed_logits[:, 2:].numpy(), actual[:, 2:].numpy(), atol=1e-6)

    def test_custom_rope_theta_and_head_dim(self):
        from paddleformers.transformers.lfm2_vl.modeling_lfm2 import Lfm2ForCausalLM

        values = get_config().text_config.to_dict()
        values["head_dim"] = 12
        values["rope_parameters"] = {"rope_type": "default", "rope_theta": 12345.0}
        config = Lfm2Config(**values)
        model = Lfm2ForCausalLM(config).eval()
        self.assertEqual(model.model.layers[1].self_attn.head_dim, 12)
        self.assertEqual(model.model.layers[1].self_attn.q_proj.weight.shape, [32, 48])
        expected = 1.0 / (12345.0 ** (np.arange(0, 12, 2, dtype="float32") / 12))
        np.testing.assert_allclose(model.model.rotary_emb.inv_freq.numpy(), expected, rtol=1e-6)
        self.assertEqual(model(input_ids=paddle.to_tensor([[5, 6, 8]])).logits.shape, [1, 3, 128])

    def test_mlp_calls_wrapped_down_projection(self):
        from paddleformers.transformers.lfm2_vl.modeling_lfm2 import Lfm2MLP

        mlp = Lfm2MLP(get_config().text_config)
        original = mlp.w2

        class WrappedProjection(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.base = original
                self.adapter = self.create_parameter([64, 32], default_initializer=paddle.nn.initializer.Constant(0.1))

            def forward(self, x):
                return self.base(x) + paddle.matmul(x, self.adapter)

        mlp.w2 = WrappedProjection()
        mlp(paddle.randn([1, 3, 32])).sum().backward()
        self.assertIsNotNone(mlp.w2.adapter.grad)
        self.assertGreater(float(paddle.abs(mlp.w2.adapter.grad).max()), 0)

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

    def test_native_multimodal_fields(self):
        model = Lfm2VlForConditionalGeneration(get_config())
        inputs = self.get_inputs()
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

    def test_text_and_vision_lfm2_lora_targets(self):
        from types import SimpleNamespace

        from paddleformers.cli.utils import get_lora_target_modules

        targets = [
            get_lora_target_modules(SimpleNamespace(config=SimpleNamespace(model_type=kind)))
            for kind in ("lfm2", "lfm2_vl")
        ]
        self.assertEqual(targets[0], targets[1])
        self.assertIn(".*in_proj.*", targets[0])

    def test_processor_rejects_image_token_without_image(self):
        processor = object.__new__(Lfm2VlProcessor)
        processor.image_token = "<image>"
        with self.assertRaisesRegex(ValueError, "image must be supplied"):
            processor(text="Describe <image>", images=None)


if __name__ == "__main__":
    unittest.main()
